package ui

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"charm.land/huh/v2"

	"github.com/jasperan/oragraphrag/gotui/internal/huhstyle"
	"github.com/jasperan/oragraphrag/gotui/internal/ogr"
)

// verbs lists the real CLI subcommands, in the order cli.py declares them, with
// the help text each one carries upstream.
func verbs() []struct {
	Verb string
	Desc string
} {
	return []struct {
		Verb string
		Desc string
	}{
		{"graphify", "Walk a folder and ingest it into the property graph"},
		{"query", "Answer a question from the indexed corpus"},
		{"sources", "List the distinct source ids currently in the graph"},
		{"export", "Export the accumulated graph as a JSONL fine-tuning corpus"},
		{"bench", "Run the benchmark harness across one or more baselines"},
		{"init-db", "Create tables, HNSW indexes, the property graph and axis rows"},
	}
}

// form is the shared wiring every form in this package needs: one theme, one
// accessibility decision, and the size of the pane it will be drawn into.
//
// Static Options only: huh's OptionsFunc is evaluated lazily and is never run
// in accessible mode, which would leave a screen-reader user with an empty
// list and an unanswerable prompt.
func form(groups ...*huh.Group) *huh.Form {
	return huh.NewForm(groups...).
		WithTheme(huh.ThemeFunc(huhstyle.Theme)).
		WithAccessible(huhstyle.Accessible())
}

// commandOptions is the dashboard's option list, extracted so the contract
// (one row per CLI verb, plus an explicit quit) is testable without depending
// on how huh scrolls a long list inside a short pane.
func commandOptions() []huh.Option[string] {
	options := make([]huh.Option[string], 0, len(verbs())+1)
	for _, v := range verbs() {
		options = append(options, huh.NewOption(fmt.Sprintf("%-9s %s", v.Verb, v.Desc), v.Verb))
	}
	return append(options, huh.NewOption("quit       Exit without running anything", ""))
}

// newCommandMenu is the dashboard: a single Select over the real CLI verbs,
// plus an explicit quit row so neither front-end needs to guess an exit key.
func newCommandMenu() (*huh.Form, *string) {
	chosen := new(string)
	options := commandOptions()

	f := form(huh.NewGroup(
		huh.NewSelect[string]().
			Title("What do you want to run?").
			Description("These drive the installed oragraphrag CLI; the answers become its arguments.").
			Options(options...).
			Value(chosen).
			Height(len(options)),
	))
	return f, chosen
}

// connectionAnswers is bound to the connection wizard's fields.
type connectionAnswers struct {
	Host     string
	Port     string
	Service  string
	Username string
	Password string
	Config   string
	// Remember persists the connection to the 0600 store when true.
	Remember bool
}

// connection renders the answers as the value the runner hands to the CLI.
//
// The DSN is rebuilt from the parts, and a blank port stays blank rather than
// becoming 1521: oracledb also accepts bare tnsnames aliases, and inventing a
// port would rewrite a working connection string.
func (a *connectionAnswers) connection() (ogr.Connection, error) {
	port := 0
	if raw := strings.TrimSpace(a.Port); raw != "" {
		p, err := strconv.Atoi(raw)
		if err != nil || p < 1 || p > 65535 {
			return ogr.Connection{}, fmt.Errorf("port must be a number between 1 and 65535")
		}
		port = p
	}

	conn := ogr.Connection{
		Username: strings.TrimSpace(a.Username),
		Password: a.Password,
		DSN:      ogr.DSN(a.Host, port, a.Service),
		Config:   strings.TrimSpace(a.Config),
	}
	if err := conn.Validate(); err != nil {
		return ogr.Connection{}, err
	}
	return conn, nil
}

// newConnectionWizard builds the Oracle connection form. existing prefills it,
// so a returning user only presses Enter.
func newConnectionWizard(existing ogr.Connection, storePath string, loaded bool) (*huh.Form, *connectionAnswers) {
	a := &connectionAnswers{Username: existing.Username, Config: existing.Config}
	if host, port, service, err := ogr.ParseDSN(existing.DSN); err == nil {
		a.Host, a.Service = host, service
		if port > 0 {
			a.Port = strconv.Itoa(port)
		}
	}
	// Only prefill the password when it came from the 0600 store we own: a
	// password typed for a different connection must not be silently reused.
	if loaded {
		a.Password = existing.Password
	}

	describeStore := "The password travels to the CLI in its environment, never on the command line."
	if storePath != "" {
		describeStore = fmt.Sprintf("Stored 0600 at %s, outside the repository.", storePath)
	}

	f := form(
		huh.NewGroup(
			huh.NewInput().
				Title("Oracle host").
				Description("Hostname, or a tnsnames alias. huh renders these as markdown, so quote any underscore.").
				Placeholder("localhost").
				Value(&a.Host).
				Validate(func(s string) error {
					if strings.TrimSpace(s) == "" {
						return fmt.Errorf("a host or alias is required")
					}
					return nil
				}),

			huh.NewInput().
				Title("Port").
				Description("Leave blank unless the listener is not on the default. Blank stays blank.").
				Placeholder("1521").
				CharLimit(5).
				Value(&a.Port).
				Validate(func(s string) error {
					if strings.TrimSpace(s) == "" {
						return nil
					}
					p, err := strconv.Atoi(strings.TrimSpace(s))
					if err != nil || p < 1 || p > 65535 {
						return fmt.Errorf("port must be a number between 1 and 65535")
					}
					return nil
				}),

			huh.NewInput().
				Title("Service name").
				Description("The PDB service, e.g. FREEPDB1. Leave blank for a plain alias.").
				Placeholder("FREEPDB1").
				Value(&a.Service),
		).Title("Oracle connection"),

		huh.NewGroup(
			huh.NewInput().
				Title("Username").
				Placeholder("ORAGRAPH").
				Value(&a.Username).
				Validate(func(s string) error {
					if strings.TrimSpace(s) == "" {
						return fmt.Errorf("a username is required")
					}
					return nil
				}),

			huh.NewInput().
				Title("Password").
				Description("Masked. Sent to the CLI in its environment, never on the command line.").
				EchoMode(huh.EchoModePassword).
				Value(&a.Password).
				Validate(func(s string) error {
					if s == "" {
						return fmt.Errorf("a password is required")
					}
					return nil
				}),
		).Title("Credentials"),

		huh.NewGroup(
			huh.NewInput().
				Title("Config file").
				Description("Optional path to config.yaml. Blank uses ./config.yaml, then defaults.").
				Placeholder("config.yaml").
				Value(&a.Config).
				Validate(func(s string) error {
					path := strings.TrimSpace(s)
					if path == "" {
						return nil
					}
					if _, err := os.Stat(path); err != nil {
						return fmt.Errorf("no such file: %s", path)
					}
					return nil
				}),

			huh.NewConfirm().
				Title("Remember this connection?").
				Description(describeStore).
				Affirmative("Yes").
				Negative("No, this session only").
				Value(&a.Remember),
		).Title("Config"),
	)
	return f, a
}

// commandAnswers holds the fields of whichever command form is open. Unused
// fields stay at their zero value, which is why one struct covers every verb.
type commandAnswers struct {
	// graphify
	Folder    string
	Reextract bool
	// query
	Question string
	DryRun   bool
	Source   string
	// export
	Out    string
	Format string
	// bench
	Suite   string
	Systems []string
	Limit   string
	// init-db
	Rebuild bool
}

// newCommandForm builds the form for one verb. sources has no options, so it
// gets a confirmation-only form; callers can rely on the returned answers
// being complete for the verb they asked for.
func newCommandForm(verb string, sources []string) (*huh.Form, *commandAnswers) {
	a := &commandAnswers{Format: ogr.ExportFormatFinetune, Systems: []string{ogr.SupportedSystems[0]}}

	// Source-scoped verbs accept a folder path or a literal src_<hex> id. The
	// suggestions are whatever `sources` last listed, so the field is useful
	// without forcing a two-step dance through the menu.
	sourceField := func() *huh.Input {
		f := huh.NewInput().
			Title("Scope to one source").
			Description("Optional. A folder path, or an id listed by the sources command.").
			Placeholder("(all sources)").
			Value(&a.Source)
		if len(sources) > 0 {
			f = f.Suggestions(sources)
		}
		return f
	}

	switch verb {
	case "graphify":
		return form(huh.NewGroup(
			huh.NewInput().
				Title("Folder to ingest").
				Description("Walked recursively; each file becomes spans in the graph.").
				Placeholder("/data/corpus").
				Value(&a.Folder).
				Validate(func(s string) error {
					path := strings.TrimSpace(s)
					if path == "" {
						return fmt.Errorf("a folder is required")
					}
					info, err := os.Stat(path)
					if err != nil {
						return fmt.Errorf("cannot read %s", path)
					}
					if !info.IsDir() {
						return fmt.Errorf("%s is not a folder", path)
					}
					return nil
				}),

			huh.NewConfirm().
				Title("Re-extract? (destructive)").
				Description("--reextract clears the ingest ledger first, so existing spans are reprocessed.").
				Affirmative("Yes, clear the ledger").
				Negative("No, incremental").
				Value(&a.Reextract),
		).Title("graphify")), a

	case "query":
		return form(
			huh.NewGroup(
				huh.NewText().
					Title("Question").
					Description("Answered from the indexed corpus.").
					Placeholder("How does the query-conditioned edge reweighting kernel work?").
					Lines(3).
					Value(&a.Question).
					Validate(func(s string) error {
						if strings.TrimSpace(s) == "" {
							return fmt.Errorf("a question is required")
						}
						return nil
					}),
			).Title("query"),

			huh.NewGroup(
				sourceField(),

				huh.NewConfirm().
					Title("Dry run?").
					Description("Prints what would be queried without touching the database or the LLM.").
					Affirmative("Yes, dry run").
					Negative("No, run it").
					Value(&a.DryRun),
			).Title("Scope"),
		), a

	case "export":
		return form(huh.NewGroup(
			huh.NewInput().
				Title("Output path").
				Description("JSONL file to write.").
				Placeholder("finetune.jsonl").
				Value(&a.Out).
				Validate(func(s string) error {
					if strings.TrimSpace(s) == "" {
						return fmt.Errorf("an output path is required")
					}
					return nil
				}),

			huh.NewSelect[string]().
				Title("Format").
				Description("Only finetune is implemented; anything else exits 2.").
				Options(huh.NewOption("finetune", ogr.ExportFormatFinetune)).
				Value(&a.Format),

			sourceField(),
		).Title("export")), a

	case "bench":
		options := make([]huh.Option[string], 0, len(ogr.SupportedSystems))
		for _, name := range ogr.SupportedSystems {
			label := name
			if name == "graphrag" || name == "lightrag" {
				label = name + " (placeholder stub upstream)"
			}
			options = append(options, huh.NewOption(label, name))
		}
		return form(huh.NewGroup(
			huh.NewInput().
				Title("Bench suite").
				Description("JSONL suite path.").
				Placeholder("benchmarks/suite.jsonl").
				Value(&a.Suite).
				Validate(func(s string) error {
					path := strings.TrimSpace(s)
					if path == "" {
						return fmt.Errorf("a suite path is required")
					}
					if _, err := os.Stat(path); err != nil {
						return fmt.Errorf("cannot read %s", path)
					}
					return nil
				}),

			huh.NewMultiSelect[string]().
				Title("Baselines").
				Description("The registry keys the runner knows.").
				Options(options...).
				Value(&a.Systems).
				Validate(func(v []string) error {
					if len(v) == 0 {
						return fmt.Errorf("pick at least one baseline")
					}
					return nil
				}),

			huh.NewInput().
				Title("Limit").
				Description("Optional cap on the number of questions, for a smoke run.").
				Placeholder("(no limit)").
				CharLimit(6).
				Value(&a.Limit).
				Validate(func(s string) error {
					raw := strings.TrimSpace(s)
					if raw == "" {
						return nil
					}
					n, err := strconv.Atoi(raw)
					if err != nil || n <= 0 {
						return fmt.Errorf("limit must be a positive integer")
					}
					return nil
				}),
		).Title("bench")), a

	case "init-db":
		return form(huh.NewGroup(
			huh.NewConfirm().
				Title("Drop and recreate the schema?").
				Description("--rebuild destroys the current tables, indexes and graph before recreating them.").
				Affirmative("Yes, rebuild").
				Negative("No, create if missing").
				Value(&a.Rebuild),
		).Title("init-db")), a

	case "sources":
		return form(huh.NewGroup(
			huh.NewNote().
				Title("sources").
				Description("Lists the distinct source ids in the property graph. No options."),
		).Title("sources")), a
	}

	// Unreachable: the menu only offers the verbs above.
	return form(huh.NewGroup(
		huh.NewNote().Title(verb).Description("Unknown command."),
	)), a
}

// buildCommand turns the form's answers into the argv the CLI will receive.
func buildCommand(verb string, a *commandAnswers, conn ogr.Connection) (ogr.Command, error) {
	var cmd ogr.Command
	switch verb {
	case "graphify":
		cmd = ogr.NewGraphify(conn.Config, a.Folder, a.Reextract)
	case "query":
		cmd = ogr.NewQuery(conn.Config, a.Question, a.DryRun, a.Source)
	case "export":
		cmd = ogr.NewExport(conn.Config, a.Out, a.Format, a.Source)
	case "bench":
		limit, err := parseLimit(a.Limit)
		if err != nil {
			return nil, err
		}
		cmd = ogr.NewBench(conn.Config, a.Suite, a.Systems, limit)
	case "init-db":
		cmd = ogr.NewInitDB(conn.Config, a.Rebuild)
	case "sources":
		cmd = ogr.NewSourcesCmd(conn.Config)
	default:
		return nil, fmt.Errorf("unknown command %q", verb)
	}
	if err := cmd.Validate(); err != nil {
		return nil, err
	}
	return cmd, nil
}

// parseLimit converts the optional limit field. A blank field means "no cap",
// which the CLI expresses by omitting --limit entirely.
func parseLimit(raw string) (*int, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil, nil
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n <= 0 {
		return nil, fmt.Errorf("limit must be a positive integer")
	}
	return &n, nil
}
