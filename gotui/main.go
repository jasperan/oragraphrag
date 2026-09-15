// Command gotui is a Go front-end for the OraGraphRAG Python CLI.
//
// It is an additional way to run the same tool, not a replacement: every
// invocation ends up as an `oragraphrag` subprocess, so a Go user and a Python
// user get the same answers. The front-end's job is to make the CLI pleasant —
// a connection wizard, real validators, and live streamed output — and it does
// that through charm.land/huh forms.
//
// Three modes, chosen automatically:
//
//  1. flag-only  when a command is given on the command line, or stdin is not a
//     terminal. No form is constructed, so it works in CI and pipes.
//  2. accessible when ACCESSIBLE is set. huh implements accessible mode only in
//     Form.Run, so this mode runs each form standalone.
//  3. TUI        the default when a human is at a terminal.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	tea "charm.land/bubbletea/v2"

	"github.com/jasperan/oragraphrag/gotui/internal/huhstyle"
	"github.com/jasperan/oragraphrag/gotui/internal/ogr"
	"github.com/jasperan/oragraphrag/gotui/internal/ui"
)

// options are the flags. Every interactive field has a flag equivalent, which
// is what makes the non-interactive path a peer of the forms rather than an
// afterthought.
type options struct {
	command string
	// formVerb, when set, opens that command's form directly instead of the
	// dashboard. It is the interactive counterpart of --command, and it is what
	// makes a scripted PTY smoke test deterministic.
	formVerb string
	bin      string
	config   string
	creds    string

	// Connection, for the flag-only path. The password is deliberately NOT a
	// flag: argv is world-readable through /proc, so it comes from the 0600
	// store or OGR__ORACLE__PASSWORD instead.
	dsn  string
	user string

	// graphify
	folder    string
	reextract bool
	// query
	question string
	dryRun   bool
	source   string
	// export
	out    string
	format string
	// bench
	suite   string
	systems string
	limit   int
	// init-db
	rebuild bool

	yes bool
}

func main() {
	opts := parseFlags()
	if err := run(opts); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(exitCodeFor(err))
	}
}

// exitCodeFor passes a subprocess failure through unchanged so the front-end is
// a drop-in for scripts that check the CLI's status.
func exitCodeFor(err error) int {
	if code, ok := err.(exitCoder); ok {
		return code.ExitCode()
	}
	return 1
}

type exitCoder interface{ ExitCode() int }

func parseFlags() options {
	var o options
	fs := flag.NewFlagSet("gotui", flag.ContinueOnError)
	fs.Usage = func() {
		out := fs.Output()
		fmt.Fprintln(out, "gotui — Go front-end for the OraGraphRAG CLI")
		fmt.Fprintln(out, "\nWith no --command and a terminal on stdin, an interactive TUI starts.")
		fmt.Fprintln(out, "With --command (or a redirected stdin) the command runs without any prompt.")
		fmt.Fprintln(out, "\nFlags:")
		fs.PrintDefaults()
	}

	fs.StringVar(&o.command, "command", "", "command to run: graphify|query|sources|export|bench|init-db")
	fs.StringVar(&o.formVerb, "form", "", "open this command's form directly instead of the dashboard")
	fs.StringVar(&o.bin, "bin", "", "path to the oragraphrag executable (default: $OGR_BIN, PATH, ./.venv/bin, uv run)")
	fs.StringVar(&o.config, "config", "", "path to config.yaml passed through to the CLI")
	fs.StringVar(&o.creds, "creds", "", "credential file (default $OGR_CREDS_FILE or the user config dir)")
	fs.StringVar(&o.dsn, "dsn", "", "Oracle connection string, e.g. localhost:1521/FREEPDB1")
	fs.StringVar(&o.user, "user", "", "Oracle username")

	fs.StringVar(&o.folder, "folder", "", "graphify: folder to ingest")
	fs.BoolVar(&o.reextract, "reextract", false, "graphify: clear the ingest ledger first (destructive)")

	fs.StringVar(&o.question, "question", "", "query: the question to answer")
	fs.BoolVar(&o.dryRun, "dry-run", false, "query: print what would be queried without touching the DB or LLM")
	fs.StringVar(&o.source, "source", "", "query/export: scope to one source (folder path or src_<hex> id)")

	fs.StringVar(&o.out, "out", "", "export: output JSONL path")
	fs.StringVar(&o.format, "format", ogr.ExportFormatFinetune, "export: output format (only finetune)")

	fs.StringVar(&o.suite, "suite", "", "bench: path to the bench suite JSONL")
	fs.StringVar(&o.systems, "systems", strings.Join(ogr.SupportedSystems[:1], ","), "bench: comma-separated baseline names")
	fs.IntVar(&o.limit, "limit", 0, "bench: cap the number of questions (0 means no cap)")

	fs.BoolVar(&o.rebuild, "rebuild", false, "init-db: drop and recreate the schema (destructive)")
	fs.BoolVar(&o.yes, "yes", false, "skip the interactive confirmation on destructive commands")

	if err := fs.Parse(os.Args[1:]); err != nil {
		os.Exit(2)
	}
	return o
}

func run(o options) error {
	runner, err := ogr.Resolve(o.bin)
	if err != nil {
		return err
	}

	// Mode 1: flag-only. A command on the command line, or no terminal on
	// stdin, means never construct a form.
	if o.command != "" || !huhstyle.Interactive() {
		return runFlagOnly(o, runner)
	}

	conn, connOK, storePath, storeLoaded, note := resolveConnection(o)

	// Mode 2: accessible. Each form runs standalone because huh only implements
	// accessible rendering inside Form.Run.
	if huhstyle.Accessible() {
		return ui.RunAccessible(runner, conn, connOK, storePath, storeLoaded, os.Stdout)
	}

	// Mode 3: the TUI.
	model := ui.New(runner, conn, connOK, storePath, storeLoaded, note)
	if o.formVerb != "" {
		model = model.WithCommandForm(o.formVerb)
	}
	_, err = tea.NewProgram(model).Run()
	return err
}

// runFlagOnly executes one command with no prompts at all.
func runFlagOnly(o options, runner ogr.Runner) error {
	if o.command == "" {
		return fmt.Errorf("stdin is not a terminal, so an interactive form is impossible; " +
			"pass --command (for example --command query --question \"...\") and see --help")
	}

	cmd, err := commandFromFlags(o)
	if err != nil {
		return err
	}

	// Destructive commands must be confirmed non-interactively too, which is why
	// --yes exists and is checked here rather than in a form.
	if destructive, ok := cmd.(interface{ Destructive() bool }); ok && destructive.Destructive() && !o.yes {
		return fmt.Errorf("%s is destructive; re-run with --yes to proceed", cmd.Verb())
	}

	conn, _, _, _, _ := resolveConnection(o)
	run := runner
	run.Env = append(append([]string{}, run.Env...), conn.Env()...)

	fmt.Fprintln(os.Stderr, "$", run.Describe(cmd.Args()))
	if err := run.Stream(context.Background(), cmd.Args(), func(l ogr.Line) {
		if l.IsStderr() {
			fmt.Fprintln(os.Stderr, l.Text)
			return
		}
		fmt.Fprintln(os.Stdout, l.Text)
	}); err != nil {
		return fmt.Errorf("%w", err)
	}
	return nil
}

// commandFromFlags builds the command from the flags alone. It is the same
// constructor the forms use, so both paths produce identical argv.
func commandFromFlags(o options) (ogr.Command, error) {
	switch o.command {
	case "graphify":
		return validate(ogr.NewGraphify(o.config, o.folder, o.reextract))
	case "query":
		return validate(ogr.NewQuery(o.config, o.question, o.dryRun, o.source))
	case "export":
		return validate(ogr.NewExport(o.config, o.out, o.format, o.source))
	case "bench":
		var limit *int
		if o.limit > 0 {
			limit = &o.limit
		}
		systems := splitCSV(o.systems)
		if len(systems) == 0 {
			systems = ogr.SupportedSystems[:1]
		}
		return validate(ogr.NewBench(o.config, o.suite, systems, limit))
	case "init-db":
		return validate(ogr.NewInitDB(o.config, o.rebuild))
	case "sources":
		return validate(ogr.NewSourcesCmd(o.config))
	default:
		return nil, fmt.Errorf("unknown command %q; want one of graphify, query, sources, export, bench, init-db", o.command)
	}
}

// validate runs the same check the forms rely on, so the flag path cannot
// construct an invocation the interactive path would have rejected.
func validate(cmd ogr.Command) (ogr.Command, error) {
	if err := cmd.Validate(); err != nil {
		return nil, err
	}
	return cmd, nil
}

func splitCSV(raw string) []string {
	var out []string
	for _, part := range strings.Split(raw, ",") {
		if trimmed := strings.TrimSpace(part); trimmed != "" {
			out = append(out, trimmed)
		}
	}
	return out
}

// resolveConnection finds credentials, in order of precedence: explicit flags,
// the 0600 store, then nothing at all.
//
// Reporting nothing is a legitimate outcome: the CLI already reads
// ./config.yaml and its own defaults, so the front-end must not force a
// connection when the repository is already configured. Only the store-loaded
// case is allowed to prefill the wizard's password field.
func resolveConnection(o options) (conn ogr.Connection, ok bool, storePath string, loaded bool, note string) {
	path := o.creds
	if path == "" {
		path, _ = ogr.StorePath()
	}
	storePath = path

	if o.dsn != "" || o.user != "" {
		conn = ogr.Connection{DSN: o.dsn, Username: o.user, Config: o.config}
		// The password is not a flag. It comes from the environment or the
		// store, so it never appears in argv or in shell history.
		conn.Password = os.Getenv("OGR__ORACLE__PASSWORD")
		if conn.Password == "" {
			if stored, found, err := ogr.Load(path); err == nil && found {
				conn.Password = stored.Password
			}
		}
		if conn.Validate() != nil {
			// Incomplete flags mean there is no usable connection; fall through
			// to the wizard rather than failing a TUI start.
			return ogr.Connection{Config: o.config}, false, storePath, false, ""
		}
		return conn, true, storePath, false, ""
	}

	stored, found, err := ogr.Load(path)
	if err == nil && found {
		if o.config != "" {
			stored.Config = o.config
		}
		return stored, true, storePath, true, ogr.PermWarning(path)
	}

	return ogr.Connection{Config: o.config}, false, storePath, false, ""
}
