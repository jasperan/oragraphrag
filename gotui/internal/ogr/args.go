// Package ogr is the pure logic layer of the OraGraphRAG Go front-end.
//
// Nothing in this package shells out or renders anything: it turns typed
// answers into the exact argv the Python CLI expects, and holds the Oracle
// connection parameters that must travel by environment rather than argv.
// Keeping it pure is what makes the front-end testable without Python.
package ogr

import (
	"fmt"
	"strconv"
	"strings"
)

// Connection is the Oracle connection material plus the optional config file.
//
// Password is deliberately the only secret here, and it never reaches argv.
// See Env: the Python side reads it from the documented OGR__* overrides.
type Connection struct {
	Username string
	Password string
	DSN      string
	// Config is an optional --config path handed to every subcommand.
	Config string
}

// DSN renders the "host:port/service" form the CLI expects.
//
// A zero port or empty service is omitted rather than defaulted, so a value
// parsed by ParseDSN round-trips byte for byte. That matters because oracledb
// also accepts bare tnsnames aliases, and rewriting "mydb_high" to
// "mydb_high:1521/" would turn a working connection string into a broken one.
func DSN(host string, port int, service string) string {
	s := strings.TrimSpace(host)
	if port > 0 {
		s += ":" + strconv.Itoa(port)
	}
	if service = strings.TrimSpace(service); service != "" {
		s += "/" + service
	}
	return s
}

// ParseDSN splits a connection string into its parts so the wizard can prefill
// itself. A port of 0 means "not specified": the caller must leave it blank
// rather than substituting 1521, or the round trip back through DSN would
// change the user's string.
func ParseDSN(dsn string) (host string, port int, service string, err error) {
	trimmed := strings.TrimSpace(dsn)
	if trimmed == "" {
		return "", 0, "", fmt.Errorf("dsn is empty")
	}

	// A tnsnames alias contains neither '/' nor ':' and is a valid DSN on its
	// own; return it as an opaque host with no port and no service.
	rest := trimmed
	if i := strings.Index(rest, "/"); i >= 0 {
		service = strings.TrimSpace(rest[i+1:])
		rest = rest[:i]
	}

	hostPart, portPart := rest, ""
	if i := strings.LastIndex(rest, ":"); i >= 0 {
		hostPart, portPart = rest[:i], rest[i+1:]
	}

	host = strings.TrimSpace(hostPart)
	if host == "" {
		return "", 0, "", fmt.Errorf("dsn %q has no host", dsn)
	}
	if portPart == "" {
		return host, 0, service, nil
	}
	p, convErr := strconv.Atoi(strings.TrimSpace(portPart))
	if convErr != nil || p <= 0 || p > 65535 {
		return "", 0, "", fmt.Errorf("dsn %q has an invalid port %q", dsn, portPart)
	}
	return host, p, service, nil
}

// Env renders the connection as the documented OGR__* environment overrides.
//
// This is the whole reason the password is safe: pydantic-settings loads
// env vars above config.yaml, so the subprocess authenticates correctly with
// the secret living only in the child's environment.
func (c Connection) Env() []string {
	out := make([]string, 0, 3)
	for _, kv := range []struct{ key, value string }{
		{"OGR__ORACLE__USERNAME", c.Username},
		{"OGR__ORACLE__PASSWORD", c.Password},
		{"OGR__ORACLE__DSN", c.DSN},
	} {
		if kv.value != "" {
			out = append(out, kv.key+"="+kv.value)
		}
	}
	return out
}

// Redacted returns a copy safe to render, log or echo into an error message.
func (c Connection) Redacted() Connection {
	safe := c
	if safe.Password != "" {
		safe.Password = "********"
	}
	return safe
}

// String never leaks the password.
func (c Connection) String() string {
	safe := c.Redacted()
	return fmt.Sprintf("%s@%s", safe.Username, safe.DSN)
}

// Validate checks the fields a connection cannot work without.
func (c Connection) Validate() error {
	if strings.TrimSpace(c.Username) == "" {
		return fmt.Errorf("username is required")
	}
	if strings.TrimSpace(c.Password) == "" {
		return fmt.Errorf("password is required")
	}
	if strings.TrimSpace(c.DSN) == "" {
		return fmt.Errorf("dsn is required")
	}
	if _, _, _, err := ParseDSN(c.DSN); err != nil {
		return err
	}
	return nil
}

// Command is something the front-end can invoke. Args are the CLI arguments
// that follow the program name.
type Command interface {
	// Args returns the exact argv tail for the Python CLI.
	Args() []string
	// Verb is the CLI subcommand, used for display.
	Verb() string
	// Validate reports why a command cannot run yet, if it cannot.
	Validate() error
}

// common carries the flags every subcommand in cli.py accepts.
type common struct {
	config string
}

func (c common) withConfig(args []string) []string {
	if strings.TrimSpace(c.config) == "" {
		return args
	}
	return append(args, "--config", c.config)
}

// SourcesCmd lists the distinct source ids in the property graph.
type SourcesCmd struct{ common }

// NewSourcesCmd builds the "sources" invocation.
func NewSourcesCmd(config string) SourcesCmd { return SourcesCmd{common{config}} }

func (SourcesCmd) Verb() string     { return "sources" }
func (SourcesCmd) Validate() error  { return nil }
func (c SourcesCmd) Args() []string { return c.withConfig([]string{"sources"}) }

// InitDB creates the schema. Rebuild is destructive: it drops what is there.
type InitDB struct {
	common
	Rebuild bool
}

// NewInitDB builds the "init-db" invocation.
func NewInitDB(config string, rebuild bool) InitDB {
	return InitDB{common{config}, rebuild}
}

func (InitDB) Verb() string { return "init-db" }
func (InitDB) Validate() error {
	return nil
}
func (c InitDB) Args() []string {
	args := []string{"init-db"}
	if c.Rebuild {
		args = append(args, "--rebuild")
	}
	return c.withConfig(args)
}

// Destructive reports whether this run drops existing data. The UI uses it to
// force an explicit confirmation before executing.
func (c InitDB) Destructive() bool { return c.Rebuild }

// Graphify walks a folder and ingests it into the property graph.
type Graphify struct {
	common
	Folder    string
	Reextract bool
}

// NewGraphify builds the "graphify" invocation.
func NewGraphify(config, folder string, reextract bool) Graphify {
	return Graphify{common{config}, folder, reextract}
}

func (Graphify) Verb() string { return "graphify" }
func (c Graphify) Validate() error {
	if strings.TrimSpace(c.Folder) == "" {
		return fmt.Errorf("folder is required")
	}
	return nil
}
func (c Graphify) Args() []string {
	args := []string{"graphify", c.Folder}
	if c.Reextract {
		args = append(args, "--reextract")
	}
	return c.withConfig(args)
}

// Destructive reports whether this run clears the ingest ledger first.
func (c Graphify) Destructive() bool { return c.Reextract }

// Query answers a question from the indexed corpus.
type Query struct {
	common
	Question string
	DryRun   bool
	Source   string
}

// NewQuery builds the "query" invocation.
func NewQuery(config, question string, dryRun bool, source string) Query {
	return Query{common{config}, question, dryRun, source}
}

func (Query) Verb() string { return "query" }
func (c Query) Validate() error {
	if strings.TrimSpace(c.Question) == "" {
		return fmt.Errorf("question is required")
	}
	return nil
}
func (c Query) Args() []string {
	args := []string{"query", c.Question}
	if c.DryRun {
		args = append(args, "--dry-run")
	}
	if s := strings.TrimSpace(c.Source); s != "" {
		args = append(args, "--source", s)
	}
	return c.withConfig(args)
}

// ExportFormatFinetune is the only format cli.py implements; anything else is
// rejected with exit code 2, so the front-end validates it up front.
const ExportFormatFinetune = "finetune"

// Export writes the accumulated graph as a JSONL fine-tuning corpus.
type Export struct {
	common
	Out    string
	Format string
	Source string
}

// NewExport builds the "export" invocation.
func NewExport(config, out, format, source string) Export {
	if strings.TrimSpace(format) == "" {
		format = ExportFormatFinetune
	}
	return Export{common{config}, out, format, source}
}

func (Export) Verb() string { return "export" }
func (c Export) Validate() error {
	if strings.TrimSpace(c.Out) == "" {
		return fmt.Errorf("output path is required")
	}
	if c.Format != ExportFormatFinetune {
		return fmt.Errorf("unsupported format %q (only %q is implemented)", c.Format, ExportFormatFinetune)
	}
	return nil
}
func (c Export) Args() []string {
	args := []string{"export", "--out", c.Out, "--format", c.Format}
	if s := strings.TrimSpace(c.Source); s != "" {
		args = append(args, "--source", s)
	}
	return c.withConfig(args)
}

// Bench runs the benchmark harness across one or more baselines.
type Bench struct {
	common
	Suite   string
	Systems []string
	// Limit is nil when the user did not cap the run.
	Limit *int
}

// NewBench builds the "bench" invocation. cli.py expects --systems as a
// comma-separated string, so the slice is joined with no spaces.
func NewBench(config, suite string, systems []string, limit *int) Bench {
	return Bench{common{config}, suite, systems, limit}
}

func (Bench) Verb() string { return "bench" }
func (c Bench) Validate() error {
	if strings.TrimSpace(c.Suite) == "" {
		return fmt.Errorf("bench suite path is required")
	}
	if len(c.Systems) == 0 {
		return fmt.Errorf("at least one system is required")
	}
	if c.Limit != nil && *c.Limit <= 0 {
		return fmt.Errorf("limit must be a positive integer")
	}
	return nil
}
func (c Bench) Args() []string {
	args := []string{"bench", "--suite", c.Suite}
	if len(c.Systems) > 0 {
		args = append(args, "--systems", strings.Join(c.Systems, ","))
	}
	if c.Limit != nil {
		args = append(args, "--limit", strconv.Itoa(*c.Limit))
	}
	return c.withConfig(args)
}

// SupportedSystems mirrors the REGISTRY keys in
// oragraphrag/bench/baselines/__init__.py, which is the authoritative set of
// --systems values. graphrag and lightrag are placeholder stubs upstream, so
// they are offered but labelled.
//
// Keep this list in step with that registry: an unknown name reaches the
// runner and fails there with a KeyError, which is a poor error to show a user.
var SupportedSystems = []string{"oragraphrag", "naive_rag", "graphrag", "lightrag"}
