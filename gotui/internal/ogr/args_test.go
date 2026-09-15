package ogr

import (
	"reflect"
	"strings"
	"testing"
)

func TestCommandArgsMatchCLIParser(t *testing.T) {
	limit := 25
	cases := []struct {
		name string
		cmd  Command
		want []string
	}{
		{
			name: "sources",
			cmd:  NewSourcesCmd(""),
			want: []string{"sources"},
		},
		{
			name: "sources with config",
			cmd:  NewSourcesCmd("/tmp/ogr.yaml"),
			want: []string{"sources", "--config", "/tmp/ogr.yaml"},
		},
		{
			name: "init-db",
			cmd:  NewInitDB("", false),
			want: []string{"init-db"},
		},
		{
			name: "init-db rebuild",
			cmd:  NewInitDB("cfg.yaml", true),
			want: []string{"init-db", "--rebuild", "--config", "cfg.yaml"},
		},
		{
			name: "graphify",
			cmd:  NewGraphify("", "/data/corpus", false),
			want: []string{"graphify", "/data/corpus"},
		},
		{
			name: "graphify reextract",
			cmd:  NewGraphify("", "/data/corpus", true),
			want: []string{"graphify", "/data/corpus", "--reextract"},
		},
		{
			name: "query dry-run",
			cmd:  NewQuery("", "what is the kernel?", true, ""),
			want: []string{"query", "what is the kernel?", "--dry-run"},
		},
		{
			name: "query scoped to a source",
			cmd:  NewQuery("", "explain the retry", false, "src_ab12"),
			want: []string{"query", "explain the retry", "--source", "src_ab12"},
		},
		{
			name: "export defaults to finetune",
			cmd:  NewExport("", "/tmp/out.jsonl", "", ""),
			want: []string{"export", "--out", "/tmp/out.jsonl", "--format", "finetune"},
		},
		{
			name: "bench with limit",
			cmd:  NewBench("", "/suite.jsonl", []string{"oragraphrag", "naive"}, &limit),
			want: []string{"bench", "--suite", "/suite.jsonl", "--systems", "oragraphrag,naive", "--limit", "25"},
		},
		{
			name: "bench without limit",
			cmd:  NewBench("", "/suite.jsonl", []string{"oragraphrag"}, nil),
			want: []string{"bench", "--suite", "/suite.jsonl", "--systems", "oragraphrag"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.cmd.Args(); !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("Args() = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestValidateRejectsIncompleteCommands(t *testing.T) {
	limit := 0
	cases := []struct {
		name string
		cmd  Command
		want string
	}{
		{"graphify without a folder", NewGraphify("", "  ", false), "folder is required"},
		{"query without a question", NewQuery("", "", false, ""), "question is required"},
		{"export without an out path", NewExport("", "", "finetune", ""), "output path is required"},
		{"export with an unimplemented format", NewExport("", "/tmp/o.jsonl", "sharegpt", ""), "unsupported format"},
		{"bench without a suite", NewBench("", "", []string{"oragraphrag"}, nil), "bench suite path is required"},
		{"bench without systems", NewBench("", "/s.jsonl", nil, nil), "at least one system is required"},
		{"bench with a zero limit", NewBench("", "/s.jsonl", []string{"oragraphrag"}, &limit), "limit must be a positive integer"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.cmd.Validate()
			if err == nil {
				t.Fatalf("Validate() = nil, want an error containing %q", tc.want)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("Validate() = %q, want it to contain %q", err, tc.want)
			}
		})
	}
}

func TestValidateAcceptsCompleteCommands(t *testing.T) {
	limit := 5
	for _, cmd := range []Command{
		NewSourcesCmd(""),
		NewInitDB("", true),
		NewGraphify("", "/data", true),
		NewQuery("", "q", true, "src_1"),
		NewExport("", "/tmp/o.jsonl", "finetune", "src_1"),
		NewBench("", "/s.jsonl", []string{"oragraphrag"}, &limit),
	} {
		if err := cmd.Validate(); err != nil {
			t.Fatalf("%s: Validate() = %v, want nil", cmd.Verb(), err)
		}
	}
}

func TestDestructiveFlagsAreReported(t *testing.T) {
	if !NewInitDB("", true).Destructive() {
		t.Error("init-db --rebuild must be reported as destructive")
	}
	if NewInitDB("", false).Destructive() {
		t.Error("plain init-db must not be reported as destructive")
	}
	if !NewGraphify("", "/d", true).Destructive() {
		t.Error("graphify --reextract must be reported as destructive")
	}
	if NewGraphify("", "/d", false).Destructive() {
		t.Error("plain graphify must not be reported as destructive")
	}
}

func TestDSNRoundTrips(t *testing.T) {
	// A zero port means "not specified" and must stay unspecified: defaulting it
	// would rewrite the user's string on the way back through DSN().
	cases := []struct {
		dsn     string
		host    string
		port    int
		service string
	}{
		{"localhost:1521/FREEPDB1", "localhost", 1521, "FREEPDB1"},
		{"db.example.com:1522/mydb_high", "db.example.com", 1522, "mydb_high"},
		{"localhost/FREEPDB1", "localhost", 0, "FREEPDB1"},
		{"adb.example.com:1522/pdb1.adb.example.com", "adb.example.com", 1522, "pdb1.adb.example.com"},
		{"mydb_high", "mydb_high", 0, ""},
		{"localhost:1521", "localhost", 1521, ""},
	}
	for _, tc := range cases {
		t.Run(tc.dsn, func(t *testing.T) {
			host, port, service, err := ParseDSN(tc.dsn)
			if err != nil {
				t.Fatalf("ParseDSN(%q) = %v", tc.dsn, err)
			}
			if host != tc.host || port != tc.port || service != tc.service {
				t.Fatalf("ParseDSN(%q) = (%q,%d,%q), want (%q,%d,%q)",
					tc.dsn, host, port, service, tc.host, tc.port, tc.service)
			}
			if rebuilt := DSN(host, port, service); rebuilt != tc.dsn {
				t.Fatalf("DSN round trip = %q, want %q", rebuilt, tc.dsn)
			}
		})
	}
}

// oracledb accepts a bare tnsnames alias, and rewriting one to
// "alias:1521/" would break a working connection string. This pins that.
func TestDSNPreservesATnsnamesAlias(t *testing.T) {
	const alias = "mydb_high"
	host, port, service, err := ParseDSN(alias)
	if err != nil {
		t.Fatalf("ParseDSN(%q) = %v", alias, err)
	}
	if got := DSN(host, port, service); got != alias {
		t.Fatalf("DSN(ParseDSN(%q)) = %q, want the alias unchanged", alias, got)
	}
}

func TestParseDSNRejectsGarbage(t *testing.T) {
	for _, dsn := range []string{"", "   ", ":1521/x", "localhost:notaport/svc", "localhost:99999/svc"} {
		if _, _, _, err := ParseDSN(dsn); err == nil {
			t.Errorf("ParseDSN(%q) = nil error, want a rejection", dsn)
		}
	}
}

func TestConnectionEnvUsesDocumentedOverrides(t *testing.T) {
	conn := Connection{Username: "ORAGRAPH", Password: "s3cret", DSN: "localhost:1521/FREEPDB1"}
	env := conn.Env()
	want := map[string]string{
		"OGR__ORACLE__USERNAME": "ORAGRAPH",
		"OGR__ORACLE__PASSWORD": "s3cret",
		"OGR__ORACLE__DSN":      "localhost:1521/FREEPDB1",
	}
	if len(env) != len(want) {
		t.Fatalf("Env() = %q, want %d entries", env, len(want))
	}
	for _, kv := range env {
		key, value, ok := strings.Cut(kv, "=")
		if !ok {
			t.Fatalf("Env() entry %q is not key=value", kv)
		}
		if want[key] != value {
			t.Errorf("Env() %s = %q, want %q", key, value, want[key])
		}
	}
}

func TestConnectionEnvOmitsEmptyFields(t *testing.T) {
	if env := (Connection{Username: "u"}).Env(); len(env) != 1 {
		t.Fatalf("Env() = %q, want only the username", env)
	}
}

// The password must never be renderable: String() feeds every log line and
// every status bar in the TUI.
func TestConnectionNeverRendersAPassword(t *testing.T) {
	conn := Connection{Username: "ORAGRAPH", Password: "correct-horse", DSN: "localhost:1521/FREEPDB1"}

	for _, got := range []string{conn.String(), conn.Redacted().Password, conn.Redacted().String()} {
		if strings.Contains(got, "correct-horse") {
			t.Fatalf("rendered %q leaks the password", got)
		}
	}
	if !strings.Contains(conn.Redacted().String(), "ORAGRAPH") {
		t.Error("redaction should keep the username, which is not a secret")
	}
}

func TestConnectionValidate(t *testing.T) {
	cases := []struct {
		name string
		conn Connection
		want string
	}{
		{"no username", Connection{Password: "p", DSN: "h:1/s"}, "username is required"},
		{"no password", Connection{Username: "u", DSN: "h:1/s"}, "password is required"},
		{"no dsn", Connection{Username: "u", Password: "p"}, "dsn is required"},
		{"malformed dsn", Connection{Username: "u", Password: "p", DSN: ":x/y"}, "dsn"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.conn.Validate()
			if err == nil {
				t.Fatalf("Validate() = nil, want an error containing %q", tc.want)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("Validate() = %q, want it to contain %q", err, tc.want)
			}
		})
	}

	good := Connection{Username: "u", Password: "p", DSN: "localhost:1521/FREEPDB1"}
	if err := good.Validate(); err != nil {
		t.Fatalf("Validate() = %v, want nil", err)
	}
}
