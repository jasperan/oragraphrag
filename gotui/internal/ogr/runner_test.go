package ogr

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestParseSourcesStripsColourAndPlaceholders(t *testing.T) {
	// typer/rich colour the output; the placeholder must not become an option,
	// and a source_id need not carry a "src_" prefix -- the development
	// database returns plain names like "default".
	output := "\x1b[36msrc_0a1b2c3d\x1b[0m\ndefault\nsrc_ffeeddcc\n[dim](no sources)[/dim]\n\nsome warning line\n"
	want := []string{"src_0a1b2c3d", "default", "src_ffeeddcc"}
	got := ParseSources(output)
	if len(got) != len(want) {
		t.Fatalf("ParseSources() = %q, want %q", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("ParseSources() = %q, want %q", got, want)
		}
	}
}

func TestParseSourcesOnEmptyOutput(t *testing.T) {
	if got := ParseSources("(no sources)\n"); len(got) != 0 {
		t.Fatalf("ParseSources() = %q, want no options", got)
	}
}

func TestResolvePrefersTheExplicitFlag(t *testing.T) {
	t.Setenv("OGR_BIN", "/from/env")
	r, err := Resolve("/from/flag")
	if err != nil {
		t.Fatalf("Resolve() = %v", err)
	}
	if r.Bin != "/from/flag" {
		t.Fatalf("Resolve() = %q, want the explicit flag to win", r.Bin)
	}
}

func TestResolvePrefersTheEnvOverride(t *testing.T) {
	t.Setenv("OGR_BIN", "/from/env")
	r, err := Resolve("")
	if err != nil {
		t.Fatalf("Resolve() = %v", err)
	}
	if r.Bin != "/from/env" {
		t.Fatalf("Resolve() = %q, want the env override", r.Bin)
	}
}

func TestDescribeIsSecretFree(t *testing.T) {
	r := Runner{Bin: "/usr/bin/oragraphrag", Prefix: []string{"run"}}
	got := r.Describe([]string{"query", "what is x?"})
	if !strings.HasPrefix(got, "/usr/bin/oragraphrag run query") {
		t.Fatalf("Describe() = %q, want the full invocation", got)
	}
	if strings.Contains(got, "ORACLE") {
		t.Fatalf("Describe() = %q, want no environment leakage", got)
	}
}

func TestStreamForwardsBothStreams(t *testing.T) {
	r := Runner{Bin: "/bin/sh", Timeout: 30 * time.Second}
	var lines []Line
	err := r.Stream(context.Background(), []string{"-c", "echo out1; echo err1 >&2; echo out2"}, func(l Line) {
		lines = append(lines, l)
	})
	if err != nil {
		t.Fatalf("Stream() = %v", err)
	}

	seen := map[string]string{}
	for _, l := range lines {
		seen[l.Text] = l.Stream
	}
	for text, stream := range map[string]string{"out1": "stdout", "out2": "stdout", "err1": "stderr"} {
		if seen[text] != stream {
			t.Fatalf("line %q arrived on %q, want %q (all lines: %+v)", text, seen[text], stream, lines)
		}
	}
}

func TestStreamReportsANonZeroExit(t *testing.T) {
	r := Runner{Bin: "/bin/sh", Timeout: 30 * time.Second}
	err := r.Stream(context.Background(), []string{"-c", "echo boom >&2; exit 3"}, nil)
	if err == nil {
		t.Fatal("Stream() = nil, want a non-zero exit to be reported")
	}
	if !strings.Contains(err.Error(), "status 3") {
		t.Fatalf("Stream() = %q, want it to name the exit status", err)
	}
}

// An unreachable Oracle must surface as a bounded error, not a hang: this is
// the failure mode the task calls out explicitly.
func TestStreamTimesOutInsteadOfHanging(t *testing.T) {
	r := Runner{Bin: "/bin/sh", Timeout: 300 * time.Millisecond}
	start := time.Now()
	err := r.Stream(context.Background(), []string{"-c", "sleep 30"}, nil)
	if err == nil {
		t.Fatal("Stream() = nil, want a timeout error")
	}
	if !strings.Contains(err.Error(), "timed out") {
		t.Fatalf("Stream() = %q, want a timeout error", err)
	}
	if elapsed := time.Since(start); elapsed > 10*time.Second {
		t.Fatalf("timeout took %s, want it bounded near the configured deadline", elapsed)
	}
}

func TestStreamCarriesTheEnvironment(t *testing.T) {
	r := Runner{Bin: "/bin/sh", Timeout: 30 * time.Second, Env: []string{"OGR__ORACLE__PASSWORD=from-env"}}
	var got string
	err := r.Stream(context.Background(), []string{"-c", "printf %s \"$OGR__ORACLE__PASSWORD\""}, func(l Line) {
		got += l.Text
	})
	if err != nil {
		t.Fatalf("Stream() = %v", err)
	}
	if got != "from-env" {
		t.Fatalf("child saw %q, want the injected environment value", got)
	}
}

func TestCaptureReturnsOutput(t *testing.T) {
	r := Runner{Bin: "/bin/sh", Timeout: 30 * time.Second}
	out, err := r.Capture(context.Background(), "-c", "echo hello")
	if err != nil {
		t.Fatalf("Capture() = %v", err)
	}
	if strings.TrimSpace(out) != "hello" {
		t.Fatalf("Capture() = %q, want %q", out, "hello")
	}
}

func TestFindVenvCLIRefusesToInventAPath(t *testing.T) {
	// No checkout above this temp dir, so resolution must report the failure
	// rather than returning a bogus path.
	dir := t.TempDir()
	old, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(dir); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chdir(old) })
	t.Setenv("OGR_BIN", "")
	t.Setenv("PATH", "")

	if got := findVenvCLI(dir); got != "" {
		t.Fatalf("findVenvCLI(%q) = %q, want \"\" with no checkout above", dir, got)
	}
	if got := findProjectDir(dir); got != "" {
		t.Fatalf("findProjectDir(%q) = %q, want \"\" with no project above", dir, got)
	}
	if _, err := Resolve(""); err == nil {
		t.Fatal("Resolve() = nil error, want ErrNoCLI when nothing is installed")
	}
}

// The front-end must work from any working directory, so resolution also looks
// above the executable. Without that, running the binary from /tmp silently
// fell through to `uv run oragraphrag`, which cannot resolve the project and
// failed with a bare exit status 2.
func TestFindVenvCLIFindsACheckoutAboveTheStart(t *testing.T) {
	root := t.TempDir()
	bin := filepath.Join(root, ".venv", "bin")
	if err := os.MkdirAll(bin, 0o755); err != nil {
		t.Fatal(err)
	}
	exe := filepath.Join(bin, "oragraphrag")
	if err := os.WriteFile(exe, []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	deep := filepath.Join(root, "a", "b", "c")
	if err := os.MkdirAll(deep, 0o755); err != nil {
		t.Fatal(err)
	}

	if got := findVenvCLI(deep); got != exe {
		t.Fatalf("findVenvCLI(%q) = %q, want %q", deep, got, exe)
	}
}

func TestFindProjectDirRequiresADeclaringPyproject(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "pyproject.toml"), []byte("name = \"other\"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := findProjectDir(root); got != "" {
		t.Fatalf("findProjectDir() = %q, want \"\" for a project that is not oragraphrag", got)
	}

	if err := os.WriteFile(filepath.Join(root, "pyproject.toml"),
		[]byte("name = \"oragraphrag\"\n[project.scripts]\noragraphrag = \"oragraphrag.cli:app\"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := findProjectDir(root); got != root {
		t.Fatalf("findProjectDir() = %q, want %q", got, root)
	}
}

// Resolution must never return a bare `uv run` without a working directory, or
// uv cannot resolve the project it is supposed to run.
func TestResolvePinsUvToTheProjectDirectory(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "pyproject.toml"),
		[]byte("name = \"oragraphrag\"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	binDir := t.TempDir()
	uv := filepath.Join(binDir, "uv")
	if err := os.WriteFile(uv, []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	old, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(root); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chdir(old) })

	t.Setenv("OGR_BIN", "")
	t.Setenv("PATH", binDir)

	r, err := Resolve("")
	if err != nil {
		t.Fatalf("Resolve() = %v", err)
	}
	if len(r.Prefix) == 0 {
		t.Fatalf("Resolve() = %+v, want the uv form", r)
	}
	if r.WorkDir != root {
		t.Fatalf("WorkDir = %q, want the discovered project %q", r.WorkDir, root)
	}
}

// A failing child must be reported with its own message, not only a status.
func TestStreamQuotesTheChildsStderr(t *testing.T) {
	r := Runner{Bin: "/bin/sh", Timeout: 30 * time.Second}
	err := r.Stream(context.Background(), []string{"-c", "echo 'error: Failed to spawn: oragraphrag' >&2; exit 2"}, nil)
	if err == nil {
		t.Fatal("Stream() = nil, want an error")
	}
	if !strings.Contains(err.Error(), "status 2") {
		t.Fatalf("Stream() = %q, want the exit status", err)
	}
	if !strings.Contains(err.Error(), "Failed to spawn") {
		t.Fatalf("Stream() = %q, want the child's own stderr quoted", err)
	}
}

// A live read-only call against the real engine, when one is reachable. This is
// the end-to-end proof that the front-end drives the real CLI, not a stub.
func TestRealCLISourcesListsRealIDs(t *testing.T) {
	r := resolveRealCLI(t)
	r.Timeout = 60 * time.Second

	out, err := r.Capture(context.Background(), NewSourcesCmd("").Args()...)
	if err != nil {
		t.Skipf("no reachable Oracle for a live sources call: %v", err)
	}

	ids := ParseSources(out)
	t.Logf("live sources: %q", ids)
	for _, id := range ids {
		if strings.ContainsAny(id, " \t") {
			t.Errorf("parsed %q as a source id, but it contains whitespace", id)
		}
	}
}

// resolveRealCLI finds the checkout's own oragraphrag, or skips. It is the
// bridge between the pure arg layer and the engine it must actually drive.
func resolveRealCLI(t *testing.T) Runner {
	t.Helper()
	root, err := filepath.Abs("../../..")
	if err != nil {
		t.Fatal(err)
	}
	bin := filepath.Join(root, ".venv", "bin", "oragraphrag")
	if info, err := os.Stat(bin); err != nil || info.IsDir() {
		t.Skipf("the checkout's oragraphrag CLI is not installed at %s", bin)
	}
	return Runner{Bin: bin, Timeout: 60 * time.Second}
}

// The strongest parity evidence: argv built by this package is accepted by the
// real Typer parser and produces engine-native output.
func TestRealCLIAcceptsGeneratedQueryArgs(t *testing.T) {
	r := resolveRealCLI(t)

	cmd := NewQuery("", "does the rewiring kernel converge?", true, "")
	if err := cmd.Validate(); err != nil {
		t.Fatalf("Validate() = %v", err)
	}

	var out strings.Builder
	var streamed []Line
	err := r.Stream(context.Background(), cmd.Args(), func(l Line) {
		streamed = append(streamed, l)
		out.WriteString(l.Text)
		out.WriteString("\n")
	})
	if err != nil {
		t.Fatalf("running the real CLI failed: %v (output: %s)", err, out.String())
	}

	text := out.String()
	if !strings.Contains(text, "dry run") {
		t.Fatalf("real CLI output = %q, want the --dry-run banner", text)
	}
	if !strings.Contains(text, "does the rewiring kernel converge?") {
		t.Fatalf("real CLI output = %q, want it to echo the question we passed", text)
	}
	if len(streamed) == 0 {
		t.Fatal("Stream() emitted no lines for the real CLI")
	}
}

func TestRealCLIRejectsNothingWeGenerate(t *testing.T) {
	r := resolveRealCLI(t)

	// --help proves every verb and flag we emit is in the real parser's table.
	help, err := r.Capture(context.Background(), "--help")
	if err != nil {
		t.Fatalf("--help failed: %v (%s)", err, help)
	}
	for _, verb := range []string{"init-db", "graphify", "query", "sources", "export", "bench"} {
		if !strings.Contains(help, verb) {
			t.Errorf("the real CLI does not know the verb %q; our Args() would be rejected", verb)
		}
	}
}

// A real `sources` run against an unreachable Oracle must produce a bounded
// error carrying the engine's own message, which is what the UI renders.
func TestRealCLIUnreachableOracleIsABoundedError(t *testing.T) {
	r := resolveRealCLI(t)
	r.Timeout = 45 * time.Second
	r.Env = []string{
		"OGR__ORACLE__USERNAME=ORAGRAPH",
		"OGR__ORACLE__PASSWORD=definitely-wrong",
		"OGR__ORACLE__DSN=127.0.0.1:1/FREEPDB1",
	}

	start := time.Now()
	out, err := r.Capture(context.Background(), NewSourcesCmd("").Args()...)
	if err == nil {
		t.Skipf("this environment can reach a DB on the configured DSN; output: %s", out)
	}
	if elapsed := time.Since(start); elapsed > r.Timeout {
		t.Fatalf("the failure took %s, want it under the %s bound", elapsed, r.Timeout)
	}
	if strings.TrimSpace(out) == "" {
		t.Error("the engine reported a failure but printed nothing; the UI would have no message to show")
	}
}
