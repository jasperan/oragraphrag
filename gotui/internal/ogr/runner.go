package ogr

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"
)

// Line is one streamed line of subprocess output.
type Line struct {
	// Stream is "stdout" or "stderr".
	Stream string
	// Text is the line without its trailing newline.
	Text string
}

// IsStderr reports whether the line came from the error stream, so the UI can
// colour it as a warning without re-deriving it.
func (l Line) IsStderr() bool { return l.Stream == "stderr" }

// Runner executes the Python CLI as a subprocess.
//
// Bin+Prefix together are the program: for a plain install that is
// ("oragraphrag", nil); for a source checkout driven by uv it is
// ("uv", ["run", "oragraphrag"]).
type Runner struct {
	Bin     string
	Prefix  []string
	Env     []string
	Timeout time.Duration
	WorkDir string
}

// DefaultTimeout bounds a single run. graphify over a large tree is the slow
// case, but an unbounded wait would turn "Oracle is unreachable" into a hang.
const DefaultTimeout = 30 * time.Minute

// SourcesTimeout bounds the short read-only listing used for suggestions.
const SourcesTimeout = 60 * time.Second

// ErrNoCLI is returned when no way to invoke oragraphrag could be found.
var ErrNoCLI = errors.New("cannot find the oragraphrag CLI; pass --bin or set OGR_BIN")

// Resolve locates the CLI, in order: explicit flag, $OGR_BIN, PATH, a
// .venv/bin/oragraphrag found above the working directory or above this
// executable, then `uv run oragraphrag` scoped to a discovered project.
//
// The search starts from two places on purpose. A front-end is a normal binary
// and gets run from anywhere -- a shell in /tmp, a cron job, an editor task --
// so looking only above the working directory would miss the checkout the
// binary was built from and fall through to a `uv run` that cannot resolve the
// project either.
func Resolve(explicit string) (Runner, error) {
	r := Runner{Timeout: DefaultTimeout}

	if explicit = strings.TrimSpace(explicit); explicit != "" {
		r.Bin = explicit
		return r, nil
	}
	if env := strings.TrimSpace(os.Getenv("OGR_BIN")); env != "" {
		r.Bin = env
		return r, nil
	}
	if path, err := exec.LookPath("oragraphrag"); err == nil {
		r.Bin = path
		return r, nil
	}

	starts := searchRoots()
	for _, start := range starts {
		if venv := findVenvCLI(start); venv != "" {
			r.Bin = venv
			return r, nil
		}
	}

	if uv, err := exec.LookPath("uv"); err == nil {
		for _, start := range starts {
			if project := findProjectDir(start); project != "" {
				// uv resolves the project from its working directory, so pin it;
				// otherwise `uv run` fails with "Failed to spawn: oragraphrag".
				r.Bin = uv
				r.Prefix = []string{"run", "oragraphrag"}
				r.WorkDir = project
				return r, nil
			}
		}
	}
	return Runner{}, ErrNoCLI
}

// searchRoots is where resolution looks for the checkout: the working
// directory first, then the directory holding this executable, which during
// development is the gotui directory inside the repository.
func searchRoots() []string {
	roots := make([]string, 0, 2)
	if cwd, err := os.Getwd(); err == nil {
		roots = append(roots, cwd)
	}
	if exe, err := os.Executable(); err == nil {
		if resolved, err := filepath.EvalSymlinks(exe); err == nil {
			exe = resolved
		}
		roots = append(roots, filepath.Dir(exe))
	}
	return roots
}

// findVenvCLI walks up from start looking for the checkout's own installed
// entry point.
func findVenvCLI(start string) string {
	dir := start
	for {
		candidate := filepath.Join(dir, ".venv", "bin", "oragraphrag")
		if info, err := os.Stat(candidate); err == nil && !info.IsDir() {
			return candidate
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return ""
		}
		dir = parent
	}
}

// findProjectDir walks up from start looking for the repository root: a
// directory that has pyproject.toml and declares the oragraphrag entry point.
func findProjectDir(start string) string {
	dir := start
	for {
		candidate := filepath.Join(dir, "pyproject.toml")
		if data, err := os.ReadFile(candidate); err == nil && bytes.Contains(data, []byte("oragraphrag")) {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return ""
		}
		dir = parent
	}
}

// Describe renders the invocation for display. It never contains a secret,
// because secrets travel by environment, not argv.
func (r Runner) Describe(args []string) string {
	parts := append(append([]string{r.Bin}, r.Prefix...), args...)
	return strings.Join(parts, " ")
}

func (r Runner) command(ctx context.Context, args []string) *exec.Cmd {
	full := append(append([]string{}, r.Prefix...), args...)
	cmd := exec.CommandContext(ctx, r.Bin, full...)
	cmd.Env = append(os.Environ(), r.Env...)
	cmd.Dir = r.WorkDir

	// A cancellation must take down the whole tree. CommandContext alone kills
	// only the direct child, and any grandchild keeps the stdout pipe open, so
	// the reader goroutine blocks past the deadline and the UI appears frozen.
	setProcessGroup(cmd)
	cmd.Cancel = func() error { return killProcessGroup(cmd) }
	// Belt and braces: even if a signal cannot be delivered, Wait stops waiting
	// for the pipes after this delay instead of blocking indefinitely.
	cmd.WaitDelay = 3 * time.Second

	return cmd
}

// Stream runs the command and calls emit for every line of stdout and stderr
// as it arrives, so a long ingest shows progress instead of a frozen screen.
//
// emit is called from one goroutine per stream and is serialized internally, so
// an implementation may safely append to shared state without its own lock.
func (r Runner) Stream(ctx context.Context, args []string, emit func(Line)) error {
	if r.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, r.Timeout)
		defer cancel()
	}

	cmd := r.command(ctx, args)

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("stdout pipe: %w", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("stderr pipe: %w", err)
	}
	if emit == nil {
		emit = func(Line) {}
	}

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("cannot start %s: %w", r.Bin, err)
	}

	var wg sync.WaitGroup
	var mu sync.Mutex
	var stderrLines []string

	record := func(l Line) {
		if !l.IsStderr() {
			return
		}
		mu.Lock()
		// A handful is enough to explain a failure; the UI already has the full
		// stream, this is for the one-line summary.
		if len(stderrLines) < 5 {
			stderrLines = append(stderrLines, strings.TrimSpace(l.Text))
		}
		mu.Unlock()
	}

	// emit is invoked from BOTH pump goroutines below, so it must be serialized
	// here. A caller that accumulates lines -- appending to a slice, updating a
	// buffer, writing to a shared model -- would otherwise race with itself and can
	// silently lose or reorder lines. This is the only place that can do it
	// correctly, because a caller cannot know that two goroutines are involved.
	// (The internal stderr summary was already guarded; the caller's callback was
	// not, and `go test -race` flagged it under load.)
	var emitMu sync.Mutex
	each := func(l Line) {
		record(l)
		emitMu.Lock()
		defer emitMu.Unlock()
		emit(l)
	}
	wg.Add(2)
	go func() { defer wg.Done(); pump(stdout, "stdout", each) }()
	go func() { defer wg.Done(); pump(stderr, "stderr", each) }()
	wg.Wait()

	waitErr := cmd.Wait()
	if ctx.Err() == context.DeadlineExceeded {
		return fmt.Errorf("timed out after %s", r.Timeout)
	}
	if waitErr != nil {
		var exit *exec.ExitError
		if errors.As(waitErr, &exit) {
			// Quote the child's own message: an exit code alone sends the user
			// hunting, and the Python CLI explains itself on stderr.
			msg := fmt.Sprintf("%s exited with status %d", r.Describe(args), exit.ExitCode())
			mu.Lock()
			if len(stderrLines) > 0 {
				msg += ": " + firstNonEmpty(stderrLines)
			}
			mu.Unlock()
			return errors.New(msg)
		}
		return waitErr
	}
	return nil
}

// firstNonEmpty returns the first line that carries any content, so a leading
// blank line does not hide the real message.
func firstNonEmpty(lines []string) string {
	for _, l := range lines {
		if l != "" {
			return l
		}
	}
	return ""
}

// Capture runs the command and returns its combined output. It is for the
// short read-only calls (sources) where the UI wants a value, not a stream.
func (r Runner) Capture(ctx context.Context, args ...string) (string, error) {
	if r.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, r.Timeout)
		defer cancel()
	}
	cmd := r.command(ctx, args)
	out, err := cmd.CombinedOutput()
	if ctx.Err() == context.DeadlineExceeded {
		return string(out), fmt.Errorf("timed out after %s", r.Timeout)
	}
	if err != nil {
		var exit *exec.ExitError
		if errors.As(err, &exit) {
			return string(out), fmt.Errorf("oragraphrag %s exited with status %d", strings.Join(args, " "), exit.ExitCode())
		}
		return string(out), err
	}
	return string(out), nil
}

// pump forwards a pipe line by line, tolerating lines longer than the scanner
// default (Oracle errors and JSON dumps both exceed 64KiB).
func pump(reader io.Reader, stream string, emit func(Line)) {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for scanner.Scan() {
		emit(Line{Stream: stream, Text: scanner.Text()})
	}
}

// ParseSources extracts the source ids from `sources` output, ignoring the
// "(no sources)" placeholder and any warning lines.
//
// A source_id is not guaranteed to carry a "src_" prefix: it is whatever
// source_id_for_folder wrote, and a hand-seeded database can contain plain
// names (the development database here returns "default", "src_one",
// "src_two"). So the test is shape-based -- a single identifier-shaped token --
// rather than prefix-based, or real ids would be silently dropped from the
// suggestions.
func ParseSources(output string) []string {
	var ids []string
	for _, raw := range strings.Split(output, "\n") {
		line := strings.TrimSpace(stripANSI(raw))
		if line == "" || strings.HasPrefix(line, "(") {
			continue
		}
		if sourceIDPattern.MatchString(line) {
			ids = append(ids, line)
		}
	}
	return ids
}

// sourceIDPattern matches one identifier-shaped token with no whitespace, which
// is what a source_id is; it excludes prose warnings and log lines.
var sourceIDPattern = regexp.MustCompile(`^[A-Za-z0-9_.:-]+$`)

// stripANSI removes SGR escape sequences so typer/rich colouring does not leak
// into a Select option.
func stripANSI(s string) string {
	var b strings.Builder
	for i := 0; i < len(s); {
		if s[i] == 0x1b && i+1 < len(s) && s[i+1] == '[' {
			j := i + 2
			for j < len(s) && !(s[j] >= 0x40 && s[j] <= 0x7e) {
				j++
			}
			i = j + 1
			continue
		}
		b.WriteByte(s[i])
		i++
	}
	return b.String()
}
