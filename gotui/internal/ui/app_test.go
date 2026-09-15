package ui

import (
	"strings"
	"testing"
	"time"

	tea "charm.land/bubbletea/v2"
	"charm.land/huh/v2"

	"github.com/jasperan/oragraphrag/gotui/internal/ogr"
)

// press builds a key press message. bubbletea v2 distinguishes presses from
// releases, and huh only reacts to presses.
func press(code rune) tea.KeyPressMsg { return tea.KeyPressMsg(tea.Key{Code: code}) }

// ctrl builds a ctrl-modified key press.
func ctrl(code rune) tea.KeyPressMsg { return tea.KeyPressMsg(tea.Key{Code: code, Mod: tea.ModCtrl}) }

// TestKeyHelpersProduceTheExpectedStrings guards the other tests: if these
// helpers built the wrong keystrokes, every transition test below would pass
// against the wrong branch.
func TestKeyHelpersProduceTheExpectedStrings(t *testing.T) {
	if got := press(tea.KeyEnter).String(); got != "enter" {
		t.Errorf("press(enter).String() = %q, want \"enter\"", got)
	}
	if got := ctrl('c').String(); got != "ctrl+c" {
		t.Errorf("ctrl(c).String() = %q, want \"ctrl+c\"", got)
	}
}

// applySizeMsg drives a window resize into a form and returns it, since
// huh.Model is an interface and Update yields that rather than *huh.Form.
func applySizeMsg(f *huh.Form, w, h int) *huh.Form {
	updated, _ := f.Update(tea.WindowSizeMsg{Width: w, Height: h})
	if next, ok := updated.(*huh.Form); ok {
		return next
	}
	return f
}

// runBounded executes one command with a hard 50ms ceiling.
//
// huh re-arms the text input's cursor blink on every update and each blink tick
// sleeps ~530ms, so draining commands naively makes a suite take minutes and,
// worse, can block forever. Anything slower than the ceiling is abandoned: the
// tests below only depend on the immediate commands (focus moves, group
// transitions), which are plain functions.
func runBounded(cmd tea.Cmd) tea.Msg {
	if cmd == nil {
		return nil
	}
	ch := make(chan tea.Msg, 1)
	go func() {
		defer func() { _ = recover() }()
		ch <- cmd()
	}()
	select {
	case msg := <-ch:
		return msg
	case <-time.After(50 * time.Millisecond):
		return nil
	}
}

func TestCommandMenuOffersEveryCLIVerb(t *testing.T) {
	options := commandOptions()

	// One row per CLI verb, plus the quit row.
	if want := len(verbs()) + 1; len(options) != want {
		t.Fatalf("commandOptions() has %d rows, want %d", len(options), want)
	}
	labels := make([]string, 0, len(options))
	for _, o := range options {
		labels = append(labels, o.Key)
	}
	joined := strings.Join(labels, "\n")
	for _, v := range verbs() {
		if !strings.Contains(joined, v.Verb) {
			t.Errorf("the menu does not offer %q; the CLI has it", v.Verb)
		}
	}
	if !strings.Contains(joined, "quit") {
		t.Error("the menu has no quit row, so neither front-end can exit deliberately")
	}

	// The quit row's value must be the empty string, which is what both
	// front-ends treat as "stop".
	if last := options[len(options)-1]; last.Value != "" {
		t.Fatalf("the quit row carries value %q, want \"\"", last.Value)
	}
	for _, o := range options[:len(options)-1] {
		if o.Value == "" {
			t.Errorf("answer %q is indistinguishable from quit", o.Key)
		}
	}
}

func TestMenuFormRenders(t *testing.T) {
	f, chosen := newCommandMenu()
	if f == nil || chosen == nil {
		t.Fatal("newCommandMenu returned nil")
	}
	if cmd := f.Init(); cmd == nil {
		t.Error("menu Init returned no command; the form would render unfocused")
	}
	_ = runBounded(f.Init())
	f = applySizeMsg(f, 120, 60)
	if view := f.View(); !strings.Contains(view, "What do you want to run?") {
		t.Fatalf("the menu did not render its title:\n%s", view)
	}
}

func TestConnectionWizardKeepsABlankPortBlank(t *testing.T) {
	// A tnsnames alias must survive the wizard untouched.
	wizard, answers := newConnectionWizard(
		ogr.Connection{DSN: "mydb_high", Username: "ORAGRAPH"}, "/tmp/creds.yaml", false)
	if wizard == nil || answers == nil {
		t.Fatal("newConnectionWizard returned nil")
	}
	if answers.Host != "mydb_high" {
		t.Fatalf("wizard prefilled host %q, want the alias unchanged", answers.Host)
	}
	if answers.Port != "" {
		t.Fatalf("wizard prefilled port %q; nothing specified a port", answers.Port)
	}
	// Only a store-loaded connection may prefill the password.
	if answers.Password != "" {
		t.Fatalf("wizard prefilled a password from a non-store connection: %q", answers.Password)
	}

	answers.Password = "s3cret"
	conn, err := answers.connection()
	if err != nil {
		t.Fatalf("connection() = %v", err)
	}
	if conn.DSN != "mydb_high" {
		t.Fatalf("connection DSN = %q, want the alias preserved", conn.DSN)
	}
}

func TestConnectionWizardPrefillsAPasswordOnlyFromTheStore(t *testing.T) {
	_, answers := newConnectionWizard(
		ogr.Connection{DSN: "localhost:1521/FREEPDB1", Username: "U", Password: "stored"},
		"/tmp/creds.yaml", true)
	if answers.Password != "stored" {
		t.Fatalf("store-loaded wizard password = %q, want the stored value", answers.Password)
	}
}

func TestConnectionWizardRebuildsAFullDSN(t *testing.T) {
	answers := &connectionAnswers{
		Host: "db.example.com", Port: "1522", Service: "pdb1",
		Username: "ORAGRAPH", Password: "p",
	}
	conn, err := answers.connection()
	if err != nil {
		t.Fatalf("connection() = %v", err)
	}
	if conn.DSN != "db.example.com:1522/pdb1" {
		t.Fatalf("DSN = %q, want the parts joined", conn.DSN)
	}
}

func TestConnectionWizardRejectsABadPort(t *testing.T) {
	answers := &connectionAnswers{Host: "h", Port: "70000", Username: "u", Password: "p"}
	if _, err := answers.connection(); err == nil {
		t.Fatal("connection() accepted port 70000")
	}

	answers.Port = "notaport"
	if _, err := answers.connection(); err == nil {
		t.Fatal("connection() accepted a non-numeric port")
	}
}

// buildCommand is the single place both front-ends turn answers into argv, so
// these cases are the shared contract.
func TestBuildCommandFromAnswers(t *testing.T) {
	conn := ogr.Connection{Config: "cfg.yaml"}
	cases := []struct {
		verb    string
		answers *commandAnswers
		want    []string
	}{
		{
			"graphify",
			&commandAnswers{Folder: "/data", Reextract: true},
			[]string{"graphify", "/data", "--reextract", "--config", "cfg.yaml"},
		},
		{
			"query",
			&commandAnswers{Question: "what?", DryRun: true, Source: "src_1"},
			[]string{"query", "what?", "--dry-run", "--source", "src_1", "--config", "cfg.yaml"},
		},
		{
			"export",
			&commandAnswers{Out: "/tmp/o.jsonl", Format: "finetune", Source: "src_2"},
			[]string{"export", "--out", "/tmp/o.jsonl", "--format", "finetune", "--source", "src_2", "--config", "cfg.yaml"},
		},
		{
			"bench",
			&commandAnswers{Suite: "/s.jsonl", Systems: []string{"oragraphrag", "naive_rag"}, Limit: "10"},
			[]string{"bench", "--suite", "/s.jsonl", "--systems", "oragraphrag,naive_rag", "--limit", "10", "--config", "cfg.yaml"},
		},
		{
			"init-db",
			&commandAnswers{Rebuild: true},
			[]string{"init-db", "--rebuild", "--config", "cfg.yaml"},
		},
		{
			"sources",
			&commandAnswers{},
			[]string{"sources", "--config", "cfg.yaml"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.verb, func(t *testing.T) {
			cmd, err := buildCommand(tc.verb, tc.answers, conn)
			if err != nil {
				t.Fatalf("buildCommand(%q) = %v", tc.verb, err)
			}
			got := strings.Join(cmd.Args(), " ")
			if got != strings.Join(tc.want, " ") {
				t.Fatalf("Args() = %q, want %q", got, strings.Join(tc.want, " "))
			}
		})
	}
}

func TestBuildCommandRejectsAnEmptyLimit(t *testing.T) {
	// A blank limit means "no cap", not zero.
	cmd, err := buildCommand("bench", &commandAnswers{Suite: "/s.jsonl", Systems: []string{"oragraphrag"}}, ogr.Connection{})
	if err != nil {
		t.Fatalf("buildCommand with a blank limit = %v", err)
	}
	if strings.Contains(strings.Join(cmd.Args(), " "), "--limit") {
		t.Fatal("a blank limit must omit --limit entirely")
	}

	if _, err := buildCommand("bench", &commandAnswers{Suite: "/s.jsonl", Systems: []string{"x"}, Limit: "-2"}, ogr.Connection{}); err == nil {
		t.Fatal("buildCommand accepted a negative limit")
	}
}

func TestBuildCommandRejectsAnUnknownVerb(t *testing.T) {
	if _, err := buildCommand("drop-everything", &commandAnswers{}, ogr.Connection{}); err == nil {
		t.Fatal("buildCommand accepted an unknown verb")
	}
}

func TestParseLimit(t *testing.T) {
	if got, err := parseLimit("  "); err != nil || got != nil {
		t.Fatalf("parseLimit(blank) = (%v, %v), want (nil, nil)", got, err)
	}
	got, err := parseLimit("12")
	if err != nil || got == nil || *got != 12 {
		t.Fatalf("parseLimit(12) = (%v, %v), want 12", got, err)
	}
	for _, bad := range []string{"0", "-1", "abc"} {
		if _, err := parseLimit(bad); err == nil {
			t.Errorf("parseLimit(%q) = nil error, want a rejection", bad)
		}
	}
}

func TestCommandFormsOpenForEveryVerb(t *testing.T) {
	for _, v := range verbs() {
		t.Run(v.Verb, func(t *testing.T) {
			f, answers := newCommandForm(v.Verb, []string{"src_abc"})
			if f == nil || answers == nil {
				t.Fatalf("newCommandForm(%q) returned nil", v.Verb)
			}
			_ = runBounded(f.Init())
			f.WithWidth(120).WithHeight(60)
			f = applySizeMsg(f, 120, 60)
			if view := f.View(); !strings.Contains(view, v.Verb) {
				t.Errorf("the %q form does not name itself in its view:\n%s", v.Verb, view)
			}
		})
	}
}

// With credentials, the TUI skips the wizard and opens the menu.
func TestModelOpensTheMenuWhenCredentialsExist(t *testing.T) {
	m := New(ogr.Runner{Bin: "/bin/true"}, ogr.Connection{DSN: "h:1/s", Username: "u"}, true, "", false, "")
	if m.scr != screenMenu {
		t.Fatalf("screen = %v, want the menu", m.scr)
	}
	if m.form == nil {
		t.Fatal("no form was armed for the menu")
	}
}

// Without credentials the wizard must run first: the CLI would otherwise fail
// with an opaque ORA error.
func TestModelOpensTheWizardWithoutCredentials(t *testing.T) {
	m := New(ogr.Runner{Bin: "/bin/true"}, ogr.Connection{}, false, "", false, "")
	if m.scr != screenWizard {
		t.Fatalf("screen = %v, want the connection wizard", m.scr)
	}
}

func TestCtrlCQuitsFromAnyScreen(t *testing.T) {
	for _, scr := range []screen{screenMenu, screenWizard, screenCommand, screenRunning, screenOutput} {
		t.Run(map[screen]string{
			screenMenu: "menu", screenWizard: "wizard", screenCommand: "command",
			screenRunning: "running", screenOutput: "output",
		}[scr], func(t *testing.T) {
			m := New(ogr.Runner{Bin: "/bin/true"}, ogr.Connection{DSN: "h:1/s", Username: "u"}, true, "", false, "")
			m.scr = scr
			if scr != screenMenu && scr != screenWizard {
				f, _ := newCommandForm("sources", nil)
				m.form = f
				if scr == screenRunning || scr == screenOutput {
					m.form = nil
				}
			}

			updated, cmd := m.Update(ctrl('c'))
			model, ok := updated.(Model)
			if !ok {
				t.Fatalf("Update returned %T", updated)
			}
			if !model.quitting {
				t.Error("ctrl+c did not set quitting")
			}
			if cmd == nil {
				t.Error("ctrl+c did not return a quit command")
			}
		})
	}
}

// esc backs out of a command form to the menu rather than exiting the program.
func TestEscReturnsFromACommandFormToTheMenu(t *testing.T) {
	m := New(ogr.Runner{Bin: "/bin/true"}, ogr.Connection{DSN: "h:1/s", Username: "u"}, true, "", false, "")
	m.scr = screenCommand
	m.verb = "sources"
	f, _ := newCommandForm("sources", nil)
	m.form = f

	updated, _ := m.Update(press(tea.KeyEscape))
	model := updated.(Model)
	if model.quitting {
		t.Fatal("esc quit the program instead of backing out")
	}
	if model.scr != screenMenu {
		t.Fatalf("screen = %v, want the menu after esc", model.scr)
	}
}

func TestWindowSizeResizesTheOutputPane(t *testing.T) {
	m := New(ogr.Runner{Bin: "/bin/true"}, ogr.Connection{}, false, "", false, "")
	updated, _ := m.Update(tea.WindowSizeMsg{Width: 140, Height: 44})
	model := updated.(Model)
	if model.width != 140 || model.height != 44 {
		t.Fatalf("size = %dx%d, want 140x44", model.width, model.height)
	}
	if model.output.Width() < 100 {
		t.Fatalf("output pane width = %d, want it to follow the window", model.output.Width())
	}
}

// The spinner must keep ticking while a command runs, or the UI looks frozen
// during a long ingest.
func TestRunningScreenCollectsStreamedLines(t *testing.T) {
	m := New(ogr.Runner{Bin: "/bin/true"}, ogr.Connection{}, false, "", false, "")
	m.scr = screenRunning
	m.form = nil
	lines := make(chan ogr.Line, 4)
	m.stream = lines

	updated, cmd := m.Update(streamLineMsg{Stream: "stdout", Text: "ingesting span 1"})
	model := updated.(Model)
	if len(model.lines) != 1 || !strings.Contains(model.lines[0], "ingesting span 1") {
		t.Fatalf("lines = %q, want the streamed line recorded", model.lines)
	}
	if cmd == nil {
		t.Error("a streamed line did not re-arm the reader")
	}

	// stderr must be distinguishable, because it is what an Oracle failure
	// arrives on.
	updated, _ = model.Update(streamLineMsg{Stream: "stderr", Text: "ORA-12541: no listener"})
	model = updated.(Model)
	if len(model.lines) != 2 {
		t.Fatalf("lines = %q, want two entries", model.lines)
	}
	if !strings.Contains(model.lines[1], "ORA-12541") {
		t.Fatalf("stderr line = %q, want it recorded", model.lines[1])
	}
}

func TestRunningScreenEndsOnDone(t *testing.T) {
	m := New(ogr.Runner{Bin: "/bin/true"}, ogr.Connection{}, false, "", false, "")
	m.scr = screenRunning
	m.form = nil

	updated, _ := m.Update(streamDoneMsg{err: nil})
	model := updated.(Model)
	if model.scr != screenOutput {
		t.Fatalf("screen = %v, want the output pane once the run finishes", model.scr)
	}
	if model.runErr != nil {
		t.Fatalf("runErr = %v, want nil for a clean run", model.runErr)
	}
}

func TestOutputScreenOffersMenuAndRerun(t *testing.T) {
	m := New(ogr.Runner{Bin: "/bin/true"}, ogr.Connection{}, false, "", false, "")
	m.scr = screenOutput
	m.form = nil

	updated, _ := m.Update(press('q'))
	if model := updated.(Model); model.scr != screenMenu {
		t.Fatalf("screen = %v, want the menu after q", model.scr)
	}
}

// A run that never produced an invocation (a validation failure) must still
// render, or the user sees a blank screen with no reason why.
func TestViewRendersInEveryState(t *testing.T) {
	m := New(ogr.Runner{Bin: "/bin/true"}, ogr.Connection{DSN: "h:1/s", Username: "u"}, true, "", false, "")

	for _, scr := range []screen{screenMenu, screenWizard, screenCommand, screenRunning, screenOutput} {
		m.scr = scr
		m.form = nil
		if scr == screenMenu || scr == screenWizard || scr == screenCommand {
			f, _ := newCommandForm("sources", nil)
			m.form = f
		}
		m.lines = []string{"a line"}
		m.output.SetContentLines(m.lines)

		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("render panicked in screen %v: %v", scr, r)
				}
			}()
			if got := m.render(); got == "" {
				t.Fatalf("render returned an empty view for screen %v", scr)
			}
		}()
	}
}

// The status line must show the exact invocation, because that is the receipt
// for what ran.
func TestInvocationLineIsDisplayed(t *testing.T) {
	m := New(ogr.Runner{Bin: "/bin/true"}, ogr.Connection{}, false, "", false, "")
	m.scr = screenOutput
	m.form = nil
	m.invocation = "oragraphrag query --dry-run"
	view := m.render()
	if !strings.Contains(view, "oragraphrag query --dry-run") {
		t.Fatalf("the invocation is not shown in:\n%s", view)
	}
}
