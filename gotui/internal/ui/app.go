// Package ui holds the interactive front-ends for the OraGraphRAG CLI.
//
// There are two, because huh's accessible mode is implemented in Form.Run and
// is bypassed entirely when a form is embedded in a Bubble Tea program:
//
//   - app.go        the Bubble Tea TUI, used when a human is at a terminal and
//     ACCESSIBLE is unset
//   - accessible.go the sequential driver, used when ACCESSIBLE is set, which
//     runs each form standalone so a screen reader gets plain prompts
//
// Both build their arguments through the same pure ogr package, so either user
// gets byte-identical CLI invocations.
package ui

import (
	"context"
	"fmt"
	"strings"

	"charm.land/bubbles/v2/spinner"
	"charm.land/bubbles/v2/viewport"
	tea "charm.land/bubbletea/v2"
	"charm.land/huh/v2"

	"github.com/jasperan/oragraphrag/gotui/internal/ogr"
)

// screen is which pane the TUI is showing.
type screen int

const (
	screenMenu screen = iota
	screenWizard
	screenCommand
	screenRunning
	screenOutput
)

// Messages.
type (
	streamLineMsg ogr.Line
	streamDoneMsg struct{ err error }
	sourcesMsg    struct {
		ids []string
		err error
	}
)

// Model is the root Bubble Tea model.
type Model struct {
	runner ogr.Runner
	conn   ogr.Connection
	// connOK is false when no credentials are known, so the wizard runs first.
	connOK bool
	// storePath is where Remember writes, empty when unavailable.
	storePath string
	// storeLoaded records whether conn came from the 0600 store, which decides
	// whether the wizard may prefill the password field.
	storeLoaded bool
	// connNote is a non-fatal caveat to display, e.g. loose file permissions.
	connNote string

	width, height int
	scr           screen

	// form is the active form; exactly one screen owns it at a time.
	form        *huh.Form
	verb        string
	pendingVerb string
	answers     *commandAnswers
	connAnswers *connectionAnswers

	spinner spinner.Model
	output  viewport.Model
	lines   []string

	invocation string
	runErr     error
	quitting   bool

	// stream carries subprocess output; done reports termination.
	stream chan ogr.Line
	done   chan error
}

// New builds the root model and arms the first screen.
//
// The arming happens here rather than in Init because Init has a value receiver
// and bubbletea only reads its returned command: a model that armed itself in
// Init would render an empty view for the first frame.
func New(runner ogr.Runner, conn ogr.Connection, connOK bool, storePath string, storeLoaded bool, note string) Model {
	sp := spinner.New()
	sp.Spinner = spinner.Dot
	sp.Style = titleStyle

	m := Model{
		runner:      runner,
		conn:        conn,
		connOK:      connOK,
		storePath:   storePath,
		storeLoaded: storeLoaded,
		connNote:    note,
		width:       80,
		height:      24,
		spinner:     sp,
		output:      viewport.New(),
	}
	m.output.SetWidth(76)
	m.output.SetHeight(18)

	if !connOK {
		f, answers := newConnectionWizard(conn, storePath, storeLoaded)
		m.applySize(f)
		m.form, m.connAnswers, m.scr = f, answers, screenWizard
	} else {
		f, chosen := newCommandMenu()
		m.applySize(f)
		m.form, m.pendingVerb, m.scr = f, *chosen, screenMenu
	}
	return m
}

func (m Model) Init() tea.Cmd {
	if m.form == nil {
		return nil
	}
	return m.form.Init()
}

// startMenu opens the command dashboard.
// WithCommandForm opens one command's form directly, skipping the dashboard.
//
// It is the interactive counterpart of --command: the same verb, but with the
// form in front of it. Unknown verbs are ignored, because Validate rejects them
// when the form is submitted.
func (m Model) WithCommandForm(verb string) Model {
	if !m.connOK {
		// The wizard must still run first: a form with no credentials would
		// fail on the first database call with an opaque ORA error.
		return m
	}
	known := false
	for _, v := range verbs() {
		if v.Verb == verb {
			known = true
			break
		}
	}
	if !known {
		return m
	}

	f, answers := newCommandForm(verb, nil)
	m.applySize(f)
	m.form, m.verb, m.answers, m.scr = f, verb, answers, screenCommand
	return m
}

func (m *Model) startMenu() tea.Cmd {
	f, chosen := newCommandMenu()
	m.applySize(f)
	m.form = f
	m.pendingVerb = *chosen
	m.scr = screenMenu
	return f.Init()
}

func (m *Model) startWizard() tea.Cmd {
	f, answers := newConnectionWizard(m.conn, m.storePath, m.storeLoaded)
	m.applySize(f)
	m.form = f
	m.connAnswers = answers
	m.scr = screenWizard
	return f.Init()
}

func (m *Model) startCommand(verb string, sources []string) tea.Cmd {
	f, answers := newCommandForm(verb, sources)
	m.applySize(f)
	m.form = f
	m.verb = verb
	m.answers = answers
	m.scr = screenCommand
	return f.Init()
}

// applySize keeps a form inside the pane it is drawn into.
func (m *Model) applySize(f *huh.Form) {
	if f == nil {
		return
	}
	w, h := m.width-4, m.height-8
	if w < 20 {
		w = 20
	}
	if h < 6 {
		h = 6
	}
	f.WithWidth(w).WithHeight(h)
}

// Update is the single message pump. While a form is open the form sees every
// message, so its own key handling stays authoritative.
func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width, m.height = msg.Width, msg.Height
		m.output.SetWidth(maxInt(20, msg.Width-4))
		m.output.SetHeight(maxInt(4, msg.Height-10))
		m.applySize(m.form)
		return m, nil

	case tea.KeyPressMsg:
		if msg.String() == "ctrl+c" {
			m.quitting = true
			return m, tea.Quit
		}

	case sourcesMsg:
		// Suggestions are a convenience: an unreachable database degrades the
		// field to free text rather than blocking the form.
		if msg.err != nil {
			return m, m.startCommand(m.pendingVerb, nil)
		}
		return m, m.startCommand(m.pendingVerb, msg.ids)
	}

	if m.form != nil {
		return m.updateForm(msg)
	}
	switch m.scr {
	case screenRunning:
		return m.updateRunning(msg)
	case screenOutput:
		return m.updateOutput(msg)
	}
	return m, nil
}

// updateForm drives the active form and reacts to its terminal states.
func (m Model) updateForm(msg tea.Msg) (tea.Model, tea.Cmd) {
	if key, ok := msg.(tea.KeyPressMsg); ok && key.String() == "esc" {
		// esc backs out of a command form to the menu; from the menu or the
		// wizard there is nothing behind it, so it quits.
		if m.scr == screenCommand {
			return m, m.startMenu()
		}
		m.quitting = true
		return m, tea.Quit
	}

	updated, cmd := m.form.Update(msg)
	if f, ok := updated.(*huh.Form); ok {
		m.form = f
	}

	switch m.form.State {
	case huh.StateCompleted:
		return m.finishForm()
	case huh.StateAborted:
		m.quitting = true
		return m, tea.Quit
	}
	return m, cmd
}

// finishForm acts on the answers a form collected.
func (m Model) finishForm() (tea.Model, tea.Cmd) {
	switch m.scr {
	case screenWizard:
		conn, err := m.connAnswers.connection()
		if err != nil {
			m.form = nil
			m.lines = []string{errorLine(err.Error())}
			m.output.SetContentLines(m.lines)
			m.scr = screenOutput
			return m, nil
		}
		m.conn = conn
		m.connOK = true

		switch {
		case m.connAnswers.Remember && m.storePath != "":
			if err := conn.Save(m.storePath); err != nil {
				m.connNote = fmt.Sprintf("could not save the connection: %v", err)
			} else {
				m.connNote = fmt.Sprintf("saved 0600 to %s", m.storePath)
			}
		case m.connAnswers.Remember:
			m.connNote = "no credential file location available; this session only"
		default:
			m.connNote = ""
		}
		return m, m.startMenu()

	case screenMenu:
		verb := m.pendingVerb
		m.form = nil
		if verb == "" {
			m.quitting = true
			return m, tea.Quit
		}
		// The source suggestions are fetched before the form opens so the field
		// can offer real ids instead of asking the user to remember them.
		if verb == "query" || verb == "export" {
			return m, m.loadSources(verb)
		}
		return m, m.startCommand(verb, nil)

	case screenCommand:
		cmd, err := buildCommand(m.verb, m.answers, m.conn)
		if err != nil {
			m.form = nil
			m.lines = []string{errorLine(err.Error())}
			m.output.SetContentLines(m.lines)
			m.scr = screenOutput
			return m, nil
		}
		m.form = nil
		return m, m.runCommand(cmd)
	}

	m.form = nil
	return m, nil
}

// loadSources asks the CLI for the current source ids. A failure is not fatal:
// the field simply degrades to free text.
func (m *Model) loadSources(verb string) tea.Cmd {
	runner := m.runner
	runner.Timeout = ogr.SourcesTimeout
	args := ogr.NewSourcesCmd(m.conn.Config).Args()
	return func() tea.Msg {
		out, err := runner.Capture(context.Background(), args...)
		if err != nil {
			return sourcesMsg{err: err}
		}
		return sourcesMsg{ids: ogr.ParseSources(out)}
	}
}

// runCommand streams a subprocess into the output pane.
func (m *Model) runCommand(cmd ogr.Command) tea.Cmd {
	args := cmd.Args()
	m.invocation = m.runner.Describe(args)
	m.lines = []string{invocationLine(m.invocation)}
	m.output.SetContentLines(m.lines)
	m.output.GotoBottom()
	m.runErr = nil
	m.scr = screenRunning

	// Credentials travel in the child's environment, never in its argv, so the
	// displayed invocation above is safe to show verbatim.
	runner := m.runner
	runner.Env = append(append([]string{}, runner.Env...), m.conn.Env()...)

	lines := make(chan ogr.Line, 128)
	done := make(chan error, 1)
	m.stream, m.done = lines, done

	go func() {
		err := runner.Stream(context.Background(), args, func(l ogr.Line) { lines <- l })
		close(lines)
		done <- err
	}()

	return tea.Batch(m.spinner.Tick, waitForStream(lines), waitForDone(done))
}

func waitForStream(ch <-chan ogr.Line) tea.Cmd {
	return func() tea.Msg {
		l, ok := <-ch
		if !ok {
			return nil
		}
		return streamLineMsg(l)
	}
}

func waitForDone(ch <-chan error) tea.Cmd {
	return func() tea.Msg { return streamDoneMsg{err: <-ch} }
}

// updateRunning collects streamed output and ends the run.
func (m Model) updateRunning(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case streamLineMsg:
		m.appendLine(string(msg.Text), ogr.Line(msg).IsStderr())
		return m, waitForStream(m.stream)

	case streamDoneMsg:
		m.runErr = msg.err
		m.scr = screenOutput
		return m, nil

	case spinner.TickMsg:
		var cmd tea.Cmd
		m.spinner, cmd = m.spinner.Update(msg)
		return m, cmd
	}
	return m, nil
}

// appendLine adds one line of subprocess output and keeps the view pinned to
// the bottom, which is where progress appears.
func (m *Model) appendLine(text string, stderr bool) {
	if stderr {
		text = errorStyle.Render(text)
	}
	m.lines = append(m.lines, text)
	m.output.SetContentLines(m.lines)
	m.output.GotoBottom()
}

// updateOutput handles the finished-run pane.
func (m Model) updateOutput(msg tea.Msg) (tea.Model, tea.Cmd) {
	key, ok := msg.(tea.KeyPressMsg)
	if !ok {
		return m, nil
	}
	switch key.String() {
	case "q", "esc":
		return m, m.startMenu()
	case "r":
		if m.verb != "" && m.answers != nil {
			if cmd, err := buildCommand(m.verb, m.answers, m.conn); err == nil {
				return m, m.runCommand(cmd)
			}
		}
		return m, m.startMenu()
	}
	return m, nil
}

// View renders the active screen.
func (m Model) View() tea.View {
	v := tea.NewView(m.render())
	v.AltScreen = true
	return v
}

func (m Model) render() string {
	if m.form != nil {
		return m.formView()
	}
	switch m.scr {
	case screenRunning:
		return m.runningView()
	case screenOutput:
		return m.outputView()
	}
	return ""
}

// formView renders the active form with a header line.
func (m Model) formView() string {
	title := map[screen]string{
		screenMenu:    "Oracle Graph RAG",
		screenWizard:  "Connection",
		screenCommand: m.verb,
	}[m.scr]

	var b strings.Builder
	b.WriteString(titleStyle.Render("  " + title + "  "))
	if m.connOK {
		b.WriteString("  ")
		b.WriteString(mutedStyle.Render(m.conn.String()))
	}
	if m.connNote != "" {
		b.WriteString("\n  ")
		b.WriteString(mutedStyle.Render(m.connNote))
	}
	b.WriteString("\n\n  ")
	b.WriteString(strings.ReplaceAll(m.form.View(), "\n", "\n  "))
	return b.String()
}

// runningView shows the spinner above the live output pane.
func (m Model) runningView() string {
	header := titleStyle.Render("  running  ") + subtextStyle.Render(m.invocation)
	status := "  " + m.spinner.View() + mutedStyle.Render("streaming output  •  ctrl+c to quit")
	return strings.Join([]string{header, "", panelStyle.Render(m.output.View()), "", status}, "\n")
}

// outputView renders the finished (or failed) run.
func (m Model) outputView() string {
	var footer string
	if m.runErr != nil {
		footer = "  " + errorStyle.Render(fmt.Sprintf("failed: %v", m.runErr))
	} else {
		footer = "  " + successStyle.Render("done")
	}
	hint := mutedStyle.Render("    q menu  •  r rerun  •  ctrl+c quit")
	if m.invocation == "" {
		// A build/validation error never reached a subprocess.
		return errorStyle.Render("  could not run") + "\n\n  " +
			strings.ReplaceAll(strings.Join(m.lines, "\n  "), "\n  \n", "\n\n  ") +
			"\n\n" + hint
	}
	return strings.Join([]string{
		titleStyle.Render("  finished  ") + subtextStyle.Render(m.invocation),
		"",
		panelStyle.Render(m.output.View()),
		"",
		footer + hint,
	}, "\n")
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
