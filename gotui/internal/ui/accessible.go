package ui

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"

	"charm.land/huh/v2"

	"github.com/jasperan/oragraphrag/gotui/internal/huhstyle"
	"github.com/jasperan/oragraphrag/gotui/internal/ogr"
)

// osCtx is the context for a foreground, user-driven run. It is cancellable so
// the runner's own deadline still applies, but nothing here cancels it early.
func osCtx() context.Context { return context.Background() }

// RunAccessible drives the same workflows as the TUI, one standalone form at a
// time.
//
// This cannot be done inside the Bubble Tea shell. huh implements accessible
// mode inside Form.Run, and Form.Init/Form.Update have no accessible branch at
// all, so WithAccessible on an embedded form is silently ignored and the user
// would get the full TUI anyway. Running each form standalone is the only way
// to get real screen-reader prompts.
//
// Two caveats, both verified against huh v2.0.3:
//
//   - huh reads each field with a fresh bufio.Scanner, so when stdin is a pipe
//     the first field swallows the entire pipe and every later field sees EOF,
//     returning empty answers. This driver is therefore for a human at a
//     terminal; automation must use the flag-only path in main.go.
//   - Prompts are written to stdout, so this mode cannot also use stdout for a
//     clean data stream. That is why it is a separate mode rather than a flag on
//     the TUI.
func RunAccessible(
	runner ogr.Runner,
	conn ogr.Connection,
	connOK bool,
	storePath string,
	storeLoaded bool,
	out io.Writer,
) error {
	if out == nil {
		out = os.Stdout
	}

	if !connOK {
		wizard, answers := newConnectionWizard(conn, storePath, storeLoaded)
		if err := wizard.Run(); err != nil {
			if errors.Is(err, huh.ErrUserAborted) {
				return nil
			}
			return err
		}
		built, err := answers.connection()
		if err != nil {
			return err
		}
		conn, connOK = built, true
		if answers.Remember && storePath != "" {
			if err := built.Save(storePath); err != nil {
				fmt.Fprintf(out, "could not save the connection: %v\n", err)
			} else {
				fmt.Fprintf(out, "saved connection to %s\n", storePath)
			}
		}
	}

	for {
		menu, chosen := newCommandMenu()
		if err := menu.Run(); err != nil {
			if errors.Is(err, huh.ErrUserAborted) {
				return nil
			}
			return err
		}
		verb := *chosen
		if verb == "" {
			return nil
		}

		// The source suggestions are best-effort: an unreachable database must
		// not stop a user from typing a path by hand.
		var sources []string
		if verb == "query" || verb == "export" {
			probe := runner
			probe.Timeout = ogr.SourcesTimeout
			probe.Env = append(append([]string{}, probe.Env...), conn.Env()...)
			if text, err := probe.Capture(osCtx(), ogr.NewSourcesCmd(conn.Config).Args()...); err == nil {
				sources = ogr.ParseSources(text)
			}
		}

		commandForm, answers := newCommandForm(verb, sources)
		if err := commandForm.Run(); err != nil {
			if errors.Is(err, huh.ErrUserAborted) {
				return nil
			}
			return err
		}

		cmd, err := buildCommand(verb, answers, conn)
		if err != nil {
			fmt.Fprintf(out, "error: %v\n", err)
			continue
		}

		fmt.Fprintf(out, "\n$ %s\n\n", runner.Describe(cmd.Args()))
		run := runner
		run.Env = append(append([]string{}, run.Env...), conn.Env()...)
		if err := run.Stream(osCtx(), cmd.Args(), func(l ogr.Line) {
			fmt.Fprintln(out, l.Text)
		}); err != nil {
			fmt.Fprintf(out, "failed: %v\n", err)
		}

		again := false
		prompt := huh.NewForm(huh.NewGroup(
			huh.NewConfirm().
				Title("Run another command?").
				Value(&again),
		)).WithAccessible(true).WithTheme(huh.ThemeFunc(huhstyle.Theme)).WithInput(os.Stdin).WithOutput(out)
		if err := prompt.Run(); err != nil {
			return nil
		}
		if !again {
			return nil
		}
	}
}
