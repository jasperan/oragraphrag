package ui

import "charm.land/lipgloss/v2"

// The approved design tokens, mirrored from docs/tui-design-tokens.md.
//
// This file exists so the TUI chrome (headers, status lines, error banners) can
// be drawn in the same palette as the huh forms, which are themed by the
// huhstyle package. These 14 values are the only colours any source file in
// this project may contain, and check_palette.py enforces that; keeping them in
// one place is what makes the rule checkable rather than aspirational.
//
// Do not use huh's built-in themes: ThemeCatppuccin() paints with auxiliary
// Catppuccin shades that are not approved tokens.
const (
	hexBG        = "#1e1e2e" // bg
	hexSurface   = "#181825" // surface
	hexElevated  = "#313244" // elevated
	hexHighest   = "#45475a" // highest
	hexText      = "#cdd6f4" // text
	hexSubtext   = "#a6adc8" // subtext
	hexMuted     = "#6c7086" // muted
	hexDim       = "#585b70" // dim
	hexPrimary   = "#89b4fa" // primary
	hexSecondary = "#cba6f7" // secondary
	hexInfo      = "#89dceb" // info
	hexSuccess   = "#a6e3a1" // success
	hexWarning   = "#f9e2af" // warning
	hexError     = "#f38ba8" // error
)

// Chrome styles. Titles are bold primary; metadata is subtext; errors are the
// only place the error token appears, because it is semantic.
var (
	titleStyle   = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color(hexPrimary))
	mutedStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color(hexMuted))
	subtextStyle = lipgloss.NewStyle().Foreground(lipgloss.Color(hexSubtext))
	successStyle = lipgloss.NewStyle().Foreground(lipgloss.Color(hexSuccess))
	errorStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color(hexError))
	infoStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color(hexInfo))
	panelStyle   = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color(hexDim)).
			Padding(0, 1)
)

// errorLine prefixes a message with the error token so a failure is visually
// unmistakable, without leaking a colour literal into the call sites.
func errorLine(msg string) string { return errorStyle.Render("error: ") + msg }

// invocationLine shows the exact command that ran. Secrets are never part of an
// argv in this front-end, so this is safe to display verbatim.
func invocationLine(invocation string) string { return mutedStyle.Render("$ " + invocation) }
