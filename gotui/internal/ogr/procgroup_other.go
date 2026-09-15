//go:build !unix

package ogr

import "os/exec"

// setProcessGroup is a no-op where process groups are unavailable. WaitDelay
// still bounds how long Stream waits for the pipes, so the timeout stays
// honoured even without a group to signal.
func setProcessGroup(*exec.Cmd) {}

// killProcessGroup falls back to killing the direct child.
func killProcessGroup(cmd *exec.Cmd) error {
	if cmd.Process == nil {
		return nil
	}
	return cmd.Process.Kill()
}
