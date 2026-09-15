//go:build unix

package ogr

import (
	"os/exec"
	"syscall"
)

// setProcessGroup puts the child in its own process group so a cancellation can
// take down whatever it spawned. The Python CLI shells out and holds DB
// connections; killing only the direct child would leave a grandchild holding
// the stdout pipe open, and the reader would block until that grandchild
// finished on its own.
func setProcessGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

// killProcessGroup signals the whole group, which closes inherited pipes
// immediately instead of waiting for the grandchildren to exit.
func killProcessGroup(cmd *exec.Cmd) error {
	if cmd.Process == nil {
		return nil
	}
	if err := syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL); err != nil {
		return cmd.Process.Kill()
	}
	return nil
}
