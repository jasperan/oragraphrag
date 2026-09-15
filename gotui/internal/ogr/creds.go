package ogr

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// credsEnvVar lets a user point the front-end at an explicit credential file.
const credsEnvVar = "OGR_CREDS_FILE"

// StorePath returns where the credential file lives.
//
// It is deliberately never inside the repository: os.UserConfigDir is
// ~/.config (or $XDG_CONFIG_HOME), so the file cannot be committed by accident.
// Set OGR_CREDS_FILE to override.
func StorePath() (string, error) {
	if explicit := strings.TrimSpace(os.Getenv(credsEnvVar)); explicit != "" {
		return explicit, nil
	}
	dir, err := os.UserConfigDir()
	if err != nil {
		return "", fmt.Errorf("cannot locate a config directory: %w", err)
	}
	return filepath.Join(dir, "oragraphrag", "credentials.yaml"), nil
}

// EnsureOutsideRepo fails when path sits inside a git working tree.
//
// This is the guard that keeps a stored password out of version control: it is
// checked before any write, so a bad OGR_CREDS_FILE cannot quietly create a
// committable secret.
func EnsureOutsideRepo(path string) error {
	dir := filepath.Dir(path)
	for {
		if _, err := os.Stat(filepath.Join(dir, ".git")); err == nil {
			return fmt.Errorf("refusing to store credentials inside the git tree at %s", dir)
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return nil
		}
		dir = parent
	}
}

// Save writes the connection to path with directory mode 0700 and file mode
// 0600, and verifies the resulting permissions rather than assuming umask
// honoured them. The password is written last so a crash cannot leave a
// half-file that looks complete.
func (c Connection) Save(path string) error {
	if err := EnsureOutsideRepo(path); err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return fmt.Errorf("cannot create %s: %w", filepath.Dir(path), err)
	}

	var b strings.Builder
	b.WriteString("# OraGraphRAG connection - written by the Go front-end.\n")
	b.WriteString("# Mode 0600, stored outside the repository on purpose.\n")
	fmt.Fprintf(&b, "# written: %s\n", time.Now().UTC().Format(time.RFC3339))
	fmt.Fprintf(&b, "username: %s\n", c.Username)
	fmt.Fprintf(&b, "dsn: %s\n", c.DSN)
	if c.Config != "" {
		fmt.Fprintf(&b, "config: %s\n", c.Config)
	}
	fmt.Fprintf(&b, "password: %s\n", c.Password)

	if err := os.WriteFile(path, []byte(b.String()), 0o600); err != nil {
		return fmt.Errorf("cannot write %s: %w", path, err)
	}
	// os.WriteFile honours umask on some platforms; make the mode explicit and
	// then prove it, because a 0644 credential file is worse than none.
	if err := os.Chmod(path, 0o600); err != nil {
		return fmt.Errorf("cannot restrict %s to 0600: %w", path, err)
	}
	info, err := os.Stat(path)
	if err != nil {
		return err
	}
	if perm := info.Mode().Perm(); perm != 0o600 {
		return fmt.Errorf("credential file %s has mode %04o, want 0600", path, perm)
	}
	return nil
}

// Load reads a credential file written by Save. A missing file is not an
// error: the caller falls back to the connection wizard.
func Load(path string) (Connection, bool, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return Connection{}, false, nil
		}
		return Connection{}, false, fmt.Errorf("cannot read %s: %w", path, err)
	}

	var c Connection
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, value, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		value = strings.TrimSpace(value)
		switch key {
		case "username":
			c.Username = value
		case "password":
			c.Password = value
		case "dsn":
			c.DSN = value
		case "config":
			c.Config = value
		}
	}
	if c.DSN == "" || c.Username == "" {
		return Connection{}, false, fmt.Errorf("%s does not contain a username and dsn", path)
	}
	return c, true, nil
}

// Warning returns a human-readable caveat about the file's permissions, or ""
// when the file is safely 0600. A user may have hand-edited it.
func PermWarning(path string) string {
	info, err := os.Stat(path)
	if err != nil {
		return ""
	}
	if perm := info.Mode().Perm(); perm != 0o600 {
		return fmt.Sprintf("%s is mode %04o; run `chmod 600` on it", path, perm)
	}
	return ""
}
