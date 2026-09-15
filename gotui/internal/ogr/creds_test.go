package ogr

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// tempOutsideRepo returns a path in t.TempDir, which is never inside a git
// tree, so EnsureOutsideRepo legitimately allows it.
func tempOutsideRepo(t *testing.T) string {
	t.Helper()
	return filepath.Join(t.TempDir(), "oragraphrag", "credentials.yaml")
}

func TestSaveWritesMode0600(t *testing.T) {
	path := tempOutsideRepo(t)
	conn := Connection{Username: "ORAGRAPH", Password: "s3cret", DSN: "localhost:1521/FREEPDB1", Config: "/tmp/cfg.yaml"}

	if err := conn.Save(path); err != nil {
		t.Fatalf("Save() = %v", err)
	}

	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	if perm := info.Mode().Perm(); perm != 0o600 {
		t.Fatalf("file mode = %04o, want 0600", perm)
	}

	dirInfo, err := os.Stat(filepath.Dir(path))
	if err != nil {
		t.Fatalf("stat dir: %v", err)
	}
	if perm := dirInfo.Mode().Perm(); perm != 0o700 {
		t.Fatalf("dir mode = %04o, want 0700", perm)
	}
}

func TestSaveLoadRoundTrip(t *testing.T) {
	path := tempOutsideRepo(t)
	want := Connection{Username: "ORAGRAPH", Password: "p@ss:with:colons", DSN: "db:1522/pdb1", Config: "/tmp/c.yaml"}

	if err := want.Save(path); err != nil {
		t.Fatalf("Save() = %v", err)
	}
	got, ok, err := Load(path)
	if err != nil {
		t.Fatalf("Load() = %v", err)
	}
	if !ok {
		t.Fatal("Load() reported the file as absent")
	}
	if got != want {
		t.Fatalf("round trip = %+v, want %+v", got, want)
	}
}

// A password containing a colon must survive, because strings.Cut splits on the
// first colon only.
func TestLoadPreservesPathologicalPasswords(t *testing.T) {
	path := tempOutsideRepo(t)
	want := Connection{Username: "u", Password: "a:b:c:d", DSN: "h:1521/s"}
	if err := want.Save(path); err != nil {
		t.Fatalf("Save() = %v", err)
	}
	got, _, err := Load(path)
	if err != nil {
		t.Fatalf("Load() = %v", err)
	}
	if got.Password != want.Password {
		t.Fatalf("password = %q, want %q", got.Password, want.Password)
	}
}

func TestLoadMissingFileIsNotAnError(t *testing.T) {
	conn, ok, err := Load(filepath.Join(t.TempDir(), "nope.yaml"))
	if err != nil {
		t.Fatalf("Load() = %v, want nil for a missing file", err)
	}
	if ok {
		t.Fatal("Load() reported a missing file as present")
	}
	if conn.Username != "" || conn.Password != "" {
		t.Fatalf("Load() returned %+v, want an empty connection", conn)
	}
}

func TestLoadRejectsAnIncompleteFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "partial.yaml")
	if err := os.WriteFile(path, []byte("password: only\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, _, err := Load(path); err == nil {
		t.Fatal("Load() = nil error, want a rejection for a file with no username/dsn")
	}
}

// The guard that keeps a stored password out of version control.
func TestEnsureOutsideRepoRefusesAGitTree(t *testing.T) {
	repo := t.TempDir()
	if err := os.MkdirAll(filepath.Join(repo, ".git"), 0o755); err != nil {
		t.Fatal(err)
	}
	inside := filepath.Join(repo, "gotui", "credentials.yaml")
	if err := os.MkdirAll(filepath.Dir(inside), 0o755); err != nil {
		t.Fatal(err)
	}

	if err := EnsureOutsideRepo(inside); err == nil {
		t.Fatal("EnsureOutsideRepo() = nil, want a refusal inside a git tree")
	}

	conn := Connection{Username: "u", Password: "p", DSN: "h:1521/s"}
	err := conn.Save(inside)
	if err == nil {
		t.Fatal("Save() wrote a credential file inside a git tree")
	}
	if !strings.Contains(err.Error(), "git tree") {
		t.Fatalf("Save() error = %q, want it to explain the git tree refusal", err)
	}
	if _, statErr := os.Stat(inside); statErr == nil {
		t.Fatal("Save() created the file despite refusing")
	}
}

func TestEnsureOutsideRepoAllowsAPlainDirectory(t *testing.T) {
	if err := EnsureOutsideRepo(filepath.Join(t.TempDir(), "x", "y.yaml")); err != nil {
		t.Fatalf("EnsureOutsideRepo() = %v, want nil outside a git tree", err)
	}
}

// Save must refuse even when a parent directory merely contains .git, which is
// the realistic case: the front-end lives inside the checkout.
func TestSaveRefusesARepoRelativePath(t *testing.T) {
	repo := t.TempDir()
	if err := os.MkdirAll(filepath.Join(repo, ".git"), 0o755); err != nil {
		t.Fatal(err)
	}
	conn := Connection{Username: "u", Password: "p", DSN: "h:1521/s"}
	if err := conn.Save(filepath.Join(repo, "gotui", "creds.yaml")); err == nil {
		t.Fatal("Save() accepted an in-repo path")
	}
}

func TestPermWarningDetectsLoosePermissions(t *testing.T) {
	path := filepath.Join(t.TempDir(), "loose.yaml")
	if err := os.WriteFile(path, []byte("username: u\ndsn: h:1/s\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(path, 0o644); err != nil {
		t.Fatal(err)
	}
	warning := PermWarning(path)
	if warning == "" {
		t.Fatal("PermWarning() = \"\", want a warning for a 0644 file")
	}
	if !strings.Contains(warning, "chmod 600") {
		t.Fatalf("PermWarning() = %q, want it to name the fix", warning)
	}

	if err := os.Chmod(path, 0o600); err != nil {
		t.Fatal(err)
	}
	if warning := PermWarning(path); warning != "" {
		t.Fatalf("PermWarning() = %q, want \"\" for a 0600 file", warning)
	}
}

func TestStorePathHonoursTheEnvOverride(t *testing.T) {
	want := filepath.Join(t.TempDir(), "custom.yaml")
	t.Setenv("OGR_CREDS_FILE", want)

	got, err := StorePath()
	if err != nil {
		t.Fatalf("StorePath() = %v", err)
	}
	if got != want {
		t.Fatalf("StorePath() = %q, want %q", got, want)
	}
}

// The default location must never be inside the repository, which is the whole
// point of not writing config.yaml next to the sources.
func TestStorePathDefaultsOutsideTheRepo(t *testing.T) {
	t.Setenv("OGR_CREDS_FILE", "")

	got, err := StorePath()
	if err != nil {
		t.Fatalf("StorePath() = %v", err)
	}
	if err := EnsureOutsideRepo(got); err != nil {
		t.Fatalf("default store path %q is not outside a git tree: %v", got, err)
	}
	if !strings.Contains(got, "oragraphrag") {
		t.Fatalf("StorePath() = %q, want it namespaced under oragraphrag", got)
	}
}
