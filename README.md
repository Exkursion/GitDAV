# GitHub ↔ Nextcloud WebDAV Repo Sync  
*(quick & dirty edition)*

A small Python tool that keeps a GitHub repository and a Nextcloud (WebDAV) folder structure in sync.

This is not trying to be a perfect bidirectional sync engine or a full Git replacement.  
It’s built for pragmatic workflows where you want:

- GitHub as the canonical source of truth  
- Nextcloud as a convenient place to edit, drop files, or collaborate  
- automatic Pull Requests when something appears in a “Push” folder  

It works well enough. It’s simple. And yes — it’s a bit quick & dirty by design.

---

## What it does

### Pull (GitHub → Nextcloud)
- Watches a GitHub branch (default: `main`)
- If the branch head changes:
  - downloads the repo as a ZIP
  - unzips it locally
  - uploads the content to WebDAV `Active/`
- Uses an “incoming folder swap” so `Active/` is never half-updated.

### Push (Nextcloud → GitHub PR)
- Watches WebDAV `Push/`
- If `Push/` changes and stays stable for `push_stable_seconds`:
  - downloads `Push/` to a temp folder
  - creates a new branch in the same repo
  - uploads all files as blobs
  - creates a commit
  - opens a Pull Request
  - archives `Push/`

### Archive / Cleanup
Moves folders into `Archive/` with timestamps:

- `Active` → `Archive/Active-YYYYMMDD-HHMMSS-<rand>/`
- `Push` → `Archive/Push-YYYYMMDD-HHMMSS-<rand>/`
- Crash leftovers like `Active.__incoming__-*`  
  → `Archive/Incoming-YYYYMMDD-HHMMSS-<rand>/`

### Excludes on Push
If there is an `exclude.conf` file in the WebDAV project root, it is respected
when creating a PR from `Push/`.

---

## Folder layout on WebDAV

For each configured repo, the tool expects something like:

```

<webdav_root>/
Active/
Push/
Archive/
exclude.conf   (optional)

```

Example:

```

Projects/MyProject/
Active/
Push/
Archive/
exclude.conf

````

---

## ⚠ Security / “quick & dirty” disclaimer

- **Secrets are stored in plaintext** in `config.json`  
  (GitHub token + Nextcloud username/password or app password).
- This is **not hardened** and not security-reviewed.
- Treat this as a **quick & dirty personal tool**, not an enterprise product.
- If you run this on a shared system: **lock down file permissions**.

Recommended:
```bash
chmod 600 config.json
````

If you want to be more serious about security, consider:

* using environment variables for secrets
* using a keyring / secret manager
* using a GitHub token with minimal scope
* using a Nextcloud app password (not your real account password)

---

## Requirements

* Python 3.9+ recommended
* `requests`

Install dependency:

```bash
python3 -m pip install requests
```

---

## Setup

Run the setup wizard:

```bash
python3 gitdav.py --setup
```

It will:

* ask for GitHub token + Nextcloud WebDAV credentials
* list your GitHub repositories
* let you pick one or more repos
* create `config.json`

---

## Running the tool

### Interactive menu

```bash
python3 gitdav.py
```

### Run once

```bash
python3 gitdav.py --once
```

### Daemon mode (polling + live reload)

```bash
python3 gitdav.py --run
```

In daemon mode it:

* pulls GitHub → `Active/` when the branch changes
* pushes `Push/` → PR when new content appears
* auto-reloads `config.json` when it changes

---

## Running in the background (Linux, systemd user service)

Typical setup: use a **systemd user service** and enable “linger”
so it runs even when you are logged out.

Example unit (adjust paths):

```ini
[Unit]
Description=gitdav (GitHub <-> Nextcloud WebDAV sync)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/daniel/worker/gitdav
ExecStart=/usr/bin/env python3 /home/daniel/worker/gitdav/gitdav.py --run --config config.json --reload-seconds 10
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
```

Enable & start:

```bash
systemctl --user daemon-reload
systemctl --user enable --now gitdav.service
```

Allow it to run after logout:

```bash
sudo loginctl enable-linger <your-user>
```

Logs:

```bash
journalctl --user -u gitdav.service -f
```

---

## exclude.conf – ignore files on Push → PR

When creating a Pull Request from the WebDAV `Push/` folder, the tool can ignore
files and folders based on patterns defined in an optional file:

```
<webdav_root>/exclude.conf
```

This file is **only used for Push → PR**.
It does **not** affect GitHub → Active pulls.

---

### How it works

* Each non-empty line is treated as a pattern
* Lines starting with `#` are comments
* Matching uses simple shell-style wildcards (`fnmatch`)

---

### Pattern rules

| Pattern         | Meaning                                                |
| --------------- | ------------------------------------------------------ |
| `*.log`         | Ignore all `.log` files anywhere                       |
| `build/`        | Ignore the entire `build` folder                       |
| `build/*`       | Ignore files inside `build`, but not the folder itself |
| `node_modules/` | Ignore `node_modules` completely                       |
| `**/*.tmp`      | Ignore all `.tmp` files in any subfolder               |
| `.DS_Store`     | Ignore macOS metadata files                            |

Notes:

* Trailing slash (`/`) means **directory rule**
* Patterns are matched against **relative paths inside Push/**
* Matching is not regex, it’s classic glob-style matching

---

### Example `exclude.conf`

```text
# Logs
*.log
*.tmp

# Python cache
__pycache__/
*.pyc

# Node stuff
node_modules/
npm-debug.log

# IDE / OS junk
.vscode/
.idea/
.DS_Store
Thumbs.db

# Build artifacts
dist/
build/
```

---

### What happens if everything is excluded?

If, after applying `exclude.conf`, **no files remain** in `Push/`,
the tool will abort the push with:

```
Push folder contains no files to push (after exclude).
```

This is intentional — it prevents creating empty commits and PRs.

---

### Typical workflow

1. Put files you want to propose into:

   ```
   <webdav_root>/Push/
   ```
2. Maintain ignore rules in:

   ```
   <webdav_root>/exclude.conf
   ```
3. The daemon detects the change, waits for stability,
   then creates a PR containing only the **non-excluded** files.

---

### Why exclude.conf lives on WebDAV

* You can tweak ignore rules **without touching the server**
* Multiple users can share the same rules
* No need to redeploy or restart the tool
  (rules are picked up automatically on next push)

---

## Notes / Limitations

* Push → PR uploads **all remaining files** as a commit.

  * No delete detection
  * No rename detection
  * It’s basically: “here is the current tree”
* Polling only:

  * no WebDAV notifications
  * no GitHub webhooks
* “Stable folder” detection is time + fingerprint based.

  * Might be imperfect for very large repos or slow uploads.

---

## Versioning / Expectations

This is a pragmatic tool:

* avoids half-uploaded states
* prefers archiving over deleting
* aims for “works reliably enough” in small workflows

If you need:

* true bidirectional sync
* conflict resolution
* merge handling

…this tool is not that 🙂

---

## License

Do whatever you want with it.
But seriously: test on a throwaway repo first.

If this thing deletes your folders,
I will feel sorry for you — but I warned you.

