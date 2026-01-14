#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GitHub <-> Nextcloud WebDAV Repo Sync Tool

Features
- Interactive menu
- Pull: GitHub branch (default: main) -> WebDAV "Active" (download ZIP, unzip locally, upload unzipped)
- Push: WebDAV "Push" -> PR on GitHub (branch + commit + PR via GitHub REST API)
- Archive:
  - Active    -> Archive/Active-YYYYMMDD-HHMMSS-<rand>
  - Push      -> Archive/Push-YYYYMMDD-HHMMSS-<rand>
  - Incoming* -> Archive/Incoming-YYYYMMDD-HHMMSS-<rand> (cleanup from crashes)
- exclude.conf in project root on WebDAV is respected on Push
- Config (including secrets) stored in config.json next to this script (may call it a antifeature)
- Live-Reload in daemon mode: config.json is auto-reloaded when changed
- Safety against half-uploaded Push: waits push_stable_seconds and re-checks fingerprint (not that good for very big Repos)
- Buzzword-Bingo:
    Nexctloud, WebDAV... it works with cloud
    It has some if statements, so you can call it AI nowadays
    You can collaborate on github, so it supports synergy effects too

Notes / Limitations
- Push->PR uploads all files from Push tree as a commit; no deletes/renames detection.
- Polling only (no Nextcloud notifications/webhooks).
"""

from __future__ import annotations

import argparse
import base64
import fnmatch
import io
import json
import logging
import shutil
import sys
import time
import zipfile
import secrets
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Any
from urllib.parse import unquote

import requests
import xml.etree.ElementTree as ET
from logging.handlers import RotatingFileHandler


CONFIG_FILE = "config.json"
LOG_FILE = "sync.log"
USER_AGENT = "gh-nc-webdav-sync/0.6"
DEFAULT_EXCLUDE_NAME = "exclude.conf"


# -----------------------------
# Logging
# -----------------------------

def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if re-imported
    if logger.handlers:
        return

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Rotate: keep logs from growing forever
    fh = RotatingFileHandler(
        log_path,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=5,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(sh)


def safe_print(msg: str, level: int = logging.INFO) -> None:
    # Always both console + logfile (handlers will decide)
    logging.log(level, msg)


# -----------------------------
# Helpers
# -----------------------------

def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M")

def now_stamp_seconds() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def rand_tag(nbytes: int = 2) -> str:
    return secrets.token_hex(nbytes)

def ensure_trailing_slash(url: str) -> str:
    return url if url.endswith("/") else url + "/"

def join_webdav(base: str, *parts: str) -> str:
    base = ensure_trailing_slash(base)
    p = "/".join([s.strip("/").replace("\\", "/") for s in parts if s is not None and s != ""])
    return base + p

def is_folder_propfind_item(href: str) -> bool:
    return href.endswith("/")

def read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def file_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0

def cfg_signature(cfg: dict) -> Tuple:
    gh = cfg.get("github", {})
    wd = cfg.get("webdav", {})
    return (
        gh.get("token", ""),
        wd.get("base_url", ""),
        wd.get("username", ""),
        wd.get("password", ""),
        bool(wd.get("verify_tls", True)),
    )

def normalize_dav_href_to_path(href: str) -> str:
    if not href:
        return ""
    return unquote(href)

def split_remote_parent(remote_path: str) -> Tuple[str, str]:
    """
    Returns (parent, name) for a WebDAV remote path, regardless of trailing slash.
    """
    rp = remote_path.strip("/")
    if "/" not in rp:
        return ("", rp)
    parent, name = rp.rsplit("/", 1)
    return (parent, name)


# -----------------------------
# WebDAV client (Nextcloud)
# -----------------------------

class WebDAVClient:
    def __init__(self, base_url: str, username: str, password: str, verify_tls: bool = True):
        self.base_url = ensure_trailing_slash(base_url)
        self.auth = (username, password)
        self.verify_tls = verify_tls
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def _req(self, method: str, url: str, **kwargs) -> requests.Response:
        resp = self.session.request(
            method=method,
            url=url,
            auth=self.auth,
            verify=self.verify_tls,
            timeout=60,
            **kwargs,
        )
        if resp.status_code >= 400 and resp.status_code != 207:
            raise RuntimeError(f"WebDAV {method} {url} -> {resp.status_code}: {resp.text[:300]}")
        return resp

    def exists(self, remote_path: str) -> bool:
        url = join_webdav(self.base_url, remote_path)
        resp = self.session.request("HEAD", url, auth=self.auth, verify=self.verify_tls, timeout=30)
        return resp.status_code in (200, 204, 207)

    def mkcol(self, remote_path: str) -> None:
        url = join_webdav(self.base_url, remote_path)
        resp = self._req("MKCOL", url)
        if resp.status_code not in (201, 405):
            raise RuntimeError(f"MKCOL unexpected: {resp.status_code} {resp.text[:200]}")

    def ensure_dir(self, remote_path: str) -> None:
        parts = [p for p in remote_path.strip("/").split("/") if p]
        cur = ""
        for p in parts:
            cur = f"{cur}/{p}" if cur else p
            if not self.exists(cur + "/"):
                self.mkcol(cur)

    def delete(self, remote_path: str) -> None:
        """
        Robust DELETE:
        - If the path exists as a directory, prefer deleting with trailing slash.
        - 404 is OK.
        """
        rp = remote_path.strip("/")
        try:
            if self.exists(rp + "/"):
                rp = rp + "/"
        except Exception:
            pass

        url = join_webdav(self.base_url, rp)
        resp = self.session.request("DELETE", url, auth=self.auth, verify=self.verify_tls, timeout=60)
        if resp.status_code in (200, 204, 404):
            return
        if resp.status_code >= 400:
            raise RuntimeError(f"WebDAV DELETE {url} -> {resp.status_code}: {resp.text[:300]}")

    def propfind(self, remote_path: str, depth: int = 1) -> List[dict]:
        url = join_webdav(self.base_url, remote_path)
        headers = {"Depth": str(depth)}
        body = """<?xml version="1.0"?>
        <d:propfind xmlns:d="DAV:">
          <d:prop>
            <d:getetag/>
            <d:getlastmodified/>
            <d:resourcetype/>
            <d:getcontentlength/>
          </d:prop>
        </d:propfind>"""
        resp = self._req("PROPFIND", url, headers=headers, data=body)

        ns = {"d": "DAV:"}
        root = ET.fromstring(resp.text)
        out: List[dict] = []
        for r in root.findall("d:response", ns):
            href = r.findtext("d:href", default="", namespaces=ns)
            propstat = r.find("d:propstat", ns)
            if propstat is None:
                continue
            prop = propstat.find("d:prop", ns)
            if prop is None:
                continue
            etag = prop.findtext("d:getetag", default="", namespaces=ns)
            lastmod = prop.findtext("d:getlastmodified", default="", namespaces=ns)
            res_type = prop.find("d:resourcetype", ns)
            is_collection = False
            if res_type is not None and res_type.find("d:collection", ns) is not None:
                is_collection = True
            clen = prop.findtext("d:getcontentlength", default="", namespaces=ns)
            out.append({
                "href": href,
                "etag": etag.strip('"'),
                "lastmodified": lastmod,
                "is_collection": is_collection,
                "content_length": int(clen) if clen.isdigit() else None,
            })
        return out

    def list_dir(self, remote_dir: str) -> List[dict]:
        remote_dir_norm = remote_dir.strip("/") + "/"
        items = self.propfind(remote_dir_norm, depth=1)
        if not items:
            return []

        want_tail = "/" + remote_dir_norm
        children: List[dict] = []
        for it in items:
            href = normalize_dav_href_to_path(it.get("href", ""))
            if href.endswith(want_tail) or href.rstrip("/") == want_tail.rstrip("/"):
                continue
            children.append(it)
        return children

    def get_text(self, remote_path: str) -> Optional[str]:
        url = join_webdav(self.base_url, remote_path)
        resp = self.session.get(url, auth=self.auth, verify=self.verify_tls, timeout=60)
        if resp.status_code == 404:
            return None
        if resp.status_code >= 400:
            raise RuntimeError(f"GET {remote_path} -> {resp.status_code}: {resp.text[:200]}")
        return resp.text

    def download_file(self, remote_path: str) -> bytes:
        url = join_webdav(self.base_url, remote_path)
        resp = self._req("GET", url)
        return resp.content

    def upload_file(self, remote_path: str, data: bytes) -> None:
        parent = "/".join(remote_path.strip("/").split("/")[:-1])
        if parent:
            self.ensure_dir(parent)
        url = join_webdav(self.base_url, remote_path)
        self._req("PUT", url, data=data)

    def move(self, src_path: str, dst_path: str, overwrite: bool = False) -> None:
        src_url = join_webdav(self.base_url, src_path)
        dst_url = join_webdav(self.base_url, dst_path)
        headers = {"Destination": dst_url, "Overwrite": "T" if overwrite else "F"}
        self._req("MOVE", src_url, headers=headers)

    def is_dir_empty(self, remote_dir: str) -> bool:
        return len(self.list_dir(remote_dir)) == 0

    def upload_tree(self, local_dir: Path, remote_dir: str, exclude_patterns: List[str] = None) -> None:
        exclude_patterns = exclude_patterns or []
        self.ensure_dir(remote_dir)

        base = local_dir.resolve()
        for p in base.rglob("*"):
            rel = p.relative_to(base).as_posix()
            if rel == "":
                continue
            if self._match_exclude(rel, exclude_patterns):
                continue

            remote_path = f"{remote_dir.rstrip('/')}/{rel}"
            if p.is_dir():
                self.ensure_dir(remote_path)
            else:
                self.upload_file(remote_path, p.read_bytes())

    def download_tree(self, remote_dir: str, local_dir: Path) -> None:
        local_dir.mkdir(parents=True, exist_ok=True)
        self._download_tree_recursive(remote_dir.rstrip("/") + "/", local_dir)

    def _download_tree_recursive(self, remote_dir: str, local_dir: Path) -> None:
        for item in self.list_dir(remote_dir):
            href = item["href"]
            name = normalize_dav_href_to_path(href).rstrip("/").split("/")[-1]
            if not name:
                continue
            if item["is_collection"] or is_folder_propfind_item(href):
                sub_remote = remote_dir.rstrip("/") + "/" + name + "/"
                sub_local = local_dir / name
                sub_local.mkdir(parents=True, exist_ok=True)
                self._download_tree_recursive(sub_remote, sub_local)
            else:
                content = self.download_file(remote_dir.rstrip("/") + "/" + name)
                (local_dir / name).write_bytes(content)

    def dir_fingerprint(self, remote_dir: str) -> str:
        parts: List[str] = []
        self._fingerprint_recursive(remote_dir.strip("/") + "/", parts)
        parts.sort()
        return "|".join(parts)

    def _fingerprint_recursive(self, remote_dir: str, parts: List[str]) -> None:
        for c in self.list_dir(remote_dir):
            href = c.get("href", "")
            name = normalize_dav_href_to_path(href).rstrip("/").split("/")[-1]
            if not name:
                continue

            et = c.get("etag") or ""
            lm = c.get("lastmodified") or ""
            is_dir = bool(c.get("is_collection")) or is_folder_propfind_item(href)

            # Prefer ETag; fallback to lastmodified
            sig = et if et else lm

            key = f"{remote_dir}{name}"
            parts.append(f"{key}:{sig}:{'d' if is_dir else 'f'}")

            if is_dir:
                sub_remote = remote_dir.rstrip("/") + "/" + name + "/"
                self._fingerprint_recursive(sub_remote, parts)

    @staticmethod
    def _match_exclude(rel_posix: str, patterns: List[str]) -> bool:
        rp = rel_posix.lstrip("./")
        for pat in patterns:
            pat = pat.strip()
            if not pat or pat.startswith("#"):
                continue
            if pat.endswith("/"):
                if rp == pat[:-1] or rp.startswith(pat):
                    return True
            else:
                if fnmatch.fnmatch(rp, pat):
                    return True
        return False


# -----------------------------
# GitHub client (REST)
# -----------------------------

class GitHubClient:
    def __init__(self, token: str, api_base: str = "https://api.github.com"):
        self.token = token
        self.api_base = api_base.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": USER_AGENT
        })

    def _req(self, method: str, url: str, **kwargs) -> Any:
        resp = self.session.request(method, url, timeout=60, **kwargs)
        if resp.status_code >= 400:
            raise RuntimeError(f"GitHub {method} {url} -> {resp.status_code}: {resp.text[:400]}")
        if resp.text.strip() == "":
            return {}
        return resp.json()

    def list_repos(self) -> List[dict]:
        repos: List[dict] = []
        url = f"{self.api_base}/user/repos?per_page=100&sort=updated"
        while url:
            resp = self.session.get(url, timeout=60)
            if resp.status_code >= 400:
                raise RuntimeError(f"GitHub GET {url} -> {resp.status_code}: {resp.text[:200]}")
            repos.extend(resp.json())

            next_url = None
            link = resp.headers.get("Link", "")
            if link:
                parts = [p.strip() for p in link.split(",")]
                for p in parts:
                    if 'rel="next"' in p:
                        next_url = p.split(";")[0].strip()[1:-1]
            url = next_url
        return repos

    def get_branch_head_sha(self, owner: str, repo: str, branch: str = "main") -> str:
        url = f"{self.api_base}/repos/{owner}/{repo}/branches/{branch}"
        data = self._req("GET", url)
        return data["commit"]["sha"]

    def download_repo_zip(self, owner: str, repo: str, branch: str = "main") -> bytes:
        url = f"{self.api_base}/repos/{owner}/{repo}/zipball/{branch}"
        resp = self.session.get(url, timeout=120)
        if resp.status_code >= 400:
            raise RuntimeError(f"GitHub ZIP {url} -> {resp.status_code}: {resp.text[:200]}")
        return resp.content

    def _create_or_update_ref(self, owner: str, repo: str, branch_name: str, sha: str) -> None:
        ref_url = f"{self.api_base}/repos/{owner}/{repo}/git/refs"
        try:
            self._req("POST", ref_url, json={"ref": f"refs/heads/{branch_name}", "sha": sha})
            return
        except RuntimeError as e:
            msg = str(e)
            if " -> 422" not in msg:
                raise
        patch_url = f"{self.api_base}/repos/{owner}/{repo}/git/refs/heads/{branch_name}"
        self._req("PATCH", patch_url, json={"sha": sha, "force": False})

    def create_pr_from_local_tree(
        self,
        owner: str,
        repo: str,
        base_branch: str,
        branch_name: str,
        title: str,
        body: str,
        tree_dir: Path,
        exclude_patterns: List[str],
    ) -> str:
        base_sha = self.get_branch_head_sha(owner, repo, base_branch)

        commit_url = f"{self.api_base}/repos/{owner}/{repo}/git/commits/{base_sha}"
        base_commit = self._req("GET", commit_url)
        base_tree_sha = base_commit["tree"]["sha"]

        entries = []
        base = tree_dir.resolve()

        for p in base.rglob("*"):
            if p.is_dir():
                continue
            rel = p.relative_to(base).as_posix()
            if WebDAVClient._match_exclude(rel, exclude_patterns):
                continue

            content = p.read_bytes()
            blob_url = f"{self.api_base}/repos/{owner}/{repo}/git/blobs"
            blob = self._req("POST", blob_url, json={
                "content": base64.b64encode(content).decode("ascii"),
                "encoding": "base64"
            })
            entries.append({
                "path": rel,
                "mode": "100644",
                "type": "blob",
                "sha": blob["sha"]
            })

        if not entries:
            raise RuntimeError("Push folder contains no files to push (after exclude).")

        tree_url = f"{self.api_base}/repos/{owner}/{repo}/git/trees"
        new_tree = self._req("POST", tree_url, json={"base_tree": base_tree_sha, "tree": entries})

        new_commit_url = f"{self.api_base}/repos/{owner}/{repo}/git/commits"
        new_commit = self._req("POST", new_commit_url, json={
            "message": title,
            "tree": new_tree["sha"],
            "parents": [base_sha]
        })

        self._create_or_update_ref(owner, repo, branch_name, new_commit["sha"])

        pr_url = f"{self.api_base}/repos/{owner}/{repo}/pulls"
        pr = self._req("POST", pr_url, json={
            "title": title,
            "head": branch_name,  # same-repo branch
            "base": base_branch,
            "body": body
        })

        return pr.get("html_url", "")


# -----------------------------
# Sync logic
# -----------------------------

@dataclass
class RepoConfig:
    name: str
    owner: str
    repo: str
    webdav_root: str
    active_dir: str = "Active"
    push_dir: str = "Push"
    archive_dir: str = "Archive"
    exclude_file: str = DEFAULT_EXCLUDE_NAME
    branch: str = "main"
    poll_minutes: int = 5
    push_stable_seconds: int = 30  # safety delay for half-uploaded Push
    enabled: bool = True
    last_main_sha: str = ""
    last_push_fp: str = ""


class RepoSync:
    def __init__(self, gh: GitHubClient, wd: WebDAVClient, work_dir: Path):
        self.gh = gh
        self.wd = wd
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def _wait_for_stable_dir(self, remote_dir: str, stable_seconds: int) -> bool:
        if stable_seconds <= 0:
            return True
        fp1 = self.wd.dir_fingerprint(remote_dir)
        time.sleep(stable_seconds)
        fp2 = self.wd.dir_fingerprint(remote_dir)
        return fp1 == fp2

    def run_once(self, rc: RepoConfig) -> Tuple[RepoConfig, List[str]]:
        logs: List[str] = []
        if not rc.enabled:
            logs.append(f"[{rc.name}] disabled, skip.")
            return rc, logs

        project_root = rc.webdav_root.strip("/")
        active_remote = f"{project_root}/{rc.active_dir}".strip("/")
        push_remote = f"{project_root}/{rc.push_dir}".strip("/")
        arch_remote = f"{project_root}/{rc.archive_dir}".strip("/")
        exclude_remote = f"{project_root}/{rc.exclude_file}".strip("/")

        self.wd.ensure_dir(active_remote)
        self.wd.ensure_dir(push_remote)
        self.wd.ensure_dir(arch_remote)

        # Pull
        try:
            main_sha = self.gh.get_branch_head_sha(rc.owner, rc.repo, rc.branch)
        except Exception as e:
            logs.append(f"[{rc.name}] ERROR reading branch head: {e}")
            return rc, logs

        if main_sha != rc.last_main_sha:
            logs.append(f"[{rc.name}] branch changed ({rc.last_main_sha[:7]} -> {main_sha[:7]}), pulling to Active…")
            try:
                self._replace_active_from_github_zip(rc, active_remote, arch_remote)
                rc.last_main_sha = main_sha
                logs.append(f"[{rc.name}] Active updated from GitHub ({main_sha[:7]}).")
            except Exception as e:
                logs.append(f"[{rc.name}] ERROR updating Active: {e}")
                logging.exception(f"[{rc.name}] Active update failed")

        # Push
        try:
            push_fp = self.wd.dir_fingerprint(push_remote)
        except Exception as e:
            logs.append(f"[{rc.name}] ERROR reading Push folder: {e}")
            logging.exception(f"[{rc.name}] Push fingerprint failed")
            return rc, logs

        try:
            push_is_empty = self.wd.is_dir_empty(push_remote)
        except Exception as e:
            logs.append(f"[{rc.name}] ERROR checking Push emptiness: {e}")
            logging.exception(f"[{rc.name}] Push empty-check failed")
            return rc, logs

        if not push_is_empty and push_fp != rc.last_push_fp:
            stable_for = max(0, int(getattr(rc, "push_stable_seconds", 30)))
            logs.append(f"[{rc.name}] Push changed, waiting {stable_for}s for stability…")

            try:
                if not self._wait_for_stable_dir(push_remote, stable_for):
                    logs.append(f"[{rc.name}] Push not stable yet (still changing). Skip this round.")
                    rc.last_push_fp = self.wd.dir_fingerprint(push_remote)
                    return rc, logs

                pr_url = self._push_folder_to_pr(rc, push_remote, exclude_remote)
                logs.append(f"[{rc.name}] PR created: {pr_url if pr_url else '(no url returned)'}")

                self._archive_folder(push_remote, arch_remote, prefix="Push")
                self.wd.ensure_dir(push_remote)

                rc.last_push_fp = self.wd.dir_fingerprint(push_remote)

            except Exception as e:
                logs.append(f"[{rc.name}] ERROR pushing PR: {e}")
                logging.exception(f"[{rc.name}] Push/PR failed")
                rc.last_push_fp = push_fp
        else:
            rc.last_push_fp = push_fp

        return rc, logs

    def _cleanup_stale_incoming_folders(self, project_root: str, active_dir_name: str, arch_remote: str) -> None:
        """
        Cleanup stale Incoming folders from crashes.
        - Try MOVE to Archive/Incoming-...
        - If MOVE fails, try DELETE.
        """
        incoming_prefix = f"{active_dir_name}.__incoming__-"
        parent = project_root.strip("/")  # might be ""

        try:
            items = self.wd.list_dir(parent)
        except Exception:
            return

        for it in items:
            href = it.get("href", "")
            name = normalize_dav_href_to_path(href).rstrip("/").split("/")[-1]
            if not name:
                continue
            is_dir = bool(it.get("is_collection")) or is_folder_propfind_item(href)
            if not is_dir:
                continue
            if not name.startswith(incoming_prefix):
                continue

            stale_remote = f"{parent}/{name}".strip("/") + "/"
            try:
                self._archive_folder(stale_remote.rstrip("/"), arch_remote, prefix="Incoming")
                safe_print(f"Archive: stale incoming moved: {stale_remote} -> {arch_remote}")
            except Exception:
                try:
                    self.wd.delete(stale_remote)
                    safe_print(f"Cleanup: stale incoming deleted: {stale_remote}", level=logging.WARNING)
                except Exception:
                    pass

    def _ensure_active_removed_or_archived(self, active_remote: str, arch_remote: str) -> None:
        """
        Ensure Active won't block the Incoming->Active MOVE:
        - If empty: DELETE
        - Else: MOVE to Archive
        In errors: never blindly delete. Prefer "try archive", else leave it.
        """
        active_dir = active_remote.strip("/") + "/"
        if not self.wd.exists(active_dir):
            return

        try:
            if self.wd.is_dir_empty(active_remote):
                self.wd.delete(active_dir)
                safe_print(f"Active: deleted empty folder {active_dir}")
                return
            self._archive_folder(active_remote, arch_remote, prefix="Active")
            safe_print(f"Archive: Active moved to archive ({active_remote} -> {arch_remote})")
            return
        except Exception:
            # Better workflow: try archive directly, without relying on exists() (which can lie under some setups).
            try:
                self._archive_folder(active_remote, arch_remote, prefix="Active")
                safe_print(f"Archive (fallback): Active moved to archive ({active_remote} -> {arch_remote})",
                           level=logging.WARNING)
                return
            except Exception:
                # Last attempt: only delete if we can confirm emptiness *now*.
                try:
                    if self.wd.is_dir_empty(active_remote):
                        self.wd.delete(active_dir)
                        safe_print(f"Active (last resort): deleted empty folder {active_dir}",
                                   level=logging.WARNING)
                except Exception:
                    safe_print(f"Active: could not archive or confirm empty; leaving as-is: {active_dir}",
                               level=logging.WARNING)
                return

    def _swap_incoming_to_active(self, incoming_remote: str, active_remote: str, arch_remote: str) -> None:
        """
        MOVE incoming -> Active, retry if destination exists or races.
        """
        src = incoming_remote.strip("/") + "/"
        dst = active_remote.strip("/") + "/"

        try:
            self.wd.move(src, dst, overwrite=False)
            safe_print(f"Active swap: MOVE {src} -> {dst}")
            return
        except Exception as e1:
            self._ensure_active_removed_or_archived(active_remote, arch_remote)
            try:
                self.wd.move(src, dst, overwrite=False)
                safe_print(f"Active swap (retry): MOVE {src} -> {dst}")
                return
            except Exception as e2:
                raise RuntimeError(f"Swap incoming->Active failed (1st={e1}, 2nd={e2})")

    def _replace_active_from_github_zip(self, rc: RepoConfig, active_remote: str, arch_remote: str) -> None:
        """
        Atomic-ish Active replace:
        0) Cleanup old Incoming leftovers
        1) Download ZIP, unzip locally
        2) Upload to Incoming folder
        3) Remove/archive Active (if needed)
        4) MOVE Incoming -> Active
        """
        project_root, active_name = split_remote_parent(active_remote)
        self._cleanup_stale_incoming_folders(project_root, active_name, arch_remote)

        zip_bytes = self.gh.download_repo_zip(rc.owner, rc.repo, rc.branch)

        tmp = self.work_dir / f"tmp_{rc.owner}_{rc.repo}_zip"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            z.extractall(tmp)

        top_dirs = [p for p in tmp.iterdir() if p.is_dir()]
        if not top_dirs:
            raise RuntimeError("ZIP does not contain a top-level folder (unexpected).")
        src_root = top_dirs[0]

        incoming_remote = f"{active_remote}.__incoming__-{now_stamp_seconds()}-{rand_tag(2)}".strip("/")
        self.wd.ensure_dir(incoming_remote)
        safe_print(f"[{rc.name}] Active pull: uploading to incoming {incoming_remote}")
        self.wd.upload_tree(src_root, incoming_remote)

        self._ensure_active_removed_or_archived(active_remote, arch_remote)
        self._swap_incoming_to_active(incoming_remote, active_remote, arch_remote)

        shutil.rmtree(tmp, ignore_errors=True)

    def _push_folder_to_pr(self, rc: RepoConfig, push_remote: str, exclude_remote: str) -> str:
        exclude_text = self.wd.get_text(exclude_remote)
        patterns: List[str] = []
        if exclude_text:
            patterns = [
                line.strip() for line in exclude_text.splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]

        tmp_push = self.work_dir / f"tmp_push_{rc.owner}_{rc.repo}"
        if tmp_push.exists():
            shutil.rmtree(tmp_push)
        tmp_push.mkdir(parents=True, exist_ok=True)

        self.wd.download_tree(push_remote, tmp_push)

        rand = secrets.token_hex(3)
        branch_name = f"webdav-push-{now_stamp_seconds()}-{rand}"
        title = f"WebDAV Push {now_stamp()}"
        body = "Automatically created from Nextcloud/WebDAV Push folder."

        pr_url = self.gh.create_pr_from_local_tree(
            owner=rc.owner,
            repo=rc.repo,
            base_branch=rc.branch,
            branch_name=branch_name,
            title=title,
            body=body,
            tree_dir=tmp_push,
            exclude_patterns=patterns
        )

        shutil.rmtree(tmp_push, ignore_errors=True)
        return pr_url

    def _archive_folder(self, folder_remote: str, archive_remote: str, prefix: str) -> None:
        stamp = f"{now_stamp_seconds()}-{rand_tag(2)}"
        src = folder_remote.strip("/") + "/"
        dst = f"{archive_remote.rstrip('/')}/{prefix}-{stamp}/"
        self.wd.ensure_dir(archive_remote)
        self.wd.move(src, dst, overwrite=False)
        safe_print(f"Archive: MOVE {src} -> {dst}")


# -----------------------------
# Config / Wizard / Menu
# -----------------------------

def prompt(msg: str, default: Optional[str] = None, secret: bool = False) -> str:
    if default is not None:
        msg2 = f"{msg} [{default}]: "
    else:
        msg2 = f"{msg}: "
    if secret:
        import getpass
        val = getpass.getpass(msg2)
    else:
        val = input(msg2).strip()
    if val == "" and default is not None:
        return default
    return val

def choose_from_list(items: List[str], prompt_text: str) -> int:
    while True:
        safe_print(prompt_text)
        for i, it in enumerate(items, start=1):
            safe_print(f"  {i}) {it}")
        sel = input("Select (number, 0=cancel): ").strip()
        if sel.isdigit():
            n = int(sel)
            if n == 0:
                return -1
            if 1 <= n <= len(items):
                return n - 1
        safe_print("Invalid. Try again.", level=logging.WARNING)

def load_repo_configs(cfg: dict) -> List[RepoConfig]:
    return [RepoConfig(**r) for r in cfg.get("repos", [])]

def dump_repo_configs(repos: List[RepoConfig]) -> List[dict]:
    return [r.__dict__ for r in repos]

def build_clients(cfg: dict) -> Tuple[GitHubClient, WebDAVClient]:
    gh = GitHubClient(cfg["github"]["token"])
    wd_cfg = cfg["webdav"]
    wd = WebDAVClient(
        base_url=wd_cfg["base_url"],
        username=wd_cfg["username"],
        password=wd_cfg["password"],
        verify_tls=bool(wd_cfg.get("verify_tls", True))
    )
    return gh, wd

def save_cfg(cfg_path: Path, cfg: dict, repos: List[RepoConfig]) -> None:
    cfg["repos"] = dump_repo_configs(repos)
    write_json(cfg_path, cfg)

def setup_wizard(config_path: Path) -> dict:
    safe_print("== Setup Wizard ==")

    gh_token = prompt("GitHub Token (needs repo + PR permissions)", secret=True)
    wd_url = prompt("Nextcloud WebDAV Base URL (e.g. https://cloud.tld/remote.php/dav/files/USER/)")
    wd_user = prompt("Nextcloud Username")
    wd_pass = prompt("Nextcloud Password / App Password", secret=True)
    verify_tls_in = prompt("Verify TLS certificates? (yes/no)", default="yes")
    verify_tls = verify_tls_in.lower().startswith("y")
    poll_minutes = int(prompt("Default polling interval (minutes)", default="5"))
    push_stable_seconds = int(prompt("Push stability delay (seconds)", default="30"))

    gh = GitHubClient(gh_token)
    repos = gh.list_repos()

    display = []
    lookup: List[Tuple[str, str]] = []
    for r in repos:
        full = r.get("full_name", "")
        private = r.get("private", False)
        display.append(f"{full} {'[private]' if private else '[public]'}")
        if "/" in full:
            owner, name = full.split("/", 1)
            lookup.append((owner, name))
        else:
            lookup.append(("", ""))

    selected: List[RepoConfig] = []
    while True:
        idx = choose_from_list(display, "Choose a repo to configure (you can add multiple):")
        if idx < 0:
            break
        owner, name = lookup[idx]
        if not owner or not name:
            safe_print("Could not parse repo name, skipping.", level=logging.WARNING)
            continue

        proj_name = prompt("Project name (folder/display)", default=name)
        webdav_root = prompt(
            "WebDAV project root (relative to Base URL), e.g. Projects/ProjectA",
            default=f"Projects/{proj_name}"
        )

        rc = RepoConfig(
            name=proj_name,
            owner=owner,
            repo=name,
            webdav_root=webdav_root,
            poll_minutes=poll_minutes,
            push_stable_seconds=push_stable_seconds,
            enabled=True,
            branch="main",
        )
        selected.append(rc)
        safe_print(f"OK: added {proj_name}\n")

        more = prompt("Add another repo? (yes/no)", default="yes")
        if not more.lower().startswith("y"):
            break

    if not selected:
        safe_print("No repo selected. Setup aborted.", level=logging.ERROR)
        sys.exit(1)

    cfg = {
        "github": {"token": gh_token},
        "webdav": {
            "base_url": wd_url,
            "username": wd_user,
            "password": wd_pass,
            "verify_tls": verify_tls
        },
        "defaults": {"poll_minutes": poll_minutes, "push_stable_seconds": push_stable_seconds},
        "repos": dump_repo_configs(selected)
    }

    write_json(config_path, cfg)
    safe_print(f"Config saved to: {config_path}")
    return cfg


def add_repo_interactive(cfg: dict, repos_cfg: List[RepoConfig]) -> Tuple[dict, List[RepoConfig]]:
    """
    Add (or update) a repo config without editing JSON manually.
    - Uses existing github token in cfg.
    - Lists GitHub repos and lets the user pick one.
    - If owner/repo already exists in config, updates that entry instead of duplicating.
    """
    if not cfg or "github" not in cfg or "token" not in cfg["github"]:
        safe_print("Config missing GitHub token. Run setup first.", level=logging.ERROR)
        return cfg, repos_cfg

    defaults = cfg.get("defaults", {})
    default_poll = int(defaults.get("poll_minutes", 5))
    default_stable = int(defaults.get("push_stable_seconds", 30))

    gh = GitHubClient(cfg["github"]["token"])
    try:
        gh_repos = gh.list_repos()
    except Exception as e:
        safe_print(f"ERROR: could not list GitHub repos: {e}", level=logging.ERROR)
        return cfg, repos_cfg

    display: List[str] = []
    lookup: List[Tuple[str, str, bool]] = []
    for r in gh_repos:
        full = r.get("full_name", "")
        private = bool(r.get("private", False))
        if "/" not in full:
            continue
        owner, name = full.split("/", 1)
        display.append(f"{full} {'[private]' if private else '[public]'}")
        lookup.append((owner, name, private))

    if not display:
        safe_print("No GitHub repos found for this token.", level=logging.WARNING)
        return cfg, repos_cfg

    idx = choose_from_list(display, "Choose a repo to add/update:")
    if idx < 0:
        return cfg, repos_cfg

    owner, name, _private = lookup[idx]

    proj_name = prompt("Project name (folder/display)", default=name)
    webdav_root = prompt(
        "WebDAV project root (relative to Base URL), e.g. Projects/ProjectA",
        default=f"Projects/{proj_name}"
    )
    branch = prompt("Branch", default="main")

    poll_minutes_in = prompt("Polling interval (minutes)", default=str(default_poll))
    poll_minutes = int(poll_minutes_in) if poll_minutes_in.isdigit() and int(poll_minutes_in) >= 1 else default_poll

    stable_in = prompt("Push stability delay (seconds)", default=str(default_stable))
    push_stable_seconds = int(stable_in) if stable_in.isdigit() and int(stable_in) >= 0 else default_stable

    enabled_in = prompt("Enable this repo? (yes/no)", default="yes")
    enabled = enabled_in.lower().startswith("y")

    new_rc = RepoConfig(
        name=proj_name,
        owner=owner,
        repo=name,
        webdav_root=webdav_root,
        branch=branch,
        poll_minutes=poll_minutes,
        push_stable_seconds=push_stable_seconds,
        enabled=enabled,
    )

    # Update if already exists by owner/repo (canonical identity), otherwise append.
    updated = False
    for i, existing in enumerate(repos_cfg):
        if existing.owner == owner and existing.repo == name:
            # Preserve state fields unless user explicitly wants to reset (we don't).
            new_rc.last_main_sha = existing.last_main_sha
            new_rc.last_push_fp = existing.last_push_fp
            repos_cfg[i] = new_rc
            updated = True
            break

    if updated:
        safe_print(f"Updated existing config for {owner}/{name} -> name='{new_rc.name}', root='{new_rc.webdav_root}'")
    else:
        repos_cfg.append(new_rc)
        safe_print(f"Added repo {owner}/{name} -> name='{new_rc.name}', root='{new_rc.webdav_root}'")

    cfg["repos"] = dump_repo_configs(repos_cfg)
    return cfg, repos_cfg


# -----------------------------
# Runner: reload often, sync on schedule
# -----------------------------

def run_scheduler_loop(cfg_path: Path, reload_seconds: int) -> int:
    cfg = read_json(cfg_path)
    if not cfg:
        safe_print("No config.json found. Run with --setup or start without flags.", level=logging.ERROR)
        return 1

    repos = load_repo_configs(cfg)
    gh, wd = build_clients(cfg)
    sync = RepoSync(gh, wd, work_dir=cfg_path.parent / ".work")

    last_mtime = file_mtime(cfg_path)
    last_sig = cfg_signature(cfg)

    next_sync_at = time.time()
    safe_print("== Daemon started (Live-Reload enabled). Stop with CTRL+C ==")

    def compute_poll_minutes(rs: List[RepoConfig]) -> int:
        enabled = [r for r in rs if r.enabled]
        if not enabled:
            return 5
        return min(max(1, r.poll_minutes) for r in enabled)

    try:
        while True:
            m = file_mtime(cfg_path)
            if m > last_mtime:
                new_cfg = read_json(cfg_path)
                if new_cfg:
                    safe_print("== Live-Reload: config.json reloaded ==")
                    cfg = new_cfg
                    repos = load_repo_configs(cfg)
                    new_sig = cfg_signature(cfg)

                    if new_sig != last_sig:
                        safe_print("== Live-Reload: credentials changed -> rebuilding clients ==",
                                   level=logging.WARNING)
                        gh, wd = build_clients(cfg)
                        sync = RepoSync(gh, wd, work_dir=cfg_path.parent / ".work")
                        last_sig = new_sig

                    # run soon after reload
                    next_sync_at = min(next_sync_at, time.time() + 1.0)
                    last_mtime = m
                else:
                    safe_print("WARN: config.json changed but could not be read. Will retry later.",
                               level=logging.WARNING)

            now = time.time()
            if now >= next_sync_at:
                updated = []
                for r in repos:
                    r2, lines = sync.run_once(r)
                    updated.append(r2)
                    for line in lines:
                        safe_print(line, level=(logging.ERROR if "ERROR" in line else logging.INFO))
                repos = updated

                cfg["repos"] = dump_repo_configs(repos)
                write_json(cfg_path, cfg)

                poll = compute_poll_minutes(repos)
                next_sync_at = now + poll * 60
                safe_print(f"-- next sync in {poll} min; reload check every ~{reload_seconds}s --")

            sleep_to_sync = max(0.0, next_sync_at - time.time())
            sleep_s = min(max(1.0, float(reload_seconds)), sleep_to_sync) if sleep_to_sync > 0 else max(1.0, float(reload_seconds))
            time.sleep(sleep_s)

    except KeyboardInterrupt:
        safe_print("Stop.")
        return 0


# -----------------------------
# Menu
# -----------------------------

def menu(cfg_path: Path, cfg: dict, reload_seconds: int) -> int:
    repos = load_repo_configs(cfg)

    while True:
        safe_print("\n== GitHub <-> Nextcloud Sync Menu ==")
        safe_print("1) Show configured repos")
        safe_print("2) Run once (all repos)")
        safe_print("3) Start daemon (polling, live-reload)")
        safe_print("4) Toggle repo enabled/disabled")
        safe_print("5) Change poll minutes for a repo")
        safe_print("6) Re-run full setup (overwrites config)")
        safe_print("7) Add repo to sync (interactive)")
        safe_print("0) Exit")

        choice = input("Select: ").strip()

        if choice == "0":
            return 0

        if choice == "1":
            safe_print("\n== Configured repos ==")
            for i, r in enumerate(repos, start=1):
                safe_print(
                    f"{i}) {r.name}: {r.owner}/{r.repo} -> {r.webdav_root} "
                    f"(enabled={r.enabled}, poll={r.poll_minutes}m, stable={r.push_stable_seconds}s, branch={r.branch})"
                )
            continue

        if choice == "6":
            cfg = setup_wizard(cfg_path)
            repos = load_repo_configs(cfg)
            continue

        if choice == "7":
            cfg, repos = add_repo_interactive(cfg, repos)
            save_cfg(cfg_path, cfg, repos)
            continue

        if choice == "2":
            gh, wd = build_clients(cfg)
            sync = RepoSync(gh, wd, work_dir=cfg_path.parent / ".work")

            updated = []
            for r in repos:
                r2, lines = sync.run_once(r)
                updated.append(r2)
                for line in lines:
                    safe_print(line, level=(logging.ERROR if "ERROR" in line else logging.INFO))
            repos = updated
            save_cfg(cfg_path, cfg, repos)
            continue

        if choice == "3":
            return run_scheduler_loop(cfg_path, reload_seconds=reload_seconds)

        if choice == "4":
            if not repos:
                safe_print("No repos configured.", level=logging.WARNING)
                continue
            items = [f"{r.name} (enabled={r.enabled})" for r in repos]
            idx = choose_from_list(items, "Choose a repo to toggle:")
            if idx < 0:
                continue
            repos[idx].enabled = not repos[idx].enabled
            safe_print(f"{repos[idx].name}: enabled={repos[idx].enabled}")
            save_cfg(cfg_path, cfg, repos)
            continue

        if choice == "5":
            if not repos:
                safe_print("No repos configured.", level=logging.WARNING)
                continue
            items = [f"{r.name} (poll={r.poll_minutes}m)" for r in repos]
            idx = choose_from_list(items, "Choose a repo:")
            if idx < 0:
                continue
            newv = prompt("New poll minutes", default=str(repos[idx].poll_minutes))
            if newv.isdigit() and int(newv) >= 1:
                repos[idx].poll_minutes = int(newv)
                safe_print(f"{repos[idx].name}: poll={repos[idx].poll_minutes}m")
                save_cfg(cfg_path, cfg, repos)
            else:
                safe_print("Invalid (must be a number >= 1).", level=logging.WARNING)
            continue

        safe_print("Unknown choice.", level=logging.WARNING)


# -----------------------------
# CLI entry
# -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="GitHub <-> Nextcloud WebDAV Repo Sync")
    ap.add_argument("--setup", action="store_true", help="Force setup wizard")
    ap.add_argument("--run", action="store_true", help="Daemon mode (polling, live-reload)")
    ap.add_argument("--once", action="store_true", help="Run once and exit")
    ap.add_argument("--list", action="store_true", help="List repos from config")
    ap.add_argument("--config", default=CONFIG_FILE, help="Path to config.json")
    ap.add_argument("--reload-seconds", type=int, default=10, help="How often to check config.json for changes")
    args = ap.parse_args()

    tool_dir = Path(__file__).resolve().parent
    cfg_path = (tool_dir / args.config).resolve()

    setup_logging(tool_dir / LOG_FILE)
    safe_print("=== GitHub <-> Nextcloud Sync started ===")

    cfg = read_json(cfg_path)
    if args.setup or cfg is None:
        cfg = setup_wizard(cfg_path)

    if args.list:
        repos = load_repo_configs(cfg)
        safe_print("== Configured repos ==")
        for r in repos:
            safe_print(f"- {r.name}: {r.owner}/{r.repo} -> {r.webdav_root} (enabled={r.enabled})")
        return 0

    # No flags => menu
    if not (args.run or args.once or args.list or args.setup):
        return menu(cfg_path, cfg, reload_seconds=args.reload_seconds)

    if args.once:
        repos = load_repo_configs(cfg)
        gh, wd = build_clients(cfg)
        sync = RepoSync(gh, wd, work_dir=tool_dir / ".work")

        updated = []
        for r in repos:
            r2, lines = sync.run_once(r)
            updated.append(r2)
            for line in lines:
                safe_print(line, level=(logging.ERROR if "ERROR" in line else logging.INFO))

        cfg["repos"] = dump_repo_configs(updated)
        write_json(cfg_path, cfg)
        return 0

    if args.run:
        return run_scheduler_loop(cfg_path, reload_seconds=args.reload_seconds)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

