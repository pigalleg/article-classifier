from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import json
import subprocess

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(cmd: list[str], cwd: Optional[Path] = None) -> Optional[str]:
    cwd = cwd or REPO_ROOT
    try:
        out = subprocess.check_output(["git", *cmd], cwd=str(cwd), stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


def get_git_commit_info(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    repo_root = Path(repo_root) if repo_root else REPO_ROOT
    commit = _git(["rev-parse", "HEAD"], cwd=repo_root)
    commit_short = _git(["rev-parse", "--short", "HEAD"], cwd=repo_root)
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    status = _git(["status", "--porcelain"], cwd=repo_root)
    dirty = bool(status)
    return {
        "commit": commit,
        "commit_short": commit_short,
        "branch": branch,
        "dirty": dirty,
    }


def _load_settings_yaml(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    try:
        p = Path(path)
        if not p.exists():
            return None
        with p.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return None


def get_run_metadata(repo_root: Optional[Path] = None, settings_path: Optional[Path] = None) -> Dict[str, Any]:
    repo_root = Path(repo_root) if repo_root else REPO_ROOT
    git = get_git_commit_info(repo_root=repo_root)
    settings = _load_settings_yaml(Path(settings_path) if settings_path else repo_root.joinpath("src/config/settings.yaml"))
    return {
        "git": git,
        "settings": settings,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
    }


def write_run_metadata_file(path: Path, info: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2, ensure_ascii=False)


def write_run_metadata(repo_root: Optional[Path], settings_path: Optional[Path], out_path: Path) -> None:
    info = get_run_metadata(repo_root=repo_root, settings_path=settings_path)
    write_run_metadata_file(out_path, info)
