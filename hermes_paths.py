"""Where Hermes keeps its home, and how a configured ``~/.hermes/...`` path maps onto it.

Hermes's home is ``~/.hermes`` only by default. ``HERMES_HOME`` moves it, a profile
moves it per context, and on Windows the default is ``%LOCALAPPDATA%\\hermes``. A
path hardcoded to ``~/.hermes`` then reads a config that does not exist and writes
logs nobody else reads. Every home-relative path in this plugin goes through here.

Standalone scripts (the dashboard, ``view_log``) import this by its bare name, so it
must not import anything from the plugin package.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_DEFAULT_PREFIXES = ("~/.hermes", "~\\.hermes")


def hermes_home() -> Path:
    """Hermes's home as Hermes itself resolves it, read fresh on every call."""
    try:
        from hermes_constants import get_hermes_home
    except Exception:
        get_hermes_home = None
    if get_hermes_home is not None:
        try:
            return Path(get_hermes_home())
        except Exception:
            pass
    configured = os.environ.get("HERMES_HOME", "").strip()
    if configured:
        return Path(os.path.expanduser(os.path.expandvars(configured)))
    if sys.platform == "win32":
        local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
        base = Path(local_appdata) if local_appdata else Path.home() / "AppData" / "Local"
        return base / "hermes"
    return Path.home() / ".hermes"


def hermes_path(value: object) -> Path:
    """A configured path, with a leading ``~/.hermes`` read as Hermes's real home."""
    text = str(value)
    for prefix in _DEFAULT_PREFIXES:
        if text == prefix:
            return hermes_home()
        if text.startswith(prefix) and text[len(prefix)] in "/\\":
            return hermes_home() / text[len(prefix) + 1:]
    return Path(os.path.expanduser(text))
