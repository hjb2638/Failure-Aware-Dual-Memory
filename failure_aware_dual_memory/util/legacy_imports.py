"""Compatibility helpers for loading legacy serialized artifacts."""

from __future__ import annotations

import importlib
import sys


def install_agent4crys_aliases() -> None:
    """Map legacy ``agent4crys`` module names onto ``failure_aware_dual_memory``."""

    root_name = "failure_aware_dual_memory"
    legacy_root = "agent4crys"

    if legacy_root in sys.modules:
        return

    root_module = importlib.import_module(root_name)
    sys.modules[legacy_root] = root_module
