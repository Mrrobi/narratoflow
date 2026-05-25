"""Verify the py.typed PEP 561 marker is bundled."""

from __future__ import annotations

from importlib import resources


def test_py_typed_marker_present() -> None:
    files = list(resources.files("narrato").iterdir())
    names = {f.name for f in files}
    assert "py.typed" in names, f"py.typed missing from narrato package; got {sorted(names)[:10]}..."
