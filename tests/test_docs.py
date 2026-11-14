"""Every Python example in the documentation must run.

Blocks that need a person at the keyboard, a published pretrained model or an existing log file
are skipped; the rest run in a fresh namespace, in a temporary folder.
"""

import os
import re
from pathlib import Path

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
DOCS = Path(__file__).parent.parent / "docs"
SKIP_MARKERS = ("HumanSession(", "zoo.load(", "session.", "read_log(")
BLOCK = re.compile(r"```python\n(.*?)```", re.S)


def _blocks():
    for page in sorted(DOCS.rglob("*.md")):
        for index, code in enumerate(BLOCK.findall(page.read_text())):
            if not any(marker in code for marker in SKIP_MARKERS):
                yield pytest.param(code, id=f"{page.relative_to(DOCS)}#{index}")


@pytest.mark.parametrize("code", list(_blocks()))
def test_documentation_example_runs(code, tmp_path, monkeypatch):
    if "render_mode=" in code or "gridhunt.hat" in code:
        pytest.importorskip("pygame")
    if "pettingzoo.make(" in code:
        import pettingzoo

        if not hasattr(pettingzoo, "make"):
            pytest.skip("PettingZoo's registry needs PettingZoo 1.27 or later")
    monkeypatch.chdir(tmp_path)
    exec(compile(code, "<docs>", "exec"), {"__name__": "__docs__"})


def test_documentation_has_examples():
    assert len(list(_blocks())) >= 10
