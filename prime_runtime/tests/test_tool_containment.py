"""The repository tools stay inside the artifact directory.

``ToolExecutor._safe_path`` must refuse ``..`` traversal, absolute paths,
sibling directories whose names share the root's prefix, and symlinks that
leave the root; ``run_tests`` must refuse a call with any target outside the
root or any option outside its allowlist, instead of dropping the refused
targets and running a wider (possibly the full) suite.
"""
from __future__ import annotations

import os
import subprocess
import sys
import types

import pytest

_TESTS = os.path.dirname(os.path.abspath(__file__))
_P = os.path.dirname(os.path.dirname(_TESTS))
for _x in (_P, os.path.join(_P, "runner")):
    if _x not in sys.path:
        sys.path.insert(0, _x)

import arms as arms_mod  # noqa: E402


@pytest.fixture()
def layout(tmp_path):
    """tmp/art (the artifact), tmp/art_evil (a sibling sharing the prefix),
    and tmp/outside, with symlinks from inside art to both."""
    art = tmp_path / "art"
    (art / "pkg").mkdir(parents=True)
    (art / "pkg" / "mod.py").write_text("x = 1\nSECRET_INSIDE = 0\n")
    (art / "tests").mkdir()
    (art / "tests" / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    (art / "setup.cfg").write_text("[metadata]\nname = demo\n")
    evil = tmp_path / "art_evil"
    evil.mkdir()
    (evil / "secret.txt").write_text("SECRET_SIBLING\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("SECRET_OUTSIDE\n")
    (art / "link_file.txt").symlink_to(outside / "secret.txt")
    (art / "link_dir").symlink_to(outside, target_is_directory=True)
    (art / "inner_link.py").symlink_to(art / "pkg" / "mod.py")
    inst = types.SimpleNamespace(artifact_dir=str(art), records={}, atoms=(),
                                 notes_for_annotators=None)
    return arms_mod.ToolExecutor(inst), art, outside


@pytest.mark.parametrize("rel", [
    "../outside/secret.txt",          # traversal
    "pkg/../../outside/secret.txt",   # traversal through a real subdirectory
    "../art_evil/secret.txt",         # sibling sharing the root's prefix
    "link_file.txt",                  # symlinked file leaving the root
    "link_dir/secret.txt",            # symlinked directory leaving the root
])
def test_safe_path_refuses_escapes(layout, rel):
    ex, _, _ = layout
    assert ex._safe_path(rel) is None
    obs = ex.execute("inspect_file", {"path": rel})
    assert obs.get("error") == "file not found" and "content" not in obs


def test_safe_path_refuses_absolute_paths(layout):
    ex, _, outside = layout
    target = str(outside / "secret.txt")
    assert ex._safe_path(target) is None
    assert "content" not in ex.execute("inspect_file", {"path": target})
    assert "content" not in ex.execute("inspect_config", {"path": target})


def test_safe_path_accepts_paths_inside(layout):
    ex, art, _ = layout
    real = os.path.realpath(str(art))
    assert ex._safe_path("pkg/mod.py") == os.path.join(real, "pkg", "mod.py")
    assert ex._safe_path("pkg/../setup.cfg") == os.path.join(real, "setup.cfg")
    # a symlink whose target stays inside the root is followed
    assert ex._safe_path("inner_link.py") == os.path.join(real, "pkg", "mod.py")
    assert ex._safe_path("") == real
    assert "x = 1" in ex.execute("inspect_file", {"path": "pkg/mod.py"})["content"]
    assert "demo" in ex.execute("inspect_config", {"path": "setup.cfg"})["content"]


def test_safe_path_when_root_is_reached_through_a_symlink(tmp_path, layout):
    _, art, _ = layout
    alias = tmp_path / "alias"
    alias.symlink_to(art, target_is_directory=True)
    inst = types.SimpleNamespace(artifact_dir=str(alias), records={}, atoms=(),
                                 notes_for_annotators=None)
    ex = arms_mod.ToolExecutor(inst)
    assert ex._safe_path("pkg/mod.py") is not None
    assert ex._safe_path("../art_evil/secret.txt") is None
    hits = ex.execute("search_repo", {"pattern": "x = 1"})["matches"]
    assert [h["path"] for h in hits] == ["inner_link.py", "pkg/mod.py"]


def test_search_repo_stays_inside(layout):
    ex, _, _ = layout
    obs = ex.execute("search_repo", {"pattern": "SECRET"})
    assert {m["path"] for m in obs["matches"]} == {"inner_link.py", "pkg/mod.py"}
    for scope in ("..", "../outside", "../art_evil", "link_dir"):
        scoped = ex.execute("search_repo", {"pattern": "SECRET", "path": scope})
        assert scoped["match_count"] == 0 and scoped.get("error") == "path not found"


def _capture_run(monkeypatch):
    seen = []

    def fake_run(cmd, **kw):
        seen.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="1 passed", stderr="")

    monkeypatch.setattr(arms_mod.subprocess, "run", fake_run)
    return seen


@pytest.mark.parametrize("suite", [
    "../outside",                      # every target outside: must not run the full suite
    "tests/test_ok.py ../outside",     # one target outside: refuse the call
    "link_dir",                        # symlink leaving the root
    "/etc",                            # absolute path
    "-p some_plugin tests",            # options that load code ...
    "--basetemp=/tmp/x tests",         # ... or touch files outside the root
    "--rootdir=.. tests",
    "-c ../outside/secret.txt",
    "--ignore=../outside tests",       # an allowed option with a path outside
])
def test_run_tests_fails_closed(layout, monkeypatch, suite):
    ex, _, _ = layout
    seen = _capture_run(monkeypatch)
    obs = ex.execute("run_tests", {"suite": suite})
    assert seen == [], "a refused call must not start pytest"
    assert obs["exit_code"] == -1 and obs["error"].startswith("refused:")


@pytest.mark.parametrize("suite, extra", [
    ("default", []),
    ("tests/test_ok.py", ["tests/test_ok.py"]),
    ("tests/test_ok.py::test_ok -q", ["tests/test_ok.py::test_ok", "-q"]),
    ("tests --ignore=tests/test_ok.py -x", ["tests", "--ignore=tests/test_ok.py", "-x"]),
    # a path that does not exist inside the root is passed on; pytest reports it
    ("all", ["all"]),
])
def test_run_tests_passes_accepted_targets(layout, monkeypatch, suite, extra):
    ex, _, _ = layout
    seen = _capture_run(monkeypatch)
    obs = ex.execute("run_tests", {"suite": suite})
    assert obs["exit_code"] == 0
    assert seen == [[sys.executable, "-m", "pytest", "-q"] + extra]
