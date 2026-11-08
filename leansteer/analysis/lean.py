"""Optional Lean type-check of generated statements against a built Mathlib.

``typecheck`` writes a batch of statements (each renamed ``thm_<i>``, proof
``sorry``) into one file under scratch and runs the project's *toolchain*
``lean`` binary directly, with ``LEAN_PATH`` pointing at the project's built
packages.  ``lake env`` is deliberately avoided: it may try to update or
re-clone packages inside the (possibly shared, read-only) project directory.

An error whose line falls inside statement *i*'s range marks it as failing.
This is a *proxy* for the benchmark's TC-Fail verdict (same kernel, possibly
a different Mathlib version, and a parse error can swallow the following
declaration); it never runs a prover.  A batch whose header fails to import
returns ``None`` (unknown) for every statement in it.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from leansteer import config

log = logging.getLogger(__name__)
HEADER = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 400000\n\nopen BigOperators Real Nat Topology Rat\n\n"
_NAME = re.compile(r"^theorem\s+\S+", re.MULTILINE)
_ERR = re.compile(r"^(?:[^\n:]*):(\d+):(\d+): error", re.MULTILINE)


def lean_binary(project_dir: str) -> str:
    tc = (Path(project_dir) / "lean-toolchain").read_text().strip() if (Path(project_dir) / "lean-toolchain").exists() else ""
    if tc:
        name = tc.replace("/", "--").replace(":", "---")
        cand = Path.home() / ".elan" / "toolchains" / name / "bin" / "lean"
        if cand.exists():
            return str(cand)
        log.warning("toolchain %s not installed under ~/.elan; falling back to `lean` on PATH", tc)
    return "lean"


def lean_path(project_dir: str) -> str:
    p = Path(project_dir)
    dirs = [p / ".lake" / "build" / "lib" / "lean"] + sorted((p / ".lake" / "packages").glob("*/.lake/build/lib/lean"))
    return ":".join(str(d) for d in dirs if d.exists())


def available(project_dir: str) -> bool:
    return any("mathlib" in d for d in lean_path(project_dir).split(":"))


_SORRY = re.compile(r"^(?:[^\n:]*):(\d+):(\d+): warning: declaration uses `sorry`", re.MULTILINE)


def check_decls(decls: List[str], project_dir: str, timeout: int = 900, allow_sorry: bool = True,
                header: str = HEADER) -> List[Optional[bool]]:
    """Compile ``decls`` (complete declarations, already renamed) in one file.
    A declaration passes when no error lands inside its line range and, unless
    ``allow_sorry``, no `sorry` warning does either.  ``None`` = could not be
    determined (header failed, timeout)."""
    tmpdir = config.SCRATCH_ROOT / "tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "LEAN_PATH": lean_path(project_dir)}
    binary = lean_binary(project_dir)
    lines, ranges = [header], []
    cur = header.count("\n") + 1
    for body in decls:
        n = body.count("\n") + 1
        ranges.append((cur, cur + n - 1))
        lines.append(body + "\n\n")
        cur += n + 1
    with tempfile.NamedTemporaryFile("w", suffix=".lean", delete=False, dir=str(tmpdir)) as f:
        f.write("".join(lines)); path = f.name
    try:
        r = subprocess.run([binary, path], cwd=str(tmpdir), env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        log.warning("lean batch timed out (%d declarations)", len(decls))
        return [None] * len(decls)
    finally:
        Path(path).unlink(missing_ok=True)
    text = r.stdout + r.stderr
    bad = [int(m.group(1)) for m in _ERR.finditer(text)]
    sorry = [int(m.group(1)) for m in _SORRY.finditer(text)]
    header_lines = header.count("\n")
    if any(l <= header_lines for l in bad) or r.returncode not in (0, 1):
        log.warning("lean batch unusable (exit %d): %s", r.returncode, text[:300])
        return [None] * len(decls)
    out = []
    for lo, hi in ranges:
        ok = not any(lo <= l <= hi for l in bad)
        if ok and not allow_sorry:
            ok = not any(lo <= l <= hi for l in sorry)
        out.append(ok)
    return out


def typecheck(statements: List[Optional[str]], project_dir: str, timeout: int = 900, batch: int = 40) -> List[Optional[bool]]:
    """Does each ``theorem ... := by sorry`` statement elaborate?"""
    out: List[Optional[bool]] = [None] * len(statements)
    todo = [i for i, s in enumerate(statements) if s]
    for b0 in range(0, len(todo), batch):
        ids = todo[b0:b0 + batch]
        decls = []
        for k, i in enumerate(ids):
            body = _NAME.sub(f"theorem thm_{k}", statements[i].strip(), count=1)
            if not body.rstrip().endswith("sorry"):
                body = body.rstrip() + " sorry"
            decls.append(body)
        for i, ok in zip(ids, check_decls(decls, project_dir, timeout, allow_sorry=True)):
            out[i] = ok
    return out


def verify_proofs(codes: List[str], project_dir: str, timeout: int = 900, batch: int = 8) -> List[Optional[bool]]:
    """Each code is one complete theorem *with* its proof (header stripped);
    passes iff it compiles with no error and no `sorry`."""
    out: List[Optional[bool]] = [None] * len(codes)
    for b0 in range(0, len(codes), batch):
        ids = list(range(b0, min(b0 + batch, len(codes))))
        decls = [_NAME.sub(f"theorem thm_{k}", codes[i].strip(), count=1) for k, i in enumerate(ids)]
        for i, ok in zip(ids, check_decls(decls, project_dir, timeout, allow_sorry=False)):
            out[i] = ok
    return out
