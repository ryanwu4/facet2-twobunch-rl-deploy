"""v1 is read-only by construction: no code in the package or the notebooks can write to the machine.

Every line of the package and of each notebook's code cells is checked against FORBIDDEN, and PV names
may appear only in pvs.py. The last tests check the patterns themselves: they must catch these write
calls and PV names and pass these reads.
"""
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

FORBIDDEN = {
    r"\bput\b": "pyepics PV.put / ca.put, p4p Context.put",
    r"caput|pvput": "pyepics caput, EPICS command-line writes",
    r"\.value\s*=(?!=)": "assigning PV.value writes the PV in pyepics",
    r"\b(Device|Motor)\b": "pyepics classes whose attribute assignment writes",
    r"monitor_delta": "pyepics writes the PV's MDEL field when a PV is given monitor_delta",
    r"\.write\(": "caproto client writes (files are written with write_text)",
    r"\brpc\b": "AIDA-PVA (multiknob) writes are RPC calls",
    r"slc_mkb|set_mkb": "the SCP multiknob wrapper",
    r"\bcaproto\b|\bp4p\b": "Channel Access / PVAccess clients other than pyepics",
    r"subprocess|os\.system|os\.popen": "shell commands (could run caput)",
}
# a string literal shaped like a PV name: at least three colon-separated parts, e.g. "KLYS:LI11:11:PREQ"
PV_NAME = r"[\"'][A-Z0-9_]+(:[A-Za-z0-9_]+){2,}"


def code_lines():
    """(where, line number, text) for every line of the package and of every notebook code cell."""
    for f in sorted((REPO / "twobunch_deploy").rglob("*.py")) + sorted(REPO.glob("*.py")):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            yield f.relative_to(REPO), i, line
    for nb in sorted(REPO.glob("*.ipynb")):
        for c, cell in enumerate(json.loads(nb.read_text())["cells"]):
            if cell["cell_type"] == "code":
                for i, line in enumerate("".join(cell["source"]).splitlines(), 1):
                    yield f"{nb.name} cell {c}", i, line


def test_no_write_calls():
    hits = [f"{where}:{i}: {line.strip()}  ({why})" for where, i, line in code_lines()
            for pattern, why in FORBIDDEN.items() if re.search(pattern, line)]
    assert not hits, "\n".join(hits)


def test_pv_names_only_in_pvs():
    hits = [f"{where}:{i}: {line.strip()}" for where, i, line in code_lines()
            if str(where) != "twobunch_deploy/pvs.py" and re.search(PV_NAME, line)]
    assert not hits, "\n".join(hits)


@pytest.mark.parametrize("line", [
    "pv.put(1.0)", "epics.caput('X', 1)", "pv.value = 2", "ctxt.put('X', 1)", "from epics import Device",
    "client_pv.write([1])", "ctx.rpc('X', args)", "slc_mkb.set_mkb('l2_phase', 1)", "import p4p",
    "subprocess.run(['caput', 'X', '1'])", "epics.PV(name, monitor_delta=0.1)"])
def test_patterns_catch_writes(line):
    assert any(re.search(p, line) for p in FORBIDDEN)


@pytest.mark.parametrize("line", [
    'epics.PV("CAMR:LI20:107:Image:ArrayData")', "get('KLYS:LI11:11:FB:FAST_PACT')", '"TCAV:LI20:2400:A"'])
def test_pv_name_pattern(line):
    assert re.search(PV_NAME, line)


@pytest.mark.parametrize("line", [
    "x = pv.get()", "r = pv.get_with_metadata()", "if pv.value == 2:", "output = compute(input)",
    "path.write_text(text)"])
def test_patterns_pass_reads(line):
    assert not any(re.search(p, line) for p in FORBIDDEN)
