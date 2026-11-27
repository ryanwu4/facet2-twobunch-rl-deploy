"""R16: one table per session day from its records, for the analysis after the run. Run here, on the records copied
back (from_srv01.sh: the day's _V2 folder and its _S5 step folder); not on the host.

    python tools/session_table.py data/srv01/2026-12-03_V2            # writes data/srv01/2026-12-03_V2/table.csv

One row per event of the day's episode logs, in time order:
- proposal: the reading (spacing, sigma_z), the goal, the policy's action, the change asked and clipped, the decision
  and what was applied, the net move, budget and step, the period since the last advance, the ledger writes that
  applied it, and from the step record of its reading: validity, the reading, frames kept per polarity, the median
  of every phase and bunch-length PV over the capture, and each polarity's median drive-witness offset on the screen
  in the first and second halves of the capture (R7: whether the beam was still settling after a write);
- hold: the failed checks and whether they were beam-state ones, with the step record's fields;
- the episode's start, goal changes, restarts, alerts, and each run's start and stop with its reason.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

CONTROLS = ("L1", "L2")


def local(path: str, data: Path) -> Path:
    """A path recorded on the host (.../data/<day>/<file>), under the copied-back data directory `data`."""
    parts = Path(path).parts
    return data.joinpath(*parts[len(parts) - 1 - parts[::-1].index("data") + 1:]) if "data" in parts else Path(path)


def step_fields(path: Path) -> dict:
    """What the table keeps from a step record; empty if it wasn't copied back."""
    if not path.exists():
        return {}
    r = json.loads(path.read_text())
    out = {"valid": r["valid"], "cut_polarity": r.get("cut_polarity"), "streak_um_per_deg": r["streak_um_per_deg"],
           "frames_plus": r["polarity"]["+"].get("kept"), "frames_minus": r["polarity"]["-"].get("kept")}
    out.update({f"read_{k}": v for k, v in (r["reading"] or {}).items()})
    for kind in ("phases", "drift"):
        out.update({f"median {name}": s.get("median") for name, s in r[kind].items()})
    out.update(halves(r.get("per_frame")))
    return out


def halves(per_frame: dict | None) -> dict:
    """R7: each polarity's median drive-witness offset x_d - x_w (um on the screen) over the kept frames of the first
    and of the second half of the capture (split at its middle in time); None where a half has none of a polarity."""
    if not per_frame or not per_frame["timestamp"]:
        return {}
    t = np.asarray(per_frame["timestamp"])
    middle = (t[0] + t[-1]) / 2
    out = {}
    for sign, label in (("+", "plus"), ("-", "minus")):
        for half, inside in (("first", t < middle), ("second", t >= middle)):
            offsets = [1e6 * (s["x_d"] - s["x_w"]) for s, tag, drop, i in
                       zip(per_frame["band_stats"], per_frame["tag"], per_frame["drop"], inside)
                       if tag == sign and not drop and s and i]
            out[f"offset_{label}_{half}_um"] = float(np.median(offsets)) if offsets else None
    return out


def pairs(prefix: str, values) -> dict:
    return {f"{prefix}_{c}": v for c, v in zip(CONTROLS, values or [None, None])}


def table(day: Path) -> list[dict]:
    data = day.parent
    ledger = [{**json.loads(line), "ledger": f.name} for f in sorted(day.glob("ledger_*.jsonl"))
              for line in f.read_text().splitlines()[1:]]
    rows = []
    for log in sorted(day.glob("episode_*.jsonl")):
        entries = [json.loads(line) for line in log.read_text().splitlines()]
        span = (entries[0]["time"], entries[-1]["time"])
        decided = {e["proposal"]: e for e in entries if "applied" in e}
        goal, record = entries[0]["goal"], entries[0]["goal_record"]
        for e in entries:
            row = {"time": e["time"], "episode": log.name, "goal": goal}
            if "goal_record" in e:
                goal, record = e.get("goal_change", goal), e["goal_record"]
                row.update(event="goal" if "goal_change" not in e else "goal change", goal=goal)
            elif "made" in e:
                d = decided.get(e["proposal"], {})
                writes = [w for w in ledger if w.get("proposal") == e["proposal"] and w["state"] != "pending"
                          and span[0] <= w["time"] <= span[1]]
                row.update(event="proposal", proposal=e["proposal"], episode_step=e["episode_step"],
                           armed=e["armed"], decision=d.get("decision"), period_s=e.get("period_s"),
                           spacing_um=1e6 * e["values"]["bunch_spacing"],
                           sigma_z_um=1e6 * e["values"]["drive_sigma_z"],
                           goal_spacing_um=1e6 * record["bunch_spacing"], goal_sigma_z_um=1e6 * record["drive_sigma_z"],
                           writes="; ".join(f"{w['kind']} {w['delta']:+.3f} {w['state']} ({w['ledger']} #{w['write']})"
                                            for w in writes), reading=e["reading"],
                           **pairs("action", e["action"]), **pairs("asked", e["asked"]), **pairs("change", e["change"]),
                           **pairs("applied", d.get("applied")), **pairs("net", e["net"]),
                           **pairs("budget", e.get("budget")), **pairs("step", e.get("step")),
                           **step_fields(local(e["reading"], data)))
            elif "held" in e:
                row.update(event="hold", failed="; ".join(e["failed"]), beam_state=e["beam_state"],
                           beam_holds=e["beam_holds"], reading=e["held"], **step_fields(local(e["held"], data)))
            elif "run" in e:
                row.update(event=f"run {e['run']}", detail=e.get("reason") or json.dumps(e.get("schedule")))
            elif "alert" in e or "restarted" in e or "ended" in e:
                row.update(event=next(k for k in ("alert", "restarted", "ended") if k in e))
            else:
                continue                                    # a decision line: in its proposal's row
            rows.append(row)
    return sorted(rows, key=lambda r: r["time"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("day", type=Path, help="a copied-back <date>_V2 folder")
    ap.add_argument("--out", type=Path, default=None, help="default: <day>/table.csv")
    a = ap.parse_args()
    rows = table(a.day)
    columns = list(dict.fromkeys(k for r in rows for k in r))
    out = a.out or a.day / "table.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows, {len(columns)} columns: {out}")


if __name__ == "__main__":
    main()
