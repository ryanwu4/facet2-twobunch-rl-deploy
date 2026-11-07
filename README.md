# facet2-twobunch-rl-deploy

Runs the FACET-II two-bunch spacing controller on the machine. The controller itself is trained in [twobunch-s2e-rl](https://github.com/ryanwu4/twobunch-s2e-rl).

**v1 is read-only.** It reads the DTOTR2 XTCAV camera and a few phase, bunch-length and magnet PVs and turns the frames into the readings the controller uses.

**v2 adds writes, in one module.** `twobunch_deploy/actuate.py` is the only code that writes: the two L1 phase setpoints and the L2 phase multiknob, each write at most one policy action, within a budget per session, only after `arm()`, each logged before and after in a ledger that `undo_all()` reverses. `tests/test_read_only.py` fails on a write call anywhere else, and pins `actuate.py`'s to its two write functions. The policy, and moves a person approves one by one, come in later stages.

## Stages

Built one stage per PR. Each PR is tested offline, run on the machine from its branch, and merged only after its check passes.

| Stage | Adds |
|---|---|
| S0 | Skeleton, environment check (no PV read) |
| S1 | Camera frames and pulse IDs |
| S2 | Frame geometry: size, orientation, ROI |
| S3 | Background, processing, `peek()` |
| S4 | TCAV readbacks and off / + / − tagging |
| S5 | Row correction and step readings |
| S6 | L1 klystron phase and amplitude readbacks, phase setpoints |
| S7 | L2 subbooster phases |
| S8 | Bunch length, drift monitors and magnets |
| S9 | The v1 console (`console.ipynb`) |
| W1 (v2) | The write path (`actuate.py`): `arm`, `move_l1`, `move_l2`, `undo_all`, the ledger; the write session with the beam stopped at TD11 (`hold_test`, `staircase`) |

## Layout

| File | Contents |
|---|---|
| `twobunch_deploy/pvs.py` | Every PV the package may read |
| `twobunch_deploy/machine.py` | Channel Access reads (pyepics) |
| `twobunch_deploy/acquire.py` | Frames for one measurement step, TCAV tagging, the PVs monitored over it |
| `twobunch_deploy/screen.py` | Image processing and step readings |
| `twobunch_deploy/session.py` | The calls the notebooks make |
| `twobunch_deploy/actuate.py` | The only writes (v2): arming, the moves, the ledger, undo |
| `twobunch_deploy/viz.py` | Plots |
| `console.ipynb` | The calls for a session at the machine, with its checklists |
| `commissioning.ipynb` | One section per stage |
| `tests/` | Offline tests; never imported by the package |

## Offline

```
pixi run test        # all tests
pixi run code-hash   # sha256 of the package source
```

The camera tests run the package's Channel Access calls against a soft IOC (`tests/soft_ioc.py`, caproto) that serves real DAQ frames under the camera's PV name, the DAQ's TCAV and L1 readbacks and bunch-length signals of each frame, its L2 and magnet values, and stand-ins for the drift PVs, on 127.0.0.1 only. The write-path tests add a stand-in for the AIDA-PVA multiknob service (`tests/mkb_server.py`, p4p), which moves the soft IOC's subbooster setpoints as the real knob does. The DAQ files are not in this repo, so these tests run only where they are (`DTOTR2_DAQ`, the TEST_15668 `images/DTOTR2` directory, with the run's `.mat` two levels up) and are skipped in CI. So are the tests that compare the image processing, TCAV tags and step readings with Yiheng Ye's DAQ analysis classes, twobunch-s2e-rl's `online/features.py` and its v7 replay of the DAQ runs: they need those, and the DAQ runs, next to this repo. The first Channel Access call starts EPICS's standard `caRepeater` helper process, which keeps running afterwards.

## On the machine

Uses the ACR production Python environment as it is; nothing is installed. Get the stage's branch onto the host, then from the repo root start Jupyter inside tmux so it survives the SSH tunnel dropping:

```
tmux new -s twobunch
# activate the ACR production Python environment
jupyter notebook --no-browser --port=<fixed port>
```

Open `commissioning.ipynb` through the tunnel and run the stage's section; once v1 is assembled, `console.ipynb` for a session. Records go to `data/` (git-ignored).

## License

Apache-2.0
