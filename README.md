# facet2-twobunch-rl-deploy

Runs the FACET-II two-bunch spacing controller on the machine. The controller itself is trained in [twobunch-s2e-rl](https://github.com/ryanwu4/twobunch-s2e-rl).

**v1 is read-only.** It reads the DTOTR2 XTCAV camera and a few phase and bunch-length PVs and turns the frames into the readings the controller uses. It contains no code that writes to the machine (`tests/test_read_only.py` fails on any write call) and no policy. Writing, with each move approved by a person (v2), comes later in its own reviewed stages.

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
| S6 | L1 phase readbacks |
| S7 | L2 subbooster phases |
| S8 | Bunch length and drift monitors |
| S9 | v1 notebook assembled |

## Layout

| File | Contents |
|---|---|
| `twobunch_deploy/pvs.py` | Every PV the package may read |
| `twobunch_deploy/machine.py` | Channel Access reads (pyepics) |
| `twobunch_deploy/acquire.py` | Frames for one measurement step, TCAV tagging |
| `twobunch_deploy/screen.py` | Image processing and step readings |
| `twobunch_deploy/replay.py` | DAQ files through the same calls, for offline runs |
| `twobunch_deploy/session.py` | The calls the notebooks make |
| `twobunch_deploy/viz.py` | Plots |
| `commissioning.ipynb` | One section per stage |
| `tests/` | Offline tests; never imported by the package |

## Offline

```
pixi run test        # all tests
pixi run code-hash   # sha256 of the package source
```

The camera tests run the package's Channel Access calls against a soft IOC (`tests/soft_ioc.py`, caproto) that serves real DAQ frames under the camera's PV name, on 127.0.0.1 only. The DAQ files are not in this repo, so these tests run only where they are (`DTOTR2_DAQ`, the TEST_15668 `images/DTOTR2` directory) and are skipped in CI. The first Channel Access call starts EPICS's standard `caRepeater` helper process, which keeps running afterwards.

## On the machine

Uses the ACR production Python environment as it is; nothing is installed. Get the stage's branch onto the host, then from the repo root start Jupyter inside tmux so it survives the SSH tunnel dropping:

```
tmux new -s twobunch
# activate the ACR production Python environment
jupyter notebook --no-browser --port=<fixed port>
```

Open `commissioning.ipynb` through the tunnel and run the stage's section. Records go to `data/` (git-ignored).

## License

Apache-2.0
