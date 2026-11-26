# The next controller: what it assumes, the handshake, and the runner changes

**Written 2026-11-25. Every decision here is Ryan's, from the 11-25 interview and its follow-up; all runner items R1–R12 are decided.**

**Who this is for.** The agent that will change the runner (this repo), and Ryan when he reviews it. The controller is trained in twobunch-s2e-rl (plan: `docs/next_controller_plan.md` on branch `rl-newframe`). It replaces `nc_zs_screen_noblen_jit_s0` (the incumbent) for the December 3–4 beam time.

**The rules for this repo still hold** (README, and Ryan's rules for machine code):
- Keep the code minimal; Ryan reviews every line.
- `actuate.py` stays the only module that writes.
- A check that can be done once per shift belongs in the console checklist, not in runtime code.
- One stage per PR. Each is tested offline, then on the machine, and merged only after its check passes.

## 1. What changes, in one paragraph

The new controller is trained on the energy-feedback simulation and its surrogate. Its training matches how this runner and the machine actually behave:
- the session budget clips its moves;
- a step takes 20–45 s;
- proposals are sometimes declined;
- L1's change is the commanded PREQ change;
- L2 changes under 0.02° are not written;
- invalid readings are held.

It sees the machine as one of many hidden states: injector state, sector-20 optics, TCAV voltage and calibration, image blur, energy setpoints and actuator gains. It is trained only with the witness trailing the drive, on spacing goals of 100–400 µm.

For the runner this means:
- a new policy directory;
- a smaller input vector (104 instead of 106), which needs no code change;
- a budget set per session;
- an alert when beam-state faults persist;
- an optional unattended mode, plus a few checklist and capture items.

## 2. What the controller assumes about the machine

Each line is something the policy was trained for. Outside these ranges its behaviour is untested.

| Item | Assumption | Trained range | Checked by | Source |
|---|---|---|---|---|
| L1 actuator | Both `KLYS:LI11:{11,21}:PREQ`, the same delta to each. PREQ's feedback off | gain beam/commanded 0.7–1.3 | `arm()` (QUIET_S), existing | W1; deploy plan |
| L1 observed change | The commanded PREQ delta, not the FAST_PACTBR readback | – | `advise.apply()`, existing | Ryan 11-25 |
| L2 actuator | `l2_phase` multiknob, a delta, moving LI12–14 1:1. LI12–14 bunch-length trim off. Energy feedbacks on | gain model/knob N(0.85, 0.12), clipped 0.6–1.1 | checklist | Ryan 11-25; Zack 11-19 (27 cos θ1 = 5 + 22 cos θ2) |
| L2 small moves | A change under 0.02° is not written and counts as 0 | same rule in training | `MIN_L2_MOVE`, existing | – |
| Budget | Net move per armed session, counted from `arm()`, detune moves included | L1 1.0–3.0°, L2 1.5–4.0°. Up to half already used by the detune when the episode starts | checklist at `arm()` (R2) | Ryan 11-25 |
| Step period | Reading start to next reading start | 20–45 s; ask ops for 10 Hz | checklist; period logged (R9) | Ryan 11-25 |
| Frames per step | Kept frames per polarity in a valid reading | 15–35 | `MIN_FRAMES` = 15, existing; capture length (R6) | commissioning 11-04 |
| Declines | Declined, expired, unanswered and shadow proposals count as action 0, change 0 | about 20% of proposals | existing | – |
| Invalid readings | No proposal, no advance; the next valid reading comes in with the last decision's outcome | the same in training (a held step is invisible to the policy; drift goes on) | existing; alert (R3) | Ryan 11-25 |
| Bunch order | Witness trails the drive, so spacing is positive | spacing goals 100–400 µm, starts readable and ≥ 100 µm | goal check (R5); negative reading (R4) | Ryan 11-25 |
| Drive σz goal | A recorded reading | the training goal distribution (the exporter writes the range) | goal check (R5) | – |
| TCAV | Phased and calibrated by ops at the session's optics. Calibration error within ±10% | 18–32 MV | checklist | Ryan 11-25 |
| Sector-20 optics | FF quads Q5FF, Q4FF, Q3FF, Q2FF, Q0FF within ±15% of design, Q1FF within ±5% (Ryan 11-25 evening: at ±15% the simulated bunch split fails in 33% of readings); BC20 sextupoles within ±35% of the 04-15 readbacks; movers within ±0.75 mm | as stated | checklist (prints BACT against design) | Ryan 11-25; 11-05 readbacks (−17% to +12%) |
| Injector | Two bunches, total charge 1.6–2.2 nC, delay 7–10 ps, laser pulses 1.5–2.3 ps each | uniform over that span | checklist (ops' numbers) | Ryan 11-25 |
| Phase drift | Beam-side L1 / L2 phase drift oscillates about a mean (feedbacks hold the cavity phases); white per-step phase wander on top | drift 0.35–1° rms, mean-reverting with τ 5–15 min, plus rare L2 jumps; wander L1 0.10–0.20°, L2 0.15–0.30° per step | – | Ryan 11-25 evening; DAQ hour 0.3–0.4° rms (10-18) |
| Energy setpoints | Energy feedbacks hold BC14/BC20 within ±4.5 / ±10 MeV per session, about 0.1% step to step | as stated | – | Ryan 11-25 |
| Image | Extra blur 0.5–1.5× what the image fits needed (about 130 µm on the screen along the streak, 1.8–2.9 px along energy) | as stated | – | Ryan 11-25 |

## 3. The handshake

### 3.1 Files

`policies/<name>/` holds what the exporter (twobunch-s2e-rl `deploy_export`) writes. These are unchanged:
- `policy.json`: the manifest.
- `weights.npz`: `obs_mean`, `obs_var`, `w*`, `b*`.
- `reference.npz`: inputs → actions from the torch policy, matched to atol 1e-5.
- `episodes.npz`: episodes from the training environment that `controller.Episode` must rebuild step by step, inputs to 1e-4.

One change on the training side: the episodes in `episodes.npz` come from the new environment. They include budget clips, declines, L2 moves under 0.02° and held invalid steps, so the tests exercise the runner's bookkeeping on those cases.

### 3.2 `policy.json`

| Field | Incumbent | New policy |
|---|---|---|
| `controls` | `[L1PhaseSet, L2PhaseSet]` | unchanged |
| `action_step` | `[0.25, 0.3]` | unchanged: the 1× step. The session's step is `action_step` × the step setting, 0.3–1.5× (R14) |
| `actuators` | `[direct_rb, direct]` | `[direct, direct]` |
| `blocks` | the seven of `controller.BLOCKS` | the seven, then `step_scale` (R14) |
| `features` | spacing, σz, `screen_valid_deployed`, `drive_charge_fraction`, the two goals (6) | spacing, σz, the two goals (4). The charge fraction is dropped (real light share 0.46–0.69 between runs and dependent on the streak sign; Ryan 11-25). The valid flag is dropped: it is always 1 at the runner and now in training too |
| `goals` | spacing (identity, scale 1e-5 m); σz (log10, scale 0.0414) | unchanged |
| `history` | 16 | unchanged |
| `episode_length` | 128 | unchanged |
| `inputs` | 106 | 106 (104 + the step setting) |
| `envelope` | – | new, see 3.4 |

### 3.3 Input vector (106)

| Block | Size | Content |
|---|---|---|
| `cumulative_change` | 2 | each control's observed change since the episode began, deg |
| `features` | 4 | spacing (identity), σz (log10), spacing goal, σz goal, each z-scored with the manifest's mean and std |
| `goal_error` | 2 | (t(reading) − t(goal)) / scale |
| `error_trend` | 2 | the goal error's change over the last step |
| `last_action` | 2 | the policy's last output; 0 at the start and after a decline |
| `gain_estimate` | 2 | 0 |
| `history` | 90 | 15 × [goal error before the step, the action before the one applied, the step's observed change], oldest first, zeros before the episode began (training's pairing, kept) |
| `step_scale` | 2 | the session's step setting per control, as a fraction of `action_step` (1 by default; R14) |

Normalisation is (x − obs_mean) / √(obs_var + 1e-5), with no clipping. The features block is read from the manifest's list (`controller.Episode._features`). `advise.values()` may return keys the policy doesn't use; they are ignored. So the smaller block needs no code change. That should be confirmed by the new policy's tests (R12).

### 3.4 The `envelope` block (new; data only, read by the console checklist)

The exporter writes the ranges the policy was trained on:

```json
"envelope": {
  "step_period_s": [20, 45],
  "kept_frames_per_polarity": [15, 35],
  "budget_deg": {"L1PhaseSet": [1.0, 3.0], "L2PhaseSet": [1.5, 4.0]},
  "step_scale": {"L1PhaseSet": [0.3, 1.5], "L2PhaseSet": [0.3, 1.5]},
  "spacing_goal_m": [1.0e-4, 4.0e-4],
  "drive_sigma_z_goal_m": ["from the training goal distribution, 5-95 %"],
  "tcav_MV": [18, 32],
  "streak_calibration_error": 0.10,
  "ff_quad_fraction": {"Q5FF": 0.15, "Q4FF": 0.15, "Q3FF": 0.15, "Q2FF": 0.15, "Q1FF": 0.05, "Q0FF": 0.15},
  "sextupole_fraction": 0.35,
  "mover_mm": 0.75,
  "charge_nC": [1.6, 2.2],
  "delay_ps": [7, 10],
  "feedbacks": {"PREQ (BC11 bunch length)": "off", "LI12-14 bunch-length trim": "off", "energy": "on"},
  "witness": "trailing"
}
```

The runner reads it only to print the pre-shift checklist (R10) and the goal check (R5). Nothing else depends on it.

### 3.5 One step, and what training does for each part

| Runner (this repo) | Training (twobunch-s2e-rl) |
|---|---|
| `step()`: one reading from the frames kept over the capture | One reading from 15–35 simulated shots per polarity, with per-shot noise, the jitter map and outliers |
| Invalid reading: no proposal, the episode doesn't advance | The env re-measures (time passes, drift goes on) without a policy step. The policy never sees an invalid reading. A beam that stays unreadable ends the episode with a penalty: the operator steps in |
| Valid reading: `advance(outcome, change, reading)`, then `act()` | The same pairing of action, change and next valid reading |
| `clip(a, step, net, budget)`; net counted from `arm()` | The same clip, with the budget drawn per episode and a pre-used net of up to ±half the budget at the start |
| L2 change under 0.02° not written: change 0 | The same dead zone |
| Declined / expired / shadow: action 0, change 0 | 20% random declines with the same bookkeeping |
| Applied: action = output; change = written delta (L1 PREQ delta; L2 multiknob delta) | L1 and L2 observed change = commanded change. The beam moves by the hidden gain times it, plus drift |
| Episode restarts after 128 steps on the same goal, history zeroed | 128-step episodes, with a new goal every 16–48 steps |
| Goal = mean of 3 valid readings, no write between | The goal is the noise-free reading at a reachable, readable control point with spacing ≥ 100 µm |

### 3.6 Conventions

- **Units.** Lengths in m inside the policy, angles in deg.
- **Spacing is signed.** It is drive minus witness z centroid. Positive means the witness trails.
  - The drive is the energy band with the majority of the charge.
  - The sign also depends on the sign of the streak calibration passed to `step()`. E331's TCAV sign pairing was reversed against E300's, so check it each session (checklist).
- **σz** is the drive's rms length from the two polarities' variances.

## 4. Runner changes (work list)

Each item gives what to change, why, where, how it's checked, and its status. Statuses: **decided** (Ryan 11-25) or **verify** (should need no code change; confirm with tests). Every item is decided.

**R1. Swap in the new policy. Decided: the new policy only.**
- `advise.POLICY` (`advise.py:32`) and the test's `NAME` (`tests/test_controller.py:11`) take the new name.
- `policies/nc_zs_screen_noblen_jit_s0/` leaves the working tree; git history keeps it.
- Done when the new policy's reference and episode tests pass.

**R2. The budget as a session setting. Decided.**
- Today the budget is a constant, `pvs.WRITE[kind]["budget"]` (1.0 / 1.5).
- Change: `actuate.arm(name, budget=None)` takes a per-kind budget, defaulting to today's values. It records the budget as the ledger's first line, and `advise.propose()` clips with the armed session's budget.
- The trained range comes from the envelope: a checklist line at `arm()`, not a refusal in code.
- Start at ±1.0 / ±1.5 and widen as trust grows (Ryan).
- Done when offline tests cover a non-default budget (the clip, the ledger line, `undo_all()` after it).

**R3. Alert on persistent beam-state faults. Decided: wait only, the operator decides. No automatic undo.**
- `propose()` already prints the failed checks and holds. Change: sort each held reading's failed checks into two kinds.
  - **Acquisition:** too few frames, no row-correction reference, beam intensity. These mean wait and re-read.
  - **Beam state:** no step cut (no energy valley), |spacing| ≤ 60 µm, edge fraction, negative spacing (R4). These mean the beam itself is unreadable.
- After 3 consecutive held readings with a beam-state fault, print one clear alert: "operator decision needed". Include the last write (from the ledger) and the net move. Nothing is proposed or written.
- The operator may `undo_all()`, move by hand, or keep waiting.
- The counter resets on a valid reading.
- Done when offline tests feed each kind of fault in sequence.

**R4. A negative spacing reading counts as a beam-state fault. Decided.**
- The policy was trained only with the witness trailing. A valid reading with spacing < 0 (bunch order reversed, or a wrong streak sign) is outside what it knows.
- Change: `values()` returns None for it and the reading is held with the check "witness trailing".
- A wrong streak sign then holds every reading, which is a safe failure the checklist catches.
- Done when the offline test of a reversed reading holds.

**R5. Goal check. Decided: warn and record.**
- `record_goal()` prints a warning when a goal's spacing is outside `envelope.spacing_goal_m` (100–400 µm) or its σz is outside `envelope.drive_sigma_z_goal_m`, saying which and by how much.
- The goal is still recorded and usable. The operator decides.

**R6. Capture length for the beam rate. Decided: version (a), the checklist.**
- Training assumes 15–35 kept frames per polarity and a 20–45 s period.
  - Machine: 10 Hz for 15 s gave a median of 17–19 kept frames per polarity.
  - 5 Hz for 30 s gave 34 + / 25 −.
  - 5 Hz for 15 s is too short.
- (a) Minimal: the console checklist sets `seconds` from the beam rate (15 s at 10 Hz, 30 s at 5 Hz). No code change.
- (b) Capture by frame count: `acquire.capture` runs until it has ≥ 20 kept frames per polarity or a time cap. This is a change in S4-era code.
- Do (a): a checklist line in `console.ipynb`. Leave `acquire.capture` unchanged.

**R7. Settling after a write. Decided: measure first.**
- Not built: the frames of the first step after a write are only dropped by the existing ≥ 1 s per-polarity-window rule.
- Write timing (W1): a PREQ put completes in about 0.02 s; a multiknob turn plus its setpoint confirmation takes 0.64–1.31 s. The beam's settling after a write has never been measured with beam.
- Change: `step(drop_after_write_s=1.0)` drops frames within that time of the step's start when a write was sent just before.
- Measure first, in the first approved-write session: compare the first and second halves of each step after a write. Build the drop above only if they differ. For now, nothing beyond a way to compare the two halves from the saved step records, if that isn't already possible.

**R8. Unattended mode. Decided that closed loop is in scope; the switch is Ryan's judgement on the day.**
- Change: `cycle(..., unattended=True)` applies the proposal without `input()`.
- Each decision is logged as "applied (unattended)".
- No step cap (Ryan). It runs until one of the stops below, or until the operator stops it.
- It drops back to approved mode on:
  - any hold;
  - a failed write;
  - the R3 alert;
  - a proposal clipped to zero on both controls (budget exhausted);
  - an operator interrupt (Ctrl-C between cycles leaves no undecided proposal).
- This is new write-path behaviour, so it gets its own PR and machine check after the approved cycle has run on two-bunch beam.

**R9. Cycle period in the log. Decided.**
- Log the time between readings in the episode log, and print a warning in the proposal line when it exceeds 45 s.
- Training assumed 20–45 s per step, and the drift per step grows with the period.
- Logging only; no refusal.

**R10. The pre-shift checklist reads the envelope. Decided.**
- A `console.ipynb` cell prints the policy's envelope next to the live values, for the operator to tick:
  - TCAV amplitude readback and ops' calibration (value and sign);
  - beam rate;
  - FF quad and sextupole BACT against design (ratio);
  - mover setpoints;
  - the feedback states as ops report them;
  - the budget about to be armed.
- Data and printing only.

**R11. v1 leftovers. Decided.**
- The −10000 sentinel of the LI12–14 PHAS readbacks enters the medians: exclude it.
- The capture-fraction figure assumes 10 Hz.
- The B5D36 supply's units.
- From the commissioning note's open items.

**R12. Inputs read from the manifest. Verify, and one test change.**
- The new policy has 4 features and `[direct, direct]` actuators. `controller.Policy` accepts both today.
- Confirm with the new policy's tests, and update `controller.READING` and the module docstring (which says 106) to match.
- `tests/test_controller.py::test_inputs_reproduce_the_training_env` must apply the L2 dead zone to the clipped change before `advance()`, as `advise.apply()` does (an L2 change under `MIN_L2_MOVE` is not written, so the observed change is 0). The new policy's exported episodes include such steps; the incumbent's training env had no dead zone, so its test never met one.
- The exported episodes may end before 40 steps when a reading stays unreadable (the runner would hold there); the test should loop over the actions it has.

**R13. The energy cut from the right polarity. Decided (Ryan 11-25 evening): implement it; the E331 replay gate is scored with it, the as-built number reported beside it. Found 11-25 in the E331 replays.**
- `screen.step_cut` takes the step's energy cut from the camera's "+" frames. On an E331-type session, where the TCAV sign pairing is reversed against E300 (11-03), those are the physically opposite polarity.
- The TCAV's energy kick changes sign with the polarity, so at large spacing their valley jumps about 18 rows to another dip. That swaps the bunch labels: the far side of the scan reads negative, and a band past the crossing reads positive.
- Taking the cut from the frames whose streak matches E300's "+" polarity fixes most of it:

  | E331 pair | 21_22 | 24_23 | 25_27 | 30_32 | 34_33 |
  |---|---|---|---|---|---|
  | Incumbent within 10 µm, as built | 0.47 | 0.47 | 0.27 | 0.44 | 0.36 |
  | Cut from the matching frames | 0.36 | 0.47 | 0.66 | 0.53 | 0.45 |

  (The 16029 / 16028 pair is skipped: unreadable at L1 −21.) Replays in twobunch-s2e-rl `results/rl_newframe/replay/`.
- Change: `step()` / `read_step()` take a `cut_polarity` ("+" or "−", default "+"). The console checklist sets it per session from the session's TCAV sign pairing, the same check that fixes the streak sign.
- Done when an offline test on an E331 pair reproduces the replay's `_cutminus` readings.

**R14. The step as a session setting. Decided (Ryan 11-25 evening).**
- Why: if the first proposals on the machine overshoot (for example an L2 gain off from training's), the operator can shrink the step for the session without retraining. The policy is trained with it: each episode's step setting is drawn (half at 1×, half log-uniform over 0.3–1.5×, per control) and is its last input, so a smaller step is in its training range and it takes more steps.
- Change (`controller.Episode(..., step_scale=...)` and the input block are done, `dd47aa1`):
  - `actuate.arm(name, budget=None, step_scale=None)` takes a per-kind setting inside the envelope's `step_scale` range, default 1, recorded in the ledger's first line beside the budget.
  - `advise.propose()` builds the episode with the armed setting and clips with `episode.step` (= `action_step` × the setting) in place of `action_step`.
  - **Note, open (Ryan decides with this work):** the hard per-write limit in `actuate` is one 1× step (`pvs.WRITE`: 0.25° on L1, 0.3° on L2). A setting above 1× needs that limit to follow the session's step, up to 1.5× (0.375° / 0.45°), which raises a safety limit ops should know about. Settings up to 1× need no change to it.
  - A policy without the `step_scale` block (the incumbent) refuses a setting other than 1.
- Checklist: arm at 1×. Lower it if the first proposals overshoot (the spacing passes the goal and comes back). Under drift keep it at 0.5× or more: at 0.3× the L2 step can't keep up with the drift in about 9% of 4-step windows, mostly the rare L2 jumps (`results/rl_newframe/diag_step_drift.py` in twobunch-s2e-rl). The 0.02° L2 dead zone is unchanged, so at 0.3× moves under 22% of a full step aren't written.
- Done when offline tests cover a non-default setting: the clip, the ledger line, the write limit and the inputs.

**Not to change** (training depends on these):
- One write per proposal, at most one session step (0.25° / 0.3° × the step setting, R14).
- The bookkeeping in 3.5: declines as 0/0, the change as written, the 0.02° L2 dead zone, holds without an advance, the episode restart, and goals as the mean of 3 readings with no write between.
- The normalisation formula and the clip formula.
- `tests/test_read_only.py`.

## 5. A session with the new policy (checklist order)

1. **Ops:** two-bunch beam, preferably at 10 Hz.
   - PREQ feedback off; LI12–14 bunch-length trim off; energy feedbacks on.
   - TCAV phased and calibrated at the session's optics. Note the value and the sign.
2. **Pre-shift cell (R10):** every live value inside the envelope, or a decision written down.
3. **Background, then shadow cycles.**
   - Witness trailing reads positive.
   - Readings are valid at the chosen capture length (R6).
   - A start that fails the crop-edge check (edge fraction ≥ 0.01): lower the TCAV voltage, or the spacing, before arming. In training, about 10% of hidden states (28–32 MV with spacing above about 300 µm) put the image at the crop edge wherever L1 / L2 start.
4. **`record_goal()`** for each goal, inside the envelope (R5).
5. **`arm(name, budget)`** with the budget inside the envelope (R2).
   - Detune moves count against it. Keep them to at most half the budget, as training assumed.
6. **`use_goal()`, then approved cycles.**
   - Unattended cycles (R8) only when Ryan decides.
7. **Holds:**
   - acquisition faults: re-read;
   - after the R3 alert: operator's decision (`undo_all()`, a manual move, or wait).
8. **Close:** `end_episode()`; `undo_all()` or keep; `disarm()`; ops restore the feedbacks.

## 6. Open questions for ops and Ryan

- Which loop writes PREQ (the BC11 bunch-length feedback?), and whether turning it back on is bumpless (Zack).
- 10 Hz for the shift.
- The TCAV amplitude to ask for: JP's advice is the best resolution that still stays on the screen. Who runs the toggler.
- The sector-20 optics in December: the 11-05 state (FF quads −17% to +12% from design) or design.
- The two-bunch set-up ops will give us (charge, delay, pulse lengths), to check against the envelope.

## References

- Interview of 2026-11-25 (Ryan): the decisions above.
- Machine facts:
  - `lab-notebook/claude/analyses/2026-10-28_twobunch-v1-commissioning.md`;
  - `lab-notebook/projects/slac-rl/experiments/2026-11-05_twobunch-v1-campaign-summary.md`;
  - `lab-notebook/claude/analyses/2026-11-06_twobunch-v2-plan.md`;
  - the W1 write session `data/srv01/2026-11-06_V2/`.
- Deploy simulation of the incumbent: twobunch-s2e-rl `results/3_rl_eval/deploy_sim_2026-11-06/summary.json`. Under the budget, 67% of steps were clipped and the median spacing error rose from 14 to 48 µm by step 128; with 40 s steps, 99 µm.
- Fit findings (optics, TCAV, blur, charge, L2 gain): `lab-notebook/claude/analyses/2026-11-18_fit-assumptions-register.md`, `lab-notebook/projects/slac-rl/experiments/2026-11-22_gpsr-energy-frame-fits-and-reduced-fit.md`.
- Training and evaluation plan: twobunch-s2e-rl `docs/next_controller_plan.md` (branch `rl-newframe`).
