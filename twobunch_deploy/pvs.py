"""Every PV the package may read, and the few it may write (v2), as literal strings.

PV names appear nowhere else in the package or the notebook (tests/test_read_only.py checks), so each
stage's PR shows exactly which PVs it adds. Only actuate.py writes, and only what WRITE lists.
"""

# The RAW plugin's array: the camera driver's output (port PS107) with no plugin in between, the same arrays
# the DAQ's HDF5 plugin saves (S2; S1 read Image:ArrayData, which goes through the ROI plugin)
CAMERA = "CAMR:LI20:107:RAW:ArrayData"

# DTOTR2 camera readbacks, read before each capture (S2). The part after the last colon is the key
# machine.camera_settings() returns; for the first ten it is also the DAQ metadata's (metadata.DTOTR2) and
# screen.TRAINED's.
CAMERA_SETTINGS = {
    "CAMR:LI20:107:MinX_RBV": "camera ROI: first sensor column",
    "CAMR:LI20:107:MinY_RBV": "camera ROI: first sensor row",
    "CAMR:LI20:107:SizeX_RBV": "camera ROI: number of columns",
    "CAMR:LI20:107:SizeY_RBV": "camera ROI: number of rows",
    "CAMR:LI20:107:X_ORIENT": "display orientation: Negative flips the columns",
    "CAMR:LI20:107:Y_ORIENT": "display orientation: Negative flips the rows",
    "CAMR:LI20:107:IS_ROTATED": "display orientation: 1 transposes",
    "CAMR:LI20:107:AcquireTime_RBV": "exposure (s)",
    "CAMR:LI20:107:Gain_RBV": "gain",
    "CAMR:LI20:107:RESOLUTION": "pixel size (um)",
    "CAMR:LI20:107:ReverseX_RBV": "driver: Yes flips the columns",
    "CAMR:LI20:107:ReverseY_RBV": "driver: Yes flips the rows",
    "CAMR:LI20:107:ArraySizeX_RBV": "driver output: number of columns, after binning",
    "CAMR:LI20:107:ArraySizeY_RBV": "driver output: number of rows, after binning",
    "CAMR:LI20:107:RAW:NDArrayPort_RBV": "port the camera array comes from (PS107: the driver)",
    "CAMR:LI20:107:RAW:ArraySize0_RBV": "camera array: number of columns (the fast axis)",
    "CAMR:LI20:107:RAW:ArraySize1_RBV": "camera array: number of rows",
}

# The XTCAV's amplitude (MV) and phase (deg at 11.424 GHz) readbacks, read with every frame and matched to it by
# pulse ID (S4). The DAQ's names (BSA_List_S20). On srv01 (11-02, by hand) they update every 12 pulse IDs (30 Hz),
# timestamped with the pulse: every camera frame (10 Hz) has its own.
TCAV_AMPLITUDE = "TCAV:LI20:2400:A"
TCAV_PHASE = "TCAV:LI20:2400:P"

# L1: the LI11 klystrons 11 and 21 (S6). Monitored during each step's capture. FAST_PACTBR and FAST_AACTBR are the
# phase and amplitude readbacks the DAQ records as FB:FAST_PACT and FB:FAST_AACT: BSA PVs, read at beam rate with the
# event definition's name (BR) appended (Zack, 11-05). On srv01 (11-05, by hand) they updated once per shot, with the
# shot's pulse ID. PREQ is the absolute phase setpoint ops move L1 with; a feedback writes the same value to both
# (Zack, 10-16), at 5 Hz on srv01 whatever the beam rate (11-05). It is in no DAQ. SSSB_PDES is the DAQ's subbooster
# phase setpoint (nonBSA_List_LINAC_SBST; -18.03 deg).
L1 = {
    "KLYS:LI11:11:FB:FAST_PACTBR": "L1 klystron 11 phase readback, deg",
    "KLYS:LI11:21:FB:FAST_PACTBR": "L1 klystron 21 phase readback, deg",
    "KLYS:LI11:11:FB:FAST_AACTBR": "L1 klystron 11 amplitude readback, norm",
    "KLYS:LI11:21:FB:FAST_AACTBR": "L1 klystron 21 amplitude readback, norm",
    "KLYS:LI11:11:PREQ": "L1 klystron 11 phase setpoint, deg",
    "KLYS:LI11:21:PREQ": "L1 klystron 21 phase setpoint, deg",
    "KLYS:LI11:11:SSSB_PDES": "L1 klystron 11 subbooster phase setpoint, deg",
    "KLYS:LI11:21:SSSB_PDES": "L1 klystron 21 subbooster phase setpoint, deg",
}

# L2: the subbooster phase setpoints of sectors 11-14, and their phase readbacks (S7). The L2 multiknob moves LI12-14's
# 1:1 (DAQ, 10-16). LI11's phases L2's klystrons 4-8 (F2_live_model) and the multiknob doesn't move it (0 throughout
# TEST_15668). Monitored during each step's capture. The DAQ's nonBSA names (nonBSA_List_LINAC_SBST).
L2 = {
    "LI11:SBST:1:PDES": "LI11 subbooster phase setpoint, deg",
    "LI12:SBST:1:PDES": "LI12 subbooster phase setpoint, deg",
    "LI13:SBST:1:PDES": "LI13 subbooster phase setpoint, deg",
    "LI14:SBST:1:PDES": "LI14 subbooster phase setpoint, deg",
    "LI11:SBST:1:PHAS": "LI11 subbooster phase readback, deg",
    "LI12:SBST:1:PHAS": "LI12 subbooster phase readback, deg",
    "LI13:SBST:1:PHAS": "LI13 subbooster phase readback, deg",
    "LI14:SBST:1:PHAS": "LI14 subbooster phase readback, deg",
}

# Bunch length (S8): the raw signals of the four bunch-length monitors, which the DAQ records as BSA PVs (BSA_List_S10,
# S11, S14 and S20, as BLEN:...:BRAW), read at beam rate like the L1 readbacks. Monitored during each step's capture.
# Their absolute counts aren't comparable across sessions, only their changes within one.
BLEN = {
    "BLEN:IN10:596:BRAWBR": "injector bunch length monitor, raw",
    "BLEN:LI11:359:BRAWBR": "BC11 bunch length monitor, raw",
    "BLEN:LI14:888:BRAWBR": "BC14 bunch length monitor, raw",
    "BLEN:LI20:3014:BRAWBR": "sector 20 bunch length monitor, raw",
}

# Drift monitors (S8), monitored during each step's capture. F2_live_model's estimates for L2 and L3 of the energy gain
# and chirp (MeV) and the fudge (design gain over the klystrons' sum), and its heartbeat: on srv01 (11-04, by hand)
# L2's changed about once a second and L3's not in a minute. The LI14 klystron pair whose phase setpoints moved by
# about 1 deg at 0.7 Hz (11-04), with the readbacks the E331 DAQ records for it (nonBSA_List_LINAC_L2KLYS).
DRIFT = {
    "SIOC:SYS1:ML02:AO922": "F2_live_model L2 energy gain, MeV",
    "SIOC:SYS1:ML02:AO926": "F2_live_model L2 chirp, MeV",
    "SIOC:SYS1:ML02:AO930": "F2_live_model L2 fudge",
    "SIOC:SYS1:ML02:AO923": "F2_live_model L3 energy gain, MeV",
    "SIOC:SYS1:ML02:AO927": "F2_live_model L3 chirp, MeV",
    "SIOC:SYS1:ML02:AO931": "F2_live_model L3 fudge",
    "PHYS:SYS1:1:MODEL_SERVER": "F2_live_model heartbeat",
    "LI14:KLYS:41:PDES": "LI14 klystron 4 phase setpoint, deg",
    "LI14:KLYS:41:PHAS": "LI14 klystron 4 phase readback, deg",
    "LI14:KLYS:41:ADES": "LI14 klystron 4 amplitude setpoint",
    "LI14:KLYS:41:AMPL": "LI14 klystron 4 amplitude readback",
    "LI14:KLYS:51:PDES": "LI14 klystron 5 phase setpoint, deg",
    "LI14:KLYS:51:PHAS": "LI14 klystron 5 phase readback, deg",
    "LI14:KLYS:51:ADES": "LI14 klystron 5 amplitude setpoint",
    "LI14:KLYS:51:AMPL": "LI14 klystron 5 amplitude readback",
}

# Magnets (S8), read once at the start of each step and by optics_test(): supply readbacks (BACT), under the DAQ's
# names (nonBSA_List_S20Magnets, nonBSA_List_S19).
# - The final-focus and spectrometer quads (kG), named as in F2_live_model's unaliased-elements.csv. The DAQ values
#   bear that map out: each supply is within 0.6% of its magnet's golden strength (twobunch-s2e-rl,
#   xtcav_tier1_configs.py), Q0D and Q2D trimmed 5.6% together.
# - The spectrometer dipole B5D36 (GeV/c), which sets the energy axis on DTOTR2 (xtcav_tier1_configs.py's supply).
# - The final-focus steering correctors and the BC20 chicane sextupoles (S3EL and S3ER are two magnets each), knobs of
#   the two-bunch campaign (sweep_params.py); the sextupoles named as in unaliased-elements.csv.
# - The two LI19 quads that launch into BC20.
MAGNETS = {
    "LI20:LGPS:3011:BACT": "Q5FF supply, kG",
    "LI20:LGPS:3311:BACT": "Q4FF supply, kG",
    "LI20:LGPS:3151:BACT": "Q3FF supply, kG",
    "LI20:LGPS:1910:BACT": "Q2FF supply, kG",
    "LI20:LGPS:3204:BACT": "Q1FF supply, kG",
    "LI20:LGPS:3031:BACT": "Q0FF supply, kG",
    "LI20:LGPS:3141:BACT": "Q0D supply, kG",
    "LI20:LGPS:3261:BACT": "Q1D supply, kG",
    "LI20:LGPS:3091:BACT": "Q2D supply, kG",
    "LI20:LGPS:3330:BACT": "B5D36 supply, GeV/c",
    "LI20:XCOR:3026:BACT": "XC1FF corrector",
    "LI20:XCOR:3086:BACT": "XC3FF corrector",
    "LI20:YCOR:3017:BACT": "YC1FF corrector",
    "LI20:YCOR:3057:BACT": "YC2FF corrector",
    "QUAD:LI19:851:BACT": "LI19 quad 851, BC20 launch",
    "QUAD:LI19:871:BACT": "LI19 quad 871, BC20 launch",
    "LI20:LGPS:2145:BACT": "sextupole S1EL supply",
    "LI20:LGPS:2165:BACT": "sextupole S2EL supply",
    "LI20:LGPS:2195:BACT": "sextupoles S3EL supply",
    "LI20:LGPS:2275:BACT": "sextupoles S3ER supply",
    "LI20:LGPS:2335:BACT": "sextupole S2ER supply",
    "LI20:LGPS:2365:BACT": "sextupole S1ER supply",
}
# Each supply's median in TEST_15668. E300_15671 and E300_15673 agree within 0.01%, the correctors within 2e-5
# (tests/test_drift.py).
MAGNETS_IN_DAQ = {
    "LI20:LGPS:3011:BACT": -72.1891,
    "LI20:LGPS:3311:BACT": -81.2477,
    "LI20:LGPS:3151:BACT": 99.2317,
    "LI20:LGPS:1910:BACT": 125.909,
    "LI20:LGPS:3204:BACT": -234.5114,
    "LI20:LGPS:3031:BACT": 125.983,
    "LI20:LGPS:3141:BACT": -115.825,
    "LI20:LGPS:3261:BACT": 180.6918,
    "LI20:LGPS:3091:BACT": -115.8243,
    "LI20:LGPS:3330:BACT": 10.000571,
    "LI20:XCOR:3026:BACT": 0.072774,
    "LI20:XCOR:3086:BACT": 0.021586,
    "LI20:YCOR:3017:BACT": 0.0849,
    "LI20:YCOR:3057:BACT": -0.039703,
    "QUAD:LI19:851:BACT": -63.763686,
    "QUAD:LI19:871:BACT": 42.499735,
    "LI20:LGPS:2145:BACT": 1400.0248,
    "LI20:LGPS:2165:BACT": -3199.927,
    "LI20:LGPS:2195:BACT": -1150.0273,
    "LI20:LGPS:2275:BACT": -1149.9828,
    "LI20:LGPS:2335:BACT": -3196.3857,
    "LI20:LGPS:2365:BACT": 1399.9952,
}

# v2 writes (W1), the only ones actuate.py makes. L1: both phase setpoints, absolute, the same delta to each, as ops
# move L1. L2: the phase multiknob, a delta, through the AIDA-PVA service ops' multiknob wrapper calls. It moves
# the LI12-14 subbooster phase setpoints 1:1 (DAQ, 10-16), which is what the readback check reads.
L1_SETPOINTS = ["KLYS:LI11:11:PREQ", "KLYS:LI11:21:PREQ"]
MULTIKNOB_SERVICE = "MKB:VAL"
MULTIKNOB = "mkb:l2_phase.mkb"          # its name as the service expects it: to confirm with Zack
MULTIKNOB_MOVES = ["LI12:SBST:1:PDES", "LI13:SBST:1:PDES", "LI14:SBST:1:PDES"]
# Per kind: the largest single write (one policy action, deg) and the net move allowed per armed session (deg; the
# first sessions', 11-06 plan)
WRITE = {"L1": {"step": 0.25, "budget": 1.0}, "L2": {"step": 0.3, "budget": 1.5}}

# PV name -> what it is and the stage that added it
READ: dict[str, str] = {
    CAMERA: "DTOTR2 XTCAV camera image, raw flat array (S1)",
    **{name: f"DTOTR2 {what} (S2)" for name, what in CAMERA_SETTINGS.items()},
    TCAV_AMPLITUDE: "XTCAV amplitude readback, MV (S4)",
    TCAV_PHASE: "XTCAV phase readback, deg (S4)",
    **{name: f"{what} (S6)" for name, what in L1.items()},
    **{name: f"{what} (S7)" for name, what in L2.items()},
    **{name: f"{what} (S8)" for name, what in {**BLEN, **DRIFT, **MAGNETS}.items()},
}
