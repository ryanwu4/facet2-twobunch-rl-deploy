"""Every PV the package may read, as literal strings.

PV names appear nowhere else in the package or the notebook (tests/test_read_only.py checks), so each
stage's PR shows exactly which PVs it adds. v1 only reads: there is no writable list.
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

# L1: the LI11 klystrons 11 and 21, which a feedback moves together (Zack, 10-16) (S6). Monitored during each step's
# capture. FAST_PACT and FAST_AACT are the DAQ's BSA names (BSA_List_S11RF), SSSB_PDES its nonBSA name
# (nonBSA_List_LINAC_SBST; -18.03 deg throughout TEST_15668). PREQ, the absolute phase setpoint ops move L1 with
# (Zack, 10-16), is in no DAQ.
L1 = {
    "KLYS:LI11:11:FB:FAST_PACT": "L1 klystron 11 phase readback, deg",
    "KLYS:LI11:21:FB:FAST_PACT": "L1 klystron 21 phase readback, deg",
    "KLYS:LI11:11:FB:FAST_AACT": "L1 klystron 11 amplitude readback",
    "KLYS:LI11:21:FB:FAST_AACT": "L1 klystron 21 amplitude readback",
    "KLYS:LI11:11:PREQ": "L1 klystron 11 phase setpoint, deg",
    "KLYS:LI11:21:PREQ": "L1 klystron 21 phase setpoint, deg",
    "KLYS:LI11:11:SSSB_PDES": "L1 klystron 11 subbooster phase setpoint, deg",
    "KLYS:LI11:21:SSSB_PDES": "L1 klystron 21 subbooster phase setpoint, deg",
}
# Each L1 phase readback and the setpoint it follows: phases_test() records their offset
L1_SETPOINTS = {"KLYS:LI11:11:FB:FAST_PACT": "KLYS:LI11:11:PREQ", "KLYS:LI11:21:FB:FAST_PACT": "KLYS:LI11:21:PREQ"}

# L2: the subbooster phase setpoints of sectors 12-14, which the L2 multiknob moves 1:1 (DAQ, 10-16), and their phase
# readbacks (S7). Monitored during each step's capture. The DAQ's nonBSA names (nonBSA_List_LINAC_SBST).
L2 = {
    "LI12:SBST:1:PDES": "LI12 subbooster phase setpoint, deg",
    "LI13:SBST:1:PDES": "LI13 subbooster phase setpoint, deg",
    "LI14:SBST:1:PDES": "LI14 subbooster phase setpoint, deg",
    "LI12:SBST:1:PHAS": "LI12 subbooster phase readback, deg",
    "LI13:SBST:1:PHAS": "LI13 subbooster phase readback, deg",
    "LI14:SBST:1:PHAS": "LI14 subbooster phase readback, deg",
}

# PV name -> what it is and the stage that added it
READ: dict[str, str] = {
    CAMERA: "DTOTR2 XTCAV camera image, raw flat array (S1)",
    **{name: f"DTOTR2 {what} (S2)" for name, what in CAMERA_SETTINGS.items()},
    TCAV_AMPLITUDE: "XTCAV amplitude readback, MV (S4)",
    TCAV_PHASE: "XTCAV phase readback, deg (S4)",
    **{name: f"{what} (S6)" for name, what in L1.items()},
    **{name: f"{what} (S7)" for name, what in L2.items()},
}
