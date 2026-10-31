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

# PV name -> what it is and the stage that added it
READ: dict[str, str] = {
    CAMERA: "DTOTR2 XTCAV camera image, raw flat array (S1)",
    **{name: f"DTOTR2 {what} (S2)" for name, what in CAMERA_SETTINGS.items()},
}
