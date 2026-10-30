"""Every PV the package may read, as literal strings.

PV names appear nowhere else in the package or the notebook (tests/test_read_only.py checks), so each
stage's PR shows exactly which PVs it adds. v1 only reads: there is no writable list.
"""

CAMERA = "CAMR:LI20:107:Image:ArrayData"

# PV name -> what it is and the stage that added it
READ: dict[str, str] = {
    CAMERA: "DTOTR2 XTCAV camera image, raw flat array (S1)",
}
