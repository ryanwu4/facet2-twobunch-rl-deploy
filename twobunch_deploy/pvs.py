"""Every PV the package may read, as literal strings.

machine.py refuses any name not listed here, so each stage's PR shows exactly which PVs it adds.
v1 only reads: there is no writable list.
"""

# PV name -> what it is and the stage that added it
READ: dict[str, str] = {}
