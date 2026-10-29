"""Channel Access reads through pyepics. v1 is read-only: this module only ever reads.

Planned:
  S1  get_frame(): one DTOTR2 image and its pulse ID (timestamp nanoseconds & 0x1ffff)
  S2  frame geometry: reshape and orient like the DAQ images
  S4  get(name): one scalar PV; refuses names not in pvs.READ
"""
