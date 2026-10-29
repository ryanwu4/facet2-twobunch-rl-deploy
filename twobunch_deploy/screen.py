"""Image processing (S3, S5): background, hand crop, median / hot-pixel / threshold filter, off-shot row
correction, energy cut, band statistics and the step reading.

Re-implements the parts of Yiheng's XTCAV analysis (xtcav.py) and of twobunch-s2e-rl's online/features.py
that the deployed analysis uses; tests check it against both on DAQ frames.
"""
