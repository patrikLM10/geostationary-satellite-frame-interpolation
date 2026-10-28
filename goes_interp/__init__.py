"""GOES-19 satellite frame interpolation.

Self-supervised frame interpolation for geostationary thermal-IR imagery:
RAFT optical flow + physics-guided Z-map + bidirectional softmax splatting
+ U-Net residual synthesis, trained and evaluated on NOAA GOES-19 ABI
Channel 13 (open S3 data).
"""

__version__ = "1.0.0"
