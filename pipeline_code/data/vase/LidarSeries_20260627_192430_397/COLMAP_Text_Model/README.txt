# Generator: SplatKing 1.0.7 (build 26)
# Generated: 2026-06-27T17:25:43Z
# World frame: right-handed, gravity-aligned, +Y up
# Pose convention: COLMAP world-to-camera (X_cam = R * X_world + t)
# Camera-local axes: +X right, +Y down, +Z forward
# Applied basis change: M = diag(1, -1, -1) (ARKit camera → COLMAP camera; 180° about +X)
# Units: meters (translations, points3D); pixels (intrinsics)
# points3D.txt sentinels: ERROR = -1, TRACK[] empty by design (no bundle adjustment, no 2D-3D correspondences)
#

SplatKing COLMAP Text Export

This folder contains a COLMAP-compatible text model generated from LiDAR mode camera poses.
Model path: sparse/0/
image_path = images/

When importing into COLMAP: File → Import model, then set image_path to the images/ subdirectory of this folder.

Files:
- sparse/0/cameras.txt
- sparse/0/images.txt
- sparse/0/points3D.txt
- images/<frame>.jpg (sensor-native orientation; cameras.txt holds one shared PINHOLE camera with per-session median intrinsics — see spec §4)
