SplatKing Capture — Quick Reference

This folder contains a 3D capture produced by SplatKing on iOS. The
contents below are intended for reconstruction workflows. If you are
sharing this folder with others, note that the captured imagery and
LiDAR data may include recognizable people, places, or location
information from the original scene.

COLMAP_Text_Model/
  Contains a COLMAP-compatible 3D reconstruction model.
  To train a Gaussian Splat:
    1. Open your training platform (LichtFeld, Postshot, Nerfstudio, etc.)
    2. Drag and drop the COLMAP_Text_Model folder into the trainer.
    3. Set the image path to COLMAP_Text_Model/images/ if prompted.
  The sparse/0/ subfolder contains camera poses and a LiDAR point cloud
  in standard COLMAP text format.

sensor_data/
  Raw per-frame sensor output: LiDAR depth maps, confidence maps, and
  camera pose/intrinsics JSON files. You do not need these to train a
  Gaussian Splat. They are provided for advanced workflows (custom depth
  fusion, research pipelines, etc.).

lidar_pointcloud_world_xyz.ply
  The full LiDAR point cloud in PLY format, when the capture session
  produced sufficient tracking data to generate one. Can be opened in
  MeshLab, CloudCompare, or any point cloud viewer. May be absent for
  sessions with limited or interrupted tracking.

splatpack.json / photo_series.json
  SplatKing metadata files. Used by SplatKing and compatible tools.
  Not required for standard Gaussian Splatting training.
