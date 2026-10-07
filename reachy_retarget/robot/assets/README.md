# Calibrated robot resources

These independent resources replace the former controller submodule's asset dependency.
The calibrated URDF is byte-for-byte preserved:
`63a1ecab3312a72c35f19a4bb1142005d74e4d7101da87822d02fea96696d448` (SHA-256).
`robot/robot.toml` binds the same geometry hash and retains measured camera/base calibration.
`collision_spheres.json` preserves its fit metadata, frozen joints and source exclusions.
The runtime explicitly enables cross-arm pairs that the original exclusion list disabled.

`packages/sources.json` records upstream repository commits and source paths for
`reachy_description`, `dynamixel_description`, and `orbita2d_description`. Their original
licenses accompany the copied meshes under each package's `LICENSE`. The robot URDF is
derived from those descriptions with the existing local calibration; no geometry was
regenerated for this migration. Prepared scene manifests remain immutable.
