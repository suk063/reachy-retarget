# ParaHome original-source execution

The first three complete original recordings were acquired and normalized on the persistent cluster on 2026-10-07. These are human–object recordings, not third-party robot retargets. Source common archives were written and checksum-verified; source availability and normalization do not establish Reachy IK or dynamics success.

| Recording | Original frames at 30 Hz | Task-related parent objects | Recorded object-part pose channels |
|---|---:|---:|---:|
| seq/s1 | 4,652 | 14 | 22 |
| seq/s2 | 5,000 | 14 | 23 |
| seq/s3 | 3,343 | 11 | 20 |

All three belong to participant `p1`. They retain their complete original recording groups `parahome/seq/s1`, `parahome/seq/s2`, and `parahome/seq/s3`; no crops or overlapping versions are counted as extra demonstrations. These are three of the publisher's 207 recordings, not full-corpus execution. The scenarios include cooking, cup/kettle transfer, washing-machine and cabinet interaction, and laptop/seated activity.

## Official source and integrity

The [official repository](https://github.com/snuvclab/ParaHome/tree/535dada556536a54d8b3a5185f2a153e6e3ccbca) is pinned to `535dada556536a54d8b3a5185f2a153e6e3ccbca`. Its public downloads remain accessible without accounts or forms. The dataset license is CC-BY-NC-SA-4.0, intended for noncommercial academic use. The [README](https://github.com/snuvclab/ParaHome/blob/535dada556536a54d8b3a5185f2a153e6e3ccbca/README.md) notes that some cabinet-retrieval object positions were manually filled and may have penetration/alignment errors; those published values are preserved.

The historical collection lock records these original payloads:

| Payload | Bytes | Historical SHA256 |
|---|---:|---|
| [seq.zip](https://drive.google.com/file/d/10MYSSM2H7f6g2n9nnXta48qmAhZ7r4yd/view) | 2,233,478,273 | `f5ba43903299e197af323c5c85822b3d59057e32c222aa1a28a061638336cd0e` |
| [scan.zip](https://drive.google.com/file/d/1-OuWvVFOFCEhut7J2t1kNbr5jv78QNFP/view) | 135,179,489 | `cd88fdec27ab317a61e621c424ea6dfabd62d0e0462c1d0114abc931500501b0` |
| [metadata.json](https://drive.google.com/file/d/1jPRCsotiep0nElHgyLQNjlkHsWgHbjhi/view) | 2,449 | `f7cea6a6f4aaf5580f3586deb8b1017b6b7d7e4f3b1314679bf94cea2a5b7140` |
| [joint_info.pkl](https://drive.google.com/file/d/15fGnZn8o4I2bzQtQF-9MliwxKc2IUdzI/view) | 2,610 | `253f69215639e93025ea8b3c66fa293c39a68855fd8fb6229adfd2b61e667fba` |

The historical catalog mislabeled the last two files as ZIP archives. Their actual JSON and restricted NumPy-pickle contents are preserved under the correct extensions. Their complete SHA256 values were reverified. For the two large ZIP files, exact HTTP 206 byte ranges and ZIP member CRC32 were verified and every acquired member received a SHA256 receipt. **Whole-archive SHA256 was not reverified**, because the complete archives were not downloaded.

Sequence spans transferred: s1 13,089,798 bytes, s2 13,413,186 bytes, s3 8,908,849 bytes. ZIP directory/metadata requests are additional. Simplified OBJ meshes were selected from the union of the three recordings' task objects: 34 members, 135,198,863 range bytes including local headers/padding, 411,706,986 decoded bytes. The union covers all 22 scanned parent objects; the saved pose channels in each recording still contain only that recording's declared task objects. No RGB, camera imagery, SMPL-X models, or third-party robot trajectories were acquired.

All transfers use `/mnt/reachy-retarget/queue/transfer.lock` and enforce the 50 decimal GB free-space reserve. A failed tiny ZIP-header request returned HTTP 500; bounded contiguous sequence-range acquisition succeeded. Partial sources and failure receipts remain available.

## Normalization and execution contract

`reachy_retarget.parahome_pipeline.fetch(root)` performs explicit acquisition. `normalize(root)` performs no network access and reuses `human.parahome`. The wrapper constructs a documented local archive view before calling the existing decoder, filtering `object_in_scene.json` using the pinned [annotation-to-item mapping](https://github.com/snuvclab/ParaHome/blob/535dada556536a54d8b3a5185f2a153e6e3ccbca/data/annot2item.json). Original downloaded JSON and state members remain immutable.

Recorded aliases reconcile release naming: `table→diningtable`, `trash can→trashbin`, `cabinet→sink`, `freezer→refrigerator`. The README describes the under-sink cabinet, and the released articulation metadata places its door parts under `sink`. All parts of selected articulated objects are retained. Objects declared absent are excluded; unmatched task labels/items are recorded explicitly, never replaced with the whole scene. All three selected recordings have no unmatched annotations/items.

The existing decoder preserves the complete camera-synchronized 30 Hz timeline, human body/hand states, source-world object transforms, articulated states, and explicit validity masks for absent object samples. The anatomical wrist mapping is a derived human reference, not a measured Reachy TCP or pad pose. Scan hashes, original member hashes, source URLs/revision, participant identity and derived archive hashes are recorded in authoritative normalized metadata sidecars.

Cluster artifacts:

- Acquired members, ranges, receipts and directory inventories: `/mnt/reachy-retarget/native/parahome/`.
- Isolated normalized workspaces: `/mnt/reachy-retarget/native/parahome/workspaces/s1`, `s2`, `s3`.
- Common source archives: `/mnt/reachy-retarget/datasets/parahome/episodes/parahome-s1-source-v1.hdf5`, `parahome-s2-source-v1.hdf5`, `parahome-s3-source-v1.hdf5`.
- Archive verification summary: `/mnt/reachy-retarget/native/parahome/common-archives.json`.
- Three ready-to-enqueue `legacy_pipeline` jobs: `/mnt/reachy-retarget/native/parahome/legacy-jobs.json`.

All three manifests were subsequently found in the persistent queue as `parahome-s1-legacy-v1`, `parahome-s2-legacy-v1`, and `parahome-s3-legacy-v1`. The queue's per-task `result.json` is authoritative for execution outcomes; this source-integration report does not infer success from enqueueing.

The queued path calls the existing IK baseline and saves derived motion in the common archive format. Source data, IK candidates and measured simulation states remain distinct. ParaHome currently has no verified task-specific dynamics adapter or recorded object mass/friction/contact; these remain explicit physics blockers after IK. No geometry scaling, source-object relocation or timing optimization was introduced by this integration.

## Validation

Four focused offline tests cover task-object selection, reuse of the actual human decoder, original timeline/source preservation, changed-member rejection and ZIP range/CRC processing. Three real cluster source archives passed `inspect_archive`; all contain 0 RGB channels and `physics_validated=false`.

| Common source archive | SHA256 |
|---|---|
| s1 | `3924891268c618d4dbefb5ce1b792d990ec2c5a19aa06a589d3c39260164215c` |
| s2 | `feddaca64271e0158332469ffa3790f74099a11d6454e9e3a43bb5d5f012abba` |
| s3 | `02951765b9bcc387ce28de8968bebaf7526f9e497a2efc51c336cc82c7cc9dd6` |
