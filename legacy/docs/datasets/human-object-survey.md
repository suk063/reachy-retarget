# Human–object datasets beyond the existing catalog

Checked: **2026-10-06**. Scope: human demonstrations, human–object motion, grasp/contact annotations, and derivatives that could contribute to Reachy manipulation. This is a documentation and metadata survey, **not a claim of downloaded, normalized, retargeted, or physically validated demonstrations**. Validation runs performed elsewhere in this change have their own reports.

The catalog at survey start contained 333 source entries, including 31 `human_object` entries. Existing families included HUMOTO, ParaHome, BEHAVE, InterCap, GRAB, OakInk/OakInk2, HOT3D, HOI4D, HO-3D, H2O, ARCTIC, TACO, InterAct, CHAIRS, HOI-M3, HO-Dome, DexYCB, OMOMO and their mirrors/derivatives. The 19 entries below were absent by source ID and URL search. They are **19 candidate releases, not 19 independent demonstration collections**: HandX, OpenEgo and HOH-Grasps reuse other collections, and AssemblyHands improves Assembly101 annotations.

Only publisher repositories, dataset cards, official project pages and papers support the findings below. [Evidence metadata](human-object-survey-evidence.json) records checked URLs, HTTP status, document SHA-256 and repository revisions where available. A reachable documentation page does not prove that every archive can be downloaded. No bulk payload was fetched by this survey task.

## Recommended order

| Priority | Source | What it adds | Main blocker before Reachy physics |
|---|---|---|---|
| 1 | HO-Cap | Small separate pose/calibration/model releases; tabletop pick/place, handover and object use | Exact MANO wrist extraction, timestamp verification, object-specific collision models |
| 2 | GigaHands object-complete subset | Broad bimanual objects and activities; explicit object tracking quality labels | Join aligned hands to valid rigid-object tracks; discard nonrigid approximations |
| 3 | CORE4D-Real v2 | Two-person collaboration, carrying, raising and handover with meshes/world trajectories | Choose tasks feasible for one Reachy; resolve license metadata conflict |
| 4 | SHOW3D hand v3 + object v1 | Public per-scene annotations, varied locations and hand/object appearances | Its “world” moves with the capture rig; coverage gaps and missing meshes |
| 5 | ContactPose / HOH-Grasps | Better contact regions and candidate grasp placements | Mostly grasp priors/keyframes; not complete execution trajectories |
| 6 | Aria Digital Twin | Scene context, dynamic objects and body trajectories in a common physical frame | Skeleton coverage and gripper-relevant finger information; custom access terms |

This ranking is an engineering assessment for this repository. It is not a publisher-provided quality ranking. Having human contact annotations helps choose where Reachy should grasp; it does not show that Reachy's two-finger hand can execute that grasp.

## Detailed source records

### 1. HO-Cap — highest value small acquisition

**Verified:** the official release lists separate poses (23.8 MB), models (52.5 MB), calibration (19.6 KB), labels (1.5 GB), and subject media. The three small Box archive URLs returned HTTP 404 to HEAD checks on this date; their advertised sizes do not establish working access. The dataset includes pick/place, handover between hands, and affordance-based object use; data are CC BY 4.0, with a separate GPL-3.0 toolkit. [Official release](https://irvlutd.github.io/HOCap/), [toolkit](https://github.com/IRVLUTD/HO-Cap).

The author HF repository lists `sequence_labels.tar` at 40,478,720 bytes, `calibration.tar` at 51,200 bytes and `models.tar` at 135,690,240 bytes. These are alternative packaging of the same collection, not additional demonstrations. Its card also declares CC BY 4.0. [Author dataset](https://huggingface.co/datasets/JWRoboticsVision/HO-Cap-Dataset).

**Adapter implications:** shared world poses and separate meshes make a rigid-object benchmark plausible. Contact forces, mass/inertia and material friction remain unverified. Reconstructed visual meshes require explicit collision processing; do not silently replace a difficult object with a simpler one. Exact pose semantics and acquisition URLs are documented in the implementation appendix below. Native frame rate must come from source timing evidence, not the viewer's sleep interval.

### 2. GigaHands — use the object-complete subset

**Verified:** the official collection describes 34 hours, 56 subjects, 417 objects and about 14k motion clips. Object poses were released in September 2025, covering about 3.3k sequences in four archives; meshes and an object-aligned MANO-derived keypoint variant are available separately. One- and five-sequence demo archives are linked. License: CC BY-NC 4.0. [Official repository and download routes](https://github.com/brown-ivl/GigaHands).

**Critical schema details:** older `keypoints_3d_mano` were recentered for motion generation and must not be paired directly with object-world tracks. Use aligned keypoints or reconstruct from source MANO parameters with the documented alignment. `optimized_pose.json` contains 6-DoF object estimates; per-scene CSVs distinguish object presence from successful tracking. Some articulated/deformable objects are represented by a rigid pose, so “tracking success” is not rigid-body validity. [Object documentation](https://github.com/brown-ivl/GigaHands/blob/main/README_object.md).

**Inference:** select rigid objects with complete jointly valid hand/object intervals first. No measured contact forces or ready-to-use Reachy collision geometry were verified. Confirm units, temporal offsets and mesh transform against the official visualizer before normalization. Multiple cameras, recentered copies and HandX/VITRA crops must share one canonical source-sequence identity.

### 3. CORE4D-Real v2 — collaborative motion with complete geometry

**Verified:** real data provide per-sequence object transforms and two people; synthetic data provide human motion, object motion and meshes. Real v2 updates human motion from v1. The official repository says CC BY 4.0, while its official HF metadata says MIT: preserve this conflict and verify applicable terms before redistribution. The real v2 motion directory lists four archives totaling approximately 39.6 GB. [Official repository](https://github.com/leolyliu/CORE4D-Instructions), [official HF v2 files](https://huggingface.co/datasets/leolyliu/CORE4D/tree/main/CORE4D_Real_human_object_motions_v2).

Object meshes use meters; `smooth_objposes.npy` is an `(N,4,4)` object-to-world trajectory. Per-person NPZ files include root rotation/translation, body rotations, 12-dimensional hand PCA parameters, 127 SMPL-X joints and vertices in world space. Tasks include cooperative moving/raising/rotation, handover, joining/leaving and obstacle variants. [File definitions](https://github.com/leolyliu/CORE4D-Instructions/blob/main/docs/file_definitions.md).

**Inference:** saved joints reduce dependence on separately licensed SMPL-X for wrist positions. Wrist orientation still requires a documented construction or model FK. Two-human cooperative lifts cannot be relabeled as a single-Reachy success without modeling the partner and support forces. Prefer a feasible single-person grasp segment first. Synthetic augmentation and v1/v2 are not independent real demonstrations.

### 4. SHOW3D — released, but moving-frame semantics matter

**Verified:** the public release has per-scene downloads and an index with hand/object/caption availability. Hand v3 was released October 3, 2026; public hand labels cover train scenes, while test labels are held out. The release describes 2,137 recordings, 20 hours, 60 fps and synchronized ego/exo views. License: CC BY-NC 4.0. [Current dataset card](https://huggingface.co/datasets/facebook/show3d-dataset).

Hand v3 supplies UmeTrack wrist rotation/translation and 21 landmarks; UmeTrack translations/landmarks use millimeters, MANO global translation uses meters. Validity requires confidence and non-null geometry. [Hand schema](https://huggingface.co/datasets/facebook/show3d-dataset/blob/main/hand_pose/README.md).

Object v1 uses world-from-object `R,t`, with `t` in millimeters. Mean object coverage is 74.1%; confidence zero is missing data, not identity. The source warns of unreliable mouse poses and mug-handle rotations. Canonical meshes for 22 aliases come from HOT3D; five other aliases lack those meshes. Hand-only `none` protocols have no object row. [Object schema](https://huggingface.co/datasets/facebook/show3d-dataset/blob/main/object_pose/README.md).

**Critical limitation:** this “world” is the moving backpack rig. Camera calibration alone does not provide an inertial physical-world trajectory. Shared hand/object relative geometry is useful, but gravity-based replay requires a verified stabilization/placement procedure with assumptions recorded. [Frame documentation](https://huggingface.co/datasets/facebook/show3d-dataset/blob/main/scenes/README.md).

### 5. Aria Digital Twin (ADT)

**Verified:** 236 sequences, single/dual-person activities, two indoor spaces and 74 dynamic object instances. The publisher gives 3–6 GB per main sequence, with heavier depth/segmentation optional. Ground truth includes object trajectories, scene information and human tracking; corresponding multi-device recordings share timestamps and a world frame. The HF card points to custom ADT terms; its YAML “privacy policy” entry is not a sufficient data license statement. The license page could not be fetched during this check. [Publisher HF card](https://huggingface.co/datasets/projectaria/aria-digital-twin).

`3d_object_pose.csv` and instance metadata describe objects; `Skeleton_*.json` and device-to-skeleton association are optional. Not every sequence has a tracked skeleton. MPS and ground-truth frames must be joined using supplied alignment information. [Official data format](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_digital_twin_dataset/data_format).

**Inference:** useful for scene-aware reaching/carrying, less immediately suitable for precise antipodal grasping than datasets with detailed hand meshes. Verify finger channels and manipulable object meshes per chosen sequence; do not infer them from the word “skeleton.” No ready physical parameters were verified. A single synchronized event observed by two devices is one demonstration, not two.

### 6. DexMV — human source plus derived robot demonstrations

**Verified:** the repository releases a human-video-to-dexterous-manipulation pipeline, simulator assets, processed demonstration files and a raw-data subset. A `relocate_mustard_example_seq` includes separate hand and object pose folders for the retargeting example; other examples include mug relocation. [Official repository](https://github.com/yzqin/dexmv-sim), [demonstration generation explanation](https://github.com/yzqin/dexmv-sim/blob/master/docs/demo_gen.md).

**Access/license:** Google Drive routes are linked; current full archive size, download completion and separate dataset license were not verified. Do not infer dataset rights from repository code licensing. The example code is a route to inspect exact coordinate conventions, not evidence that every processed demonstration retains original human/object tracking.

**Inference:** valuable as an end-to-end comparison and small initial sample. Existing robot-specific actions are not Reachy commands; translation, temporal alignment and inverse dynamics must be redone for Reachy. Grasp topology differs significantly from the source dexterous hand. Raw video, estimated human tracks and retargeted robot demonstrations require shared lineage and separate validation stages.

### 7. FPHA / First-Person Hand Action Benchmark

**Verified:** provides 21-joint hand trajectories and 6-D object annotations for four objects: juice carton, milk bottle, salt and liquid soap. Hand coordinates are millimeters, textured PLY object models meters. Each object-pose line contains a frame index and a column-major 4×4 transform. The example loader defines projection/calibration. Access requires a form; terms restrict use to academic noncommercial research and explicitly restrict modification/redistribution. [Official data instructions](https://github.com/guiggh/hand_pose_action), [coordinate conversion example](https://github.com/guiggh/hand_pose_action/blob/master/load_example.py).

**Inference:** a compact rigid-object route after access and derivative-use terms are resolved. Object-pose coverage must be restricted to the annotated subset, not all action classes. A wrist orientation derived from hand joints must be labeled derived. Exact selected-archive size and successful access remain unverified. A registered sample in the repository is not proof of unrestricted access to the full collection.

### 8. HOH — real human-to-human handovers

**Verified:** 2,720 handovers, 136 objects and 20 giver/receiver pairs, with event labels for grasp, transfer and release, hand/object segments and object models. The official website currently says access is password protected and asks users to email authors; no request was sent. [Official project](https://tars-home.github.io/hohdataset/).

The original release is approximately 9.51 TB, dominated by media/point clouds. `3dModelAlignments` stores object-model transformations per time step, and separate filtered point-cloud/metadata routes could avoid full media. Object mesh terms vary by asset; do not collapse them into one permissive license. [Publisher supplementary dataset description](https://proceedings.neurips.cc/paper_files/paper/2023/file/d8c6a37c4c94e9a63e53d296f1f668ae-Supplemental-Datasets_and_Benchmarks.pdf).

**Inference:** strong scenario coverage for handover, with more work than ordinary pick/place: hand poses may require fitting/registration from point clouds, exact wrist orientation and contact constraints must be derived, and the receiving agent must be modeled. No claim of current payload access or directly executable Reachy trajectories is made. HOH-Grasps below is a derived subset, not a new capture collection.

### 9. ContactPose — contact-region supervision

**Verified:** hand/object poses, contact maps, object models and RGB-D grasp images. Data other than object models use MIT; object models have per-model licenses. Original Dropbox downloads are no longer valid according to the official repository; the replacement is IEEE DataPort, with a separate public Google Drive sample. [Official repository and current access warning](https://github.com/facebookresearch/ContactPose).

The documented grasp package contains 3-D joints, MANO fits and calibration; old documentation estimates it at about 126 MB, versus approximately 2.5 TB for complete RGB-D. Documentation defines a transform tree, 21-joint convention and contact-map format. The old package size is historical, not a verified size for the replacement route. [Data and transform documentation](https://github.com/facebookresearch/ContactPose/blob/main/docs/doc.md).

**Inference:** particularly useful for choosing a stable, task-appropriate gripper contact region without changing the object. Thermal/contact observations are not measured Reachy force closure and do not establish a complete pick/place trajectory. Use as a grasp prior, then test candidate Reachy wrist/gripper configurations in free-body dynamics. Repeated views of the same grasp do not increase demonstration count.

### 10. HOGraspNet — useful annotations, restrictive access and lost raw data

**Verified:** processed annotations and scanned models for 30 objects; subject-specific/annotation-only download modes exist after a form. The official repository states raw data became inaccessible after storage hardware failure and the planned v2 update is suspended. Terms are academic noncommercial and restrict redistribution/modification; there is no demonstrated unrestricted direct acquisition route. [Official repository, access and status](https://github.com/kaist-uvr-lab/HOGraspNet).

**Schema/gaps:** JSON labels include hand/object information; exact object pose convention and temporal completeness require inspection of the released processed sample and loader. Camera-space grasp annotations cannot simply become world-space motion. Its taxonomy is useful for grasp selection, but scan availability does not establish collision-ready assets, mass, friction or measured contact forces.

**Inference:** lower priority than HO-Cap for immediate retargeting. Keep as an access-required grasp-analysis candidate. Avoid claiming that unavailable raw channels can be recovered by the downloader, and do not count eventual v2 as an existing independent collection.

### 11. AffordPose — static affordance-conditioned grasp prior

**Verified:** 26.7k manually adjusted hand–object interactions, 641 objects, 13 categories and 8 affordances. JSON includes hand mesh/configuration, palm transform, object mesh with affordance labels and canonical object transform. Geometry data are offered as an approximately 855 MB archive with category selection; rendered images are separate. Dataset license is CC BY-NC-ND 4.0, while repository code is GPL-3.0. [Official dataset schema and license](https://affordpose.github.io/), [official code/unit warning](https://github.com/GentlesJan/AffordPose).

**Inference:** useful for locating handles, support regions or graspable surfaces. These are grasp configurations, not temporal demonstrations. Hand-model geometry is documented in millimeters; preserve exact source transforms and verify quaternion convention before adapting. Reachy still needs its own feasible grasp synthesis, approach path and free-body test. The no-derivatives condition must not be treated as permission to publish converted data; applicable derivative-use terms need resolution.

### 12. SHOWMe — shape/contact geometry, not a full manipulation benchmark

**Verified:** the publisher supplies annotated hand/object textured 3-D scans, videos, depth and camera intrinsics; dataset license CC BY-NC-SA 4.0. [Official download page](https://europe.naverlabs.com/research/showme/). The paper describes 96 sequences and grasping protocols aimed at hand–object reconstruction, including use and handover-style grasps. [Original paper](https://openaccess.thecvf.com/content/ICCV2023W/ACVR/papers/Swamy_SHOWMe_Benchmarking_Object-Agnostic_Hand-Object_3D_Reconstruction_ICCVW_2023_paper.pdf).

**Inference:** good for validating visual mesh/contact geometry and testing grasp placement against unusual objects. Capture of a held grasp and changing viewpoint must not be reported as an independent pick/place execution. Exact archive sizes, full per-frame object/hand schema and an isolated small sample were not verified; a guessed README URL returned 404. Preserve object-relative and camera-relative transforms and derive no unobserved release/lift events.

### 13. EgoDex — strong hand trajectories, no documented object state

**Verified:** Apple's release has 829 hours of egocentric manipulation, 194 tasks, 30 Hz video and paired HDF5 head/upper-body/hand poses. The five training archives are approximately 300 GB each; the test archive is 16 GB and additional data approximately 200 GB. Pose transforms start in the ARKit origin frame, with code illustrating camera-frame conversion. Dataset terms are CC BY-NC-ND, separate from code licensing. [Official repository](https://github.com/apple-aiml-research/ml-egodex).

**Gap:** the documented release does not provide tracked object 6-DoF states, object meshes, collision models or forces. An object appearing in RGB or task text is insufficient. **Recommendation:** exclude from the current object-complete retarget benchmark until a separately validated reconstruction layer supplies these fields, with estimation uncertainty and provenance. It can still support hand-motion pretraining; that is a different objective. Crops, aliases and OpenEgo/HandX-style repackaging do not add independent demonstrations.

### 14. HoloAssist — instructional manipulation, missing object 6-DoF labels

**Verified:** first-person task recordings include hands, head/camera-related modalities, depth, gaze and task annotations. The release describes modality-specific rates and `_synced` files aligned to RGB; data terms are CDLA-Permissive-2.0. Download lists allow recording-level selection. [Official project](https://holoassist.github.io/), [data format](https://holoassist.github.io/data_links/README.html).

The original paper identifies object-pose annotation as future work. Consequently, hand/head tracking and task labels should not be presented as object-state supervision. [Original paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_HoloAssist_an_Egocentric_Human_Interaction_Dataset_for_Interactive_AI_Assistants_ICCV_2023_paper.pdf).

**Inference:** useful for task decomposition, approach intent and language; insufficient for this repository's object-complete physical benchmark without reconstruction. Exact metric frame convention must be obtained from PSI recording metadata and projection code before importing hand tracks. No small archive byte size was verified. HandX and OpenEgo conversions retain source identity and cannot fill absent measured object states by renaming fields.

### 15. AssemblyHands — assembly hand pose, not object dynamics

**Verified:** the toolkit documents 42 hand joints, world coordinates, camera calibration, train/validation annotations and Google Drive routes. It explicitly describes the toolkit as CC BY-NC 4.0; verify dataset-specific terms at acquisition rather than treating the code license as blanket permission. [Official toolkit](https://github.com/facebookresearch/assemblyhands-toolkit). AssemblyHands refines hand annotations of Assembly101, so the same source videos are overlapping observations. [Original paper](https://arxiv.org/abs/2304.12301).

**Gap:** no object 6-DoF trajectory or object mesh channel was established in the documented release. Mechanical assembly often also needs articulated-part state, fit tolerance and contact geometry; task names and 2-D images do not supply them. **Recommendation:** exclude from current object-complete retarget execution tests, retain as a reconstruction/pose-quality candidate. An eventual adapter must preserve keypoint ordering, units, camera extrinsics, invalid-joint flags and source-sequence identity. Size of a minimal annotation-only download was not verified.

### 16. Ego-Exo4D — scene/task richness, incomplete manipulation state

**Verified:** EgoPose supplies global 3-D hand/body keypoints, validity information and camera calibration; hand and body annotations are separate, with manual and automatic labels distinguished. [Official pose schema](https://docs.ego-exo4d-data.org/annotations/ego_pose/). Download requires an agreement and configured credentials. The documented full annotation bundle is roughly 10.5 GB; selectors can restrict benchmark, split and take IDs, avoiding multi-TB video. [Official downloader](https://docs.ego-exo4d-data.org/download/).

**Gap:** segmentation and hand pose do not establish metric, time-varying object 6-DoF and matched meshes. **Recommendation:** not an immediate object-complete retarget source. Use only after reconstruction is evaluated separately. Frame-level sparsity, manual versus pseudo ground truth and original capture/take identities must survive conversion. Multiple cameras and new mesh fits of existing takes must not be counted as new demonstrations. Dataset license is its custom agreement; exact applicable terms need review at access, and no credentials were requested or used.

### 17. HandX — mixed-source derivative, not an object-complete replacement

**Verified:** the public release contains 84,729 training and 9,648 test entries of canonicalized `(60,2,21,3)` hand motion, MANO parameters and text. Source mapping is supplied. It redistributes permitted portions of GigaHands, HOT3D and HoloAssist, but does not redistribute ARCTIC or H2O source motion. License is Snap's noncommercial research terms plus source-specific conditions. [Current release card](https://huggingface.co/datasets/alexzhang598/HandX), [provenance instructions](https://github.com/handx-project/HandX).

**Gap:** the documented NPZ schema has no object trajectory/mesh fields; canonicalized short clips can lose the source world and full approach/release context. The GitHub instructions and HF package use different source-metadata filenames, so follow the pinned payload's manifest.

**Recommendation:** useful as a conversion reference or language/hand-motion resource, not a new object-complete execution dataset. Count only unique original capture intervals; do not count overlapping 60-frame windows, train/test packaging, or multiple MANO representations as independent demonstrations. Exact NPZ sizes were not verified.

### 18. OpenEgo — aggregation rather than new measurements

**Verified:** its authors describe consolidation of CaptainCook4D, HOI4D, HoloAssist, EgoDex, HOT3D and HO-Cap, normalized hand keypoints in camera coordinates and action primitives with timestamps. Their table lists source-specific licenses; these must remain separate. [Official OpenEgo repository](https://github.com/ahadjawaid/openego).

**Inference:** useful for comparing annotation conversion and locating action intervals, but no independent capture count should be added. Camera-relative hand normalization alone does not preserve the original objects, physical world, geometry or force information. Confirm each chosen source-specific exported channel instead of interpreting unified task/object-name labels as 6-DoF states. For HO-Cap/HOT3D, acquire missing object data from the canonical upstream source and join using explicit source IDs. Current archive sizes and full payload access were not verified.

### 19. HOH-Grasps — directly relevant grasp candidates, derived keyframes

**Verified:** derived from HOH, with object/hand point clouds at pregrasp and handover, object-to-scene transforms, contact-aware candidate grasps, widths/depths, collision labels, tolerance and grasp quality. Units are meters/radians. The release has three archives totaling approximately 53.8 GB; its HF card lacks license metadata, so original object terms and derivative terms need resolution. [Author dataset schema](https://huggingface.co/datasets/tars-home/HOH-Grasps), [author repository](https://github.com/Terascale-All-sensing-Research-Studio/HOH-Grasps-Dataset).

**Inference:** particularly useful for ranking Reachy grasp locations that avoid the other person's hand and retain geometric margin. Its parallel-gripper candidates require checking Reachy's actual curved fingers, jaw span, wrist reach and collisions. There are only two extracted keyframes, not an executed trajectory between them. The standardized scene frame differs from HOH; use explicit transforms rather than matching raw coordinate values. All entries retain HOH lineage, and sampled robot grasps are hypotheses rather than additional human demonstrations.

## What the current repository can and cannot ingest

At survey start `normalize.py` dispatches explicit adapters for robomimic, HUMOTO and ParaHome, with LeRobot fallback. `human.py` has HUMOTO glTF and ParaHome adapters; it does not provide generic MANO, SMPL-X, ADT, GigaHands or arbitrary CSV support. Acquiring any source above therefore does not establish conversion readiness. The existing robomimic adapter explicitly checks the PickPlaceCan sensor ordering and version; it must not silently interpret another task's observation vector as Can.

| New adapter family | Required source operations | Canonical outputs | Missing information to retain explicitly |
|---|---|---|---|
| HO-Cap | Read sequence metadata; decode MANO/root and object transforms; join calibration and timing | Both hands; all tracked object poses; matched textured mesh references | Forces, friction, inertia; unavailable wrist or timing evidence |
| GigaHands | Join scene/sequence IDs, aligned hands, object metadata and valid tracks | Bimanual hand poses/joints; object pose and geometry; validity masks | Nonrigid/articulated dynamics, force, source tracking gaps |
| CORE4D | Load numeric arrays through restricted deserialization; retain both people; choose role | World wrists/body, object poses/meshes, partner motion | Partner forces, precise wrist orientation if derived, task feasibility |
| SHOW3D | Pin hand v3/object v1; mm-to-m; intersect validity; resolve mesh alias; stabilize frame | Hand/object relative pose and any verified world transform | Inertial frame if unavailable, no-pose intervals, missing canonical meshes |
| ADT | Join timestamps/object IDs, optional skeleton and scene metadata | World hand/body approximations, objects, scene geometry | Finger/contact detail and missing skeletons |
| ContactPose / HOH-Grasps / AffordPose | Preserve source grasp frame and mesh geometry; map candidate contact regions | Grasp hypotheses with source provenance | Complete motion, actuated robot control, physical success |
| EgoDex / HoloAssist / AssemblyHands / Ego-Exo4D | Hand-motion ingestion is possible, but separate object reconstruction is required | Explicitly incomplete source records only | Object pose, geometry, dynamics; no object-complete episode admission |

An object-complete dataset gate should require **a usable temporal hand target, tracked moving-object identity and pose in a documented common frame, and a matched geometry route**, with missing intervals explicitly represented. Video-only object names, task labels, point-cloud centroids and fabricated stationary object poses must not satisfy that gate. Static grasp-prior datasets should remain a separate resource class.

## HO-Cap implementation appendix

Pinned upstream code: `576c63ebf3b84dfec8744ba0f021234213bf0dab`. The official [download config](https://github.com/IRVLUTD/HO-Cap/blob/576c63ebf3b84dfec8744ba0f021234213bf0dab/config/hocap_recordings.yaml) lists these small archives, but all three returned **HTTP 404 on HEAD checks** on 2026-10-06:

- [poses ZIP](https://utdallas.box.com/shared/static/2lofbp2yd005d8o213ns77mdrtxg8eep.zip)
- [models ZIP](https://utdallas.box.com/shared/static/con44iqej33weg9f3rpxof61eh3x2x21.zip)
- [calibration ZIP](https://utdallas.box.com/shared/static/nlp4c6vtd0n8o0entxlh1vxdpcdeh0h8.zip)

Prefer inspecting the reachable [author HF inventory pinned at `2b24836d5e51ad39e56ed4db3fc0c166e755332e`](https://huggingface.co/datasets/JWRoboticsVision/HO-Cap-Dataset/tree/2b24836d5e51ad39e56ed4db3fc0c166e755332e). Its `sequence_labels.tar` has publisher LFS SHA-256 `be3be5f9e5beacab45d75d996c96d3fc959a6d822e31e402214ca899342416e2`. Metadata inspection alone did not establish that this alternative archive has precisely the same members as the older poses ZIP.

The general upstream downloader always also requests labels and subject media; use this project's explicit bounded fetch mechanism for a selected sample instead. Record final URL, source revision, exact bytes, archive/member SHA-256, extraction selection and license. Preserve at least 50 decimal GB free space before and throughout any new transfer.

The [renderer](https://github.com/IRVLUTD/HO-Cap/blob/576c63ebf3b84dfec8744ba0f021234213bf0dab/hocap_toolkit/renderers/sequence_renderer.py) reads `poses_o.npy` as object-major frame sequences and converts quaternion-plus-translation to 4×4 transforms. It selects right-hand slot 0 and left-hand slot 1 from `poses_m.npy`; sequence metadata identifies which sides are present. The [transform utility](https://github.com/IRVLUTD/HO-Cap/blob/576c63ebf3b84dfec8744ba0f021234213bf0dab/hocap_toolkit/utils/transforms.py) defines quaternion order as `qx,qy,qz,qw`, followed by translation.

The [MANO group layer](https://github.com/IRVLUTD/HO-Cap/blob/576c63ebf3b84dfec8744ba0f021234213bf0dab/hocap_toolkit/layers/mano_group_layer.py) splits each 51-value block into 48 pose values and 3 translation values. The [MANO wrapper](https://github.com/IRVLUTD/HO-Cap/blob/576c63ebf3b84dfec8744ba0f021234213bf0dab/hocap_toolkit/layers/mano_layer.py) uses 45 PCA components and a nonflat mean, and converts generated geometry to meters. **The last three values are MANO translation parameters, not automatically the measured wrist coordinate.** Obtain exact wrist position from supplied joints or correctly evaluated MANO, retaining the fitted subject shape. No robot or capture SDK is needed to parse saved numeric states.

The [sequence loader](https://github.com/IRVLUTD/HO-Cap/blob/576c63ebf3b84dfec8744ba0f021234213bf0dab/hocap_toolkit/loaders/sequence_loader.py) maps camera coordinates into the `tag_1` world frame using `inverse(tag_1_to_master) @ camera_to_master`. It identifies `textured_mesh.obj` and `cleaned_mesh_10000.obj` per object. The cleaned model is not automatically a validated collision mesh. The original paper locates the world origin near table center. [Pose frame definition](https://arxiv.org/html/2406.06843v2).

The official annotation pipeline sets `mp_hand.frame_rate` to 30 and synthesizes frame intervals from that value. This supports a **declared 30 Hz source-processing convention**, while measured per-sequence timestamps still take precedence. The viewer's 0.067-second sleep is not acquisition-rate evidence. [Annotation configuration](https://github.com/IRVLUTD/HO-Cap-Annotation/blob/main/config/config.yaml), [timestamp construction](https://github.com/IRVLUTD/HO-Cap-Annotation/blob/main/hocap_annotation/wrappers/mediapipe.py). The paper changed from 70 videos in v2 to 64 in v3; use the actual pinned payload's sequence inventory, not an older headline. [Updated paper](https://arxiv.org/html/2406.06843v3).

## Validation and counting requirements

1. Pin a source revision and select complete source episodes before testing. Keep full episodes and explicit crops linked; never count cameras, mirrors, v1/v2 updates or overlapping windows as independent demonstrations.
2. Check units, axis directions, quaternion order, wrist origin, time base and object mesh scaling using source-overlay or geometric consistency checks. Do not “repair” a mismatch by silently editing source data.
3. Report normalized, kinematically retargeted and physically validated counts separately. Distinguish unreachable geometry from controller failures and absent data.
4. Derive Reachy grasp candidates from unchanged object geometry: antipodal/contact-region feasibility, jaw opening, finger curvature, wrist approach clearance and arm reach. Human contact locations are constraints or priors, not guaranteed Reachy grasps.
5. Test actual actuator-driven dynamics with an unconstrained moving object. No weld, teleport, pose overwrite or prescribed object trajectory during physical validation. Record assumed mass, inertia, friction and collision approximation separately from measured source fields.
6. Measure bilateral contact, penetration, relative slip, task completion, release stability, tracking errors and hardware-model limits. Save failed attempts and report denominators per dataset, object and scenario. Multiple parameter sweeps of one source episode remain one source demonstration.

The most useful next increase in evidence is a small number of **object-complete, source-diverse episodes with explicit failures**, rather than a larger catalog count. HO-Cap and a rigid, quality-filtered GigaHands subset are the first new acquisitions to implement; static contact datasets can improve grasp placement without changing the manipulated object.
