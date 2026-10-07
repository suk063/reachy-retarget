"""State-only recording for actuator rollouts, including incomplete attempts.

Commands are position-servo targets; observed deltas are not mislabeled as
issued commands. No image is rendered or stored. No object state is assigned.
"""
import copy
import json
from pathlib import Path
import h5py
import mujoco
import numpy as np
from .agent_dataset import write_archive
from .state_observations import StateObserver, stack_observations
from .store import json_write

# Explicit task fixtures; unrelated scene decorations/other pegs are excluded.
TASK_FIXTURES = {
    'PickPlaceCan': {'pickup_bin':'bin1', 'receptacle':'bin2'},
    'Lift': {'support_table':'table'},
    'Stack_D0': {'support_table':'table'},
    'Threading_D0': {'support_table':'table'},
    'NutAssemblySquare': {'support_table':'table', 'target_peg':'peg1'},
    'PickCube-v1': {'support_table':'table-workspace'},
}


def synchronize_observation(model,data):
    """Refresh only derived transforms/velocities, never qpos/qvel/integration state."""
    mujoco.mj_fwdPosition(model,data)
    mujoco.mj_fwdVelocity(model,data)
    mujoco.mj_fwdActuation(model,data)


def _source_provenance(source_metadata, supplied=None):
    """Promote explicit, checksum-linked source evidence without guessing URLs."""
    result = {"source_urls": copy.deepcopy(source_metadata.get("source_urls") or []),
              "source_revision": source_metadata.get("source_revision"),
              "provenance": copy.deepcopy(source_metadata.get("provenance") or [])}
    if supplied is not None:
        if not isinstance(supplied, dict) or set(supplied) - set(result):
            raise ValueError("source_provenance accepts source_urls, source_revision and provenance only")
        incoming = supplied.get("provenance", [])
        if not isinstance(incoming, list) or not incoming:
            raise ValueError("Supplied source provenance needs checksum-linked raw records")
        known = {row.get("sha256") for row in result["provenance"] if isinstance(row, dict)} - {None}
        if not known or any(not isinstance(row, dict) or row.get("sha256") not in known for row in incoming):
            raise ValueError("Supplied raw source SHA256 does not match inherited provenance")
        for row in incoming:
            if row not in result["provenance"]:
                result["provenance"].append(copy.deepcopy(row))
        if "source_urls" in supplied:
            urls = supplied["source_urls"]
            if not isinstance(urls, list) or any(not isinstance(url, str) or not url for url in urls):
                raise ValueError("Source URLs must be explicitly supplied nonempty strings")
            result["source_urls"] = list(dict.fromkeys(result["source_urls"] + urls))
        if "source_revision" in supplied:
            revision = supplied["source_revision"]
            if revision is not None and (not isinstance(revision, str) or not revision):
                raise ValueError("Source revision must be a nonempty string or null")
            if result["source_revision"] and revision != result["source_revision"]:
                raise ValueError("Supplied source revision conflicts with inherited provenance")
            result["source_revision"] = revision
    result["provenance_missing_fields"] = [key for key in ("source_urls", "source_revision") if not result[key]]
    return result


def _recording_metadata(observer_metadata, source_metadata, timestamps, report,
                        source_episode, output, *, source_provenance=None):
    """Keep source-only descriptions nested; describe the measured target rows."""
    meta = copy.deepcopy(observer_metadata)
    times = np.asarray(timestamps, dtype=float)
    delta = np.diff(times)
    uniform = bool(len(delta) and np.all(np.isfinite(delta)) and np.all(delta > 0)
                   and np.allclose(delta, np.median(delta), rtol=1e-8, atol=1e-12))
    rate = float(1 / np.median(delta)) if uniform else None
    meta["clock"] = dict(meta["clock"], sampling_rate_hz=rate,
                         sampling_rate_source="measured timestamp differences",
                         uniform_sampling=uniform if len(delta) else None)
    meta.update(robot_type="Reachy2", fps=rate,
                source_sequence=source_metadata["source_sequence"],
                source_group=source_metadata.get("source_group", source_episode),
                source_episode_id=source_episode, source_metadata=copy.deepcopy(source_metadata),
                action_semantics="Recorded absolute Reachy2 joint position servo targets; exact actuator controls before each physics substep",
                control_labels={"desired_joint_targets": "command/joint_position; ordered by command_joint_names",
                                "desired_ee_targets": "No Cartesian commands issued by this position-servo rollout",
                                "controller_state": "controller_state_json; custom servo/feedback snapshot, not a native reachy-agent controller",
                                "substep_controls": "applied_controls with applied_controls_valid; exact controls before each mj_step"},
                target_model={"format": "MuJoCo XML", "scene_path": str(Path(output) / "scene.xml"),
                              "scene_sha256": meta["model_identity"]["scene_sha256"],
                              "source_urdf_path": report.get("physics", {}).get("source_urdf"),
                              "source_urdf_sha256": meta["model_identity"]["source_urdf_sha256"]},
                derived_fields={"fps": "Reciprocal median timestamp difference only when measured sampling is uniform; otherwise null"})
    for key in ("source_id", "source_format"):
        if key in source_metadata:
            meta[key] = copy.deepcopy(source_metadata[key])
    if source_metadata.get("hdf5_sha256"):
        meta["source_normalized_hdf5_sha256"] = source_metadata["hdf5_sha256"]
    meta.update(_source_provenance(source_metadata, source_provenance))
    return meta


class RolloutRecorder:
    def __init__(self, model, manifest, details, output):
        self.model,self.output=model,Path(output)
        task=details['task']
        if task not in TASK_FIXTURES:
            raise ValueError('Task needs an explicit observation-object manifest')
        objects={name:spec['body'] for name,spec in manifest['objects'].items()}
        objects.update(TASK_FIXTURES[task])
        roles={'pickup':details['object_id']}
        if 'receptacle' in objects:roles['receptacle']='receptacle'
        elif task=='Stack_D0':roles['receptacle']='cubeB'
        elif task=='Threading_D0':roles['receptacle']='tripod_obj'
        self.observer=StateObserver(model,task_objects=objects,robot_body_root='base_link',
            base_body='base_link',head_body='head',tcp_sites={'left':'l_arm_tip_tcp','right':'r_arm_tip_tcp'},
            object_roles=roles,model_identity={'scene_sha256':manifest['scene_sha256'],
                                             'source_urdf_sha256':manifest['source_urdf_sha256']})
        self.frames=[];self.command=[];self.controller=[];self.controls=[];self.durations=[]
        self.pending=None;self.pending_controls=[]
        self.metadata=dict(self.observer.metadata,
            observation_timing='pre-integration-state-after-command-selection',
            command_profile='position-servos-with-recorded-feedback-memory-v1',
            command_joint_names=[model.joint(int(model.actuator_trnid[i,0])).name for i in range(model.nu)],
            command_joint_units=['m' if int(model.jnt_type[int(model.actuator_trnid[i,0])])==2 else 'rad' for i in range(model.nu)],
            command_semantics={'command/joint_position':{
                'kind':'joint_position','representation':'absolute','timing':'pre_action',
                'joint_names':[model.joint(int(model.actuator_trnid[i,0])).name for i in range(model.nu)]}},
            issued_control_modes=['joint_position'],
            control_notes={'applied_controls':'Exact actuator controls before every mj_step',
                           'controller_state_json':'Retarget servo/feedback history, not a reachy-agent native controller snapshot',
                           'velocity_commands':'Not issued by this position-servo rollout; measured velocity remains an observation'},
            rgb_stored=False)

    def begin(self,data,controller_state):
        if self.pending is not None:raise RuntimeError('Previous interval was not finalized')
        synchronize_observation(self.model,data)
        self.pending=self.observer.capture(data)
        self.pending_command=data.ctrl.copy()
        self.pending_controller=json.dumps(controller_state,sort_keys=True,allow_nan=False)
        self.pending_controls=[]

    def before_step(self,data):
        if self.pending is None:raise RuntimeError('No current observation interval')
        self.pending_controls.append(data.ctrl.copy())

    def end(self,data):
        if self.pending is None:return
        self.frames.append(self.pending);self.command.append(self.pending_command)
        self.controller.append(self.pending_controller);self.controls.append(np.asarray(self.pending_controls))
        self.durations.append(float(data.time-self.pending['arrays']['timestamp']))
        self.pending=None

    def finish(self,data,source_metadata,report,source_episode,*,source_provenance=None):
        self.end(data)
        if not self.frames:return {'status':'no_observation_frames'}
        result=stack_observations(self.frames)
        arrays=result['arrays']
        max_substeps=max(len(v) for v in self.controls)
        applied=np.full((len(self.controls),max_substeps,self.model.nu),np.nan)
        mask=np.zeros((len(self.controls),max_substeps),dtype=bool)
        for i,control in enumerate(self.controls):
            applied[i,:len(control)]=control;mask[i,:len(control)]=True
        arrays.update({'command/joint_position':np.asarray(self.command),
                       'controller_state_json':np.asarray(self.controller),
                       'applied_controls':applied,'applied_controls_valid':mask,
                       'control_interval_s':np.asarray(self.durations)})
        synchronize_observation(self.model,data)
        terminal=self.observer.capture(data)['arrays']
        arrays.update({'terminal/'+name:value for name,value in terminal.items()})
        # Preserve even a single-frame failure as an attempt artifact. Canonical
        # sequence publication still requires >=2 rows and honest timing.
        raw=self.output/'state-attempt.hdf5'
        fallback=dict(schema='reachy-retarget-raw-measured-state-v1',metadata_complete=False,
                      source_episode_id=source_episode,physics_validated=False,
                      scope='Retained measured arrays and issued controls; metadata/provenance finalization is pending, so this is not a canonical episode')
        with h5py.File(raw,'x') as f:
            for name,value in arrays.items():
                value=np.asarray(value)
                if value.dtype.kind in 'USO':
                    f.create_dataset(name,data=value.astype(object),dtype=h5py.string_dtype('utf-8'))
                else:f.create_dataset(name,data=value,compression='gzip' if value.ndim and value.size else None)
            f.attrs['metadata_json']=json.dumps(fallback,allow_nan=False)
        try:
            meta=_recording_metadata(self.metadata, source_metadata, arrays['timestamp'], report,
                                     source_episode, self.output, source_provenance=source_provenance)
            meta.update(simulation_assumptions=report['physics'].get('simulation_assumptions',[]),
                        status=report['status'],physics_validated=report['physics_validated'],
                        physics_report=report,missing_fields=result['missing_fields'],
                        final_state_storage='terminal/* is the measured boundary after the last applied interval; not another independent row',
                        validity_masks={'applied_controls':'applied_controls_valid'})
            encoded=json.dumps(meta,allow_nan=False)
        except Exception as error:
            fallback.update(metadata_error=type(error).__name__+': '+str(error),
                            scope='Measured arrays retained after metadata/provenance failure; not a canonical episode')
            with h5py.File(raw,'r+') as f:f.attrs['metadata_json']=json.dumps(fallback,allow_nan=False)
            raise
        with h5py.File(raw,'r+') as f:f.attrs['metadata_json']=encoded
        if len(self.frames)<2:return {'status':'partial_attempt','path':str(raw),'rows':len(self.frames)}
        archive=write_archive(self.output/'common','reachy-rollout',source_episode,arrays,meta)
        return {'status':'recorded','archive':str(archive),'rows':len(self.frames),
                'task_objects':list(self.metadata['objects']),'neck_joint_indices':self.metadata['joint_groups']['neck'],
                'rgb_stored':False,'physics_validated':report['physics_validated']}
