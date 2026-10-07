import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile

import h5py
import mujoco
import numpy as np
import pytest

from reachy_retarget.frozen_parent import materialize, model_equivalence


MESH = b'v 0 0 0\nv .1 0 0\nv 0 .1 0\nv 0 0 .1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n'
SHA = lambda raw: hashlib.sha256(raw).hexdigest()


def fixture(tmp_path, *, external=False, corrupt=None):
    origin = Path('/unavailable-original-pod/spool')
    workspace = Path('workspaces/base')
    source = Path('data/normalized/source/episode.h5')
    attempt = workspace/'runs/dynamics/aligned/episode'
    asset_relative = workspace/'data/assets/generated.obj'
    shared = tmp_path/'shared'; shared.mkdir()
    asset_path = shared/'generated.obj' if external else origin/asset_relative
    if external: asset_path.write_bytes(MESH)
    xml = f'''<mujoco><option timestep=".001"/><asset><mesh name="object_mesh" file="{asset_path}"/></asset>
      <worldbody><geom name="floor" type="plane" size="2 2 .1"/>
       <body name="robot" pos="0 0 1"><joint name="r_shoulder_pitch" axis="0 1 0"/><geom type="sphere" size=".05" pos=".1 0 0"/></body>
       <body name="object" pos="1 0 .5"><freejoint name="object_free"/><geom type="sphere" size=".03"/></body>
       <geom name="mesh_visual" type="mesh" mesh="object_mesh" pos="3 0 0" contype="0" conaffinity="0"/>
      </worldbody><actuator><position joint="r_shoulder_pitch" kp="10"/></actuator></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml, assets={str(asset_path): MESH})
    data = mujoco.MjData(model)
    init = {k:getattr(data,k).copy() for k in ('qpos','qvel','ctrl')}
    values = {k:[] for k in ('qpos','qvel','ctrl')}
    clock = []
    for target in (.01, .02, -.01, 0.):
        data.ctrl[:] = target
        for _ in range(10): mujoco.mj_step(model,data)
        mujoco.mj_forward(model,data)
        clock.append(data.time)
        for key in values: values[key].append(getattr(data,key).copy())
    def hdf5(entries):
        buffer = io.BytesIO()
        with h5py.File(buffer,'w') as f:
            for key, value in entries.items(): f[key] = value
        return buffer.getvalue()
    source_bytes = hdf5({'time_s':np.arange(3)*.02})
    replay = {'time_s':clock,**{'initial/'+k:v for k,v in init.items()},
              'simulation/qpos':values['qpos'], 'simulation/qvel':values['qvel'],
              'simulation/actuator_control':values['ctrl']}
    report = {'source_hdf5_sha256':SHA(source_bytes), 'scene_sha256':SHA(xml.encode()),
              'plan':{'alignment_variant':'pad_translation'}, 'success':False,
              'physics':{'scene_asset_hashes':{'meshes':{str(asset_path):SHA(MESH)},'textures':{},'hfields':{}}}}
    audit = {'actuator_replay_pass':True,'scene_sha256':report['scene_sha256'],'frames':4}
    files = {str(workspace/source):source_bytes,str(workspace/source.with_suffix('.json')):b'{"source": "preserved"}\n',
             str(workspace/'data/retargeted/source/episode/motion.h5'):hdf5({'time_s':np.arange(3)*.02}),
             str(workspace/'data/retargeted/source/episode/validation.json'):b'{"kinematic":true}\n',
             str(attempt/'scene.xml'):xml.encode(),str(attempt/'result.json'):json.dumps(report).encode(),
             str(attempt/'plan.h5'):hdf5({'time_s':np.arange(3)*.02}),
             str(attempt/'replay.h5'):hdf5(replay),str(attempt/'actuator-replay-audit.json'):json.dumps(audit).encode()}
    if not external: files[str(asset_relative)] = MESH
    if corrupt == 'asset': files[str(asset_relative)] += b'# corrupt\n'
    if corrupt == 'missing_asset': files.pop(str(asset_relative))
    if corrupt == 'source': files[str(workspace/source)] += b'changed'
    archive = tmp_path/'artifact.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        root = tarfile.TarInfo('.'); root.type = tarfile.DIRTYPE; tar.addfile(root)
        for path,raw in files.items():
            info=tarfile.TarInfo('./'+path);info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
    kwargs=dict(archive_sha256=SHA(archive.read_bytes()), original_spool_root=str(origin),
                workspace_relative=str(workspace),source_relative=str(source),attempt_relative=str(attempt),
                shared_roots=(shared,),minimum_free_bytes=0,replay_steps=25)
    return archive,kwargs,files


def test_archived_generated_assets_remain_portable_without_original_pod(tmp_path):
    archive,kwargs,originals=fixture(tmp_path)
    before=archive.read_bytes();out=tmp_path/'published'
    result=materialize(archive,out,**kwargs)
    assert result['ready'] and result['model_equivalence']['equal']
    assert result['actuator_prefix_equivalence']['physics_steps']==25
    assert result['actuator_prefix_equivalence']['relocated_max_state_difference']==0
    assert result['actuator_prefix_equivalence']['recorded_prefix_max_state_error']==0
    assert archive.read_bytes()==before==(out/'original-artifact.tar.gz').read_bytes()
    for path,raw in originals.items():
        if '/data/assets/' not in path:
            assert (out/'original-inputs'/path).read_bytes()==raw
    parent=Path(result['parent_workspace']);attempt=Path(result['parent_attempt'])
    assert (parent/kwargs['source_relative']).read_bytes()==originals[kwargs['workspace_relative']+'/'+kwargs['source_relative']]
    assert (attempt/'plan.h5').read_bytes()==originals[kwargs['attempt_relative']+'/plan.h5']
    model=mujoco.MjModel.from_xml_path(str(attempt/'scene.xml'))
    assert model.nmesh==1
    assert '/unavailable-original-pod/' not in (attempt/'scene.xml').read_text()
    receipt=(out/'materialization.json').read_bytes()
    assert materialize(archive,out,**kwargs)==result
    assert (out/'materialization.json').read_bytes()==receipt


def test_verified_shared_asset_copied_and_no_longer_required(tmp_path):
    archive,kwargs,_=fixture(tmp_path,external=True)
    result=materialize(archive,tmp_path/'published',**kwargs)
    assert result['asset_relocations'][0]['verified_source']=='explicit_shared_source_root'
    shutil.rmtree(tmp_path/'shared')
    model=mujoco.MjModel.from_xml_path(str(Path(result['parent_attempt'])/'scene.xml'))
    assert model.nmesh==1
    assert materialize(archive,tmp_path/'published',**kwargs)['ready']


@pytest.mark.parametrize('corruption',['asset','missing_asset','source'])
def test_incomplete_or_corrupt_inputs_never_publish_ready_parent(tmp_path,corruption):
    archive,kwargs,_=fixture(tmp_path,corrupt=corruption)
    out=tmp_path/'failed'
    with pytest.raises(ValueError):materialize(archive,out,**kwargs)
    assert not (out/'materialization.json').exists()
    assert json.loads((out/'failure.json').read_text())['ready'] is False
    assert (out/'original-artifact.tar.gz').read_bytes()==archive.read_bytes()
    with pytest.raises(ValueError,match='Incomplete'):materialize(archive,out,**kwargs)


def test_archive_hash_and_postpublication_asset_tampering_rejected(tmp_path):
    archive,kwargs,_=fixture(tmp_path)
    with pytest.raises(ValueError,match='archive checksum'):
        materialize(archive,tmp_path/'bad',**dict(kwargs,archive_sha256='bad'))
    assert not (tmp_path/'bad').exists()
    result=materialize(archive,tmp_path/'good',**kwargs)
    asset=Path(result['asset_relocations'][0]['materialized_path']);asset.write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum changed'):
        materialize(archive,tmp_path/'good',**kwargs)


def test_compiled_physical_property_change_cannot_pass_path_only_proof():
    xml='<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>'
    before=mujoco.MjModel.from_xml_string(xml)
    after=mujoco.MjModel.from_xml_string(xml.replace('mass="1"','mass="2"'))
    with pytest.raises(ValueError,match='Compiled model changed'):
        model_equivalence(before,after)
