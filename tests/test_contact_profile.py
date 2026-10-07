import mujoco
import numpy as np
from reachy_retarget.contact_profile import prepare


def test_global_contact_profile_changes_only_declared_compliance_and_timestep(tmp_path):
    xml='''<mujoco><option timestep=".002"/><default><geom solref=".02 1"/></default>
      <worldbody><geom name="floor" type="plane" size="1 1 .1" friction=".4 .02 .003"/>
      <body pos="0 0 .04"><freejoint/><geom name="box" type="box" size=".02 .02 .02" mass=".1"/></body>
      </worldbody><contact><pair geom1="floor" geom2="box" solref=".01 .8"/></contact></mujoco>'''
    model=mujoco.MjModel.from_xml_string(xml)
    original=[model,xml,{'simulation_assumptions':[]},None,None,{}]+[object() for _ in range(6)]
    result=prepare(tuple(original),tmp_path/'profile')
    np.testing.assert_array_equal(model.geom_solref[:,0],.02)
    np.testing.assert_array_equal(result[0].geom_solref[:,0],.004)
    np.testing.assert_array_equal(result[0].pair_solref,[[.004,.8]])
    for key in ('geom_friction','geom_size','body_mass','body_inertia','qpos0','geom_solimp'):
        np.testing.assert_array_equal(getattr(result[0],key),getattr(model,key))
    assert result[0].opt.timestep==.001
    assert result[5]['simulation_profile']['source_fidelity'] is False
    assert not original[2]['simulation_assumptions']
    for i in range(6,12):assert result[i] is original[i]
