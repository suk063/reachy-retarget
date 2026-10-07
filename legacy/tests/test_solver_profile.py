import mujoco
import numpy as np
import pytest
from reachy_retarget.solver_profile import prepare, PROFILES


@pytest.mark.parametrize('profile', PROFILES)
def test_solver_sensitivity_preserves_materials_and_initial_physical_state(tmp_path, profile):
    xml='''<mujoco><option cone="elliptic" timestep=".002"/>
    <worldbody><geom name="floor" type="plane" size="1 1 .1" friction=".3 .01 .001"/>
    <body pos="0 0 .02"><freejoint/><geom name="cube" type="box" size=".02 .02 .02" mass=".1"/></body>
    </worldbody></mujoco>'''
    model=mujoco.MjModel.from_xml_string(xml)
    prepared=(model,xml,{'simulation_assumptions':[]},None,None,{})+tuple(object() for _ in range(6))
    changed=prepare(prepared,tmp_path/'profile',profile)
    for name in ('geom_friction','geom_solref','geom_solimp','body_inertia','body_mass','qpos0'):
        np.testing.assert_array_equal(getattr(model,name),getattr(changed[0],name))
    assert changed[0].opt.timestep==model.opt.timestep
    assert changed[0].opt.tolerance==1e-10
    assert changed[5]['solver_sensitivity']['source_fidelity'] is False
    assert not prepared[2]['simulation_assumptions']
    for i in range(6,12):assert changed[i] is prepared[i]
