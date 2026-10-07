"""Behavioral checks for common geometry decisions and unassisted search."""

from dataclasses import asdict
import json
import xml.etree.ElementTree as ET

import h5py
import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from reachy_retarget.interaction import phases, contact_contract, contact_gates, geometry_seed, add_search_sensors, supported_release
from reachy_retarget.trajectory_search import SearchConfig, optimize
from reachy_retarget.physics import BASE, ARMS


def geometry(name="arbitrary", mass=.1, kind="box", size=".02 .035 .04"):
    xml = f'<mujoco><worldbody><body name="{name}"><freejoint/><geom name="{name}_shape" type="{kind}" size="{size}" mass="{mass}" friction=".3 .01 .001"/></body></worldbody></mujoco>'
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def seed(model, data):
    return geometry_seed(model, data, 1, np.array([.05, 0, 0]),
                         Rotation.from_euler("z", .4).as_matrix(), np.array([0, 1, 0]))


def test_names_do_not_select_grasp_or_force():
    a = seed(*geometry("Can"))
    b = seed(*geometry("Threading_D0"))
    for key in ("point", "yaw_deg", "normal_force_n", "extents_m"):
        np.testing.assert_array_equal(a[key], b[key])


def test_force_responds_to_mass_without_changing_geometry_or_model():
    light = seed(*geometry(mass=.05))
    model, data = geometry(mass=.2)
    mass, inertia, friction, size = [x.copy() for x in (model.body_mass,model.body_inertia,model.geom_friction,model.geom_size)]
    heavy = seed(model, data)
    assert heavy["normal_force_n"] > light["normal_force_n"]
    for actual, expected in zip((model.body_mass,model.body_inertia,model.geom_friction,model.geom_size), (mass,inertia,friction,size)):
        np.testing.assert_array_equal(actual, expected)


def test_cylinder_has_no_arbitrary_yaw():
    result = seed(*geometry(kind="cylinder", size=".025 .04"))
    assert result["yaw_deg"] == 0


def test_contact_schedule_preserves_repeated_grasps_and_terminal_intent():
    grip = [-1, -1, 1, 1, -1, 1, 1]
    assert [(p.start,p.stop,p.mode) for p in phases(grip)] == [(0,2,"free"),(2,4,"grasp"),(4,5,"free"),(5,7,"grasp")]
    assert contact_contract(grip)["terminal_contact"]
    assert contact_contract([-1,1,1,-1])["release_required"]


def test_contact_modes_cannot_silently_inherit_grasp_success():
    with pytest.raises(ValueError, match="complete physical contact contract"):
        contact_gates("push", acquired=False, fraction=0, translation=0, rotation=0)
    gates = contact_gates("grasp_carry", acquired=True, fraction=.94, translation=.004, rotation=.1)
    assert gates["bilateral_lifted_grasp"] and not gates["carry_contact_95pct"]
    assert not gates["grasp_translation_3mm"] and not gates["grasp_rotation_3deg"]


@pytest.mark.parametrize("position,force,lifted,near,expected", [
    ([0,0,1],1.,True,True,True),
    ([0,0,1],0.,True,True,False),
    ([.1,0,1],1.,True,True,False),
    ([0,0,1.1],1.,True,True,False),
    ([0,0,1],1.,False,True,False),
    ([0,0,1],1.,True,False,False),
])
def test_release_requires_goal_region_actual_support_and_correct_phase(position,force,lifted,near,expected):
    assert supported_release(position,[0,0,1],force,1.,lifted=lifted,near_release=near) is expected


def toy_attempt(path):
    path.mkdir()
    root = ET.Element("mujoco")
    ET.SubElement(root,"option",timestep=".002",gravity="0 0 -9.81")
    world=ET.SubElement(root,"worldbody")
    robot=ET.SubElement(world,"body",name="base_link",pos="0 0 1")
    ET.SubElement(robot,"geom",type="sphere",size=".02",mass="1",contype="0",conaffinity="0")
    actuator=ET.SubElement(root,"actuator")
    moving=robot
    for i, name in enumerate((*BASE,*ARMS,"r_hand_finger")):
        moving=ET.SubElement(moving,"body",name=f"segment_{i}")
        ET.SubElement(moving,"geom",type="sphere",size=".005",mass=".01",contype="0",conaffinity="0")
        ET.SubElement(moving,"joint",name=name,type="slide" if i<2 else "hinge",axis="0 0 1",armature=".1",damping="1")
        ET.SubElement(actuator,"position",name=name,joint=name,kp="10")
    ET.SubElement(robot,"site",name="r_arm_tip_tcp",pos="0 0 .1",size=".001")
    for i, name in enumerate(("r_hand_distal_link","r_hand_distal_mimic_link")):
        finger=ET.SubElement(robot,"body",name=name,pos=f"0 {i*.1} 0")
        ET.SubElement(finger,"geom",type="sphere",size=".01",mass=".01",contype="0",conaffinity="0")
    obj=ET.SubElement(world,"body",name="object",pos="2 0 1")
    ET.SubElement(obj,"freejoint",name="object_joint")
    ET.SubElement(obj,"geom",type="sphere",size=".02",mass=".1")
    xml=add_search_sensors(ET.tostring(root,encoding="unicode"),"object")
    (path/"scene.xml").write_text(xml)
    model=mujoco.MjModel.from_xml_string(xml)
    data=mujoco.MjData(model)
    data.qpos[model.joint("r_hand_finger").qposadr[0]]=2
    data.ctrl[model.actuator("r_hand_finger").id]=2
    mujoco.mj_forward(model,data)
    times=np.arange(0,.061,.01)
    with h5py.File(path/"plan.h5","w") as f:
        f["time_s"]=times; f["reference"]=np.zeros((len(times),17)); f["gripper"]=np.full(len(times),2.)
        f["object_goals"]=np.tile([2,0,1,1,0,0,0],(len(times),1))
        f["hand_goals"]=np.tile([0,0,1.1,1,0,0,0],(len(times),1))
        f["initial/qpos"]=data.qpos; f["initial/qvel"]=data.qvel; f["initial/ctrl"]=data.ctrl
    (path/"result.json").write_text(json.dumps({"plan":{"object_id":"object"},"physics":{"objects":{"object":{"joint":"object_joint"}}}}))
    return model


def test_incremental_native_search_never_tracks_by_overwriting_object(tmp_path):
    attempt=tmp_path/"seed"; model=toy_attempt(attempt)
    config=SearchConfig(samples=3,iterations=1,elite=1,knot_s=.02,horizon_increment_s=.03,threads=1)
    selected=optimize(attempt,tmp_path/"search",config)
    summary=json.loads((selected.parent/"search.json").read_text())
    assert summary["all_previous_knots_reoptimized"]
    assert not summary["physics_validated"]
    assert len(summary["samples"])==6
    address=1+int(model.joint("object_joint").qposadr[0])+2
    starts=[]
    for record in summary["samples"]:
        with np.load(selected.parent/record["path"]) as sample:
            z=sample["state"][:,address]
            # Free fall proceeds physically despite a stationary object goal.
            assert np.all(np.diff(z)<0) and z[0]<1
            starts.append(z[0])
    np.testing.assert_array_equal(starts, np.full(len(starts),starts[0]))
    with pytest.raises(FileExistsError):
        optimize(attempt,selected.parent,config)


@pytest.mark.parametrize("kwargs", [{"samples":2},{"elite":8},{"knot_s":0},{"iterations":0}])
def test_invalid_sampling_budgets_rejected(kwargs):
    with pytest.raises(ValueError):
        SearchConfig(**kwargs).validate()
