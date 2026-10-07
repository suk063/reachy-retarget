"""Explicit global contact-model sensitivity, separate from source fidelity.

This changes virtual contact compliance, not an object's shape, inertia,
friction, initial pose or state trajectory. Results must name this assumption.
"""
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from .store import json_write


NAME='global-contact-4ms'
REFERENCE='https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters'


def prepare(prepared, output, timestep=.001):
    if timestep not in (.001,.0005):raise ValueError('Undeclared numerical profile')
    original,xml=prepared[:2];tree=ET.fromstring(xml)
    option=tree.find('option')
    if option is None:option=ET.SubElement(tree,'option')
    option.set('timestep',str(timestep))
    changes=[]
    for node in tree.findall('./worldbody//geom'):
        name=node.get('name')
        if not name:raise ValueError('Global profile requires named collision/visual geoms')
        values=original.geom_solref[original.geom(name).id].copy()
        if values[0]<=0:raise ValueError('Direct stiffness/damping contact is not covered')
        if values[0]>.004:
            changes.append(dict(kind='geom',name=name,before=values.tolist(),after=[.004,float(values[1])]))
            node.set('solref',f'.004 {values[1]:.17g}')
    for node in tree.findall('./contact/pair'):
        geoms={original.geom(node.get(k)).id for k in ('geom1','geom2')}
        indices=[i for i in range(original.npair) if {int(original.pair_geom1[i]),int(original.pair_geom2[i])}==geoms]
        if len(indices)!=1:raise ValueError('Ambiguous explicit contact pair')
        values=original.pair_solref[indices[0]].copy()
        if values[0]<=0:raise ValueError('Direct stiffness/damping pair is not covered')
        if values[0]>.004:
            changes.append(dict(kind='pair',geoms=[node.get('geom1'),node.get('geom2')],before=values.tolist(),after=[.004,float(values[1])]))
            node.set('solref',f'.004 {values[1]:.17g}')
    modified_xml=ET.tostring(tree,encoding='unicode');model=mujoco.MjModel.from_xml_string(modified_xml)
    fixed=[]
    for name in dir(original):
        value=getattr(original,name)
        if not isinstance(value,np.ndarray):continue
        if name in ('geom_solref','pair_solref'):
            expected=value.copy();expected[:,0]=np.minimum(expected[:,0],.004)
            if not np.array_equal(getattr(model,name),expected):raise ValueError('Incomplete global compliance change: '+name)
        else:
            if not np.array_equal(getattr(model,name),value):raise ValueError('Undeclared model change: '+name)
            fixed.append(name)
    report=dict(profile=NAME,timestep_s=timestep,contact_time_constant_cap_s=.004,
        changed_contacts=changes,unchanged_model_arrays=fixed,source_fidelity=False,
        physics_validated=False,reference=REFERENCE,
        meaning='Alternative global virtual contact stiffness assumption; never promoted as a pass under original source contact parameters')
    output=Path(output);output.mkdir(parents=True,exist_ok=False);json_write(output/'profile.json',report)
    result=list(prepared);result[0]=model;result[1]=modified_xml
    manifest=deepcopy(prepared[2]);manifest['simulation_profile']=report
    manifest.setdefault('simulation_assumptions',[]).append('Global positive contact time constants capped at4ms; original damping, impedance, friction and all object state/geometry/inertia preserved. Distinct from source-fidelity validation.')
    manifest['physics_timestep_s']=timestep
    manifest['physics_hz']=round(1/timestep)
    manifest['source_material_contact_policy']='Original source physical fields are preserved as provenance; the explicitly named alternative simulation profile replaces only contact time constants in the executed model'
    details=deepcopy(prepared[5]);details['simulation_profile']=report
    result[2]=manifest;result[5]=details
    return tuple(result)
