"""Explicit friction-regularization sensitivity, never a source-model pass."""
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from .store import json_write


PROFILES = {
    'impratio10': dict(impratio=10., noslip_iterations=0),
    'impratio100': dict(impratio=100., noslip_iterations=0),
    'noslip1': dict(impratio=1., noslip_iterations=1),
    'noslip3': dict(impratio=1., noslip_iterations=3),
}


def prepare(prepared, output, profile):
    if profile not in PROFILES:
        raise ValueError('Unknown friction solver sensitivity')
    original, xml = prepared[:2]
    if original.opt.cone != int(mujoco.mjtCone.mjCONE_ELLIPTIC):
        raise ValueError('Sensitivity requires an existing elliptic friction cone')
    settings = dict(PROFILES[profile], tolerance=1e-10, solver='Newton')
    tree = ET.fromstring(xml)
    option = tree.find('option')
    if option is None:
        option = ET.SubElement(tree, 'option')
    for key, value in settings.items():
        option.set(key, str(value))
    changed = ET.tostring(tree, encoding='unicode')
    model = mujoco.MjModel.from_xml_string(changed)
    fixed = []
    for name in dir(original):
        value = getattr(original, name)
        if isinstance(value, np.ndarray):
            if not np.array_equal(getattr(model, name), value):
                raise ValueError('Undeclared physical model change: '+name)
            fixed.append(name)
    before = {key: getattr(original.opt, key) for key in settings}
    report = dict(profile=profile, settings=settings, before=before,
        unchanged_model_arrays=fixed, source_fidelity=False, physics_validated=False,
        reference='https://mujoco.readthedocs.io/en/stable/modeling.html#preventing-slip',
        meaning='Friction regularization sensitivity, not changed material friction. The exact original model remains the primary validation protocol.')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    json_write(output/'profile.json', report)
    result = list(prepared); result[0] = model; result[1] = changed
    manifest = deepcopy(prepared[2]); manifest['solver_sensitivity'] = report
    manifest.setdefault('simulation_assumptions', []).append(
        'Explicit alternative friction regularization '+profile+'; geometry, mass, inertia, material friction, solref/solimp, timestep and initial state preserved.')
    details = deepcopy(prepared[5]); details['solver_sensitivity'] = report
    result[2] = manifest; result[5] = details
    return tuple(result)
