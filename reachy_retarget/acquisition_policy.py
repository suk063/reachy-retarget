"""Explicit geometry adapters sharing one measured acquisition clock."""
from importlib import import_module


SCHEMAS = {'cylindrical-acquisition-v1': 'cylindrical_acquisition',
           'box-acquisition-v1': 'box_acquisition'}


def adapter_name(metadata):
    schema = metadata.get('schema')
    if schema not in SCHEMAS:
        raise ValueError('Unknown measured acquisition geometry schema')
    return SCHEMAS[schema]


def adapter(metadata):
    return import_module('.'+adapter_name(metadata), __package__)


def from_details(details):
    keys = [key for key in SCHEMAS.values() if key in details]
    if len(keys) != 1:
        raise ValueError('Exactly one measured acquisition geometry policy is required')
    key = keys[0]
    metadata = details[key]
    if adapter_name(metadata) != key:
        raise ValueError('Acquisition metadata key and geometry schema disagree')
    return key, metadata
