"""Geometry dispatch cannot silently substitute a different measured guard."""
import pytest
from reachy_retarget.acquisition_policy import adapter_name, from_details


@pytest.mark.parametrize('schema,key', [('cylindrical-acquisition-v1','cylindrical_acquisition'),
                                      ('box-acquisition-v1','box_acquisition')])
def test_only_explicit_geometry_policies_select_their_own_metadata(schema,key):
    metadata={'schema':schema}
    assert adapter_name(metadata)==key
    assert from_details({key:metadata})==(key,metadata)
    wrong='box_acquisition' if key=='cylindrical_acquisition' else 'cylindrical_acquisition'
    with pytest.raises(ValueError,match='disagree'):
        from_details({wrong:metadata})
    with pytest.raises(ValueError,match='Exactly one'):
        from_details({key:metadata,wrong:metadata})


def test_unknown_schema_cannot_import_an_arbitrary_module():
    for metadata in ({}, {'schema':'os'}, {'schema':'unqualified-future-policy'}):
        with pytest.raises(ValueError,match='Unknown'):
            adapter_name(metadata)
