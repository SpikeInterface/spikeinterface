from spikeinterface.curation.curation_format import validate_curation_dict
import pytest


"""example = {
    'unit_ids': List[str, int],
    'labels_definition': {
        'category_key1':
        {'name': str,
         'labels': List[str],
         'auto_eclusive': bool}
    },
    'manual_labels': [
        {'unit_id': str or int,
         'label_category_key': str,
         'label_category_value': list or str
         }
    ],
    'merged_unit_groups': List[List[unit_ids]],  # one cell goes into at most one list
    'removed_units': List[unit_ids]  # Can not be  in the merged_units
}
"""

valid_int = {
    'unit_ids': [1, 2, 3, 6, 10, 14, 20, 31, 42],
    'labels_definition': {
        'quality':
        {'name': 'quality',
         'labels': ['good', 'noise', 'MUA', 'artifact'],
         'auto_eclusive': True},
        'experimental':
        {'name': 'experimental',
         'labels': ['acute', 'chronic', 'headfixed', 'freelymoving'],
         'auto_eclusive': False}
    },
    'manual_labels': [
        {'unit_id': 1,
         'label_category_key': 'quality',
         'label_category_value': 'good'
         },
        {'unit_id': 2,
         'label_category_key': 'quality',
         'label_category_value': 'noise'
         },
        {'unit_id': 2,
         'label_category_key': 'experimental',
         'label_category_value': ['chronic', 'headfixed']
         },
    ],
    'merged_unit_groups': [[3, 6], [10, 14, 20]],  # one cell goes into at most one list
    'removed_units': [31, 42]  # Can not be  in the merged_units
}


valid_str = {
    'unit_ids': ["u1", "u2", "u3", "u6", "u10", "u14", "u20", "u31", "u42"],
    'labels_definition': {
        'quality':
        {'name': 'quality',
         'labels': ['good', 'noise', 'MUA', 'artifact'],
         'auto_eclusive': True},
        'experimental':
        {'name': 'experimental',
         'labels': ['acute', 'chronic', 'headfixed', 'freelymoving'],
         'auto_eclusive': False}
    },
    'manual_labels': [
        {'unit_id': "u1",
         'label_category_key': 'quality',
         'label_category_value': 'good'
         },
        {'unit_id': "u2",
         'label_category_key': 'quality',
         'label_category_value': 'noise'
         },
        {'unit_id': "u2",
         'label_category_key': 'experimental',
         'label_category_value': ['chronic', 'headfixed']
         },
    ],
    'merged_unit_groups': [["u3", "u6"], ["u10", "u14", "u20"]],  # one cell goes into at most one list
    'removed_units': ["u31", "u42"]  # Can not be  in the merged_units
}

# This is a failure example
duplicate_merge = {
    'unit_ids': [1, 2, 3, 6, 10, 14, 20, 31, 42],
    'labels_definition': {
        'quality':
        {'name': 'quality',
         'labels': ['good', 'noise', 'MUA', 'artifact'],
         'auto_eclusive': True},
        'experimental':
        {'name': 'experimental',
         'labels': ['acute', 'chronic', 'headfixed', 'freelymoving'],
         'auto_eclusive': False}
    },
    'manual_labels': [
        {'unit_id': 1,
         'label_category_key': 'quality',
         'label_category_value': 'good'
         },
        {'unit_id': 2,
         'label_category_key': 'quality',
         'label_category_value': 'noise'
         },
        {'unit_id': 2,
         'label_category_key': 'experimental',
         'label_category_value': ['chronic', 'headfixed']
         },
    ],
    'merged_unit_groups': [[3, 6, 10], [10, 14, 20]],  # one cell goes into at most one list
    'removed_units': [31, 42]  # Can not be  in the merged_units
}


# This is a failure example
merged_and_removed = {
    'unit_ids': [1, 2, 3, 6, 10, 14, 20, 31, 42],
    'labels_definition': {
        'quality':
        {'name': 'quality',
         'labels': ['good', 'noise', 'MUA', 'artifact'],
         'auto_eclusive': True},
        'experimental':
        {'name': 'experimental',
         'labels': ['acute', 'chronic', 'headfixed', 'freelymoving'],
         'auto_eclusive': False}
    },
    'manual_labels': [
        {'unit_id': 1,
         'label_category_key': 'quality',
         'label_category_value': 'good'
         },
        {'unit_id': 2,
         'label_category_key': 'quality',
         'label_category_value': 'noise'
         },
        {'unit_id': 2,
         'label_category_key': 'experimental',
         'label_category_value': ['chronic', 'headfixed']
         },
    ],
    'merged_unit_groups': [[3, 6], [10, 14, 20]],  # one cell goes into at most one list
    'removed_units': [3, 31, 42]  # Can not be  in the merged_units
}


unknown_merged_unit = {
    'unit_ids': [1, 2, 3, 6, 10, 14, 20, 31, 42],
    'labels_definition': {
        'quality':
        {'name': 'quality',
         'labels': ['good', 'noise', 'MUA', 'artifact'],
         'auto_eclusive': True},
        'experimental':
        {'name': 'experimental',
         'labels': ['acute', 'chronic', 'headfixed', 'freelymoving'],
         'auto_eclusive': False}
    },
    'manual_labels': [
        {'unit_id': 1,
         'label_category_key': 'quality',
         'label_category_value': 'good'
         },
        {'unit_id': 2,
         'label_category_key': 'quality',
         'label_category_value': 'noise'
         },
        {'unit_id': 2,
         'label_category_key': 'experimental',
         'label_category_value': ['chronic', 'headfixed']
         },
    ],
    'merged_unit_groups': [[3, 6, 99], [10, 14, 20]],  # one cell goes into at most one list
    'removed_units': [31, 42]  # Can not be  in the merged_units
}


unknown_removed_unit = {
    'unit_ids': [1, 2, 3, 6, 10, 14, 20, 31, 42],
    'labels_definition': {
        'quality':
        {'name': 'quality',
         'labels': ['good', 'noise', 'MUA', 'artifact'],
         'auto_eclusive': True},
        'experimental':
        {'name': 'experimental',
         'labels': ['acute', 'chronic', 'headfixed', 'freelymoving'],
         'auto_eclusive': False}
    },
    'manual_labels': [
        {'unit_id': 1,
         'label_category_key': 'quality',
         'label_category_value': 'good'
         },
        {'unit_id': 2,
         'label_category_key': 'quality',
         'label_category_value': 'noise'
         },
        {'unit_id': 2,
         'label_category_key': 'experimental',
         'label_category_value': ['chronic', 'headfixed']
         },
    ],
    'merged_unit_groups': [[3, 6], [10, 14, 20]],  # one cell goes into at most one list
    'removed_units': [31, 42, 99]  # Can not be  in the merged_units
}


def test_curation_format_validation():
    assert validate_curation_dict(valid_int)
    assert validate_curation_dict(valid_str)
    with pytest.raises(ValueError):
        # Raised because duplicated merged units
        validate_curation_dict(duplicate_merge)
    with pytest.raises(ValueError):
        # Raised because Some units belong to multiple merge groups"
        validate_curation_dict(merged_and_removed)
    with pytest.raises(ValueError):
        # Some merged units are not in the unit list
        validate_curation_dict(unknown_merged_unit)
    with pytest.raises(ValueError):
        # Raise beecause Some removed units are not in the unit list
        validate_curation_dict(unknown_removed_unit)
