from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Dict, Union, Optional, Literal
from itertools import combinations, chain
import numpy as np


supported_curation_format_versions = {"1"}


class LabelDefinition(BaseModel):
    name: str = Field(..., description="Name of the label")
    label_options: List[str] = Field(..., description="List of possible label options", min_length=2)
    exclusive: bool = Field(..., description="Whether the label is exclusive")


class ManualLabel(BaseModel):
    unit_id: Union[int, str] = Field(..., description="ID of the unit")
    labels: Dict[str, List[str]] = Field(..., description="Dictionary of labels for the unit")


class Merge(BaseModel):
    merge_unit_group: List[Union[int, str]] = Field(..., description="List of groups of units to be merged")
    merge_new_unit_ids: Optional[Union[int, str]] = Field(default=None, description="New unit IDs for the merge group")


class Split(BaseModel):
    unit_id: Union[int, str] = Field(..., description="ID of the unit")
    split_mode: Literal["indices", "labels"] = Field(
        default="indices",
        description=(
            "Mode of the split. The split can be defined by indices or labels. "
            "If indices, the split is defined by the a list of lists of indices of spikes within spikes "
            "belonging to the unit (`split_indices`). "
            "If labels, the split is defined by a list of labels for each spike (`split_labels`). "
        ),
    )
    split_indices: Optional[Union[List[List[int]]]] = Field(default=None, description="List of indices for the split")
    split_labels: Optional[List[int]] = Field(default=None, description="List of labels for the split")
    split_new_unit_ids: Optional[List[Union[int, str]]] = Field(
        default=None, description="List of new unit IDs for each split"
    )


class CurationModel(BaseModel):
    format_version: str = Field(..., description="Version of the curation format")
    unit_ids: List[Union[int, str]] = Field(..., description="List of unit IDs")
    label_definitions: Optional[Dict[str, LabelDefinition]] = Field(
        default=None, description="Dictionary of label definitions"
    )
    manual_labels: Optional[List[ManualLabel]] = Field(default=None, description="List of manual labels")
    removed: Optional[List[Union[int, str]]] = Field(default=None, description="List of removed unit IDs")
    merges: Optional[List[Merge]] = Field(default=None, description="List of merges")
    splits: Optional[List[Split]] = Field(default=None, description="List of splits")

    @field_validator("format_version")
    def check_format_version(cls, v):
        if v not in supported_curation_format_versions:
            raise ValueError(f"Format version ({v}) not supported. Only {supported_curation_format_versions} are valid")
        return v

    @model_validator(mode="before")
    def add_label_definition_name(cls, values):
        label_definitions = values.get("label_definitions")
        if label_definitions is None:
            values["label_definitions"] = {}
            return values
        if isinstance(values["label_definitions"], dict):
            if label_definitions is None:
                label_definitions = {}
            else:
                for key in list(label_definitions.keys()):
                    if isinstance(label_definitions[key], dict):
                        label_definitions[key]["name"] = key
            values["label_definitions"] = label_definitions
        return values

    @model_validator(mode="before")
    def check_manual_labels(cls, values):
        unit_ids = values["unit_ids"]
        manual_labels = values.get("manual_labels")
        if manual_labels is None:
            values["manual_labels"] = []
        else:
            for manual_label in manual_labels:
                unit_id = manual_label["unit_id"]
                labels = manual_label.get("labels")
                if labels is None:
                    labels = set(manual_label.keys()) - {"unit_id"}
                    manual_label["labels"] = {}
                for label in labels:
                    if label not in values["label_definitions"]:
                        raise ValueError(f"Manual label {unit_id} has an unknown label {label}")
                    if label not in manual_label["labels"]:
                        if label in manual_label:
                            manual_label["labels"][label] = manual_label[label]
                        else:
                            raise ValueError(f"Manual label {unit_id} has no value for label {label}")
                if unit_id not in unit_ids:
                    raise ValueError(f"Manual label unit_id {unit_id} is not in the unit list")
        return values

    @model_validator(mode="before")
    def check_merges(cls, values):
        unit_ids = values["unit_ids"]
        merges = values.get("merges")
        if merges is None:
            values["merges"] = []
            return values
        elif isinstance(merges, list):
            # Convert list of lists to Merge objects
            for i, merge in enumerate(merges):
                if isinstance(merge, list):
                    merges[i] = Merge(merge_unit_group=merge)
        elif isinstance(merges, dict):
            # Convert dict format to list of Merge objects if needed
            merge_list = []
            for merge_new_id, merge_group in merges.items():
                merge_list.append({"merge_unit_group": merge_group, "merge_new_unit_ids": merge_new_id})
            merges = merge_list
            values["merges"] = merges

        # Convert dict items to Merge objects if needed
        for i, merge in enumerate(merges):
            if isinstance(merge, dict):
                merges[i] = Merge(**merge)

        for merge in merges:
            # Check unit ids exist
            for unit_id in merge.merge_unit_group:
                if unit_id not in unit_ids:
                    raise ValueError(f"Merge unit group unit_id {unit_id} is not in the unit list")

            # Check minimum group size
            if len(merge.merge_unit_group) < 2:
                raise ValueError("Merge unit groups must have at least 2 elements")

            # Check new unit id not already used
            if merge.merge_new_unit_ids is not None:
                if merge.merge_new_unit_ids in unit_ids:
                    raise ValueError(f"New unit ID {merge.merge_new_unit_ids} is already in the unit list")

        return values

    @model_validator(mode="before")
    def check_splits(cls, values):
        unit_ids = values["unit_ids"]
        splits = values.get("splits")
        if splits is None:
            values["splits"] = []
            return values

        # Convert dict format to list of Split objects if needed
        if isinstance(splits, dict):
            split_list = []
            for unit_id, split_data in splits.items():
                # If split_data is a list of indices, assume indices mode
                if isinstance(split_data[0], (list, np.ndarray)) if split_data else False:
                    split_list.append({"unit_id": unit_id, "split_mode": "indices", "split_indices": split_data})
                # Otherwise assume it's a list of labels
                else:
                    split_list.append({"unit_id": unit_id, "split_mode": "labels", "split_labels": split_data})
            splits = split_list
            values["splits"] = splits

        # Convert dict items to Split objects if needed
        for i, split in enumerate(splits):
            if isinstance(split, dict):
                splits[i] = Split(**split)

        for split in splits:
            # Check unit exists
            if split.unit_id not in unit_ids:
                raise ValueError(f"Split unit_id {split.unit_id} is not in the unit list")

            # Check split definition based on mode
            if split.split_mode == "indices":
                if split.split_indices is None:
                    raise ValueError(f"Split unit {split.unit_id} has no split_indices defined")
                if len(split.split_indices) < 1:
                    raise ValueError(f"Split unit {split.unit_id} has empty split_indices")
                # Check no duplicate indices across splits
                all_indices = list(chain.from_iterable(split.split_indices))
                if len(all_indices) != len(set(all_indices)):
                    raise ValueError(f"Split unit {split.unit_id} has duplicate indices")

            elif split.split_mode == "labels":
                if split.split_labels is None:
                    raise ValueError(f"Split unit {split.unit_id} has no split_labels defined")
                if len(split.split_labels) == 0:
                    raise ValueError(f"Split unit {split.unit_id} has empty split_labels")

            # Check new unit ids if provided
            if split.split_new_unit_ids is not None:
                if split.split_mode == "indices":
                    if len(split.split_new_unit_ids) != len(split.split_indices):
                        raise ValueError(
                            f"Number of new unit IDs does not match number of splits for unit {split.unit_id}"
                        )
                elif split.split_mode == "labels":
                    if len(split.split_new_unit_ids) != len(set(split.split_labels)):
                        raise ValueError(
                            f"Number of new unit IDs does not match number of unique labels for unit {split.unit_id}"
                        )

                # Check new ids not already used
                for new_id in split.split_new_unit_ids:
                    if new_id in unit_ids:
                        raise ValueError(f"New unit ID {new_id} is already in the unit list")

        return values

    @model_validator(mode="before")
    def check_removed(cls, values):
        unit_ids = values["unit_ids"]
        removed = values.get("removed", [])
        if removed is None:

            for unit_id in removed:
                if unit_id not in unit_ids:
                    raise ValueError(f"Removed unit_id {unit_id} is not in the unit list")

        else:
            values["removed"] = removed

        return values

    @model_validator(mode="after")
    def validate_curation_dict(cls, values):
        labeled_unit_set = set([lbl.unit_id for lbl in values.manual_labels]) if values.manual_labels else set()
        merged_units_set = (
            set(chain.from_iterable(merge.merge_unit_group for merge in values.merges)) if values.merges else set()
        )
        split_units_set = set(split.unit_id for split in values.splits) if values.splits else set()
        removed_set = set(values.removed) if values.removed else set()
        unit_ids = values.unit_ids

        unit_set = set(unit_ids)
        if not labeled_unit_set.issubset(unit_set):
            raise ValueError("Curation format: some labeled units are not in the unit list")
        if not merged_units_set.issubset(unit_set):
            raise ValueError("Curation format: some merged units are not in the unit list")
        if not split_units_set.issubset(unit_set):
            raise ValueError("Curation format: some split units are not in the unit list")
        if not removed_set.issubset(unit_set):
            raise ValueError("Curation format: some removed units are not in the unit list")

        # Check for units being merged multiple times
        all_merging_groups = [set(merge.merge_unit_group) for merge in values.merges] if values.merges else []
        for gp_1, gp_2 in combinations(all_merging_groups, 2):
            if len(gp_1.intersection(gp_2)) != 0:
                raise ValueError("Curation format: some units belong to multiple merge groups")

        # Check no overlaps between operations
        if len(removed_set.intersection(merged_units_set)) != 0:
            raise ValueError("Curation format: some units were merged and deleted")
        if len(removed_set.intersection(split_units_set)) != 0:
            raise ValueError("Curation format: some units were split and deleted")
        if len(merged_units_set.intersection(split_units_set)) != 0:
            raise ValueError("Curation format: some units were both merged and split")

        for manual_label in values.manual_labels:
            for label_key in values.label_definitions.keys():
                if label_key in manual_label.labels:
                    unit_id = manual_label.unit_id
                    label_value = manual_label.labels[label_key]
                    if not isinstance(label_value, list):
                        raise ValueError(f"Curation format: manual_labels {unit_id} is invalid should be a list")

                    is_exclusive = values.label_definitions[label_key].exclusive

                    if is_exclusive and not len(label_value) <= 1:
                        raise ValueError(
                            f"Curation format: manual_labels {unit_id} {label_key} are exclusive labels. {label_value} is invalid"
                        )

        return values
