from pydantic import BaseModel, Field, model_validator, field_validator
from typing import List, Dict, Union, Optional, Literal, Tuple
from itertools import chain, combinations
import numpy as np

from spikeinterface import BaseSorting


class LabelDefinition(BaseModel):
    name: str = Field(..., description="Name of the label")
    label_options: List[str] = Field(..., description="List of possible label options", min_length=2)
    exclusive: bool = Field(..., description="Whether the label is exclusive")


class ManualLabel(BaseModel):
    unit_id: Union[int, str] = Field(..., description="ID of the unit")
    labels: Dict[str, List[str]] = Field(..., description="Dictionary of labels for the unit")


class Merge(BaseModel):
    merge_unit_group: List[Union[int, str]] = Field(..., description="List of groups of units to be merged")
    merge_new_unit_id: Optional[Union[int, str]] = Field(default=None, description="New unit IDs for the merge group")


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

    def get_full_spike_indices(self, sorting: BaseSorting):
        """
        Get the full indices of the spikes in the split for different split modes.
        """
        num_spikes = sorting.count_num_spikes_per_unit()[self.unit_id]
        if self.split_mode == "indices":
            # check the sum of split_indices is equal to num_spikes
            num_spikes_in_split = sum(len(indices) for indices in self.split_indices)
            if num_spikes_in_split != num_spikes:
                # add remaining spike indices
                full_spike_indices = list(self.split_indices)
                existing_indices = np.concatenate(self.split_indices)
                remaining_indices = np.setdiff1d(np.arange(num_spikes), existing_indices)
                full_spike_indices.append(remaining_indices)
            else:
                full_spike_indices = self.split_indices
        elif self.split_mode == "labels":
            assert len(self.split_labels) == num_spikes, (
                f"In 'labels' mode, the number of split_labels ({len(self.split_labels)}) "
                f"must match the number of spikes in the unit ({num_spikes})"
            )
            # convert to spike indices
            full_spike_indices = []
            for label in np.unique(self.split_labels):
                label_indices = np.where(self.split_labels == label)[0]
                full_spike_indices.append(label_indices)

        return full_spike_indices


class CurationModel(BaseModel):
    supported_versions: Tuple[Literal["1"], Literal["2"]] = Field(
        default=["1", "2"], description="Supported versions of the curation format"
    )
    format_version: str = Field(..., description="Version of the curation format")
    unit_ids: List[Union[int, str]] = Field(..., description="List of unit IDs")
    label_definitions: Optional[Dict[str, LabelDefinition]] = Field(
        default=None, description="Dictionary of label definitions"
    )
    manual_labels: Optional[List[ManualLabel]] = Field(default=None, description="List of manual labels")
    removed: Optional[List[Union[int, str]]] = Field(default=None, description="List of removed unit IDs")
    merges: Optional[List[Merge]] = Field(default=None, description="List of merges")
    splits: Optional[List[Split]] = Field(default=None, description="List of splits")

    @field_validator("label_definitions", mode="before")
    def add_label_definition_name(cls, label_definitions):
        if label_definitions is None:
            return {}
        if isinstance(label_definitions, dict):
            label_definitions = dict(label_definitions)
            for key in list(label_definitions.keys()):
                if isinstance(label_definitions[key], dict):
                    label_definitions[key] = dict(label_definitions[key])
                    label_definitions[key]["name"] = key
            return label_definitions
        return label_definitions

    @classmethod
    def check_manual_labels(cls, values):
        unit_ids = list(values["unit_ids"])
        manual_labels = values.get("manual_labels")
        if manual_labels is None:
            values["manual_labels"] = []
        else:
            manual_labels = list(manual_labels)
            for i, manual_label in enumerate(manual_labels):
                manual_label = dict(manual_label)
                unit_id = manual_label["unit_id"]
                labels = manual_label.get("labels")
                if labels is None:
                    labels = set(manual_label.keys()) - {"unit_id"}
                    manual_label["labels"] = {}
                else:
                    manual_label["labels"] = {k: list(v) for k, v in labels.items()}
                for label in labels:
                    if label not in values["label_definitions"]:
                        raise ValueError(f"Manual label {unit_id} has an unknown label {label}")
                    if label not in manual_label["labels"]:
                        if label in manual_label:
                            manual_label["labels"][label] = list(manual_label[label])
                        else:
                            raise ValueError(f"Manual label {unit_id} has no value for label {label}")
                if unit_id not in unit_ids:
                    raise ValueError(f"Manual label unit_id {unit_id} is not in the unit list")
                manual_labels[i] = manual_label
            values["manual_labels"] = manual_labels
        return values

    @classmethod
    def check_merges(cls, values):
        unit_ids = list(values["unit_ids"])
        merges = values.get("merges")
        if merges is None:
            values["merges"] = []
            return values

        if isinstance(merges, dict):
            # Convert dict format to list of Merge objects
            merge_list = []
            for merge_new_id, merge_group in merges.items():
                merge_list.append({"merge_unit_group": list(merge_group), "merge_new_unit_id": merge_new_id})
            merges = merge_list

        # Make a copy of the list
        merges = list(merges)

        # Convert items to Merge objects
        for i, merge in enumerate(merges):
            if isinstance(merge, list):
                merge = {"merge_unit_group": list(merge)}
            if isinstance(merge, dict):
                merge = dict(merge)
                if "merge_unit_group" in merge:
                    merge["merge_unit_group"] = list(merge["merge_unit_group"])
                merges[i] = Merge(**merge)

        # Validate merges
        for merge in merges:
            # Check unit ids exist
            for unit_id in merge.merge_unit_group:
                if unit_id not in unit_ids:
                    raise ValueError(f"Merge unit group unit_id {unit_id} is not in the unit list")

            # Check minimum group size
            if len(merge.merge_unit_group) < 2:
                raise ValueError("Merge unit groups must have at least 2 elements")

            # Check new unit id not already used
            if merge.merge_new_unit_id is not None:
                if merge.merge_new_unit_id in unit_ids:
                    raise ValueError(f"New unit ID {merge.merge_new_unit_id} is already in the unit list")

        values["merges"] = merges
        return values

    @classmethod
    def check_splits(cls, values):
        unit_ids = list(values["unit_ids"])
        splits = values.get("splits")
        if splits is None:
            values["splits"] = []
            return values

        # Convert dict format to list format
        if isinstance(splits, dict):
            split_list = []
            for unit_id, split_data in splits.items():
                if isinstance(split_data[0], (list, np.ndarray)) if split_data else False:
                    split_list.append(
                        {
                            "unit_id": unit_id,
                            "split_mode": "indices",
                            "split_indices": [list(indices) for indices in split_data],
                        }
                    )
                else:
                    split_list.append({"unit_id": unit_id, "split_mode": "labels", "split_labels": list(split_data)})
            splits = split_list

        # Make a copy of the list
        splits = list(splits)

        # Convert items to Split objects
        for i, split in enumerate(splits):
            if isinstance(split, dict):
                split = dict(split)
                if "split_indices" in split:
                    split["split_indices"] = [list(indices) for indices in split["split_indices"]]
                if "split_labels" in split:
                    split["split_labels"] = list(split["split_labels"])
                if "split_new_unit_ids" in split:
                    split["split_new_unit_ids"] = list(split["split_new_unit_ids"])
                splits[i] = Split(**split)

        # Validate splits
        for split in splits:
            # Check unit exists
            if split.unit_id not in unit_ids:
                raise ValueError(f"Split unit_id {split.unit_id} is not in the unit list")

            # Validate based on mode
            if split.split_mode == "indices":
                if split.split_indices is None:
                    raise ValueError(f"Split unit {split.unit_id} has no split_indices defined")
                if len(split.split_indices) < 1:
                    raise ValueError(f"Split unit {split.unit_id} has empty split_indices")
                # Check no duplicate indices
                all_indices = list(chain.from_iterable(split.split_indices))
                if len(all_indices) != len(set(all_indices)):
                    raise ValueError(f"Split unit {split.unit_id} has duplicate indices")

            elif split.split_mode == "labels":
                if split.split_labels is None:
                    raise ValueError(f"Split unit {split.unit_id} has no split_labels defined")
                if len(split.split_labels) == 0:
                    raise ValueError(f"Split unit {split.unit_id} has empty split_labels")

            # Validate new unit IDs
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

                for new_id in split.split_new_unit_ids:
                    if new_id in unit_ids:
                        raise ValueError(f"New unit ID {new_id} is already in the unit list")

        values["splits"] = splits
        return values

    @classmethod
    def check_removed(cls, values):
        unit_ids = list(values["unit_ids"])
        removed = values.get("removed")
        if removed is None:
            values["removed"] = []
        else:
            removed = list(removed)
            for unit_id in removed:
                if unit_id not in unit_ids:
                    raise ValueError(f"Removed unit_id {unit_id} is not in the unit list")
            values["removed"] = removed
        return values

    @classmethod
    def convert_old_format(cls, values):
        format_version = values.get("format_version", "0")
        if format_version == "0":
            print("Conversion from format version v0 (sortingview) to v2")
            if "mergeGroups" not in values.keys():
                values["mergeGroups"] = []
            merge_groups = values["mergeGroups"]

            first_unit_id = next(iter(values["labelsByUnit"].keys()))
            if str.isdigit(first_unit_id):
                unit_id_type = int
            else:
                unit_id_type = str

            all_units = []
            all_labels = []
            manual_labels = []
            general_cat = "all_labels"
            for unit_id_, l_labels in values["labelsByUnit"].items():
                all_labels.extend(l_labels)
                unit_id = unit_id_type(unit_id_)
                if unit_id not in all_units:
                    all_units.append(unit_id)
                manual_labels.append({"unit_id": unit_id, general_cat: list(l_labels)})
            labels_def = {
                "all_labels": {"name": "all_labels", "label_options": list(set(all_labels)), "exclusive": False}
            }
            for merge_group in merge_groups:
                all_units.extend(merge_group)
            all_units = list(set(all_units))

            values = {
                "format_version": "2",
                "unit_ids": values.get("unit_ids", all_units),
                "label_definitions": labels_def,
                "manual_labels": list(manual_labels),
                "merges": [{"merge_unit_group": list(group)} for group in merge_groups],
                "splits": [],
                "removed": [],
            }
        elif values["format_version"] == "1":
            merge_unit_groups = values.get("merge_unit_groups")
            if merge_unit_groups is not None:
                values["merges"] = [{"merge_unit_group": list(group)} for group in merge_unit_groups]
            removed_units = values.get("removed_units")
            if removed_units is not None:
                values["removed"] = list(removed_units)
        return values

    @model_validator(mode="before")
    def validate_fields(cls, values):
        values = dict(values)
        values["label_definitions"] = values.get("label_definitions", {})
        values = cls.convert_old_format(values)
        values = cls.check_manual_labels(values)
        values = cls.check_merges(values)
        values = cls.check_splits(values)
        values = cls.check_removed(values)
        return values

    @model_validator(mode="after")
    def validate_curation_dict(cls, values):
        if values.format_version not in values.supported_versions:
            raise ValueError(
                f"Format version {values.format_version} not supported. Only {values.supported_versions} are valid"
            )

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
