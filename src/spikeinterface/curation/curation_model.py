from pydantic import BaseModel, Field, model_validator, field_validator
from typing import List, Dict, Union, Optional, Literal, Tuple
from itertools import chain, combinations
import numpy as np

from spikeinterface import BaseSorting


class LabelDefinition(BaseModel):
    name: str = Field(description="Name of the label")
    label_options: List[str] = Field(description="List of possible label options", min_length=2)
    exclusive: bool = Field(description="Whether the label is exclusive")


class ManualLabel(BaseModel):
    unit_id: Union[int, str] = Field(description="ID of the unit")
    labels: Dict[str, List[str]] = Field(description="Dictionary of labels for the unit")


class Merge(BaseModel):
    unit_ids: List[Union[int, str]] = Field(description="List of unit ids to be merged")
    new_unit_id: Optional[Union[int, str]] = Field(default=None, description="New unit IDs for the merge group")


class Split(BaseModel):
    unit_id: Union[int, str] = Field(description="ID of the unit")
    mode: Literal["indices", "labels"] = Field(
        default="indices",
        description=(
            "Mode of the split. The split can be defined by indices or labels. "
            "If indices, the split is defined by the a list of lists of indices of spikes within spikes "
            "belonging to the unit (`indices`). "
            "If labels, the split is defined by a list of labels for each spike (`labels`). "
        ),
    )
    indices: Optional[List[List[int]]] = Field(
        default=None,
        description=(
            "List of indices for the split. The unit is split in multiple groups (one for each list of indices), "
            "plus an optional extra if the spike train has more spikes than the sum of the indices in the lists."
        ),
    )
    labels: Optional[List[int]] = Field(default=None, description="List of labels for the split")
    new_unit_ids: Optional[List[Union[int, str]]] = Field(
        default=None, description="List of new unit IDs for each split"
    )

    def get_full_spike_indices(self, sorting: BaseSorting):
        """
        Get the full indices of the spikes in the split for different split modes.
        """
        num_spikes = sorting.count_num_spikes_per_unit()[self.unit_id]
        if self.mode == "indices":
            # check the sum of split_indices is equal to num_spikes
            num_spikes_in_split = sum(len(indices) for indices in self.indices)
            if num_spikes_in_split != num_spikes:
                # add remaining spike indices
                full_spike_indices = list(self.indices)
                existing_indices = np.concatenate(self.indices)
                remaining_indices = np.setdiff1d(np.arange(num_spikes), existing_indices)
                full_spike_indices.append(remaining_indices)
            else:
                full_spike_indices = self.indices
        elif self.mode == "labels":
            assert len(self.labels) == num_spikes, (
                f"In 'labels' mode, the number of.labels ({len(self.labels)}) "
                f"must match the number of spikes in the unit ({num_spikes})"
            )
            # convert to spike indices
            full_spike_indices = []
            for label in np.unique(self.labels):
                label_indices = np.where(self.labels == label)[0]
                full_spike_indices.append(label_indices)

        return full_spike_indices


class CurationModel(BaseModel):
    supported_versions: Tuple[Literal["1"], Literal["2"]] = Field(
        default=["1", "2"], description="Supported versions of the curation format"
    )
    format_version: str = Field(description="Version of the curation format")
    unit_ids: List[Union[int, str]] = Field(description="List of unit IDs")
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
        """
        Checks and validates the manual labels in the curation model.

          * Checks if the unit_ids in each manual label exist in the unit_ids list.
          * Validates that each label in the manual labels exists in the label_definitions.

        """
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
        """
        Checks and validates the merges in the curation model.

          * Checks if the unit_ids in each merge group exist in the unit_ids list.
          * Validates that each merge group has at least two unit IDs.
          * Ensures that any new_unit_id provided does not already exist in the unit_ids list.
          * Converts merges from dict format to list of Merge objects if necessary.

        """
        unit_ids = list(values["unit_ids"])
        merges = values.get("merges")
        if merges is None:
            values["merges"] = []
            return values

        if isinstance(merges, dict):
            # Convert dict format to list of Merge objects
            merge_list = []
            for merge_new_id, merge_group in merges.items():
                merge_list.append({"unit_ids": list(merge_group), "new_unit_id": merge_new_id})
            merges = merge_list

        # Make a copy of the list
        merges = list(merges)

        # Convert items to Merge objects
        for i, merge in enumerate(merges):
            if isinstance(merge, list):
                merge = {"unit_ids": list(merge)}
            if isinstance(merge, dict):
                merge = dict(merge)
                if "unit_ids" in merge:
                    merge["unit_ids"] = list(merge["unit_ids"])
                merges[i] = Merge(**merge)

        # Validate merges
        for merge in merges:
            # Check unit ids exist
            for unit_id in merge.unit_ids:
                if unit_id not in unit_ids:
                    raise ValueError(f"Merge unit group unit_id {unit_id} is not in the unit list")

            # Check minimum group size
            if len(merge.unit_ids) < 2:
                raise ValueError("Merge unit groups must have at least 2 elements")

            # Check new unit id not already used
            if merge.new_unit_id is not None:
                if merge.new_unit_id in unit_ids:
                    raise ValueError(f"New unit ID {merge.new_unit_id} is already in the unit list")

        values["merges"] = merges
        return values

    @classmethod
    def check_splits(cls, values):
        """
        Checks and validates the splits in the curation model.

          * Checks if the unit_id exists in the unit_ids list.
          * Validates the mode (indices or labels).
          * If mode is indices, checks that indices are defined and not empty, and that there are no duplicate indices.
          * If mode is labels, checks that labels are defined and not empty.
          * | Validates new unit IDs if provided, ensuring they are not already in the unit_ids list and match the
            | number of splits.

        """
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
                            "mode": "indices",
                            "indices": [list(indices) for indices in split_data],
                        }
                    )
                else:
                    split_list.append({"unit_id": unit_id, "mode": "labels", "labels": list(split_data)})
            splits = split_list

        # Make a copy of the list
        splits = list(splits)

        # Convert items to Split objects
        for i, split in enumerate(splits):
            if isinstance(split, dict):
                split = dict(split)
                if "indices" in split:
                    split["indices"] = [list(indices) for indices in split["indices"]]
                if "labels" in split:
                    split["labels"] = list(split["labels"])
                if "new_unit_ids" in split:
                    split["new_unit_ids"] = list(split["new_unit_ids"])
                splits[i] = Split(**split)

        # Validate splits
        for split in splits:
            # Check unit exists
            if split.unit_id not in unit_ids:
                raise ValueError(f"Split unit_id {split.unit_id} is not in the unit list")

            # Validate based on mode
            if split.mode == "indices":
                if split.indices is None:
                    raise ValueError(f"Split unit {split.unit_id} has no indices defined")
                if len(split.indices) < 1:
                    raise ValueError(f"Split unit {split.unit_id} has empty indices")
                # Check no duplicate indices
                all_indices = list(chain.from_iterable(split.indices))
                if len(all_indices) != len(set(all_indices)):
                    raise ValueError(f"Split unit {split.unit_id} has duplicate indices")

            elif split.mode == "labels":
                if split.labels is None:
                    raise ValueError(f"Split unit {split.unit_id} has no labels defined")
                if len(split.labels) == 0:
                    raise ValueError(f"Split unit {split.unit_id} has empty labels")

            # Validate new unit IDs
            if split.new_unit_ids is not None:
                if split.mode == "indices":
                    if (
                        len(split.new_unit_ids) != len(split.indices)
                        and len(split.new_unit_ids) != len(split.indices) + 1
                    ):
                        raise ValueError(
                            f"Number of new unit IDs does not match number of splits for unit {split.unit_id}"
                        )
                elif split.mode == "labels":
                    if len(split.new_unit_ids) != len(set(split.labels)):
                        raise ValueError(
                            f"Number of new unit IDs does not match number of unique labels for unit {split.unit_id}"
                        )

                for new_id in split.new_unit_ids:
                    if new_id in unit_ids:
                        raise ValueError(f"New unit ID {new_id} is already in the unit list")

        values["splits"] = splits
        return values

    @classmethod
    def check_removed(cls, values):
        """
        Checks and validates the removed units in the curation model.
        If `removed` is None, it initializes it as an empty list.
        It then checks that each unit ID in `removed` exists in the `unit_ids` list.
        """
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
        """
        Converts old curation formats (v0 and v1) to the current format (v2).
        v0 (sortingview) format is converted to v2 by extracting labels, merges, and unit IDs.
        v1 format is updated to v2 by renaming fields and ensuring the structure matches the v2 format.
        """
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
                "merges": [{"unit_ids": list(group)} for group in merge_groups],
                "splits": [],
                "removed": [],
            }
        elif values["format_version"] == "1":
            merge_unit_groups = values.get("merge_unit_groups")
            if merge_unit_groups is not None:
                values["merges"] = [{"unit_ids": list(group)} for group in merge_unit_groups]
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
    def validate_curation_dict(self):
        if self.format_version not in self.supported_versions:
            raise ValueError(
                f"Format version {self.format_version} not supported. Only {self.supported_versions} are valid"
            )

        labeled_unit_set = set([lbl.unit_id for lbl in self.manual_labels]) if self.manual_labels else set()
        merged_units_set = set(chain.from_iterable(merge.unit_ids for merge in self.merges)) if self.merges else set()
        split_units_set = set(split.unit_id for split in self.splits) if self.splits else set()
        removed_set = set(self.removed) if self.removed else set()
        unit_ids = self.unit_ids

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
        all_merging_groups = [set(merge.unit_ids) for merge in self.merges] if self.merges else []
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

        for manual_label in self.manual_labels:
            for label_key in self.label_definitions.keys():
                if label_key in manual_label.labels:
                    unit_id = manual_label.unit_id
                    label_value = manual_label.labels[label_key]
                    if not isinstance(label_value, list):
                        raise ValueError(f"Curation format: manual_labels {unit_id} is invalid should be a list")

                    is_exclusive = self.label_definitions[label_key].exclusive

                    if is_exclusive and not len(label_value) <= 1:
                        raise ValueError(
                            f"Curation format: manual_labels {unit_id} {label_key} are exclusive labels. {label_value} is invalid"
                        )

        return self
