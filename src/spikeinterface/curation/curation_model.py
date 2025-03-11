from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Dict, Union, Optional
from itertools import combinations

supported_curation_format_versions = {"1"}


class LabelDefinition(BaseModel):
    name: str = Field(..., description="Name of the label")
    label_options: List[str] = Field(..., description="List of possible label options")
    exclusive: bool = Field(..., description="Whether the label is exclusive")


class ManualLabel(BaseModel):
    unit_id: Union[int, str] = Field(..., description="ID of the unit")
    labels: Dict[str, List[str]] = Field(..., description="Dictionary of labels for the unit")


class CurationModel(BaseModel):
    format_version: str = Field(..., description="Version of the curation format")
    unit_ids: List[Union[int, str]] = Field(..., description="List of unit IDs")
    label_definitions: Dict[str, LabelDefinition] = Field(..., description="Dictionary of label definitions")
    manual_labels: List[ManualLabel] = Field(..., description="List of manual labels")
    merge_unit_groups: List[List[Union[int, str]]] = Field(..., description="List of groups of units to be merged")
    removed_units: List[Union[int, str]] = Field(..., description="List of removed unit IDs")
    merge_new_unit_ids: Optional[List[Union[int, str]]] = Field(
        default=None, description="List of new unit IDs after merging"
    )

    @field_validator("format_version")
    def check_format_version(cls, v):
        if v not in supported_curation_format_versions:
            raise ValueError(f"Format version ({v}) not supported. Only {supported_curation_format_versions} are valid")
        return v

    @field_validator("label_definitions", mode="before")
    def add_label_definition_name(cls, v):
        if v is None:
            v = {}
        else:
            for key in list(v.keys()):
                v[key]["name"] = key
        return v

    @model_validator(mode="before")
    def check_manual_labels(cls, values):
        unit_ids = values["unit_ids"]
        manual_labels = values["manual_labels"]
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
                    manual_label["labels"][label] = manual_label[label]
                if unit_id not in unit_ids:
                    raise ValueError(f"Manual label unit_id {unit_id} is not in the unit list")
        return values

    @model_validator(mode="before")
    def check_merge_unit_groups(cls, values):
        unit_ids = values["unit_ids"]
        merge_unit_groups = values.get("merge_unit_groups", [])
        for merge_group in merge_unit_groups:
            for unit_id in merge_group:
                if unit_id not in unit_ids:
                    raise ValueError(f"Merge unit group unit_id {unit_id} is not in the unit list")
            if len(merge_group) < 2:
                raise ValueError("Merge unit groups must have at least 2 elements")
        return values

    @model_validator(mode="before")
    def check_merge_new_unit_ids(cls, values):
        unit_ids = values["unit_ids"]
        merge_new_unit_ids = values.get("merge_new_unit_ids")
        if merge_new_unit_ids is not None:
            merge_unit_groups = values.get("merge_unit_groups")
            assert merge_unit_groups is not None, "Merge unit groups must be defined if merge new unit ids are defined"
            if len(merge_unit_groups) != len(merge_new_unit_ids):
                raise ValueError("Merge unit groups and new unit ids must have the same length")
            if len(merge_new_unit_ids) > 0:
                for new_unit_id in merge_new_unit_ids:
                    if new_unit_id in unit_ids:
                        raise ValueError(f"New unit ID {new_unit_id} is already in the unit list")
        return values

    @model_validator(mode="before")
    def check_removed_units(cls, values):
        unit_ids = values["unit_ids"]
        removed_units = values.get("removed_units", [])
        for unit_id in removed_units:
            if unit_id not in unit_ids:
                raise ValueError(f"Removed unit_id {unit_id} is not in the unit list")
        return values

    @model_validator(mode="after")
    def validate_curation_dict(cls, values):
        labeled_unit_set = set([lbl.unit_id for lbl in values.manual_labels])
        merged_units_set = set(sum(values.merge_unit_groups, []))
        removed_units_set = set(values.removed_units)
        unit_ids = values.unit_ids

        unit_set = set(unit_ids)
        if not labeled_unit_set.issubset(unit_set):
            raise ValueError("Curation format: some labeled units are not in the unit list")
        if not merged_units_set.issubset(unit_set):
            raise ValueError("Curation format: some merged units are not in the unit list")
        if not removed_units_set.issubset(unit_set):
            raise ValueError("Curation format: some removed units are not in the unit list")

        for group in values.merge_unit_groups:
            if len(group) < 2:
                raise ValueError("Curation format: 'merge_unit_groups' must be list of list with at least 2 elements")

        all_merging_groups = [set(group) for group in values.merge_unit_groups]
        for gp_1, gp_2 in combinations(all_merging_groups, 2):
            if len(gp_1.intersection(gp_2)) != 0:
                raise ValueError("Curation format: some units belong to multiple merge groups")
        if len(removed_units_set.intersection(merged_units_set)) != 0:
            raise ValueError("Curation format: some units were merged and deleted")

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
