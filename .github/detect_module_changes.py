from pathlib import Path
import argparse
import os


# We get the list of files change as an input
parser = argparse.ArgumentParser()
parser.add_argument("changed_files", nargs="*", help="List of changed files")
args = parser.parse_args()

changed_files = args.changed_files
changed_files_paths = [Path(file) for file in changed_files]

# We assume nothing has been changed
conditions_changed = {
    "CORE_CHANGED": False,
    "PYPROJECT_TOML_CHANGED": False,
    "NEOBASEEXTRACTOR_CHANGED": False,
    "EXTRACTORS_CHANGED": False,
    "PLEXON2_CHANGED": False,
    "PREPROCESSING_CHANGED": False,
    "POSTPROCESSING_CHANGED": False,
    "QUALITYMETRICS_CHANGED": False,
    "SORTERS_CHANGED": False,
    "SORTERS_EXTERNAL_CHANGED": False,
    "SORTERS_INTERNAL_CHANGED": False,
    "COMPARISON_CHANGED": False,
    "CURATION_CHANGED": False,
    "WIDGETS_CHANGED": False,
    "EXPORTERS_CHANGED": False,
    "SORTINGCOMPONENTS_CHANGED": False,
    "GENERATION_CHANGED": False,
}


for changed_file in changed_files_paths:

    file_is_in_src = changed_file.parts[0] == "src"

    if not file_is_in_src:

        if changed_file.name == "pyproject.toml":
            conditions_changed["PYPROJECT_TOML_CHANGED"] = True

    else:
        if changed_file.name == "neobaseextractor.py":
            conditions_changed["NEOBASEEXTRACTOR_CHANGED"] = True
        elif changed_file.name == "plexon2.py":
            conditions_changed["PLEXON2_CHANGED"] = True
        elif "core" in changed_file.parts:
            conditions_changed["CORE_CHANGED"] = True
        elif "extractors" in changed_file.parts:
            conditions_changed["EXTRACTORS_CHANGED"] = True
        elif "preprocessing" in changed_file.parts:
            conditions_changed["PREPROCESSING_CHANGED"] = True
        elif "postprocessing" in changed_file.parts:
            conditions_changed["POSTPROCESSING_CHANGED"] = True
        elif "qualitymetrics" in changed_file.parts:
            conditions_changed["QUALITYMETRICS_CHANGED"] = True
        elif "comparison" in changed_file.parts:
            conditions_changed["COMPARISON_CHANGED"] = True
        elif "curation" in changed_file.parts:
            conditions_changed["CURATION_CHANGED"] = True
        elif "widgets" in changed_file.parts:
            conditions_changed["WIDGETS_CHANGED"] = True
        elif "exporters" in changed_file.parts:
            conditions_changed["EXPORTERS_CHANGED"] = True
        elif "sortingcomponents" in changed_file.parts:
            conditions_changed["SORTINGCOMPONENTS_CHANGED"] = True
        elif "generation" in changed_file.parts:
            conditions_changed["GENERATION_CHANGED"] = True
        elif "sorters" in changed_file.parts:
            conditions_changed["SORTERS_CHANGED"] = True
            if "external" in changed_file.parts:
                conditions_changed["SORTERS_EXTERNAL_CHANGED"] = True
            elif "internal" in changed_file.parts:
                conditions_changed["SORTERS_INTERNAL_CHANGED"] = True


# Write the conditions to the GITHUB_ENV file
env_file = os.getenv("GITHUB_ENV")
with open(env_file, "a") as f:
    for key, value in conditions_changed.items():
        f.write(f"{key}={value}\n")


# test_to_run = {
#     "RUN_EVERYTHING": conditions_changed["CORE_CHANGED"]
#     or conditions_changed["PYPROJECT_TOML_CHANGED"]
#     or conditions_changed["NEOBASEEXTRACTOR_CHANGED"],
#     "RUN_EXTRACTOR_TEST": conditions_changed["Extractors_CHANGED"],
#     "RUN_PREPROCESSING_TEST": conditions_changed["PREPROCESSING_CHANGED"],
#     "RUN_POSTPROCESSING_TEST": conditions_changed["POSTPROCESSING_CHANGED"],
#     "RUN_QUALITYMETRICS_TEST": conditions_changed["QUALITYMETRICS_CHANGED"],
#     "RUN_SORTERS_TEST": conditions_changed["SORTERS_CHANGED"],
#     "RUN_SORTERS_EXTERNAL_TEST": conditions_changed["SORTERS_EXTERNAL_CHANGED"],
#     "RUN_SORTERS_INTERNAL_TEST": conditions_changed["SORTERS_INTERNAL_CHANGED"],
#     "RUN_COMPARISON_TEST": conditions_changed["COMPARISON_CHANGED"],
#     "RUN_CURATION_TEST": conditions_changed["CURATION_CHANGED"],
#     "RUN_WIDGETS_TEST": conditions_changed["WIDGETS_CHANGED"],
#     "RUN_EXPORTERS_TEST": conditions_changed["EXPORTERS_CHANGED"],
#     "RUN_SORTINGCOMPONENTS_TEST": conditions_changed["SORTINGCOMPONENTS_CHANGED"],
#     "RUN_GENERATION_TEST": conditions_changed["GENERATION_CHANGED"],
# }
