from pathlib import Path
import argparse
import os


# We get the list of files change as an input
parser = argparse.ArgumentParser()
parser.add_argument("changed_files_in_the_pull_request", nargs="*", help="List of changed files")
args = parser.parse_args()

changed_files_in_the_pull_request = args.changed_files_in_the_pull_request
changed_files_in_the_pull_request_paths = [Path(file) for file in changed_files_in_the_pull_request]

# We assume nothing has been changed

core_changed = False
pyproject_toml_changed = False
neobaseextractor_changed = False
extractors_changed = False
plexon2_changed = False
preprocessing_changed = False
postprocessing_changed = False
qualitymetrics_changed = False
sorters_changed = False
sorters_external_changed = False
sorters_internal_changed = False
comparison_changed = False
curation_changed = False
widgets_changed = False
exporters_changed = False
sortingcomponents_changed = False
generation_changed = False
stream_extractors_changed = False


for changed_file in changed_files_in_the_pull_request_paths:

    file_is_in_src = changed_file.parts[0] == "src"

    if changed_file.name == "pyproject.toml":
        pyproject_toml_changed = True
    elif changed_file.name == "neobaseextractor.py":
        neobaseextractor_changed = True
        extractors_changed = True
    elif changed_file.name == "plexon2.py":
        plexon2_changed = True
    elif changed_file.name == "nwbextractors.py":
        extractors_changed = True  # There are NWB tests that are not streaming
        stream_extractors_changed = True
    elif changed_file.name == "iblextractors.py":
        stream_extractors_changed = True
    elif "core" in changed_file.parts:
        core_changed = True
    elif "extractors" in changed_file.parts:
        extractors_changed = True
    elif "preprocessing" in changed_file.parts:
        preprocessing_changed = True
    elif "postprocessing" in changed_file.parts:
        postprocessing_changed = True
    elif "qualitymetrics" in changed_file.parts:
        qualitymetrics_changed = True
    elif "comparison" in changed_file.parts:
        comparison_changed = True
    elif "curation" in changed_file.parts:
        curation_changed = True
    elif "widgets" in changed_file.parts:
        widgets_changed = True
    elif "exporters" in changed_file.parts:
        exporters_changed = True
    elif "sortingcomponents" in changed_file.parts:
        sortingcomponents_changed = True
    elif "generation" in changed_file.parts:
        generation_changed = True
    elif "sorters" in changed_file.parts:
        if "external" in changed_file.parts:
            sorters_external_changed = True
        elif "internal" in changed_file.parts:
            sorters_internal_changed = True
        else:
            sorters_changed = True


run_everything = core_changed or pyproject_toml_changed or neobaseextractor_changed
run_generation_tests = run_everything or generation_changed
run_extractor_tests = run_everything or extractors_changed or plexon2_changed
run_preprocessing_tests = run_everything or preprocessing_changed
run_postprocessing_tests = run_everything or postprocessing_changed
run_qualitymetrics_tests = run_everything or qualitymetrics_changed
run_curation_tests = run_everything or curation_changed
run_sortingcomponents_tests = run_everything or sortingcomponents_changed

run_comparison_test = run_everything or run_generation_tests or comparison_changed
run_widgets_test = run_everything or run_qualitymetrics_tests or run_preprocessing_tests or widgets_changed
run_exporters_test = run_everything or run_widgets_test or exporters_changed

run_sorters_test = run_everything or sorters_changed
run_internal_sorters_test = run_everything or run_sortingcomponents_tests or sorters_internal_changed

run_streaming_extractors_test = stream_extractors_changed

install_plexon_dependencies = plexon2_changed


environment_varaiables_to_add = {
    "RUN_EXTRACTORS_TESTS": run_extractor_tests,
    "RUN_PREPROCESSING_TESTS": run_preprocessing_tests,
    "RUN_POSTPROCESSING_TESTS": run_postprocessing_tests,
    "RUN_QUALITYMETRICS_TESTS": run_qualitymetrics_tests,
    "RUN_CURATION_TESTS": run_curation_tests,
    "RUN_SORTINGCOMPONENTS_TESTS": run_sortingcomponents_tests,
    "RUN_GENERATION_TESTS": run_generation_tests,
    "RUN_COMPARISON_TESTS": run_comparison_test,
    "RUN_WIDGETS_TESTS": run_widgets_test,
    "RUN_EXPORTERS_TESTS": run_exporters_test,
    "RUN_SORTERS_TESTS": run_sorters_test,
    "RUN_INTERNAL_SORTERS_TESTS": run_internal_sorters_test,
    "INSTALL_PLEXON_DEPENDENCIES": install_plexon_dependencies,
    "RUN_STREAMING_EXTRACTORS_TESTS": run_streaming_extractors_test,
}

# Write the conditions to the GITHUB_ENV file
env_file = os.getenv("GITHUB_ENV")
with open(env_file, "a") as f:
    for key, value in environment_varaiables_to_add.items():
        f.write(f"{key}={value}\n")
