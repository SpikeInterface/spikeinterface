import os
import pathlib

changes = {
    'CORE_CHANGED': ['pyproject.toml', '/core/', '/extractors/neoextractors/neobaseextractor.py'],
    'EXTRACTORS_CHANGED': ['/extractors/'],
    'PLEXON2_CHANGED': ['plexon2'],
    'PREPROCESSING_CHANGED': ['/preprocessing/'],
    'POSTPROCESSING_CHANGED': ['/postprocessing/'],
    'QUALITYMETRICS_CHANGED': ['/qualitymetrics/'],
    'SORTERS_CHANGED': ['/sorters/'],
    'SORTERS_EXTERNAL_CHANGED': ['/sorters/external'],
    'SORTERS_INTERNAL_CHANGED': ['/sorters/internal'],
    'COMPARISON_CHANGED': ['/comparison/'],
    'CURATION_CHANGED': ['/curation/'],
    'WIDGETS_CHANGED': ['/widgets/'],
    'EXPORTERS_CHANGED': ['/exporters/'],
    'SORTINGCOMPONENTS_CHANGED': ['/sortingcomponents/'],
    'GENERATION_CHANGED': ['/generation/'],
}

changed_files = os.getenv('CHANGED_FILES', '').split()

for change, patterns in changes.items():
    for file in changed_files:
        if any(pathlib.PurePath(file).match(pattern) for pattern in patterns):
            print(f'{change}=true')
            with open(os.getenv('GITHUB_ENV'), 'a') as env_file:
                env_file.write(f'{change}=true\n')
            break
