import os
import re
from pathlib import Path
import requests
import json


def get_pypi_versions(package_name):
    url = f"https://pypi.org/pypi/{package_name}/json"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    return list(sorted(data["releases"].keys()))


if __name__ == "__main__":
    package_name = "kilosort"
    versions = get_pypi_versions(package_name)
    with open(Path(os.path.realpath(__file__)).parent / "kilosort4-latest-version.json", "w") as f:
        json.dump(versions, f)
