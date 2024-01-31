"""
This file is for backwards compatibility with the old zarr extractor file structure.
"""

from __future__ import annotations

from pathlib import Path
from .zarrextractors import ZarrRecordingExtractor as ZarrRecordingExtractorNew


class ZarrRecordingExtractor(ZarrRecordingExtractorNew):
    def __init__(self, root_path: Path | str, storage_options: dict | None = None):
        super().__init__(folder_path=root_path, storage_options=storage_options)
