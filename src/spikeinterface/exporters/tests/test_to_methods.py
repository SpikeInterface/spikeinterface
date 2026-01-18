"""Tests for export_to_methods function."""

from __future__ import annotations

import pytest
from pathlib import Path

from spikeinterface.exporters import export_to_methods
from spikeinterface.exporters.tests.common import make_sorting_analyzer


class TestExportToMethods:
    """Test the export_to_methods function."""

    @pytest.fixture(scope="class")
    def sorting_analyzer(self):
        """Create a sorting analyzer for testing."""
        return make_sorting_analyzer(sparse=False)

    def test_export_to_methods_markdown(self, sorting_analyzer):
        """Test markdown output format."""
        result = export_to_methods(sorting_analyzer, format="markdown")

        assert isinstance(result, str)
        assert len(result) > 0
        # Check for markdown header and prose content
        assert "## Spike Sorting Methods" in result
        assert "Extracellular recordings were acquired" in result
        assert "### References" in result

    def test_export_to_methods_latex(self, sorting_analyzer):
        """Test LaTeX output format."""
        result = export_to_methods(sorting_analyzer, format="latex")

        assert isinstance(result, str)
        assert len(result) > 0
        # Check for LaTeX sections
        assert "\\section{Spike Sorting Methods}" in result
        assert "Extracellular recordings were acquired" in result
        assert "\\subsection*{References}" in result

    def test_export_to_methods_text(self, sorting_analyzer):
        """Test plain text output format."""
        result = export_to_methods(sorting_analyzer, format="text")

        assert isinstance(result, str)
        assert len(result) > 0
        assert "SPIKE SORTING METHODS" in result
        assert "Extracellular recordings were acquired" in result

    def test_export_to_methods_invalid_format(self, sorting_analyzer):
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError, match="format must be"):
            export_to_methods(sorting_analyzer, format="invalid")

    def test_export_to_methods_invalid_detail_level(self, sorting_analyzer):
        """Test that invalid detail_level raises ValueError."""
        with pytest.raises(ValueError, match="detail_level must be"):
            export_to_methods(sorting_analyzer, detail_level="invalid")

    def test_export_to_methods_detail_levels(self, sorting_analyzer):
        """Test different detail levels produce different output lengths."""
        brief = export_to_methods(sorting_analyzer, detail_level="brief")
        standard = export_to_methods(sorting_analyzer, detail_level="standard")
        detailed = export_to_methods(sorting_analyzer, detail_level="detailed")

        # Brief should be shortest, detailed should be longest
        assert len(brief) <= len(standard)
        assert len(standard) <= len(detailed)

    def test_export_to_methods_with_citations(self, sorting_analyzer):
        """Test that citations are included when requested."""
        with_citations = export_to_methods(sorting_analyzer, include_citations=True)
        without_citations = export_to_methods(sorting_analyzer, include_citations=False)

        # With citations should be longer
        assert len(with_citations) > len(without_citations)
        # Should include SpikeInterface citation
        assert "SpikeInterface" in with_citations or "spikeinterface" in with_citations.lower()

    def test_export_to_methods_bombcell_citation(self, sorting_analyzer):
        """Test that Bombcell citation is included when quality metrics are present."""
        # The sorting_analyzer from make_sorting_analyzer has quality_metrics computed
        result = export_to_methods(sorting_analyzer, include_citations=True)

        # Should include Bombcell citation since quality_metrics is present
        assert "Bombcell" in result or "bombcell" in result.lower()
        assert "Fabre" in result  # First author of Bombcell paper

    def test_export_to_methods_contains_recording_info(self, sorting_analyzer):
        """Test that recording information is included."""
        result = export_to_methods(sorting_analyzer)

        # Should contain sampling frequency
        assert "Hz" in result
        # Should contain channel count
        assert "channels" in result.lower() or "channel" in result.lower()

    def test_export_to_methods_contains_extensions(self, sorting_analyzer):
        """Test that computed extensions are listed."""
        result = export_to_methods(sorting_analyzer)

        # The sorting_analyzer from make_sorting_analyzer has these extensions
        assert "waveforms" in result.lower() or "Waveforms" in result
        assert "templates" in result.lower() or "Templates" in result
        assert "quality" in result.lower() or "Quality" in result

    def test_export_to_methods_write_to_file(self, sorting_analyzer, tmp_path):
        """Test writing output to a file."""
        output_file = tmp_path / "methods.md"
        result = export_to_methods(sorting_analyzer, output_file=output_file)

        # File should be created
        assert output_file.exists()

        # File content should match returned string
        file_content = output_file.read_text(encoding="utf-8")
        assert file_content == result

    def test_export_to_methods_write_to_nested_path(self, sorting_analyzer, tmp_path):
        """Test writing to a nested path that doesn't exist."""
        output_file = tmp_path / "nested" / "path" / "methods.md"
        result = export_to_methods(sorting_analyzer, output_file=output_file)

        # File and parent directories should be created
        assert output_file.exists()
        assert output_file.read_text(encoding="utf-8") == result


class TestExportToMethodsWithoutSortingInfo:
    """Test export_to_methods when sorting_info is not available."""

    @pytest.fixture(scope="class")
    def sorting_analyzer_no_info(self):
        """Create a sorting analyzer without sorting_info."""
        analyzer = make_sorting_analyzer(sparse=False)
        # The sorting from generate_ground_truth_recording doesn't have sorting_info
        return analyzer

    def test_handles_missing_sorting_info(self, sorting_analyzer_no_info):
        """Test that missing sorting_info is handled gracefully."""
        result = export_to_methods(sorting_analyzer_no_info)

        assert isinstance(result, str)
        assert len(result) > 0
        # Should mention that info is not available
        assert "not available" in result.lower() or "Spike Sorting" in result


if __name__ == "__main__":
    # Quick manual test
    analyzer = make_sorting_analyzer(sparse=False)
    result = export_to_methods(analyzer, detail_level="detailed")
    print(result)
