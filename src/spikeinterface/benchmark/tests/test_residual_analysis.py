import shutil
import pytest
from pathlib import Path


from spikeinterface.benchmark.tests.common_benchmark_testing import make_dataset

from spikeinterface.benchmark import analyse_residual

job_kwargs = dict(n_jobs=-1, progress_bar=True)


@pytest.mark.skip()
def test_analyse_residual():
    _, _, analyzer = make_dataset()
    if not analyzer.has_extension("amplitude_scalings"):
        analyzer.compute("amplitude_scalings", **job_kwargs)
    print(analyzer)
    residual, peaks = analyse_residual(analyzer, **job_kwargs)
    # print(residual)
    # print(peaks)


if __name__ == "__main__":
    test_analyse_residual()
