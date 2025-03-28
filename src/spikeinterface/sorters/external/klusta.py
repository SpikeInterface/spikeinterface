from __future__ import annotations

from pathlib import Path
import sys
import shutil
import importlib.util
from importlib.metadata import version

from spikeinterface.sorters.basesorter import BaseSorter, get_job_kwargs
from spikeinterface.sorters.utils import ShellScript

from probeinterface import write_prb

from spikeinterface.core import write_binary_recording
from spikeinterface.extractors import KlustaSortingExtractor


class KlustaSorter(BaseSorter):
    """Klusta Sorter object."""

    sorter_name = "klusta"

    requires_locations = False
    requires_binary_data = True

    _default_params = {
        "adjacency_radius": None,
        "threshold_strong_std_factor": 5,
        "threshold_weak_std_factor": 2,
        "detect_sign": -1,
        "extract_s_before": 16,
        "extract_s_after": 32,
        "n_features_per_channel": 3,
        "pca_n_waveforms_max": 10000,
        "num_starting_clusters": 50,
    }

    _params_description = {
        "adjacency_radius": "Radius in um to build channel neighborhood ",
        "threshold_strong_std_factor": "Strong threshold for spike detection",
        "threshold_weak_std_factor": "Weak threshold for spike detection",
        "detect_sign": "Use -1 (negative), 1 (positive) or 0 (both) depending "
        "on the sign of the spikes in the recording",
        "extract_s_before": "Number of samples to cut out before the peak",
        "extract_s_after": "Number of samples to cut out after the peak",
        "n_features_per_channel": "Number of PCA features per channel",
        "pca_n_waveforms_max": "Maximum number of waveforms for PCA",
        "num_starting_clusters": "Number of initial clusters",
    }

    sorter_description = """Klusta is a density-based spike sorter that uses a masked EM approach for clustering.
    For more information see https://doi.org/10.1038/nn.4268"""

    installation_mesg = """\nTo use Klusta run:\n
       >>> pip install Cython h5py tqdm
       >>> pip install click klusta klustakwik2

    More information on klusta at:
      * https://github.com/kwikteam/phy
      * https://github.com/kwikteam/klusta
    """

    @classmethod
    def is_installed(cls):
        klusta_spec = importlib.util.find_spec("klusta")
        klustakwik2_spec = importlib.util.find_spec("klustakwik2")
        if klusta_spec is not None and klustakwik2_spec is not None:
            HAVE_KLUSTA = True
        else:
            HAVE_KLUSTA = False
        return HAVE_KLUSTA

    @classmethod
    def get_sorter_version(cls):
        return version("klusta")

    @classmethod
    def _setup_recording(cls, recording, sorter_output_folder, params, verbose):
        source_dir = Path(__file__).parent

        # alias to params
        p = params

        experiment_name = sorter_output_folder / "recording"

        # save prb file
        prb_file = sorter_output_folder / "probe.prb"
        probegroup = recording.get_probegroup()
        write_prb(prb_file, probegroup, radius=p["adjacency_radius"])

        # source file
        if recording.binary_compatible_with(time_axis=0, file_offset=0):
            # no copy
            d = recording.get_binary_description()
            raw_filename = str(d["file_paths"][0])
            dtype = str(d["dtype"])
            if not recording.binary_compatible_with(file_suffix=".dat"):
                # copy and change suffix
                print("Binary file is not a .dat file. Making a copy!")
                shutil.copy(raw_filename, sorter_output_folder / "recording.dat")
                raw_filename = sorter_output_folder / "recording.dat"
        else:
            # save binary file (chunk by chunk) into a new file
            raw_filename = sorter_output_folder / "recording.dat"
            dtype = "int16"
            write_binary_recording(recording, file_paths=[raw_filename], dtype=dtype, **get_job_kwargs(params, verbose))

        if p["detect_sign"] < 0:
            detect_sign = "negative"
        elif p["detect_sign"] > 0:
            detect_sign = "positive"
        else:
            detect_sign = "both"

        # set up klusta config file
        with (source_dir / "klusta_config_default.prm").open("r") as f:
            klusta_config = f.readlines()

        # Note: should use format with dict approach here
        klusta_config = "".join(klusta_config).format(
            experiment_name,
            prb_file,
            raw_filename,
            float(recording.get_sampling_frequency()),
            recording.get_num_channels(),
            "'{}'".format(dtype),
            p["threshold_strong_std_factor"],
            p["threshold_weak_std_factor"],
            "'" + detect_sign + "'",
            p["extract_s_before"],
            p["extract_s_after"],
            p["n_features_per_channel"],
            p["pca_n_waveforms_max"],
            p["num_starting_clusters"],
        )

        with (sorter_output_folder / "config.prm").open("w") as f:
            f.writelines(klusta_config)

    @classmethod
    def _run_from_folder(cls, sorter_output_folder, params, verbose):
        if "win" in sys.platform and sys.platform != "darwin":
            shell_cmd = """
                        klusta --overwrite {klusta_config}
                    """.format(
                klusta_config=sorter_output_folder / "config.prm"
            )
        else:
            shell_cmd = """
                        #!/bin/bash
                        klusta {klusta_config} --overwrite
                    """.format(
                klusta_config=sorter_output_folder / "config.prm"
            )

        shell_script = ShellScript(
            shell_cmd,
            script_path=sorter_output_folder / f"run_{cls.sorter_name}",
            log_path=sorter_output_folder / f"{cls.sorter_name}.log",
            verbose=verbose,
        )
        shell_script.start()

        retcode = shell_script.wait()

        if retcode != 0:
            raise Exception("klusta returned a non-zero exit code")

        if not (sorter_output_folder / "recording.kwik").is_file():
            raise Exception("Klusta did not run successfully")

    @classmethod
    def _get_result_from_folder(cls, sorter_output_folder):
        sorting = KlustaSortingExtractor(file_or_folder_path=Path(sorter_output_folder) / "recording.kwik")
        return sorting
