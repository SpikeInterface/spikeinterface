# AGENTS.md

## 项目定位 / Project Focus

本工作区用于围绕 SpikeInterface 开展 SiNAPS-NHP 高密度电生理数据分析，当前重点是：

- 对 1024-channel SiNAPS-NHP probe 数据进行 drift / motion correction。
- 使用 SpikeInterface 的 motion correction pipeline，尤其是 DREDge 相关 preset。
- 在 spike sorting 前，验证 probe geometry、bad channels、noise levels、peak detection/localization 和 motion estimate 是否可信。

Primary goal: help the researcher build a reproducible SiNAPS drift-correction workflow before downstream spike sorting or curation.

## 当前本地上下文 / Local Context

Important local files:

- `localise_peaks.ipynb`: 当前主要分析 notebook。
- `30Connell_0.dat`: 大型 raw binary recording，不要移动、重写或提交。
- `SiNAPS_v_1_6_kilosortChanMap.prb`: SiNAPS probe/channel map reference metadata。
- `Papers/`: 相关论文，包括 SiNAPS-NHP probe 和 DREDge motion correction 文献。
- `src/spikeinterface/`: SpikeInterface source tree。

Known recording assumptions currently used in the notebook:

- Sampling frequency: `20000.0 Hz`
- Number of channels: `1024`
- dtype: `int16`
- gain: `5.85 uV`
- offset: `150 uV`
- Probe: SiNAPS-p1024s1NHP style 4 columns x 256 rows

Critical geometry note:

- The notebook currently uses hand-written probe spacing of `29 um`.
- The `.prb` file and SiNAPS-NHP paper indicate a regular 4 x 256 array with `30 um` pitch.
- Do not silently replace one geometry with the other. Always flag this difference and ask/justify before changing geometry-dependent analysis.

## 工作原则 / Working Principles

When helping in this workspace:

- Prefer reproducible analysis over one-off notebook hacking.
- Preserve raw data and user-generated notebooks unless explicitly asked to edit them.
- Keep parameter choices visible: presets, thresholds, seeds, time ranges, selected channels, removed channels, and output folders.
- Treat motion correction as an experimental result that needs validation, not just a preprocessing checkbox.
- Use SpikeInterface APIs rather than ad hoc binary parsing where possible.
- Avoid whitening before motion estimation, because it can disrupt spatial amplitude information.
- For large recordings, test first on short time windows before launching full-session computations.

## 推荐分析流程 / Recommended Drift-Correction Workflow

A typical SiNAPS drift-correction workflow should follow this order:

1. Load recording with `BinaryRecordingExtractor`.
2. Attach and visually inspect the SiNAPS probe geometry.
3. Inspect representative traces across depth and columns.
4. Detect bad or abnormal channels.
5. Remove bad/zero-noise channels before motion estimation.
6. Apply AP-band preprocessing for motion estimation, usually bandpass filtering around `300-6000 Hz`.
7. Estimate noise levels with multiple random seeds/chunks.
8. Run a fast control motion estimate first, then DREDge-style estimate.
9. Save `motion`, `motion_info`, parameters, and diagnostic plots.
10. Only then consider interpolation or downstream sorting.

SpikeInterface high-level entry points:

```python
from spikeinterface.preprocessing import compute_motion, correct_motion
from spikeinterface.preprocessing import get_motion_parameters_preset, get_motion_presets
```

Useful comparison presets:

- `"dredge"`: official DREDge-style AP preset.
- `"dredge_fast"`: faster SpikeInterface variant.
- `"rigid_fast"`: quick control to check whether drift is present.
- `"nonrigid_accurate"`: slower comparison path.

## DARTsort Usage / 如何使用 DARTsort

DARTsort source links checked on 2026-07-08:

- Repository: `https://github.com/cwindolf/dartsort`
- Documentation: `https://dartsort.github.io/`
- Main API: `https://dartsort.github.io/main_api/`

DARTsort is a modular spike sorter written in Python/PyTorch around statistical clustering and probe motion. The upstream project currently labels it as work in progress and says it is not yet recommended for production spike sorting. In this workspace, treat DARTsort as an exploratory comparison tool for SiNAPS drift/motion behavior and sorting candidates, not as the default trusted final sorter.

Installation notes:

- Prefer a separate environment from the SpikeInterface development checkout, especially on GPU machines.
- If PyTorch already works in the environment: `pip install dartsort`.
- For visualization or tests: `pip install "dartsort[test,vis]"`.
- If setting up from scratch, follow the upstream conda-forge/mamba path and then install `dartsort`.

Input requirements:

- DARTsort expects a SpikeInterface `BaseRecording` as `recording`.
- Always attach/verify the SiNAPS probe before running it.
- Remove or document bad/zero-noise channels before running it.
- Do not pass raw `int16` SiNAPS data with default config and assume it is safe: upstream defaults to `preprocessing="none"`, and DARTsort expects standardized input.
- Either let DARTsort preprocess with `preprocessing="ibllikecmr"` / `"ibllike"`, or pass an already standardized SpikeInterface recording with `preprocessing="none"` and document exactly what preprocessing was done.

Recommended SiNAPS pilot pattern:

```python
from pathlib import Path
import dartsort

output_dir = Path("dartsort_outputs") / "pilot_dredge_only"

cfg = dartsort.DARTsortUserConfig(
    preprocessing="ibllikecmr",
    do_motion_estimation=True,
    dredge_only=True,
    work_in_tmpdir=True,
    copy_recording_to_tmpdir=False,
    device="cuda",  # use "cpu" if CUDA is not available
)

dartsort_result = dartsort.dartsort(
    recording_for_dartsort,
    output_dir,
    cfg=cfg,
    overwrite=False,
)

sorting = dartsort_result["sorting"]
motion = dartsort_result["motion"]
```

Use `dredge_only=True` for a motion/localization pilot before full sorting. After the pilot looks reasonable, rerun with `dredge_only=False` for full DARTsort sorting.

If SpikeInterface motion was already computed and validated, pass it explicitly rather than re-estimating:

```python
dartsort_result = dartsort.dartsort(
    recording_for_dartsort,
    output_dir,
    cfg=dartsort.DARTsortUserConfig(
        preprocessing="none",
        do_motion_estimation=False,
    ),
    si_motion=motion,
    overwrite=False,
)
```

Loading, exporting, and visualizing:

```python
import dartsort
import dartsort.vis as dartvis

sorting = dartsort.load(output_dir)
motion = dartsort.try_load_motion_info(output_dir)

si_sorting = sorting.to_numpy_sorting()

dartvis.visualize_sorting(
    recording_for_dartsort,
    sorting,
    Path("dartsort_outputs") / "pilot_vis",
    motion=motion,
    make_unit_summaries=False,
)
```

Expected DARTsort outputs include:

- `dartsort_sorting.npz`: final spike train arrays.
- `matching1.h5`: final matching-step features, amplitudes, localizations, and related data.
- `motion_info.pkl`: DARTsort motion information.
- `models/`: PyTorch model weights and learned modeling artifacts when produced.

DARTsort validation rules for this project:

- Start with a short time window or reduced pilot before full-session SiNAPS data.
- Compare DARTsort motion with SpikeInterface `compute_motion(..., preset="dredge")` when possible.
- Check whether DARTsort's localization/depth trends agree with the probe geometry and raw trace snippets.
- Record every non-default config value, especially `preprocessing`, `dredge_only`, `do_motion_estimation`, `device`, thresholds, and output folder.
- Use `dartsort -h` for command-line/TOML configuration discovery, but prefer Python calls in notebooks so the config is visible and reproducible.

## Motion-Correction Validation Checklist

Before trusting a correction, check:

- Probe geometry is plausible when plotted.
- Channel order matches device/channel map expectations.
- Bad channels and zero-noise channels are removed or explained.
- Noise estimates are finite and positive.
- Detected peaks are distributed across expected depth ranges.
- Peak localization does not collapse onto a few channels/depths.
- Motion estimate is smooth enough to be biologically/mechanically plausible.
- Rigid and non-rigid estimates agree on broad trends when appropriate.
- Diagnostic plots are saved with parameter labels.
- Any large displacement is checked against raw traces and known experiment events.

Useful plots include:

- Probe layout.
- Raw trace snippets across depth.
- Bad-channel labels.
- Noise level histogram by channel/depth.
- Peak activity map.
- Motion displacement over time/depth.
- Motion-corrected peak depth raster when available.

## 参数记录 / Parameter Logging

Every analysis run should record:

- Data file path and size.
- Sampling frequency, dtype, gain, offset, channel count.
- Probe geometry source: hand-written, `.prb`, or other.
- Bad-channel method and thresholds.
- Removed channel IDs.
- Filter settings.
- Noise estimation settings and seeds.
- Motion preset and overridden parameters.
- DARTsort version, config, preprocessing mode, and whether `dredge_only` was used.
- Output folder.
- Runtime notes and failures.

Do not overwrite a previous motion output folder unless the user explicitly asks. Prefer timestamped or named folders such as:

```text
motion_outputs/
  2026-07-08_dredge_30um/
  2026-07-08_dredge_fast_29um/
```

## Coding Conventions

This repository is SpikeInterface.

Follow existing project style:

- Python source lives under `src/spikeinterface/`.
- Tests live near modules in `tests/` folders.
- Use Black formatting with line length `120`.
- Prefer NumPy-style docstrings for public functions.
- Use descriptive names such as `recording`, `channel_indices`, `folder_path`.
- Avoid unnecessary abbreviations like `rec`, `idx`, or single-letter variables in production code.

For tests, prefer focused commands such as:

```bash
pytest src/spikeinterface/preprocessing/tests/test_motion.py
pytest src/spikeinterface/sortingcomponents/motion/tests
```

For broader dependency-managed runs, use the existing `uv run --extra ... --group ... pytest ...` pattern described in the development docs.

## Data Safety

Never commit or casually modify:

- `*.dat`
- generated motion folders
- large binary/cache outputs
- notebook checkpoints
- sorter output folders
- DARTsort output folders such as `dartsort_outputs/`
- DARTsort output files such as `dartsort_sorting.npz`, `matching1.h5`, `motion_info.pkl`, and `models/`
- temporary folders such as `motion_tmp/`

Before suggesting commits, check for large files and generated artifacts.

Raw data should be treated as read-only unless the user explicitly requests conversion, copying, or export.

## Literature Anchors

Relevant local papers:

- SiNAPS-NHP probe paper: high-density identified cell recordings from macaque motor cortex using 1024-channel SiNAPS-NHP probes.
- DREDge paper: robust motion correction for high-density extracellular recordings across species.

Use these papers to guide assumptions about:

- 1024-channel SiNAPS geometry.
- 4 columns x 256 rows.
- 30 um electrode pitch.
- NHP acute insertion / motor cortex context.
- Large motion and non-rigid drift expectations.

Do not overclaim biological interpretation from drift-correction outputs alone. Treat motion estimates as preprocessing diagnostics unless validated against spikes, behavior, stimulation timing, or downstream sorting quality.

## Assistant Behavior

When asked to help with this project:

- First inspect the notebook and current git status.
- Do not overwrite notebooks without showing the intended change.
- Explain parameter choices in research terms, not just software terms.
- If a result looks suspicious, propose a minimal diagnostic before a full rerun.
- Keep Chinese explanations natural, but preserve exact English API names.
- When unsure whether to prioritize speed or accuracy, default to a small-window pilot analysis first.

Default research stance:

> 先确认 probe geometry 和信号质量，再估计 drift；先小范围验证，再跑完整数据；先保存 motion evidence，再进入 sorting。
