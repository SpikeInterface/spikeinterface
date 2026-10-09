"""Standalone RMS pattern-motion analysis and public DANDI drift experiment.

Run from the repository root:
    python playing/rms_drift_experiment.py --stage all
    python playing/rms_drift_experiment.py --stage analyse

Dependencies: this SpikeInterface checkout, numpy, scipy, matplotlib, h5py,
remfile, and requests. No spike sorting, resampling, or motion correction is
performed. The fixed public source is DANDI 000957 / ZYE-0057. Its motion gates
occur at 2108.9--2625.5 s, so the default excerpt is 1920--2820 s (32--47 min).
The first quarter of the original recording does not contain imposed motion.

The excerpt retains all sample values and all 384 channels. Compressed HDF5
byte ranges are downloaded with an ETag guard; gzip payloads are checked while
decompressing. The manifest records the source identity and the excerpt's own
SHA-256 (which is distinct from the published full-NWB checksum).

The NP Ultra geometry is eight columns by 48 rows. RMS is computed using all
channels, and the same +/-3 soft-matching rule is applied within each column.
The session estimate averages all columnwise displacement estimates. Cached
RMS and figures are written beside the excerpt; --sigma-r adjusts matching
tolerance without rereading the raw data.
"""

import numpy as np

SOURCE = {
    "dandiset": "000957", "version": "0.240731.0249",
    "asset_id": "2918120f-dc7f-436d-93d9-e017525fe86e",
    "url": "https://dandiarchive.s3.amazonaws.com/blobs/73a/dc8/73adc836-e4bf-456e-90b5-75c31d9b84b8",
    "path": "sub-ZYE-0057/sub-ZYE-0057_ecephys+image.nwb",
    "full_asset_sha256": "4f97295342902a089c426c7ed05ed0fa2a06910b3f5ae8f7527960ee9f54a333",
    "series": "acquisition/ElectricalSeriesAP",
    "protocol_url": "https://dandiarchive.org/dandiset/000957/0.240731.0249",
}


def compute_columnwise_rms_motion(rms, times, locations, sigma_r=None):
    """Match vertically within each column, then average matching depth rows."""
    x, y = np.asarray(locations).T
    columns = [np.flatnonzero(x == value) for value in np.unique(x)]
    columns = [indices[np.argsort(-y[indices], kind="stable")] for indices in columns]
    if any(not np.array_equal(y[indices], y[columns[0]]) for indices in columns):
        raise ValueError("Columnwise RMS averaging needs columns with matching depths.")
    if sigma_r is None:
        sigma_r = max(float(np.median(np.abs(np.diff(rms, axis=0)))), np.finfo(float).eps)
    estimates = [compute_rms_motion(rms[:, indices], times, sigma_r=sigma_r) for indices in columns]
    motion = estimates[0].copy()
    weights = np.mean([item["weights"] for item in estimates], axis=0)
    displacement = np.sum(weights * motion["offsets"], axis=2)
    motion.update({
        "weights": weights, "displacement_channels": displacement,
        "velocity_channels_per_second": displacement / np.diff(times)[:, None],
        "offset_spread_channels": np.sqrt(np.sum(weights * (motion["offsets"] - displacement[:, :, None])**2, axis=2)),
        "session_velocity_channels_per_second": displacement.mean(axis=1) / np.diff(times),
        "cumulative_displacement_channels": np.r_[0.0, np.cumsum(displacement.mean(axis=1))],
        "column_channel_indices": np.stack(columns),
        "row_depths_um": y[columns[0]],
        "displacements_by_column": np.stack([item["displacement_channels"] for item in estimates]),
    })
    return np.mean([rms[:, indices] for indices in columns], axis=0), motion


def compute_rms_motion(rms, times, sigma_r=None, radius=3):
    """Soft matches to the next chunk, in channels ordered from top to bottom.

    Weights are exp(-RMS_difference**2 / (2*sigma_r**2)) / (1 + offset**2).
    Positive displacement is toward the bottom of the probe. Symmetric
    neighborhoods shrink at the ends to avoid an artificial inward bias.
    This describes RMS pattern motion; it is not a calibrated tissue displacement.
    """
    rms = np.asarray(rms, dtype=float)
    times = np.asarray(times, dtype=float)
    if rms.ndim != 2 or len(times) != len(rms) or len(times) < 2 or rms.shape[1] < 1:
        raise ValueError("RMS must have shape (at least 2 times, channels), with matching times.")
    if not np.all(np.isfinite(rms)) or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("RMS and times must be finite, with strictly increasing times.")
    if not isinstance(radius, (int, np.integer)) or radius < 0:
        raise ValueError("radius must be a nonnegative integer.")
    if sigma_r is None:
        # Typical temporal RMS change, shared by all channels in this segment.
        sigma_r = max(float(np.median(np.abs(np.diff(rms, axis=0)))),
                      np.finfo(float).eps * max(1.0, float(np.max(np.abs(rms)))))
    if not np.isfinite(sigma_r) or sigma_r <= 0:
        raise ValueError("sigma_r must be finite and positive.")

    offsets = np.arange(-radius, radius + 1)
    channels = np.arange(rms.shape[1])
    local_radius = np.minimum(radius, np.minimum(channels, rms.shape[1] - 1 - channels))
    valid = np.abs(offsets)[None, :] <= local_radius[:, None]
    candidates = np.clip(channels[:, None] + offsets, 0, rms.shape[1] - 1)
    differences = rms[1:, candidates] - rms[:-1, :, None]
    log_weights = -0.5 * (differences / sigma_r) ** 2 - np.log1p(offsets ** 2)
    log_weights = np.where(valid[None, :, :], log_weights, -np.inf)
    log_weights -= np.max(log_weights, axis=2, keepdims=True)
    weights = np.exp(log_weights)
    weights /= weights.sum(axis=2, keepdims=True)

    displacement = np.sum(weights * offsets, axis=2)
    spread = np.sqrt(np.sum(weights * (offsets - displacement[:, :, None]) ** 2, axis=2))
    dt = np.diff(times)
    mean_displacement = displacement.mean(axis=1)
    return {
        "sigma_r": float(sigma_r), "offsets": offsets, "weights": weights,
        "displacement_channels": displacement,
        "velocity_channels_per_second": displacement / dt[:, None],
        "offset_spread_channels": spread,
        "transition_times": (times[:-1] + times[1:]) / 2,
        "session_velocity_channels_per_second": mean_displacement / dt,
        "cumulative_displacement_channels": np.r_[0.0, np.cumsum(mean_displacement)],
    }


def plot_rms_motion(rms, times, motion, title="RMS pattern motion"):
    """Full heatmap, local vector detail, mean velocity, and integrated drift."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(17, 10), constrained_layout=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    edges = np.r_[times[0] - (times[1] - times[0]) / 2,
                  (times[:-1] + times[1:]) / 2,
                  times[-1] + (times[-1] - times[-2]) / 2]
    channel_edges = np.arange(rms.shape[1] + 1) - 0.5
    vmin, vmax = np.percentile(rms, [1, 99])
    if vmin == vmax:
        vmax = vmin + 1

    def draw(ax, channel_indices, transition_indices):
        heatmap = ax.pcolormesh(edges, channel_edges, rms.T, cmap="magma",
                               vmin=vmin, vmax=vmax, rasterized=True)
        x, y = np.meshgrid(times[transition_indices], channel_indices)
        u = np.broadcast_to(np.diff(times)[transition_indices], x.shape)
        v = motion["displacement_channels"][np.ix_(transition_indices, channel_indices)].T
        ax.quiver(x, y, u, v, angles="xy", scale_units="xy", scale=1,
                  color="cyan", alpha=0.8, width=0.0014, headwidth=3, headlength=4)
        ax.set(xlabel="Time (s)", ylabel="Channel position (top to bottom)")
        ax.set_ylim(rms.shape[1] - 0.5, -0.5)
        return heatmap

    stride = max(1, int(np.ceil(rms.shape[1] / 48)))
    heatmap = draw(axes[0, 0], np.arange(0, rms.shape[1], stride), np.arange(len(times) - 1))
    axes[0, 0].set_title(f"RMS + vectors at every time step (every {stride} channel(s))")
    n_channels = min(24, rms.shape[1])
    n_steps = min(16, len(times) - 1)
    c0 = (rms.shape[1] - n_channels) // 2
    t0 = (len(times) - 1 - n_steps) // 2
    draw(axes[0, 1], np.arange(c0, c0 + n_channels), np.arange(t0, t0 + n_steps))
    axes[0, 1].set(xlim=(edges[t0], edges[t0 + n_steps + 1]),
                   ylim=(c0 + n_channels - 0.5, c0 - 0.5),
                   title="Central detail: every channel and time step")
    fig.colorbar(heatmap, ax=list(axes[0]), label="RMS (µV), colour limits: 1st–99th percentile",
                 extend="both", shrink=0.85)

    axes[1, 0].plot(motion["transition_times"], motion["session_velocity_channels_per_second"])
    axes[1, 0].axhline(0, color="grey", lw=0.8)
    axes[1, 0].set(xlabel="Time (s)", ylabel="Channels / s", title="Session drift: mean over channels")
    axes[1, 1].plot(times, motion["cumulative_displacement_channels"])
    axes[1, 1].axhline(0, color="grey", lw=0.8)
    axes[1, 1].set(xlabel="Time (s)", ylabel="Channel positions", title="Cumulative mean displacement")
    fig.suptitle(f"{title}\nPositive = toward probe bottom; sigma_R = {motion['sigma_r']:.3g} µV")
    return fig


def _write_json(path, value):
    import json
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def download_excerpt(output, start_seconds=1920, end_seconds=2820, workers=6):
    """Fetch only relevant gzip HDF5 chunks; preserve all 384 channels.

    The source uses gzip only (no shuffle/checksum filters). Each compressed
    chunk is decompressed and assembled using its HDF5 byte offset. Completed
    time blocks are checkpointed, so an interrupted download can resume.
    """
    import hashlib
    import json
    import time
    import zlib
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path
    import h5py
    import remfile
    import requests

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "motion_excerpt.dat"
    manifest_path = output / "motion_excerpt.json"
    state_path = output / "download_progress.json"
    part = output / "motion_excerpt.dat.part"
    if target.exists():
        info = json.loads(manifest_path.read_text())
        if (info["start_seconds"] != start_seconds or info["end_seconds"] != end_seconds
                or target.stat().st_size != info["size_bytes"]):
            raise ValueError("Existing excerpt has different bounds or size; use a new output directory.")
        return target, info

    header = requests.head(SOURCE["url"], timeout=60)
    header.raise_for_status()
    etag = header.headers["ETag"]
    source_size = int(header.headers["Content-Length"])
    print("Reading source geometry and compressed-chunk index...", flush=True)
    with h5py.File(remfile.File(SOURCE["url"]), "r") as f:
        series = f[SOURCE["series"]]
        data = series["data"]
        if data.compression != "gzip" or data.shuffle or data.fletcher32 or data.scaleoffset is not None:
            raise ValueError("This downloader requires gzip-only HDF5 chunks.")
        rate = float(series["starting_time"].attrs["rate"])
        first, stop = round(start_seconds * rate), round(end_seconds * rate)
        if not 0 <= first < stop <= data.shape[0]:
            raise ValueError("Requested interval is outside the recording.")
        rows, columns = data.chunks
        n_channels = data.shape[1]
        if n_channels % columns:
            raise ValueError("Channel count must divide exactly into stored channel blocks.")
        electrodes = series["electrodes"][:]
        table = f["general/extracellular_ephys/electrodes"]
        conversion = float(data.attrs["conversion"])
        channel_conversion = series["channel_conversion"][:] if "channel_conversion" in series else np.ones(n_channels)
        info = dict(SOURCE, start_seconds=start_seconds, end_seconds=end_seconds,
                    sampling_frequency_hz=rate, start_frame=first, end_frame=stop,
                    num_frames=stop-first, num_channels=n_channels, dtype=data.dtype.str,
                    gain_to_uV=(channel_conversion * conversion * 1e6).tolist(),
                    offset_uV=float(data.attrs.get("offset", 0)) * 1e6,
                    x_um=table["rel_x"][:][electrodes].tolist(),
                    y_um=table["rel_y"][:][electrodes].tolist(),
                    electrode_rows=electrodes.tolist(),
                    source_etag=etag, source_size_bytes=source_size,
                    size_bytes=(stop-first)*n_channels*data.dtype.itemsize)
        blocks = []
        for row in range(first // rows * rows, stop, rows):
            chunks = []
            for col in range(0, n_channels, columns):
                entry = data.id.get_chunk_info_by_coord((row, col))
                if entry.filter_mask != 0 or entry.byte_offset is None:
                    raise ValueError("Unexpected unfiltered or missing source chunk.")
                chunks.append({"column": col, "offset": entry.byte_offset, "size": entry.size})
            blocks.append({"row": row, "chunks": chunks})
    info["hdf5_chunk_shape"] = [rows, columns]
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["info"] != info or not part.exists() or part.stat().st_size != info["size_bytes"]:
            raise ValueError("Partial download metadata does not match this request.")
    else:
        if part.exists():
            raise ValueError("Untracked partial excerpt exists; choose a new output directory.")
        with part.open("wb") as stream:
            stream.truncate(info["size_bytes"])
        state = {"info": info, "completed": {}}
        _write_json(state_path, state)

    # Revalidate completed blocks before trusting a resumed download.
    with part.open("rb") as stream:
        for key, digest in list(state["completed"].items()):
            block = blocks[int(key)]
            a, b = max(first, block["row"]), min(stop, block["row"] + rows)
            stream.seek((a-first)*n_channels*2)
            if hashlib.sha256(stream.read((b-a)*n_channels*2)).hexdigest() != digest:
                del state["completed"][key]

    def fetch(index):
        block = blocks[index]
        lo = min(c["offset"] for c in block["chunks"])
        hi = max(c["offset"]+c["size"] for c in block["chunks"])
        for attempt in range(4):
            try:
                response = requests.get(SOURCE["url"], headers={
                    "Range": f"bytes={lo}-{hi-1}", "If-Match": etag,
                }, timeout=(30, 180))
                response.raise_for_status()
                if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {lo}-{hi-1}/{source_size}":
                    raise ValueError("Server did not return the exact requested range.")
                payload = response.content
                if len(payload) != hi-lo:
                    raise ValueError("Truncated compressed source block.")
                values = np.empty((rows, n_channels), dtype=info["dtype"])
                for chunk in block["chunks"]:
                    start = chunk["offset"] - lo
                    decoded = zlib.decompress(payload[start:start+chunk["size"]])
                    values[:, chunk["column"]:chunk["column"]+columns] = np.frombuffer(
                        decoded, dtype=info["dtype"]).reshape(rows, columns)
                a, b = max(first, block["row"]), min(stop, block["row"]+rows)
                cropped = values[a-block["row"]:b-block["row"]]
                with part.open("r+b") as stream:
                    stream.seek((a-first)*n_channels*values.dtype.itemsize)
                    stream.write(memoryview(cropped).cast("B"))
                    stream.flush()
                return index, hashlib.sha256(cropped).hexdigest()
            except (requests.RequestException, ValueError, zlib.error):
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)

    pending = [i for i in range(len(blocks)) if str(i) not in state["completed"]]
    print(f"Downloading {len(pending)}/{len(blocks)} time blocks; excerpt {info['size_bytes']/1e9:.2f} GB", flush=True)
    last_print = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(fetch, i) for i in pending]):
            index, digest = future.result()
            state["completed"][str(index)] = digest
            _write_json(state_path, state)
            if time.monotonic()-last_print > 10 or len(state["completed"]) == len(blocks):
                print(f"Downloaded {len(state['completed'])}/{len(blocks)} blocks", flush=True)
                last_print = time.monotonic()
    print("Hashing completed excerpt...", flush=True)
    digest = hashlib.sha256()
    with part.open("rb") as stream:
        for block in iter(lambda: stream.read(16*1024**2), b""):
            digest.update(block)
    info["excerpt_sha256"] = digest.hexdigest()
    _write_json(manifest_path, info)
    part.replace(target)
    return target, info


def load_manipulator_summary(output):
    """Extract the 5 V motion-gate pulses, preserving original recording times."""
    from pathlib import Path
    path = Path(output) / "manipulator_trigger_summary.npz"
    if not path.exists():
        import h5py
        import remfile
        with h5py.File(remfile.File(SOURCE["url"]), "r") as f:
            group = f["acquisition/manipulator_trigger"]
            values, times = group["data"][:], group["timestamps"][:]
        indices = np.flatnonzero(values > 1)
        starts = indices[np.r_[True, np.diff(indices) > 1]]
        ends = indices[np.r_[np.diff(indices) > 1, True]]
        n = len(values) // 100
        np.savez(path, times=times[:n*100].reshape(n, 100).mean(axis=1),
                 mean=values[:n*100].reshape(n, 100).mean(axis=1),
                 minimum=values[:n*100].reshape(n, 100).min(axis=1),
                 maximum=values[:n*100].reshape(n, 100).max(axis=1),
                 pulse_start_times=times[starts], pulse_end_times=times[ends])
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def analyse_excerpt(output, chunk_seconds=3.0, workers=2, sigma_r=None):
    """Run the original RMS estimator separately down each physical column.

    Preprocessing matches trying-new-metrics.py: 300 Hz highpass, global median
    reference, then demeaned RMS. The final chunk is dropped as before. Outputs
    use original-session timestamps and distinguish inferred pattern motion
    from the manipulator's commanded position.
    """
    import json
    import sys
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Use this checkout, not an older site-packages installation.
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    import spikeinterface as si
    import spikeinterface.preprocessing as spre

    output = Path(output)
    info = json.loads((output / "motion_excerpt.json").read_text())
    path = output / "motion_excerpt.dat"
    if path.stat().st_size != info["size_bytes"]:
        raise ValueError("Excerpt size does not match its manifest.")
    fs = info["sampling_frequency_hz"]
    chunk_size = round(chunk_seconds * fs)
    if chunk_size < 1:
        raise ValueError("Chunk duration must be positive.")
    slices = [(a, min(a+chunk_size, info["num_frames"]))
              for a in range(0, info["num_frames"], chunk_size)][:-1]
    if len(slices) < 2:
        raise ValueError("At least two retained chunks are required.")
    signature = json.dumps({"source_sha256": info["excerpt_sha256"], "chunk_size": chunk_size,
                            "highpass_hz": 300, "reference": "global_median",
                            "spikeinterface": si.__version__}, sort_keys=True)
    cache = output / "rms_chunks.npz"
    partial = output / "rms_chunks.partial.npz"
    rms = np.full((len(slices), info["num_channels"]), np.nan)
    completed = np.zeros(len(slices), dtype=bool)
    times = np.array([(a+b-1)/(2*fs) + info["start_seconds"] for a, b in slices])
    existing = cache if cache.exists() else partial
    if existing.exists():
        with np.load(existing) as data:
            if str(data["signature"]) != signature:
                raise ValueError("RMS cache parameters changed; use a new output directory.")
            rms, completed = data["rms"].copy(), data["completed"].copy()

    if not completed.all():
        recording = si.read_binary(file_paths=path, sampling_frequency=fs,
                                   num_channels=info["num_channels"], dtype=info["dtype"])
        recording.set_channel_gains(info["gain_to_uV"])
        recording.set_channel_offsets(info["offset_uV"])
        recording = spre.highpass_filter(recording, freq_min=300)
        recording = spre.common_reference(recording, operator="median")

        def compute(index):
            a, b = slices[index]
            traces = recording.get_traces(start_frame=a, end_frame=b, return_in_uV=True)
            return index, np.std(traces, axis=0, dtype=np.float64)

        last_print = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in as_completed([pool.submit(compute, int(i)) for i in np.flatnonzero(~completed)]):
                index, values = future.result()
                rms[index], completed[index] = values, True
                if time.monotonic()-last_print > 15 or completed.all():
                    with partial.open("wb") as stream:
                        np.savez(stream, rms=rms, completed=completed, times=times, signature=signature)
                    print(f"RMS: {completed.sum()}/{len(slices)} chunks", flush=True)
                    last_print = time.monotonic()
        partial.replace(cache)

    x, y = np.asarray(info["x_um"]), np.asarray(info["y_um"])
    columns = [np.flatnonzero(x == value) for value in np.unique(x)]
    columns = [indices[np.argsort(-y[indices], kind="stable")] for indices in columns]
    row_depths = y[columns[0]]
    if any(not np.array_equal(y[indices], row_depths) for indices in columns):
        raise ValueError("The cross-column average requires columns with matching depths.")
    spacing = float(np.median(np.abs(np.diff(row_depths))))
    if sigma_r is None:
        sigma_r = max(float(np.median(np.abs(np.diff(rms, axis=0)))), np.finfo(float).eps)
    column_motion = [compute_rms_motion(rms[:, indices], times, sigma_r=sigma_r) for indices in columns]
    mean_rms = np.mean([rms[:, indices] for indices in columns], axis=0)
    motion = column_motion[0].copy()
    weights = np.mean([item["weights"] for item in column_motion], axis=0)
    displacement = np.sum(weights * motion["offsets"], axis=2)
    motion.update({
        "weights": weights, "displacement_channels": displacement,
        "velocity_channels_per_second": displacement / np.diff(times)[:, None],
        "offset_spread_channels": np.sqrt(np.sum(weights * (motion["offsets"] - displacement[:, :, None])**2, axis=2)),
        "session_velocity_channels_per_second": displacement.mean(axis=1) / np.diff(times),
        "cumulative_displacement_channels": np.r_[0.0, np.cumsum(displacement.mean(axis=1))],
    })
    trigger = load_manipulator_summary(output)
    starts, ends = trigger["pulse_start_times"], trigger["pulse_end_times"]
    np.savez_compressed(output / "rms_motion_results.npz", rms=rms, mean_rms=mean_rms,
                        times=times, column_channel_indices=np.stack(columns), row_depths_um=row_depths,
                        displacements_by_column=np.stack([m["displacement_channels"] for m in column_motion]),
                        trigger_starts=starts, trigger_ends=ends, **motion)
    fig = plot_rms_motion(mean_rms, times, motion,
                          title="DANDI 000957 / ZYE-0057: columnwise RMS motion, averaged over 8 columns")
    for ax in fig.axes[:2]:
        ax.set_ylabel(f"Depth row ({spacing:g} µm spacing; top to bottom)")
    fig.axes[2].set_ylabel("Depth rows / s")
    fig.axes[3].set_ylabel("Depth rows")
    for ax in fig.axes[:4]:
        ax.axvspan(starts[0], ends[-1], color="limegreen", alpha=0.12, zorder=0)
    fig.savefig(output / "rms_motion.png", dpi=180)
    plt.close(fig)

    # The TTL identifies each movement interval, not actual tissue displacement.
    # Reconstruct the documented alternating 25-um commands for visual context.
    command = np.zeros_like(times)
    for index, (start, end) in enumerate(zip(starts, ends)):
        sign = (-1) ** index
        command += sign * 25 * np.clip((times-start)/(end-start), 0, 1)
    command_velocity = np.diff(command) / np.diff(times)
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True, constrained_layout=True)
    axes[0].plot(motion["transition_times"]/60, motion["session_velocity_channels_per_second"]*spacing)
    axes[0].set(ylabel="Estimated µm/s", title="Mean RMS-pattern velocity across all columns")
    axes[1].plot(times/60, motion["cumulative_displacement_channels"]*spacing)
    axes[1].set(ylabel="Estimated µm", title="Accumulated RMS-pattern displacement")
    axes[2].plot(times/60, command, color="black")
    axes[2].set(xlabel="Original recording time (min)", ylabel="Commanded µm",
                title="Alternating 25-µm probe commands reconstructed from motion gates (independent sign convention)")
    for ax in axes:
        for start, end in zip(starts, ends):
            ax.axvspan(start/60, end/60, color="grey", alpha=0.12)
        ax.axhline(0, color="grey", lw=0.6)
    fig.savefig(output / "session_drift_vs_protocol.png", dpi=180)
    plt.close(fig)
    moving = command_velocity != 0
    correlation = float(np.corrcoef(command_velocity[moving],
                        motion["session_velocity_channels_per_second"][moving])[0, 1])
    summary = {
        "source": SOURCE, "excerpt_minutes": [info["start_seconds"]/60, info["end_seconds"]/60],
        "trigger_minutes": [float(starts[0]/60), float(ends[-1]/60)],
        "motion_gates": len(starts), "retained_chunks": len(times), "channels": len(x),
        "columns": len(columns), "rows_per_column": len(columns[0]), "spacing_um": spacing,
        "sigma_r_uV": sigma_r, "chunk_seconds": chunk_seconds,
        "mean_velocity_um_per_second": float(np.mean(motion["session_velocity_channels_per_second"])*spacing),
        "final_accumulated_pattern_displacement_um": float(motion["cumulative_displacement_channels"][-1]*spacing),
        "command_velocity_correlation_during_motion": correlation,
        "interpretation": "RMS-pattern motion is heuristic. TTL gates and the documented command sequence are not measured tissue motion; signs are independent.",
    }
    _write_json(output / "analysis_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "recordings/dandi_000957")
    parser.add_argument("--stage", choices=["download", "analyse", "all"], default="all")
    parser.add_argument("--start-seconds", type=float, default=1920)
    parser.add_argument("--end-seconds", type=float, default=2820)
    parser.add_argument("--download-workers", type=int, default=6)
    parser.add_argument("--analysis-workers", type=int, default=2)
    parser.add_argument("--chunk-seconds", type=float, default=3)
    parser.add_argument("--sigma-r", type=float)
    args = parser.parse_args()
    if args.stage in ("download", "all"):
        download_excerpt(args.output, args.start_seconds, args.end_seconds, args.download_workers)
        load_manipulator_summary(args.output)
    if args.stage in ("analyse", "all"):
        analyse_excerpt(args.output, args.chunk_seconds, args.analysis_workers, args.sigma_r)
