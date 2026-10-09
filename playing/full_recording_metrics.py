"""Download the complete DANDI source and prepare its AP samples for metrics.

Run: python playing/full_recording_metrics.py
The original NWB is retained and checked against its published SHA-256.
The additional binary copy makes repeated full-session analyses faster.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import time

from rms_drift_experiment import SOURCE


def prepare_full_recording():
    import requests
    import h5py
    import numpy as np

    folder = Path(__file__).resolve().parent / "recordings" / "dandi_000957_full"
    folder.mkdir(exist_ok=True)
    target = folder / "sub-ZYE-0057_ecephys+image.nwb"
    partial = target.with_suffix(".nwb.part")
    checkpoint = folder / "download_progress.json"
    if not target.exists():
        response = requests.head(SOURCE["url"], timeout=60)
        response.raise_for_status()
        size, etag = int(response.headers["Content-Length"]), response.headers["ETag"]
        block_size = 64 * 1024**2
        progress = {"size": size, "etag": etag, "blocks": {}}
        if checkpoint.exists():
            progress = json.loads(checkpoint.read_text())
            if progress["size"] != size or progress["etag"] != etag:
                raise ValueError("Remote source changed; download checkpoint cannot be reused.")
        if not partial.exists():
            with partial.open("wb") as stream:
                stream.truncate(size)
            progress["blocks"] = {}
        blocks = [(i, a, min(a+block_size, size)) for i, a in enumerate(range(0, size, block_size))]
        # Check resumed blocks before trusting their completion records.
        with partial.open("rb") as stream:
            for i, a, b in blocks:
                if str(i) in progress["blocks"]:
                    stream.seek(a)
                    if hashlib.sha256(stream.read(b-a)).hexdigest() != progress["blocks"][str(i)]:
                        del progress["blocks"][str(i)]

        def fetch(block):
            i, a, b = block
            for attempt in range(5):
                try:
                    r = requests.get(SOURCE["url"], headers={"Range": f"bytes={a}-{b-1}", "If-Match": etag}, timeout=(30, 180))
                    r.raise_for_status()
                    if r.status_code != 206 or r.headers.get("Content-Range") != f"bytes {a}-{b-1}/{size}" or len(r.content) != b-a:
                        raise ValueError("Invalid range response")
                    digest = hashlib.sha256(r.content).hexdigest()
                    with partial.open("r+b") as stream:
                        stream.seek(a)
                        stream.write(r.content)
                    return str(i), digest
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(2**attempt)

        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(fetch, block) for block in blocks if str(block[0]) not in progress["blocks"]]
            for future in as_completed(futures):
                i, digest = future.result()
                progress["blocks"][i] = digest
                temp = checkpoint.with_suffix(".tmp")
                temp.write_text(json.dumps(progress))
                temp.replace(checkpoint)
                if len(progress["blocks"]) % 10 == 0:
                    print(f"Downloaded {len(progress['blocks'])}/{len(blocks)} blocks", flush=True)
        print("Verifying full NWB SHA-256...", flush=True)
        with partial.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != SOURCE["full_asset_sha256"]:
            raise ValueError(f"Full NWB checksum mismatch: {digest}")
        partial.replace(target)
        (folder / "nwb_verified.json").write_text(json.dumps({"sha256": digest, "size_bytes": size}))

    binary = folder / "recording.dat"
    manifest = folder / "recording.json"
    if binary.exists() and manifest.exists():
        return folder
    binary_partial = binary.with_suffix(".dat.part")
    with h5py.File(target, "r") as f:
        series = f[SOURCE["series"]]
        data = series["data"]
        fs = float(series["starting_time"].attrs["rate"])
        electrodes = series["electrodes"][:]
        table = f["general/extracellular_ephys/electrodes"]
        gains = series["channel_conversion"][:] if "channel_conversion" in series else np.ones(data.shape[1])
        metadata = dict(SOURCE, sampling_frequency_hz=fs, num_channels=data.shape[1],
                        num_frames=data.shape[0], dtype=data.dtype.str,
                        gain_to_uV=(gains * float(data.attrs["conversion"]) * 1e6).tolist(),
                        offset_uV=float(data.attrs.get("offset", 0)) * 1e6,
                        x_um=table["rel_x"][:][electrodes].tolist(),
                        y_um=table["rel_y"][:][electrodes].tolist(),
                        start_seconds=float(series["starting_time"][()]),
                        duration_seconds=data.shape[0]/fs)
        print(f"Extracting all {metadata['duration_seconds']/60:.2f} minutes of AP samples...", flush=True)
        step = data.chunks[0]
        with binary_partial.open("wb") as stream:
            for a in range(0, data.shape[0], step):
                data[a:a+step].tofile(stream)
                if (a // step) % 100 == 0:
                    print(f"Extracted {a/data.shape[0]:.1%}", flush=True)
        # Independently compare two slices of the binary against the NWB.
        extracted = np.memmap(binary_partial, mode="r", dtype=data.dtype, shape=data.shape)
        for a in (0, data.shape[0]//2, data.shape[0]-100):
            np.testing.assert_array_equal(extracted[a:a+100], data[a:a+100])
        del extracted
    binary_partial.replace(binary)
    manifest.write_text(json.dumps(metadata, indent=2))
    print(f"Full recording ready: {folder}", flush=True)
    return folder


if __name__ == "__main__":
    prepare_full_recording()
