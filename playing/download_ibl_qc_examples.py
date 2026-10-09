#!/usr/bin/env python3
"""Save three low-noise and two high-noise IBL AP recordings from DANDI.

The labels refer ONLY to IBL's processed AP RMS noise metric. IBL's RIGOR
guideline uses median AP RMS < 40 uV. Alyx exposes the 10th and 90th
percentiles; good recordings pass conservatively when p90 < 40 uV.
For potential bad recordings, the script downloads IBL's small per-channel
AP RMS QC array and requires its actual median to be >= 40 uV.
These labels do not summarize unit, behavior, or histology quality.

By default each output is a five-minute SpikeInterface binary folder, streamed
from Dandiset 000409's raw NWB files. Pass --full for whole recordings, which
may require hundreds of GB. A manifest records asset and QC provenance.

Install: python -m pip install spikeinterface h5py remfile
Preview: python playing/download_ibl_qc_examples.py --plan-only
Run:     python playing/download_ibl_qc_examples.py
"""

import argparse
import io
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DANDI_API = "https://api.dandiarchive.org/api"
DANDISET = "000409"
DANDI_VERSION = "draft"
ALYX_URL = "https://openalyx.internationalbrainlab.org"
NOISE_LIMIT_VOLTS = 40e-6
RAW_ASSET_RE = re.compile(r"_ses-([0-9a-f-]{36})_desc-raw_ecephys\.nwb$", re.IGNORECASE)
PROBE_RE = re.compile(r"^probe(\d+)$", re.IGNORECASE)


def fetch_json(url, token=None):
    request = Request(url, headers={"Authorization": f"Token {token}"} if token else {})
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def alyx_token():
    # Public OpenAlyx account documented by IBL.
    credentials = urlencode({"username": "intbrainlab", "password": "international"}).encode()
    with urlopen(Request(f"{ALYX_URL}/auth-token", data=credentials), timeout=60) as response:
        return json.load(response)["token"]


def insertions_for_session(eid, token):
    url = f"{ALYX_URL}/insertions?{urlencode({'session': eid})}"
    insertions = []
    while url:
        page = fetch_json(url, token)
        insertions.extend(page["results"])
        url = page.get("next")
    return eid, insertions


def raw_assets_by_session():
    """Index only raw ephys NWB assets, one DANDI listing pass."""
    url = f"{DANDI_API}/dandisets/{DANDISET}/versions/{DANDI_VERSION}/assets/?page_size=100"
    found = {}
    while url:
        page = fetch_json(url)
        for asset in page["results"]:
            path = asset["path"]
            match = RAW_ASSET_RE.search(path)
            if match:
                eid = match.group(1).lower()
                asset_id = asset.get("asset_id") or asset.get("id")
                if not asset_id:
                    raise ValueError(f"DANDI asset has no ID: {path}")
                found[eid] = {"asset_id": asset_id, "path": path}
        url = page.get("next")
    return found


def numeric_metric(extended_qc, key):
    try:
        value = float(extended_qc[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def classify_qc(insertion):
    """Return a low-noise label or a candidate requiring the median QC file."""
    qc = (insertion.get("json") or {}).get("extended_qc") or {}
    p10 = numeric_metric(qc, "apRms_p10_proc")
    p90 = numeric_metric(qc, "apRms_p90_proc")
    if p10 is None or p90 is None or p10 < 0 or p90 < p10:
        return None
    if p90 < NOISE_LIMIT_VOLTS:
        label = "good"
    else:
        label = "needs_median"
    return {"label": label, "ap_rms_p10_proc_v": p10, "ap_rms_p90_proc_v": p90,
            "insertion_qc": (insertion.get("json") or {}).get("qc")}


def candidates(assets, token):
    good, possible_bad = [], []
    with ThreadPoolExecutor(max_workers=8) as executor:
        all_insertions = executor.map(lambda eid: insertions_for_session(eid, token), sorted(assets))
        for index, (eid, insertions) in enumerate(all_insertions, 1):
            if index % 50 == 0:
                print(f"Checked {index}/{len(assets)} DANDI sessions", flush=True)
            asset = assets[eid]
            for insertion in insertions:
                probe = PROBE_RE.fullmatch(insertion.get("name") or "")
                result = classify_qc(insertion)
                if not probe or result is None:
                    continue
                item = {
                    **result, "eid": eid, "pid": insertion["id"],
                    "probe": insertion["name"], "asset_id": asset["asset_id"],
                    "asset_path": asset["path"],
                    "electrical_series_path": f"acquisition/ElectricalSeriesProbe{int(probe.group(1)):02d}AP",
                }
                (good if result["label"] == "good" else possible_bad).append(item)
    # Independent sessions; reproducible ordering by noise and ID.
    good.sort(key=lambda x: (x["ap_rms_p90_proc_v"], x["eid"], x["probe"]))
    possible_bad.sort(key=lambda x: (-x["ap_rms_p90_proc_v"], x["eid"], x["probe"]))
    return good, possible_bad


def processed_ap_rms_median(item, token):
    """Read IBL's compact AP RMS QC array, not the raw traces."""
    import numpy as np

    query = urlencode({"session": item["eid"], "name": "_iblqc_ephysChannels.apRMS.npy"})
    url = f"{ALYX_URL}/datasets?{query}"
    collection = f"raw_ephys_data/{item['probe']}"
    while url:
        page = fetch_json(url, token)
        for dataset in page["results"]:
            if dataset.get("collection") != collection:
                continue
            for record in dataset.get("file_records") or []:
                data_url = record.get("data_url")
                if data_url and record.get("exists", True):
                    with urlopen(data_url, timeout=60) as response:
                        rms = np.load(io.BytesIO(response.read()), allow_pickle=False)
                    if rms.ndim != 2 or rms.shape[0] != 2:
                        raise ValueError(f"Unexpected AP RMS shape {rms.shape} for {item['pid']}")
                    return float(np.median(rms[1]))
        url = page.get("next")
    return None


def confirmed_bad(possible_bad, token):
    bad = []
    used_sessions = set()
    for item in possible_bad:
        if item["eid"] in used_sessions:
            continue
        median = processed_ap_rms_median(item, token)
        if median is not None and median >= NOISE_LIMIT_VOLTS:
            bad.append({**item, "label": "bad", "ap_rms_median_proc_v": median})
            used_sessions.add(item["eid"])
            if len(bad) == 2:
                break
    return bad


def choose_independent(good, bad):
    chosen, used_sessions = [], set()
    for pool, count in ((good, 3), (bad, 2)):
        selected = []
        for item in pool:
            if item["eid"] not in used_sessions:
                selected.append(item)
                used_sessions.add(item["eid"])
                if len(selected) == count:
                    break
        if len(selected) != count:
            raise RuntimeError(
                f"Only {len(selected)} of {count} {pool[0]['label'] if pool else 'qualifying'} "
                "independent raw NWB sessions found. No QC threshold was relaxed."
            )
        chosen.extend(selected)
    return chosen


def asset_url(asset_id):
    metadata = fetch_json(f"{DANDI_API}/assets/{asset_id}/")
    return next(
        (url for url in metadata.get("contentUrl", []) if "dandiarchive.s3.amazonaws.com" in url),
        f"{DANDI_API}/assets/{asset_id}/download/",
    )


def save_example(item, output, duration_seconds):
    import spikeinterface.extractors as se

    folder = output / f"{item['label']}_{item['eid']}_{item['probe']}"
    marker = folder / "download_complete.json"
    if marker.exists():
        print(f"Already saved: {folder}", flush=True)
        return
    if folder.exists():
        raise RuntimeError(f"Incomplete output exists: {folder}. Move it before retrying.")

    print(f"Streaming {item['label']}: {item['asset_path']} {item['probe']}", flush=True)
    recording = se.read_nwb_recording(
        file_path=asset_url(item["asset_id"]),
        electrical_series_path=item["electrical_series_path"],
        stream_mode="remfile",
        load_channel_properties=True,
        load_time_vector=True,
    )
    if duration_seconds is not None:
        end_frame = min(recording.get_num_samples(), int(duration_seconds * recording.get_sampling_frequency()))
        if end_frame <= 0:
            raise RuntimeError(f"No samples in {item['asset_path']}")
        recording = recording.frame_slice(0, end_frame)

    recording.save(folder=folder, format="binary", n_jobs=1, chunk_duration="1s", progress_bar=True)
    marker.write_text(json.dumps({
        **item,
        "qc_rule": "processed AP RMS p90 < 40 uV (good), median >= 40 uV (bad)",
        "dandiset": DANDISET, "dandi_version": DANDI_VERSION,
        "num_channels": recording.get_num_channels(),
        "num_samples": recording.get_num_samples(),
        "sampling_frequency_hz": recording.get_sampling_frequency(),
    }, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("playing/recordings/ibl_qc_examples"))
    parser.add_argument("--duration-seconds", type=float, default=300)
    parser.add_argument("--full", action="store_true", help="Save whole recordings instead of five-minute cuts")
    parser.add_argument("--plan-only", action="store_true", help="Select and print five assets without trace transfer")
    args = parser.parse_args()
    if args.duration_seconds <= 0:
        parser.error("--duration-seconds must be positive")

    token = alyx_token()
    assets = raw_assets_by_session()
    print(f"Found {len(assets)} DANDI raw ephys sessions", flush=True)
    good, possible_bad = candidates(assets, token)
    print(f"QC candidates: {len(good)} low-noise, {len(possible_bad)} requiring median check", flush=True)
    bad = confirmed_bad(possible_bad, token)
    chosen = choose_independent(good, bad)
    for item in chosen:
        print(f"{item['label']:4s} {item['eid']} {item['probe']} "
              f"p10={item['ap_rms_p10_proc_v'] * 1e6:.1f} uV "
              f"p90={item['ap_rms_p90_proc_v'] * 1e6:.1f} uV "
              f"asset={item['asset_id']}")
    if args.plan_only:
        return

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "selection.json").write_text(json.dumps(chosen, indent=2), encoding="utf-8")
    for item in chosen:
        save_example(item, args.output, None if args.full else args.duration_seconds)


if __name__ == "__main__":
    main()
