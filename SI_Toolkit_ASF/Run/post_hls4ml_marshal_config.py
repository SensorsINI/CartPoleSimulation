"""
Post-conversion hook: emit nn_marshal_config.h into the deployed FPGA network folder.

Called from Convert_Network_With_hls4ml.py after SI_Toolkit's convert_with_hls4ml().
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


def _repo_root() -> Path:
    # .../Driver/CartPoleSimulation/SI_Toolkit_ASF/Run/post_hls4ml_marshal_config.py
    return Path(__file__).resolve().parents[4]


def _load_config(asf_dir: Path) -> dict:
    with (asf_dir / "config_hls.yml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_fpga_network_dir(repo_root: Path, cfg: dict) -> Path | None:
    nn_root = repo_root / "FPGA" / "NeuralNetworks"
    if not nn_root.is_dir():
        return None

    tag = cfg.get("hls4ml_output_folder_name")
    candidates = sorted(nn_root.glob("hls4ml_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None

    if tag:
        tagged = [p for p in candidates if tag in p.name]
        if tagged:
            return tagged[0]

    return candidates[0]


def _resolve_norm_csv_dir(asf_dir: Path, cfg: dict) -> Path:
    models = Path(cfg["path_to_models"])
    if not models.is_absolute():
        models = asf_dir / models
    return models / cfg["net_name"] / "norm_vectors"


def generate_marshal_config_after_hls4ml(
    *,
    vhd_dir: Path | None = None,
    norm_csv_dir: Path | None = None,
) -> Path | None:
    """
    Generate nn_marshal_config.h. When paths are omitted they are inferred from
    config_hls.yml and the newest FPGA/NeuralNetworks/hls4ml_* folder.
    """
    repo_root = _repo_root()
    asf_dir = Path(__file__).resolve().parent.parent
    cfg = _load_config(asf_dir)

    vhd_dir = vhd_dir or _resolve_fpga_network_dir(repo_root, cfg)
    if vhd_dir is None or not (vhd_dir / "myproject.vhd").is_file():
        print(
            "post_hls4ml_marshal_config: no deployed FPGA network folder found; "
            "skipping nn_marshal_config.h generation",
            file=sys.stderr,
        )
        return None

    norm_csv_dir = norm_csv_dir or _resolve_norm_csv_dir(asf_dir, cfg)
    if not norm_csv_dir.is_dir():
        print(
            f"post_hls4ml_marshal_config: norm_vectors not found at {norm_csv_dir}; "
            "skipping nn_marshal_config.h generation",
            file=sys.stderr,
        )
        return None

    script = repo_root / "FPGA" / "scripts" / "generate_nn_marshal_config.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--vhd-dir",
            str(vhd_dir),
            "--norm-csv-dir",
            str(norm_csv_dir),
        ],
        check=True,
    )
    return vhd_dir / "nn_marshal_config.h"
