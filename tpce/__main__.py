"""Historical time-induced TPCE CLI; for current PCE use own_own_costs."""

import argparse, json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from . import TPCEConfig, Calibration, estimate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help=".npy (N,2) or .npz observations")
    p.add_argument("--key", help="NPZ array key")
    p.add_argument("--output", required=True)
    p.add_argument("--config", help="TPCEConfig JSON")
    p.add_argument("--calibration", help="frozen calibration JSON")
    p.add_argument(
        "--anchors", help=".npy embedding-start indices; default all valid points"
    )
    a = p.parse_args()
    source = np.load(a.input, allow_pickle=False)
    if isinstance(source, np.lib.npyio.NpzFile):
        if not a.key:
            p.error("--key is required for NPZ input")
        data = source[a.key]
    else:
        data = source
    cfg = (
        TPCEConfig(**json.loads(Path(a.config).read_text()))
        if a.config
        else TPCEConfig()
    )
    cal = (
        Calibration.from_dict(json.loads(Path(a.calibration).read_text()))
        if a.calibration
        else None
    )
    r = estimate(
        data,
        cfg,
        anchors=np.load(a.anchors, allow_pickle=False) if a.anchors else None,
        calibration=cal,
    )
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "TPCE.npz",
        anchors=r.anchors,
        cost=r.cost,
        S_PCE=r.S_PCE,
        denominator=r.denominator,
        S_R=r.S_R,
        neighbors=r.neighbors,
        h1_counts=r.h1_counts,
        h0_degenerate=r.h0_degenerate,
    )
    (out / "settings.json").write_text(
        json.dumps(
            {
                "method": "TPCE_target_neighborhood",
                "config": asdict(cfg),
                "calibration": asdict(r.calibration),
                "axes": "cost[H,target_XY,time]; S_PCE[H,time]",
                "H1_empty_policy": "mass_coarsening",
                "sign": "positive X->Y convention; not an identification theorem",
            },
            indent=2,
        )
    )
    print(out / "TPCE.npz")


if __name__ == "__main__":
    main()
