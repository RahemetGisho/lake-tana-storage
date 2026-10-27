"""Run the whole project:  python run_all.py   (add --synthetic only for an offline smoke test)."""

import argparse
from dataclasses import replace

from tana.config import Config
from tana.pipeline import run


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--synthetic", action="store_true", help="fake climate data, smoke test only"
    )
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--cap-months", type=float)
    ap.add_argument("--demand-frac", type=float)
    ap.add_argument(
        "--no-battery",
        action="store_true",
        help="skip the daily-solar battery experiment",
    )
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    cfg = Config()
    changes = {
        k: v
        for k, v in {
            "lat": args.lat,
            "lon": args.lon,
            "cap_months": args.cap_months,
            "demand_frac": args.demand_frac,
        }.items()
        if v is not None
    }
    cfg = replace(cfg, **changes)
    R, out, figs = run(
        cfg, synthetic=args.synthetic, outdir=args.outdir, battery=not args.no_battery
    )
    print(f"\nDone. Results in {out}/")
    print(
        R["table"][["shortfall_months", "unmet_mm3", "spill_mm3", "generation_gwh"]]
        .round(1)
        .to_string()
    )
    if "battery" in R:
        print(
            R["battery"]["table"][
                [
                    "peak_hydro_mwh",
                    "overnight_grid_mwh",
                    "weighted_cost",
                    "saving_vs_best_static_pct",
                ]
            ]
            .round(1)
            .to_string()
        )
    for f in figs:
        print(" ", f)


if __name__ == "__main__":
    main()
