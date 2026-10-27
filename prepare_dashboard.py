"""Build the dashboard's forecast cache once:  python prepare_dashboard.py
Commit the data/ folder afterwards so the deployed app starts instantly."""
import argparse

from tana.config import BatteryConfig, Config
from tana.dashboard_data import load_forecasts, load_inputs

ap = argparse.ArgumentParser()
ap.add_argument("--synthetic", action="store_true")
args = ap.parse_args()

cfg, bc = Config(), BatteryConfig()
monthly, daily = load_inputs(cfg, bc, args.synthetic, "data")
fcs, built = load_forecasts(cfg, monthly, args.synthetic, "data")
print("forecast cache", "built" if built else "already up to date", "| daily solar:", "yes" if daily is not None else "NO")
