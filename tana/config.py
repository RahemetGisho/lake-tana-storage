"""All assumptions in one place. Every number here is an assumption you can change."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # --- Location: approximate centroid of the Lake Tana / upper Blue Nile highlands ---
    lat: float = 11.85
    lon: float = 37.45
    start_year: int = 1990
    end_year: int = 2025

    # --- Forecast evaluation ---
    test_start: str = "2016-01"      # everything before this is training data
    horizon: int = 3                 # months ahead the operator looks

    # --- Toy catchment: inflow follows lagged rainfall ---
    runoff_weights: tuple = (0.35, 0.40, 0.25)   # share of rainfall reaching the reservoir 0, 1, 2 months later
    mean_annual_inflow_mm3: float = 3800.0       # assumed mean annual inflow, million m3 (order of magnitude only)

    # --- Toy reservoir (all sized relative to mean monthly inflow) ---
    cap_months: float = 6.0          # storage capacity = this many months of mean inflow
    dead_frac: float = 0.10          # share of capacity that cannot be released
    demand_frac: float = 0.80        # firm monthly release target, as a share of mean monthly inflow
    rmax_mult: float = 1.5           # max turbine release = this x demand
    init_frac: float = 0.50          # starting storage as a share of capacity

    # --- Illustrative hydropower plant ---
    gwh_per_mm3: float = 0.245       # ~100 m head x 90% efficiency (1 Mm3 falling 1 m = 0.002725 GWh)

    # --- Solar / wind backup assumptions ---
    pv_perf_ratio: float = 0.78      # system losses for utility PV
    wind_shear: float = 0.14         # power-law exponent, 50 m -> hub height
    hub_height_m: float = 100.0
    wind_losses: float = 0.85        # wake, availability, electrical losses
    cut_in: float = 3.0
    rated: float = 12.0
    cut_out: float = 25.0
    weak_season_quantile: float = 0.25   # JJAS rainfall below this quantile = "weak rainy season"


@dataclass(frozen=True)
class BatteryConfig:
    """Battery dispatch experiment. The load profile and plant sizes are SYNTHETIC assumptions."""
    daily_start_year: int = 2010     # daily solar record used for forecasting (test years follow Config.test_start)
    pv_mwp: float = 40.0             # solar PV capacity next to the battery
    pv_perf_ratio: float = 0.78
    load_mean_mw: float = 10.0       # mean load of the feeder the battery supports (synthetic)
    cap_mwh: float = 50.0            # usable battery energy
    power_mw: float = 12.5           # max charge / discharge power
    eta_charge: float = 0.92
    eta_discharge: float = 0.92      # round trip = 0.92 x 0.92 = 0.846
    sunrise: int = 6                 # PV is spread over a clear-sky bell between these hours
    sunset: int = 18
    critical_hours: tuple = (18, 19, 20, 21)   # evening peak the battery is reserved for
    peak_weight: float = 2.0         # cost of one MWh of evening peak that hydro must supply
    overnight_weight: float = 1.0    # cost of one MWh drawn overnight to top up the battery
    margin_mwh: float = 0.0          # extra reserve the forecast-aware rule keeps (safety margin)
