from dataclasses import asdict, dataclass


ATTR_LST = [
    "p_mean",
    "pet_mean",
    "p_seasonality",
    "frac_snow",
    "aridity",
    "high_prec_freq",
    "high_prec_dur",
    "low_prec_freq",
    "low_prec_dur",
    "elev_mean",
    "slope_mean",
    "area_gages2",
    "frac_forest",
    "lai_max",
    "lai_diff",
    "gvf_max",
    "gvf_diff",
    "dom_land_cover_frac",
    "dom_land_cover",
    "root_depth_50",
    "soil_depth_pelletier",
    "soil_depth_statsgo",
    "soil_porosity",
    "soil_conductivity",
    "max_water_content",
    "sand_frac",
    "silt_frac",
    "clay_frac",
    "geol_1st_class",
    "glim_1st_class_frac",
    "geol_2nd_class",
    "glim_2nd_class_frac",
    "carbonate_rocks_frac",
    "geol_porostiy",
    "geol_permeability",
]


@dataclass
class ExperimentConfig:
    train_start: str = "1980-10-01"
    train_end: str = "1995-10-01"
    test_start: str = "1995-10-01"
    test_end: str = "2010-10-01"
    forcing_name: str = "daymet"
    batch_size: int = 16
    rho: int = 365
    bufftime: int = 365
    epochs: int = 2
    hidden_size: int = 64
    nmul: int = 4
    max_iter_ep: int = 8
    save_every: int = 1
    learning_rate: float = 1.0
    adadelta_rho: float = 0.9
    alpha_rmse_comb: float = 0.25
    seed: int = 111111
    gpu_id: int = 0
    use_gpu: bool = True
    component_routing: bool = True
    lgdyn: bool = True
    lgdynweight: float = 0.6
    dynamic_sq: bool = True
    dynamic_etgam: bool = True
    dynamic_partition: bool = True
    dynamic_cfmax_snow: bool = True
    dynamic_routing_scale: bool = False
    checkpoint_path: str | None = None

    def to_dict(self):
        return asdict(self)


DEFAULT_CONFIG = ExperimentConfig()
