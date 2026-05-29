import glob
import numpy as np
import pandas as pd

from darts import TimeSeries

def get_kss_score(nf) : 
    kss = nf.split("_")
    return float(kss[3])

def apply_fixed_scaling(series, absolute_min, absolute_max):
    scaled_series = (series - absolute_min) / (absolute_max - absolute_min)
    return np.clip(scaled_series, 0.0, 1.0)

def reverse_fixed_scaling(series, absolute_min, absolute_max):
    unscaled_series = series * (absolute_max - absolute_min) + absolute_min
    return unscaled_series

def load_data_as_ts_list(ft, file):
    folders = glob.glob(f"data_{ft}/*")
    target_ts_list = []
    past_cov_ts_list = []

    for f in folders :
        df = pd.read_csv(f"{f}/{file}")
        df["kss_score"] = get_kss_score(f)
        df["log_time"] = pd.to_datetime(df['log_time'])
        df = df.set_index("log_time")
        df = df.resample("1s").mean().interpolate(method="linear")
        df = df.reset_index()

        df["kss_score"] = apply_fixed_scaling(df["kss_score"], 1, 9)
        df["breath_rate"] = apply_fixed_scaling(df["breath_rate"], 5, 40)
        df["heart_rate"] = apply_fixed_scaling(df["heart_rate"], 40, 200)

        target_ts = TimeSeries.from_dataframe(df, time_col="log_time", value_cols=["kss_score"])
        past_cov_ts = TimeSeries.from_dataframe(df, time_col="log_time", value_cols=["breath_rate", "heart_rate"])
        target_ts_list.append(target_ts)
        past_cov_ts_list.append(past_cov_ts)
    return target_ts_list, past_cov_ts_list
