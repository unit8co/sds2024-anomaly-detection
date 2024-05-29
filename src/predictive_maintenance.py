from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts import concatenate, TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.utils.data.tabularization import create_lagged_data
from darts.utils.utils import generate_index


FREQ = pd.tseries.frequencies.to_offset("10min")
DEV_LO, DEV_MD, DEV_HI, DEV_AGG = ["dev_lo", "dev_md", "dev_hi", "dev_agg"]
ANOM_LO, ANOM_HI, ANOM_AGG = ["anom_lo", "anom_hi", "anom_agg"]


def df_anom_to_series(df: pd.DataFrame):
    """Converts the wind turbine failure DataFrame (rows with anomaly start and end dates)
    into a TimeSeries of binary anomalies."""
    # create time series with binary anomalies
    time_index = generate_index(
        start=pd.Timestamp(df["start"].min()),
        end=pd.Timestamp(df["end"].max()),
        freq=FREQ
    ).tz_localize(None)

    turbine_ids = sorted(df["Turbine_ID"].unique())
    df_anom = pd.DataFrame(
        index=time_index,
        data={t_id: 0. for t_id in turbine_ids},
    )

    for t_id in turbine_ids:
        anom_times = df.loc[df["Turbine_ID"] == t_id, ["start", "end"]]
        for start, end in anom_times.values:
            df_anom.loc[start:end, t_id] = 1.

    return TimeSeries.from_dataframe(df_anom)


def compute_anomalies(
    model,
    series: TimeSeries,
    pred_series: TimeSeries,
    quantiles: List[float],
    min_value: float = 0.,
    anom_window: int = 3,
    min_anom_prob: float = 1.0,
):
    """
    Computes the anomalies based on the predicted normal operating range and the actual values of the feature.

    - Ignores points which were only slightly outside the interval (threshold `min_value`)
    - Scans the residuals in fixed size windows and counts how many points were out-of bounds in each window
      (`anom_window`)
    - For each window we can set a minimum out-of-bounds probability below which we do not consider the window as
      anomalous (`min_anom_prob`)

    Parameters
    ----------
    model
        A trained probabilistic forecasting model that predicts three quantiles.
    series
        The actual target series values
    pred_series
        The predicted target series quantile values
    quantiles
         The predicted quantiles.
    min_value
        Ignore points where the actual values are less than `min_value` outside the predicted interval.
    anom_window
        The window size to detect anomalies on.
    min_anom_prob
        For each window it is the minimum value for the fraction between the number of points that are
        out-of-bounds and `anom_window`.

    Returns
    -------
    anom_pred
        the residuals ("dev_lo", ...) and final anomaly flags ("anom_lo", ...) as a `TimeSeries`.
    df_anom_pred
        a `DataFrame` where each row represents an anomaly, including the start, end date and some statistics
    ql
        the quantile loss for the historical forecast (as a metric how good the quantile predictions are, 0.
        being the best score).
    """
    n_quantiles = pred_series.n_components // series.n_components
    # repeat target column so when can compute the residuals per quantile
    series_ext = concatenate([series] * n_quantiles, axis=1)
    residuals = model.residuals(
        series=series_ext,
        historical_forecasts=pred_series,
        last_points_only=True,
    )
    # quantile loss (metric) per quantile
    ql = quantile_loss(residuals, quantiles)

    # ignore residuals where y_true was within high and low quantile
    df = residuals.pd_dataframe(copy=True)
    df.columns = [DEV_LO, DEV_MD, DEV_HI]

    df = df.fillna(0.)
    df.loc[df[DEV_LO] > -min_value, DEV_LO] = 0.
    df.loc[df[DEV_HI] < min_value, DEV_HI] = 0.
    df.loc[:, DEV_AGG] = df[DEV_LO] + df[DEV_HI]

    # ignore residuals where the anomaly didn't last for a couple time steps
    df = _find_anomaly_periods(df, anom_window, min_anom_prob)
    # generate a dataframe where each row
    anom = _compute_anomaly_table(df)
    return TimeSeries.from_dataframe(df), anom, ql


def _find_anomaly_periods(
    df: pd.DataFrame,
    anom_window: int = 3,
    min_anom_prob: float = 1.0,
):
    # ignore residuals where the anomaly didn't last for a couple time steps
    df_out = df.copy()
    for col_dev, col_anom in zip([DEV_LO, DEV_HI, DEV_AGG], [ANOM_LO, ANOM_HI, ANOM_AGG]):
        is_anomaly = df_out[col_dev] != 0

        # windowed anomalies
        windows_anom = create_lagged_data(
            target_series=TimeSeries.from_series(is_anomaly),
            lags=[i for i in range(-anom_window, 0)],
            uses_static_covariates=False,
            is_training=False,
        )[0]
        # windowing results in n - anom_window windows -> repeat the first window and prepend
        windows_anom = np.concatenate([
            np.zeros((anom_window - 1, anom_window, 1)),
            windows_anom,
        ])
        # get the average number of anomalous steps per window
        windows_anom = windows_anom.mean(axis=1)[:, 0]
        # compute the actual anomalous time frames, with respect to probability
        idx_anom = np.argwhere(windows_anom >= min_anom_prob)[:, 0]

        # reset anomaly flags
        windows_anom = np.zeros(shape=windows_anom.shape)
        for i in range(anom_window):
            windows_anom[idx_anom - i] = 1.

        windows_anom = pd.Series(windows_anom, index=is_anomaly.index).astype(bool)
        # add binary anomalies
        df_out.loc[:, col_anom] = windows_anom
        # remove deviations from too short anomalies
        df_out.loc[~windows_anom, col_dev] = 0.0
    return df_out


def _compute_anomaly_table(df: pd.DataFrame) -> pd.DataFrame:
    """Computes start and end dates for each anomaly in `df`. The returned DataFrame has columns:
        - "start": anomaly time stamp
        - "end": anomaly end time stamp
        - "n_steps": how many steps the anomaly lasted
        - "sum": sum of all deviations (y_true above the high and/or below the low predicted quantile)
        - "name" "anom_lo", "anom_hi" or "anom_agg"

    Parameters
    ----------
    df
        the pandas DataFrame containing anomaly columns `columns`
    """
    interval_dfs = []
    for col in [ANOM_LO, ANOM_HI, ANOM_AGG]:
        # make [0, 0, 1, 1, 0] groupable -> [0, 0, 1, 1, 2]
        blocks = df[col].diff().ne(0).cumsum()
        # remove groups with initial zeros, so that we can group only the ones-groups
        blocks = blocks[df[col] > 0]
        # group all ones and get the start and end dates (index)
        anom_df = df.loc[df[col] > 0]
        interval_df = (
            anom_df[col].index.to_frame()
            .groupby(blocks, sort=True)
            .agg(["min", "max", "size"])
            .rename(columns={"min": "start", "max": "end", "size": "n_steps"})
            .reset_index(drop=True)
        )
        # drop the time tag from columns
        interval_df.columns = interval_df.columns.droplevel(0)
        # deviance column
        dev_col = col.replace("anom", "dev")
        dev_df = (
            anom_df[[dev_col]]
            .groupby(blocks, sort=True)
            .agg("sum")
            .reset_index(drop=True)
            .rename(columns={dev_col: "dev_tot"})
        )
        interval_df = pd.concat([interval_df, dev_df], axis=1)
        interval_df["name"] = col
        interval_dfs.append(interval_df)
    return pd.concat(interval_dfs, ignore_index=True)


def quantile_loss(residuals: TimeSeries, quantiles: List[float]):
    """Computes the quantile loss per predicted quantile."""
    errors = residuals.values(copy=False)
    qs = np.array(quantiles)
    losses = 2.0 * np.maximum((qs - 1) * errors, qs * errors)
    return pd.DataFrame(np.nanmean(losses, axis=0), index=quantiles).T


def plot_predicted_anomalies(
    df: pd.DataFrame,
    series: TimeSeries,
    covs: TimeSeries,
    hist_fc: TimeSeries,
    anom_true: TimeSeries,
    anom_pred: TimeSeries,
    turbine_id: str,
    max_plots: int = 5,
):
    """For the first `max_plots` predicted anomaly in `df`, it plots the preceding and following 3 days of:

    - actual target values
    - historical forecast boundaries
    - predicted anomaly as a green capsule
    - actual anomaly as a red capsule (if there is any in the time frame)
    - and the covariates scaled to a value range (0, 1).
    """

    for idx, (start, end) in enumerate(df[["start", "end"]].values):
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        p_start = start - FREQ * 6 * 24 * 3
        p_end = end + FREQ * 6 * 24 * 3
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 9.6), sharex=True)

        series[p_start:p_end].plot(ax=ax1)
        plot_intervals(hist_fc[p_start:p_end], ax=ax1, plot_med=True, alpha=0.5)
        plot_actual_anomalies(anom_true[turbine_id][p_start:p_end] * 100, ax=ax1)
        plot_actual_anomalies(anom_pred[ANOM_AGG][p_start:p_end] * 100, ax=ax1)
        anom_pred[DEV_AGG][p_start:p_end].plot(ax=ax1)

        Scaler().fit_transform(covs)[p_start:p_end].plot(ax=ax2)
        ax1.set_title(f"Anom start: {start.round(FREQ)}, end: {end.round(FREQ)}")
        plt.show()
        if idx == max_plots - 1:
            break


def plot_intervals(
    series: TimeSeries,
    ax=None,
    plot_med: bool = True,
    alpha: float=0.25,
    c: Optional[str] = None
):
    """Plot historical quantile forecasts as intervals."""

    if ax is None:
        fig, ax = plt.subplots()

    vals = series.values(copy=False)
    if plot_med:
        median_p = ax.plot(
            series.time_index,
            vals[:, 1],
            label=series.columns[1]
        )
    else:
        median_p = ax.plot([], [])
    color_used = c or median_p[0].get_color()

    ax.fill_between(
        series.time_index,
        vals[:, 0],
        vals[:, -1],
        color=color_used,
        alpha=alpha,
    )
    ax.legend()


def plot_actual_anomalies(
    series: TimeSeries,
    ax=None,
    alpha: float = 0.25,
    c: Optional[str] = None
):
    """Plots a binary anomaly `series` as capsule."""
    if ax is None:
        fig, ax = plt.subplots()

    vals = series.values(copy=False)
    vals_zero = np.zeros(vals.shape)
    if c is None:
        empty_p = ax.plot([], [])
        color_used = empty_p[0].get_color()
    else:
        color_used = c

    ax.fill_between(
        series.time_index,
        vals_zero[:, 0],
        vals[:, 0],
        color=color_used,
        alpha=alpha,
        label=series.columns[0] + "_anom",
    )
    ax.legend()
