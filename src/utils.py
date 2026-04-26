import numpy as np


def normalize(series):
    if series.max() == series.min():
        return series
    return (
        series - series.min()
    ) / (
        series.max() - series.min()
    )


def time_decay(days, decay=0.002):
    return np.exp(
        -decay * days
    )


def hybrid_score(
    cf,
    content,
    popularity
):
    return (
        0.40 * cf +
        0.35 * content +
        0.25 * popularity
    )


def validate_user(
    user_id,
    df
):
    return user_id in df.userId.unique()
