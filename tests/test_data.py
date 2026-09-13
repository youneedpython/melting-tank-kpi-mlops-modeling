from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from melting_tank.config import load_config
from melting_tank.data import build_minute_sequences


def _minute(timestamp: str, ng_count: int) -> list[dict]:
    rows = []
    for index in range(10):
        is_ng = index < ng_count
        rows.append(
            {
                "STD_DT": pd.Timestamp(timestamp),
                "MELT_TEMP": 400 + index,
                "MOTORSPEED": 100,
                "MELT_WEIGHT": 500,
                "TAG": "NG" if is_ng else "OK",
                "target_ng": int(is_ng),
            }
        )
    return rows


def test_sequence_predicts_next_minute_majority_ng() -> None:
    rows = _minute("2020-03-04 00:00:00", 0) + _minute("2020-03-04 00:01:00", 5)
    X, y, times = build_minute_sequences(pd.DataFrame(rows), load_config())
    assert X.shape == (1, 10, 3)
    assert y.tolist() == [1]
    assert times[0] == np.datetime64("2020-03-04T00:00:00")


def test_four_ng_values_do_not_make_majority_ng() -> None:
    rows = _minute("2020-03-04 00:00:00", 0) + _minute("2020-03-04 00:01:00", 4)
    _, y, _ = build_minute_sequences(pd.DataFrame(rows), load_config())
    assert y.tolist() == [0]


def test_missing_minute_is_not_connected_as_next_minute() -> None:
    rows = _minute("2020-03-04 00:00:00", 0) + _minute("2020-03-04 00:02:00", 10)
    X, y, times = build_minute_sequences(pd.DataFrame(rows), load_config())
    assert X.shape[0] == len(y) == len(times) == 0


def test_unsupported_target_rule_fails() -> None:
    config = load_config()
    invalid = replace(config, data=replace(config.data, target_rule="any-ng"))
    with pytest.raises(ValueError, match="target_rule"):
        build_minute_sequences(pd.DataFrame(_minute("2020-03-04 00:00:00", 0)), invalid)

