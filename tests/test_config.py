from melting_tank.config import load_config


def test_v2_config_matches_notebook_baseline() -> None:
    config = load_config()
    assert config.version == "0.2.0"
    assert config.data.target_rule == "next-minute-majority"
    assert config.data.features == ("MELT_TEMP", "MOTORSPEED", "MELT_WEIGHT")
    assert config.data.sequence_length == 10
    assert config.model.epochs == 30

