import pytest
import hydra
from pathlib import Path

import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Import train function
from src.train import train


@pytest.fixture
def config():
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(
            config_name="train",
            overrides=["experiment=dogbreed_ex_train"],
        )
        return cfg


def test_dogbreed_ex_training(config, tmp_path):
    # Update output and log directories to use temporary path
    config.paths.output_dir = str(tmp_path)
    config.paths.log_dir = str(tmp_path / "logs")
    train(config)