import pytest
import hydra
from pathlib import Path

import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Import train function
from src.infer import infer


@pytest.fixture
def config():
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(
            config_name="infer",
            overrides=["callbacks.model_checkpoint.filename=/workspace/model_storage/epoch-checkpoint.ckpt"],
        )
        return cfg


def test_dogbreed_ex_infer(config, tmp_path):
    # Update output and log directories to use temporary path
    config.paths.output_dir = str(tmp_path)
    config.paths.log_dir = str(tmp_path / "logs")
    infer(config)