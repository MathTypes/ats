
from typing import Any

import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig

from vss.interactive.app_utils import ModuleManager
from hydra.core.config_store import ConfigStore

@hydra.main(config_path="../../../conf", config_name="search")
def main(cfg: DictConfig) -> Any:
    ModuleManager(cfg).run_app()

if __name__ == "__main__":
    main()
