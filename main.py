import hydra
from omegaconf import DictConfig
import importlib
from typing import Optional

@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(cfg: DictConfig) -> Optional[float]:
    method = cfg.method.name

    method_module = importlib.import_module(f"scripts.run_{method}")
    
    return method_module.main(cfg)

if __name__ == "__main__":
    main()