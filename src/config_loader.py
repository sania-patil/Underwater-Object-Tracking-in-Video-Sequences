# Configuration loader for pipeline

import yaml
from src.models import Config, ConfigError

REQUIRED_KEYS = {"fine_tuned_checkpoint", "output_dir", "dataset_yaml"}


class ConfigLoader:
    def validate(self, cfg_dict: dict) -> None:
        missing = REQUIRED_KEYS - cfg_dict.keys()
        if missing:
            raise ConfigError(f"Missing required config keys: {', '.join(sorted(missing))}")

    def load(self, path: str) -> Config:
        with open(path) as f:
            cfg_dict = yaml.safe_load(f)
        self.validate(cfg_dict)
        known_fields = {f for f in Config.__dataclass_fields__}
        filtered = {k: v for k, v in cfg_dict.items() if k in known_fields}
        return Config(**filtered)
