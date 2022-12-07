import yaml

from pathlib import Path
from pydantic import BaseModel, Extra


class ArrayAction(BaseModel, extra=Extra.forbid):
    name: str
    method: str
    sampling_rate: str | None = None
    cutoff: float | None = None


class ArrayConfig(BaseModel, extra=Extra.forbid):
    name: str
    new_name: str
    actions: list[ArrayAction] | None = None


class SeriesConfig(BaseModel, extra=Extra.forbid):
    name: str
    array_configs: list[ArrayConfig]


class DatasetConfig(BaseModel, extra=Extra.forbid):
    series_configs: list[SeriesConfig]


def parse_config(config_path: Path) -> DatasetConfig:
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    return DatasetConfig.parse_obj(cfg)