"""CLI for extracting and preprocessing a subset of data in sleeplab format."""
import argparse
import logging

from pathlib import Path
from sleeplab_extractor import config, preprocess
from sleeplab_format import reader, writer


logger = logging.getLogger(__name__)


def extract(src_dir: Path, dst_dir: Path, config_path: Path) -> None:
    """Read, preprocess, and write data in sleeplab format.""" 
    logger.info(f'Reading config from {config_path}')
    cfg = config.parse_config(config_path)
    
    logger.info(f'Reading dataset from {src_dir}')
    series_names = [series_config.name for series_config in cfg.series_configs]
    ds = reader.read_dataset(src_dir, series_names=series_names)
    
    updated_series = {}

    for series_config in cfg.series_configs:
        logger.info(f'Creating updated series {series_config.name}')
        _series = preprocess.process_series(ds.series[series_config.name], series_config)
        updated_series[series_config.name] = _series
    
    logger.info('Creating updated Dataset')
    ds = ds.copy(update={'series': updated_series})

    logger.info(f'Applying preprocessing and writing dataset to {dst_dir}')
    writer.write_dataset(ds, dst_dir)


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--src_dir', required=True)
    parser.add_argument('-d', '--dst_dir', required=True)
    parser.add_argument('-c', '--config_path', required=True)

    return parser


def run_cli():
    parser = get_parser()
    args = parser.parse_args()
    extract(
        Path(args.src_dir),
        Path(args.dst_dir),
        Path(args.config_path)
    )


if __name__ == '__main__':
    run_cli()
