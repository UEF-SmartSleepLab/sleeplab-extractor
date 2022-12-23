import pytest

from pathlib import Path
from sleeplab_format.test_utils.fixtures import *


@pytest.fixture(scope='session')
def example_config_path():
    data_dir = Path(__file__).parent / 'data'
    return data_dir / 'example_config.yml'
