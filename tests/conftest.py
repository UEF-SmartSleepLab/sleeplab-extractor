import pytest

from pathlib import Path
from sleeplab_format.models import *


@pytest.fixture(scope='session')
def example_config_path():
    data_dir = Path(__file__).parent / 'data'
    return data_dir / 'example_config.yml'


@pytest.fixture(scope='session')
def subject_ids():
    return ['10001', '10002', '10003']


@pytest.fixture
def subjects(subject_ids):
    subjs = {}
    for sid in subject_ids:
        metadata = SubjectMetadata(**subject_metadata(sid))
        
        dict_arrays = sample_arrays()
        arrays = {
            k: SampleArray(
                attributes=ArrayAttributes(**v['attributes']),
                values_func=lambda v=v: v['values'])
            for k, v in dict_arrays.items()
        }

        subjs[sid] = Subject(
            metadata=metadata,
            sample_arrays=arrays,
            annotations=None,
            study_logs=None)

    return subjs


@pytest.fixture
def series(subjects):
    series = Series(
        name='series1',
        subjects=subjects
    )
    return {'series1': series}


@pytest.fixture
def dataset(series):
    dataset = Dataset(
        name='dataset1',
        series=series
    )
    return dataset
