from sleeplab_extractor import extract
from sleeplab_format import reader


def test_extract_preprocess(ds_dir, tmp_path, example_config_path):
    orig_ds = reader.read_dataset(ds_dir)
    dst_dir = tmp_path / 'extracted_datasets'
    
    extract.extract(ds_dir, dst_dir, example_config_path)

    extr_ds = reader.read_dataset(dst_dir / 'dataset1')

    assert orig_ds.name == extr_ds.name

    old_subj = orig_ds.series['series1'].subjects['10001']
    new_subj = extr_ds.series['series1'].subjects['10001']

    assert new_subj.sample_arrays['s1_8Hz'].attributes.name == 's1_8Hz'

    old_shape = old_subj.sample_arrays['s1'].values_func().shape
    new_shape = new_subj.sample_arrays['s1_8Hz'].values_func().shape
 
    assert old_shape[0] == 4 * new_shape[0]
