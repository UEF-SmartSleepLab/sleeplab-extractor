from importlib import import_module
from sleeplab_extractor.config import ArrayConfig, SeriesConfig
from sleeplab_format.models import SampleArray, Series


def import_function(func_str):
    module_str, func_name = func_str.rsplit('.', maxsplits=1)
    module = import_module(module_str)
    func = getattr(module, func_name)
    return func


def add_resampling(values_func, cfg):
    """Use a closure/decorator to add resampling to values_func."""
    _func = import_function(cfg.method)



def process_array(arr: SampleArray, cfg: ArrayConfig) -> SampleArray:
    # TODO: Do we even need to use action names or would it be better to
    # require a method and a dict of parameters for it?
    # i. e. have a function add_action(sample_array, cfg), and cfg would contain something like
    # attributes: {sampling_rate: 32} or {cutoff: 0.3}, and it would be responsibility of
    # cfg.method to validate and use the args.
    # This way, any function could be used for preprocessing without modifying sleeplab-extractor source code!
    for action in cfg.actions:
        if action == 'resample':
            arr.values_func = add_resampling(arr.values_func, cfg)
            arr.attributes.sampling_rate = cfg.sampling_rate
        else:
            raise AttributeError(f'process_array: unknown action {action}')

    arr.attributes.name = cfg.new_name
    return arr


def process_series(series: Series, cfg: SeriesConfig) -> Series:
    for _, subj in series.subjects.items():
        _sample_arrays = {}

        for array_cfg in cfg.array_configs:
            if array_cfg.name in subj.sample_arrays.keys():
                _sample_arrays[array_cfg.new_name] = process_array(
                    subj.sample_arrays[array_cfg.name], array_cfg)

        # Substitute old sample arrays with processed
        subj.sample_arrays = _sample_arrays

    return series
