import numpy as np
import scipy.signal

from importlib import import_module
from sleeplab_extractor.config import ArrayAction, ArrayConfig, SeriesConfig
from sleeplab_format.models import ArrayAttributes, SampleArray, Series
from typing import Callable


def import_function(func_str: str) -> Callable:
    module_str, func_name = func_str.rsplit('.', maxsplits=1)
    module = import_module(module_str)
    func = getattr(module, func_name)
    return func


def chain_action(
        orig_func: Callable,
        orig_attrs: ArrayAttributes,
        action: ArrayAction) -> Callable:
    """Use a closure to chain orig_func with an action."""
    def inner():
        return _func(orig_func(), orig_attrs, **action.kwargs)

    _func = import_function(action.method)
    return inner


def process_array(arr: SampleArray, cfg: ArrayConfig) -> SampleArray:
    # TODO: Do we even need to use action names or would it be better to
    # require a method and a dict of parameters for it?
    # i. e. have a function add_action(sample_array, cfg), and cfg would contain something like
    # attributes: {sampling_rate: 32} or {cutoff: 0.3}, and it would be responsibility of
    # cfg.method to validate and use the args.
    # This way, any function could be used for preprocessing without modifying sleeplab-extractor source code!
    arr = arr.copy(update={'name': cfg.new_name})
    for action in cfg.actions:
        _values_func = chain_action(arr.values_func, arr.attributes, action)
        _attributes = arr.attributes.copy(updated=cfg.updated_attributes)
        arr = arr.copy(update={'attributes': _attributes, 'values_func': _values_func})

    return arr


def process_series(series: Series, cfg: SeriesConfig) -> Series:
    updated_subjects = {}
    for sid, subj in series.subjects.items():
        _sample_arrays = {}

        for array_cfg in cfg.array_configs:
            if array_cfg.name in subj.sample_arrays.keys():
                _sample_arrays[array_cfg.new_name] = process_array(
                    subj.sample_arrays[array_cfg.name], array_cfg)

        updated_subjects[sid] = subj.copy(update={'sample_arrays': _sample_arrays})

    return series.copy(update={'subjects': updated_subjects})


def is_power_of_two(x: float) -> bool:
    return np.log2(x) % 1 == 0.0


def _decimate(s: np.array, factor: int) -> np.array:
    """Implement decimation of powers of two by consecutive decimation by 2.
    
    If higher factors are used, a lot of noise will be induced in the signals.
    """
    assert is_power_of_two(factor)
    if factor < 4:
        return scipy.signal.decimate(s, factor)
    else:
        return _decimate(decimate(s, 2), factor // 2)


def decimate(
        s: np.array,
        attributes: ArrayAttributes,
        fs_new: float,
        dtype: np.dtype = np.float32) -> np.array:
    ds_factor = int(attributes.sampling_rate // fs_new)
    return _decimate(s, ds_factor).astype(dtype)


def cheby2_highpass_filtfilt(
        s: np.array,
        fs: float,
        cutoff: float,
        order: int = 5,
        rs: float = 40.0) -> np.array:
    """Chebyshev type1 highpass filtering.
    
    Args:
        s: the signal
        fs: sampling freq in Hz
        cutoff: cutoff freq in Hz
    Returns:
        the filtered signal
    """
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    sos = scipy.signal.cheby2(order, rs, norm_cutoff, btype='highpass', output='sos')
    return scipy.signal.sosfiltfilt(sos, s)


def highpass(
        s: np.array,
        attributes: ArrayAttributes,
        cutoff: float) -> np.array:
    return cheby2_highpass_filtfilt(s, attributes.sampling_rate, cutoff)
