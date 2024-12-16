"""Some stuff for handling configuration for datasets/models/training runs/preprocessing steps"""

import collections
import copy
import logging
import sys
from os import makedirs
from os.path import exists, split

import psutil
import toml
import torch
from easydict import EasyDict

logger = logging.getLogger(__name__)
log_handler_stdout = logging.StreamHandler(sys.stdout)
logger.addHandler(log_handler_stdout)


def is_pow_two(x: int):
    return x and (not (x & (x - 1)))


def deep_dict_merge(dct1, dct2, override=True) -> dict:
    """
    dct1: First dict to merge
    dct2: Second dict to merge
    override: on conflict dict2.key overrides dict1.key (default=True)
    return: The merge dictionary

    https://stackoverflow.com/a/46020972
    Modified to keep the exact contents of lists with matching keys, where the original would 'merge' lists by concatenating them
    """
    merged = copy.deepcopy(dct1)
    for k, v2 in dct2.items():
        if k in merged:
            v1 = merged[k]
            if isinstance(v1, dict) and isinstance(v2, collections.abc.Mapping):
                merged[k] = deep_dict_merge(v1, v2, override)
            else:
                if override:
                    merged[k] = copy.deepcopy(v2)
        else:
            merged[k] = copy.deepcopy(v2)
    return merged


def deep_dict_compare_schema(schema: dict, extant: dict):
    """
    Check that the structure & types of 'extant_dict' match a sub-set of 'schema'

    schema: expected structure value-inferred types
    extant: dict object to check
    return: true if all keys in extant are present in schema and have matching inferred types. see below for treatment of lists/floats

    list
      if either list is empty    --> True
      if lists are the same size --> all elements must match types
      if lists are diff sizes    --> all elements of the shorter list must match types in the sublist of the longer one

    int/float
      schema:float, extant:int   --> True  (type promotion)
      schema:int, extant:float   --> False (avoids loss of precision)

    Dervied from here: https://stackoverflow.com/a/45812573
    Modified to use the inferred types of the values in both schema and extant_dict
    """

    def _compare_schema(schema, extant):
        if isinstance(schema, dict) and isinstance(extant, dict):
            if (extant.keys() & schema.keys()) != extant.keys():
                logger.error(f"Schema mismatch - extra key(s): {', '.join(extant.keys() - schema.keys())}")
                return False
            return all(k in schema and _compare_schema(schema[k], extant[k]) for k in extant)
        if isinstance(schema, list) and isinstance(extant, list):
            if len(schema) != 0 and len(extant) != 0:
                return all(_compare_schema(schema[idx], extant[idx]) for idx in range(min(len(schema), len(extant))))
            return True
        if isinstance(extant, int) and isinstance(schema, float):
            return True
        elif isinstance(extant, type(schema)):
            return True
        else:
            logger.warning(f"Type mismatch type(extant)!=type(ref): {extant}({type(extant)} != {schema}({type(schema)})")
            return False

    return _compare_schema(schema, extant)


def deep_dict_compare_schema_values(ref, extant):
    if not deep_dict_compare_schema(ref, extant):
        return False

    def _compare_values(k, ref, extant):
        if isinstance(ref, dict):
            return all(_compare_values(k, ref[k], extant[k]) for k in ref)
        if isinstance(ref, list):
            return all(_compare_values(f"{k}[{idx}]", ref[idx], extant[idx]) for idx in range(len(ref)))

        if ref != extant:
            logger.warning(f"Value mismatch extant!=ref:  {extant} != {ref}, key='{k}'")
        return ref == extant

    return _compare_values("root", ref, extant)


def ensure_file(path: str, default_dict: dict):
    """
    Ensure that TOML config file exists at 'path', that any existing config file matches the schema of 'default_dict' (see deep_dict_compare_schema)
    and will return a dict with default values from 'default_dict' in place of any keys missing from the config in the file
    path: path to the expected config file.
    default_dict: dict object with the full default configuration schema."""
    parent_path, _ = split(path)
    if not exists(parent_path) and not parent_path == "":
        logger.warning(f"Directory does not exist. Creating it: '{parent_path}'")
        makedirs(parent_path, exist_ok=True)
    if not exists(path):
        logger.warning(f"Path does not exist. Creating it: '{path}'")
        with open(path, "x") as file:
            toml.dump(default_dict, file, encoder=toml.TomlNumpyEncoder())
            return EasyDict(default_dict)
    else:
        with open(path, "r") as file:
            extant_config = toml.load(file)
            if not deep_dict_compare_schema(default_dict, extant_config):
                logger.error("Config file does not match the expected structure")
                raise ValueError("Config file does not match the expected structure")
            return EasyDict(deep_dict_merge(extant_config, default_dict, override=False))


def save_file(path: str, data: dict, overwrite: bool):
    if exists(path) and not overwrite:
        raise FileExistsError("File already exists, but overwrite is set to false")

    with open(path, "w") as file:
        toml.dump(data, file, encoder=toml.TomlNumpyEncoder())


def load_file(path: str, default_dict: dict = None, quiet=False, require_exists=True):
    if not exists(path):
        if require_exists or default_dict is None:
            raise FileNotFoundError(f"Path does not exist: '{path}'")
        logger.warning(f"Path does not exist: '{path}'")
        return EasyDict(default_dict)
    else:
        with open(path, "r") as file:
            return EasyDict(deep_dict_merge(toml.load(file), default_dict if default_dict is not None else {}, override=False))


def sizeof_fmt(num, suffix="B", metric=False):
    """https://stackoverflow.com/questions/1094841/get-a-human-readable-version-of-a-file-size#1094933"""
    units = ("", "K", "M", "G", "T", "P", "E", "Z")
    basis = 1000.0
    if not metric:
        units = [u + ("i" if idx > 0 else "") for idx, u in enumerate(units)]
        basis = 1024.0
    for unit in units:
        if abs(num) < basis:
            return f"{num:3.1f}{unit}{suffix}"
        num /= basis
    return f"{num:.1f}Yi{suffix}"


def td_to_txt(tdiff):
    """Returns t1-t0 as EasyDict({'d': ..., 'h': ..., 'm': ..., 's': ... })"""
    mins_secs = divmod(tdiff.days * (60 * 60 * 24) + tdiff.seconds, 60)
    hours_mins = divmod(mins_secs[0], 60)
    days_hours = divmod(hours_mins[0], 24)
    return EasyDict({"d": days_hours[0], "h": days_hours[1], "m": hours_mins[1], "s": mins_secs[1]})


def td_txt(t0, t1):
    d = td_to_txt(t1 - t0)
    return f"{d.d}-{d.h:02}:{d.m:02}:{d.s:02}"


def _load_resourceutilization(idx, total):
    """Print the total CPU and GPU memory use while loading data"""
    cu = torch.cuda.mem_get_info()
    ram = psutil.virtual_memory()
    proc = psutil.Process()

    logger.info(f"file {idx:>6}/{total}, Free Mem: GPU={100*cu[0]/cu[1]:4.1f}% of {sizeof_fmt(cu[1])}, RAM={100-ram.percent:4.2f}% of {sizeof_fmt(ram.total)}")
    logger.info(f"file {idx:>6}/{total}, Mem Usage: GPU=, CPU={sizeof_fmt(proc.memory_info().rss)}")
