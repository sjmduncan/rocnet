"""Some stuff for handling configuration for datasets/models/training runs/preprocessing steps"""

import collections
import copy
import glob
import logging
import sys
from os import makedirs
from os.path import exists, join, normpath, split

import numpy as np
import toml
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

    int/flouat
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
            logger.error(f"Type mismatch: {extant}({type(extant)} != {schema}({type(schema)})")
            return False

    return _compare_schema(schema, extant)


def deep_dict_compare_schema_values(ref, extant):
    if not deep_dict_compare_schema(ref, extant):
        return False

    def _compare_values(schema, extant):
        if isinstance(schema, dict):
            return all(_compare_values(schema[k], extant[k]) for k in schema)
        if isinstance(schema, list):
            return all(_compare_values(schema[idx], extant[idx]) for idx in range(len(schema)))
        return schema == extant

    return _compare_values(ref, extant)


def write_file(path: str, data: dict, overwrite_ok: bool = False):
    if exists(path) and not overwrite_ok:
        raise FileExistsError(f"File exists bu overwrite_ok is false: {path}")
    with open(path, "w") as f:
        toml.dump(data, f)


def ensure_file(path: str, default_config: dict):
    """
    Ensure that TOML config file exists at 'path', that any existing config file matches the schema of 'default_config' (see deep_dict_compare_schema)
    and will return a dict with default values from 'default_config' in place of any keys missing from the config in the file
    path: path to the expected config file.
    default_config: dict object with the full default configuration schema."""
    parent_path, _ = split(path)
    if not exists(parent_path) and not parent_path == "":
        logger.warning(f"Directory does not exist. Creating it: '{parent_path}'")
        makedirs(parent_path, exist_ok=True)
    if not exists(path):
        logger.warning(f"Path does not exist. Creating it: '{path}'")
        with open(path, "x") as file:
            toml.dump(default_config, file)
            return EasyDict(default_config)
    else:
        logger.info(f"Loading config file {path}")
        with open(path, "r") as file:
            extant_config = toml.load(file)
            if not deep_dict_compare_schema(default_config, extant_config):
                logger.error("Config file does not match the expected structure")
                raise ValueError("Config file does not match the expected structure")
            return EasyDict(deep_dict_merge(extant_config, default_config, override=False))


def save_file(path: str, data: dict, overwrite: bool):
    if exists(path) and not overwrite:
        raise FileExistsError("File already exists, but overwrite is set to false")

    with open(path, "w") as file:
        toml.dump(data, file)


def load_file(path: str, default_config: dict = None, quiet=False):
    if not exists(path):
        logger.warning(f"Path does not exist: '{path}'")
        raise FileNotFoundError(f"Path does not exist: '{path}'")
    else:
        if not quiet:
            logger.info(f"Loading config file {path}")
        with open(path, "r") as file:
            return EasyDict(deep_dict_merge(toml.load(file), default_config if default_config is not None else {}, override=False))


def get_most_epoch_model(base_path, suffix, require=True, max_epochs=-1):
    prefix = normpath(base_path)
    files = glob.glob(base_path + "*" + suffix)
    epochs = [int(normpath(s).split(prefix)[1].split(suffix)[0]) for s in files]
    if max_epochs > 0:
        epochs = [e for e in epochs if e <= max_epochs]
    if epochs == [] and require:
        msg = f"Model required but file not found (max_epoch={max_epochs}), glob='{base_path}*{suffix}'"
        logger.info(msg)
        raise FileNotFoundError(msg)
    if epochs == []:
        return 0, f"{join(split(base_path)[0], 'model.toml')}"
    return max(epochs), f"{base_path}{str(max(epochs))}.pth"


# This cell will print out a list of lists of convolutional layers which will
# go from a given in_dim to a given out_dim within a max number of layers and with some max kernel size/padding/stride settings
# This is based on the equations from PyTorch documentation: https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html#conv3d
def layer_config(in_dim, out_dim, max_layers, max_ksize=4, max_pad=2, max_stride=2):
    """
    in_dim: this is the leaf_dim in rocnet (one edge of a cubic volume)
    out_dim: dimensions of the node in rocnet (also one edge of a cubic volume)
    max_layers: maximum allowable convolutional layers
    max_ksize: maximum kernel size for any convolutional layer
    max_pad: maximum padding for any convolutional layer
    max_stride: maximum stride for any convolutional layer

    Output is in the form of a tree, see print_tree for how this is interpreted.
    """
    ksize = range(1, max_ksize + 1)
    pad = range(0, max_pad + 1)
    stride = range(1, max_stride + 1)

    def next_layer(lvl_in, k_in, p_in, s_in, level):
        children = []
        for k in ksize:
            for p in pad:
                for s in stride:
                    DILATION = 1.0
                    lvl_out = (lvl_in + 2 * p - DILATION * (k - 1) - 1) / s + 1
                    if not float(lvl_out).is_integer():
                        pass
                    elif lvl_out < out_dim or lvl_out >= lvl_in:
                        pass
                    elif level + 1 < max_layers:
                        children.append(next_layer(lvl_out, k, p, s, level + 1))
                    elif level + 1 == max_layers and lvl_out == out_dim:
                        children.append(next_layer(lvl_out, k, p, s, level + 1))
        return {"in": lvl_in, "level": level, "k": k_in, "p": p_in, "s": s_in, "children": children}

    return next_layer(in_dim, 0, 0, 0, 0)


def print_tree(tree, _prefix=[]):
    """
      Print a tree produced by the layer_config function.
      This is recursive, leave the _prefix argument alone when calling this.
      Output is printed sequence of arrays, where each array defines a set of convolution
      parameters which start at leaf_dim (the length of one edge of a leaf node in voxels)
      and end at whatever dimension the internal rocnet node size is (in this case 4)
    [[ 0. 16.  0.  0.  0.] <- init condition (leaf_dim=16 === in_dim of next layer)
     [ 1. 15.  2.  0.  1.] <- first layer (k=2, p=0, s=1, out_dim=15 === in_dim of next layer)
     [ 2.  8.  1.  0.  2.] <- second layer (k=1, p=1, s=2, out_dim=8 === in_dim of next layer)
     [ 3.  4.  2.  0.  2.]] <- third layer (k=2, p=0, s=2, out_dim=4 === input to the node encoder)
    [[ 0. 16.  0.  0.  0.] <- init condition for the next possible configuration
     [ 1.  8.  2.  0.  2.]
     [ 2.  7.  2.  0.  1.]
     ...
    """
    lvl = [tree["level"], tree["in"], tree["k"], tree["p"], tree["s"]]
    for c in tree["children"]:
        print_tree(c, _prefix + lvl)
    if len(tree["children"]) == 0 and tree["in"] == 4:
        print(np.array(_prefix + lvl).reshape(-1, 5))


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
