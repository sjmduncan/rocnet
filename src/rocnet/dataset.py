"""Use rocnet.Octree in a PyTorch dataset"""

import glob
import logging
import random
from os.path import exists, getsize, join

import toml
import torch
import torch.utils
from easydict import EasyDict as ed
from numpy import loadtxt, savetxt

import rocnet.data
from rocnet.octree import Octree

logger = logging.getLogger(__name__)

DEFAULT_METADATA = {"recurse": False, "grid_dim": 64, "vox_size": 1.0, "files_in": [""], "transforms": [[], []]}


def filelist(folder, train=False, max_samples=-1, min_size=-1, file_list="", recurse=False):
    if file_list != "" and exists(file_list):
        files = loadtxt(file_list, dtype=str)
        if max_samples > 0 and max_samples != len(files):
            raise ValueError(f"Both file_list (n={len(files)}) and max_samples={max_samples} provided, but the number of files does not match.")
        return files
    else:
        midfix = "train" if train else "test"
        all_files = glob.glob(join(folder, "*", midfix, "*.npy")) if recurse else glob.glob(join(folder, midfix, "*.npy"))
        if min_size > 0:
            all_files = [f for f in all_files if getsize(f) > min_size]
        if max_samples > 0:
            files = random.sample(all_files, min(int(max_samples), len(all_files)))
            if file_list != "" and not exists(file_list):
                savetxt(file_list, files, fmt="%s")
            return files
        return all_files


def write_metadata(out_dir: str, meta: dict):
    """Write metadata to the dataset dir, the only required key in meta is 'grid_dim'"""
    if not exists(join(out_dir, "meta.toml")):
        with open(join(out_dir, "meta.toml"), "w") as f:
            toml.dump(meta, f)
    else:
        logger.warning(f"File already exists: {join(out_dir, 'meta.toml')}")


class Dataset(torch.utils.data.Dataset):
    """Access a collection of Octrees as a torch dataset"""

    def __init__(self, folder: str, model_grid_dim: int, train=True, max_samples: int = -1, file_list: str = ""):
        """Construct a dataset and load all of the members into memory
        folder: top-level folder containing 'train' and 'test' subfolders containing the respective subsets
        leaf_dim: edge length of a cube of voxels at the maximum depth of the octree
        grid_dim: edge length of the voxel grid to construct from the data
        train: set to True for training subset, and False for testing
        centre: indicate whether to centre the models on the voxel grid or not
        max_samples: If this is a positive integer then this is the maximum nuber of samples which will be retrieved for this dataset
        recurse: set to true for datasets which are split into categories (e.g. ModelNet)
        """
        self.trees = []
        self.leaves = []
        self.midfix = "train" if train else "test"
        self.prefix = join(folder, self.midfix)
        if exists(join(folder, "meta.toml")):
            with open(join(folder, "meta.toml")) as f:
                self.metadata = toml.load(f)
            self.metadata = ed(self.metadata)
            if "grid_dim" in self.metadata and self.metadata.grid_dim < model_grid_dim:
                raise ValueError(f"Dataset grid_dim check failed: expected dataset.grid_dim <= model.grid_dim (found dataset.grid_dim={self.metadata.grid_dim}, model.grid_dim={model_grid_dim})")
            elif "grid_dim" in self.metadata and self.metadata.grid_dim > model_grid_dim:
                self.grid_div = self.metadata.grid_dim / model_grid_dim
            else:
                self.grid_div = None
        else:
            raise FileNotFoundError(f"metadata file not found: {join(folder, 'meta.toml')}")
        self.files = filelist(folder, train, max_samples, recurse=self.metadata.recurse if "recurse" in self.metadata else False, file_list=file_list)
        self.max_samples = max_samples
        assert len(self.files) > 0

    def load(self, grid_dim, leaf_dim):
        self.read_files(grid_dim, leaf_dim)

    def read_files(self, grid_dim, leaf_dim):
        print(f"read_files {len(self.files)}:", end="", flush=True)
        incr_modulo = max(len(self.files) // 10, 1)
        for idx, f in enumerate(self.files):
            grid = rocnet.data.load_as_occupancy(f, grid_dim, scale=1.0 / self.grid_div if self.grid_div is not None else None)
            features, labels = rocnet.data.occupancy_to_features(grid, leaf_dim)
            tree = Octree(features.float(), labels.int())
            self.trees.append(tree)
            if (idx % incr_modulo == 0) or idx == len(self.files) - 1:
                print(f" {idx + 1}", end="", flush=True)
        print("")

    def get_fpath(self, index):
        return self.files[index]

    def __getitem__(self, index):
        return self.trees[index]

    def __len__(self):
        return len(self.trees)
