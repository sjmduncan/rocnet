"""Use rocnet.Octree in a PyTorch dataset"""

import glob
import logging
import random
from os.path import exists, getsize, join, splitext

import laspy as lp
import numpy as np
import toml
import torch
import torch.utils
from numpy import loadtxt, savetxt
from tqdm import tqdm

import rocnet.utils as utils
from rocnet.octree import Octree, points_to_features

logger = logging.getLogger(__name__)

DEFAULT_METADATA = {
    "type": "tileset",
    "rel_path": "",
    "recurse": False,
    "grid_dim": 64,
    "vox_size": 1.0,
    "vox_attrib": "RGB",
    "files_in": [["", ""]],
    "transforms": [[], []],
}


def load_npy(file_path, scale, grid_dim):
    pts = np.load(file_path, allow_pickle=True)
    return pts.astype("float32")


def load_laz_as_voxel_indices(filepath, vox_size):
    laz = lp.read(filepath)
    return np.unique(laz.xyz // vox_size, axis=0)


def load_points(file_path, grid_dim=None, scale=None, vox_size=None):
    ext = splitext(file_path)[1]
    if ext in [".laz", ".las"]:
        return load_laz_as_voxel_indices(file_path, vox_size)
    elif ext == ".npy":
        return load_npy(file_path, scale, grid_dim)
    else:
        raise ValueError(f"File type not understood. Extension is {ext}, should be one of .las, .laz, or .npy")


def filelist(folder, train=False, max_samples=-1, min_size=-1, file_list=None, recurse=False):
    if file_list is not None and exists(file_list):
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
            if file_list is not None and not exists(file_list):
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

    def __init__(self, folder: str, model_grid_dim: int, train=True, max_samples: int = None, file_list: str = None):
        """Parse dataset metadata without loading all the files into memory (use read_files for thta)

        folder: folder containing meta.toml and/or train/ and test/ subfolders
        model_grid_dim: model input size of the model this data is for
        model_leaf_dim: model leaf size of the model this data is for
        train: set to True for training subset, and False for testing
        max_samples: If this is a positive integer then this is the maximum nuber of samples which will be retrieved for this dataset
        file_list: If max_samples is used a random subset of famples will be used and saved to the path specified here, or if the file already exists then the list of samples in this file will be used
        vox_size: voxel size to use for quantisation, only required for datasets consisting of .laz files
        """
        self.trees = []
        self.leaves = []
        self.grid_div = 1.0
        self.folder = folder
        self.train = train
        self.max_samples = max_samples
        self.metadata = utils.ensure_file(join(folder, "meta.toml"), DEFAULT_METADATA)  # , True, True)
        if "grid_dim" in self.metadata and self.metadata.grid_dim < model_grid_dim:
            raise ValueError(f"Dataset grid_dim check failed: expected dataset.grid_dim <= model.grid_dim (found dataset.grid_dim={self.metadata.grid_dim}, model.grid_dim={model_grid_dim})")
        elif "grid_dim" in self.metadata and self.metadata.grid_dim > model_grid_dim:
            self.grid_div = self.metadata.grid_dim / model_grid_dim

        if self.metadata.type == "tileset":
            self.__init_tileset(file_list)
        elif self.metadata.type == "lazset":
            self.__init_lazset()
        else:
            raise ValueError(f"Dataset type '{self.metadata.type}'not understood")

        assert len(self.files) > 0

    def __init_tileset(self, file_list):
        """Initialise a dataset consisting of tiles divided into train/ and test subsets"""
        self.midfix = "train" if self.train else "test"
        self.prefix = join(self.folder, self.midfix)
        self.files = filelist(self.folder, self.train, self.max_samples, recurse=self.metadata.recurse if "recurse" in self.metadata else False, file_list=file_list)

    def __init_lazset(self):
        """Initialise a dataset consisting of a set of .laz files with train/test subsets defined in meta.toml

        The .laz files might be located in a relative path
        """
        self.midfix = "train" if self.train else "test"
        self.prefix = join(self.folder, self.midfix)
        if self.max_samples is not None:
            self.files = utils.ensure_file(join(self.folder, "file_lists.toml"), {"train": [], "test": []})[self.midfix][: self.max_samples]
        else:
            self.files = utils.ensure_file(join(self.folder, "file_lists.toml"), {"train": [], "test": []})[self.midfix]

    def load(self, grid_dim, leaf_dim):
        self.read_files(grid_dim, leaf_dim)

    def read_files(self, grid_dim, leaf_dim):
        print(f"read_files {len(self.files)}:", end="", flush=True)
        if self.metadata.type == "tileset":
            for f in tqdm(self.files):
                indices = load_npy(f, 1.0 / self.grid_div, grid_dim)
                indices[:, 3:] = indices[:, 3:] / 256
                features, labels = points_to_features(indices, grid_dim, leaf_dim, indices.shape[1] - 3)
                tree = Octree(features.float(), labels.int())
                self.trees.append(tree)
        else:
            for f in tqdm(self.files):
                indices = load_laz_as_voxel_indices(f, vox_size=self.metadata.vox_size)
                features, labels = points_to_features(indices, grid_dim, leaf_dim)
                tree = Octree(features.float(), labels.int())
                self.trees.append(tree)

    def __getitem__(self, index):
        return self.trees[index]

    def __len__(self):
        return len(self.trees)
