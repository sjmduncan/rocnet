"""Use rocnet.Octree in a PyTorch dataset"""

import glob
import logging
import random
import sys
from os.path import exists, getsize, join, splitext

import laspy as lp
import numpy as np
import toml
import torch
import torch.utils
from numpy import loadtxt, savetxt

import rocnet.utils as utils
from rocnet.octree import Octree, points_to_features

logger = logging.getLogger(__name__)
log_handler_stdout = logging.StreamHandler(sys.stdout)
logger.addHandler(log_handler_stdout)

DEFAULT_METADATA = {
    "type": "tileset",
    "rel_path": "",
    "recurse": False,
    "grid_dim": 64,
    "vox_size": 1.0,
    "train_fraction": 0.85,
    "vox_attrib": "RGB",
}


def load_npy(file_path, scale):
    """Load a .npy file containing a list of occupied voxel grid indices, and optionally byte-encoded RGB colors

    If scale<0 then the scaled indices will be truncated to integer values, and if there is color data then all
    colors with indices that round to the same value will be averaged. In this case the output will be the unique
    rounded indices with their matching average colors.

    file_path: File produced by numpy.save
    scale: scale factor to apply to indices
    """
    pts = np.load(file_path, allow_pickle=True).astype("float32")
    assert pts.shape[1] in [3, 6], "Only occupancy and colour point attributes are supported (3-vectors or 6-vectors in .npy files)"
    pts[:, :3] = pts[:, :3] * scale
    pts[:, 3:] = pts[:, 3:] / 256.0
    if scale < 1.0:
        indices = (pts[:, :3]).astype("int")
        uniques = np.unique(indices, axis=0)
        colors = [np.mean(pts[np.all(indices == u, axis=1), 3:], axis=0) for u in uniques]
        pts = np.concat([uniques, colors], axis=1)
    return pts


def load_laz_as_voxel_indices(filepath, vox_size):
    """Load a .laz file, round to the voxel size, and return one index for each occupied voxel. Attributes are not returned."""
    laz = lp.read(filepath)
    return np.unique(laz.xyz // vox_size, axis=0)


def load_points(file_path, scale=1.0, vox_size=None):
    ext = splitext(file_path)[1]
    if ext in [".laz", ".las"]:
        return load_laz_as_voxel_indices(file_path, vox_size)
    elif ext == ".npy":
        return load_npy(file_path, scale)
    else:
        raise ValueError(f"File type not understood. Extension is {ext}, should be .las, .laz, or .npy")


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
        if max_samples:
            files = random.sample(all_files, min(int(max_samples), len(all_files)))
            if file_list is not None and not exists(file_list) and max_samples == len(files):
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
        logger.info(f"Loading dataset metadata file: {join(folder, 'meta.toml')}, train={train}")
        self.metadata = utils.ensure_file(join(folder, "meta.toml"), DEFAULT_METADATA)  # , True, True)
        if "grid_dim" in self.metadata and self.metadata.grid_dim < model_grid_dim:
            raise ValueError(f"Dataset grid_dim check failed: expected dataset.grid_dim <= model.grid_dim (found dataset.grid_dim={self.metadata.grid_dim}, model.grid_dim={model_grid_dim})")
        elif "grid_dim" in self.metadata and self.metadata.grid_dim > model_grid_dim:
            self.grid_div = self.metadata.grid_dim / model_grid_dim

        logger.info(f"grid_div={self.grid_div}")
        self.max_samples = max_samples
        logger.info(f"max_samples={self.max_samples}")

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
        logger.info(f"Init 'tileset' dataset with {len(self.files)} files {self.prefix}")

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
        logger.info(f"Init 'lazset' dataset with {len(self.files)} files {self.prefix}")

    def load(self, grid_dim, leaf_dim):
        self.read_files(grid_dim, leaf_dim)

    def read_files(self, grid_dim, leaf_dim):
        logger.info(f"Loading {len(self.files)} files")
        print_mod = int(len(self.files) / 10) + 1
        if self.metadata.type == "tileset":
            for idx, f in enumerate(self.files):
                if idx % print_mod == 0:
                    utils._load_resourceutilization(idx, len(self.files))
                indices = load_npy(f, 1.0 / self.grid_div)
                features, labels = points_to_features(indices, grid_dim, leaf_dim, indices.shape[1] - 3)
                tree = Octree(features.float(), labels.int())
                self.trees.append(tree)
        else:
            for idx, f in enumerate(self.files):
                if idx % print_mod == 0:
                    utils._load_resourceutilization(idx, len(self.files))
                indices = load_laz_as_voxel_indices(f, vox_size=self.metadata.vox_size)
                features, labels = points_to_features(indices, grid_dim, leaf_dim)
                tree = Octree(features.float(), labels.int())
                self.trees.append(tree)

    def __getitem__(self, index):
        return self.trees[index]

    def __len__(self):
        return len(self.trees)
