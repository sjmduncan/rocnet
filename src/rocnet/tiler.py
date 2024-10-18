"""
Convert large/multi-file LIDAR datasets to 'tiles' which can be used to train RocNet

Pick the largest grid_dim of 64, 128, or 256,  with a matching vox_size that
preserves surface continuity in the LIDAR grid.
The LIDAR point cloud is first transformed (if a transform/transform list is provided
to produce_dataset or process_file), then quantised to vox_size, and then divided into
cuboid 'tiles'. The tile origins in the horizontal plane are fixed to the voxel grid,
while the vertical origin is determined by the lowest occupied voxel in the tile.
This is because LIDAR scans won't return anythign below ground level, and it maximizes
the possibl occupancy of each voxel grid.

Each tile is an array (dtype=uint8) of the voxel grid indices of the occupied voxels.
Tiles are saved as .npy files, where the file name starts with the global origin of
the tile of the form 'x_y_z_...'

rocnet.data has utilities for loading tiles, and for parsing filenames to parse file
names to recover the voxel origin.

Assumptions:
- tiles are powers of two, probably 64, 128, or 256
- voxel resolution is fixed for a dataset, and for a trained model.
- vertical axis orientation is fixed, and it's the axis with the least extents (i.e. not necessarily the Z-axis)
- where tiles are taller than grid_dim voxels, two vertically stacked tiles will be produced.

Transforms:
- pass in a list of compact transforms of the form [t_x, t_y, t_z, yaw_angle] to produce_dataset or process_file
- only translate the pointcloud in the horizontal plane
- avoid transforms that are powers of two or multiples of the expected model leaf_dim
- include sub-voxel transforms so that the grid is re-sampled
- include at a sub-voxel shift even when you only want to rotate the grid
"""

import copy
import glob
from os import makedirs
from os.path import basename, exists, join

import laspy as lp
import numpy as np
import open3d as o3d

import rocnet.utils as util
from rocnet.data import load_tile_as_pt_array, parse_tile_filename, pc_to_tiles, tile_filename
from rocnet.dataset import DEFAULT_METADATA

DEFAULT_CONFIG = {
    "flat_div": 1.5,
    "clean": True,
    "ratios": [0.7, 0.15, 0.15],  # [train, validate, test]
    "in_dir": "tbd",
    "out_dir_parent": "tbd",
    "dataset": "tbd",
}


def laz_to_points(path: str, clean: bool, save_intermediate: bool) -> o3d.geometry.PointCloud:
    """Load a .laz file, optionally clean outliers and if cleaning optionally save the cleaned file for future use

    Load a .laz file as a point cloud, optionally perform statistical outlier removal.
    If clean=true and save_intermediate=true then the cleaned point cloud is saved with the
    file name '{path}.clean.ply' to avoid having to re-clean the pointcloud in the future.

    path: .laz file to load
    clean: enable outlier cleaning
    save_intermediate:
    """

    cleaned_pts_path = f"{path}.clean.ply"
    if clean and exists(cleaned_pts_path):
        return o3d.io.read_point_cloud(cleaned_pts_path)
    else:
        laz = lp.read(path)
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(laz.xyz))
        if clean:
            _, ind = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

            pc = pc.select_by_index(ind)
            if save_intermediate:
                o3d.io.write_point_cloud(cleaned_pts_path, pc)

        return pc


def tiles_folder_name(path, vox_sz, grid_dim, shift=[0, 0, 0, 0]):
    return f"{path}_{vox_sz}v_{grid_dim}t_{shift[0]}_{shift[1]}_{shift[2]}_{shift[3]}_tiles"


def points_to_tiles(pts: np.array, tiled_pts_path: str, save_intermediate: bool, vox_sz: float, grid_dim: int) -> tuple[list, list]:
    """Load/clean a .laz file, return a list of 'tiles' and their bottom-left corners

    Invokes laz_to_points with path and clean, scales the resulting cloud by 1/vox_sz
    and divides the scaled cloud into cube-shaped 'tiles' with edge lengths of grid_dim.

    To avoid re-computing tiles (e.g. to re-run the tile filter, or to res-sort train/test subsets)
    the the un-sorted tiles can be saved in '{path}_{vox_sz}v_{grid_dim}t_tiles/' with file names
    computed by tile_filename.

    path: the .laz file
    clean: clean flag to pass to laz_to_points
    vox_sz: the voxel size, which should match the point spacing of the .laz file
    grid_dim: the edge length of a cube-shaped tile
    shift: 3D shift vector to apply to the voxel grid before tiling, array of int
    """

    if exists(tiled_pts_path):
        files = glob.glob(join(tiled_pts_path, "*.npy"))
        tile_pts = [load_tile_as_pt_array(f, grid_dim) for f in files]
        tile_bl = [parse_tile_filename(f) for f in files]
        tileset = list(zip(tile_bl, tile_pts))
    else:
        tileset = pc_to_tiles(pts, vox_sz, grid_dim)
        if save_intermediate:
            makedirs(tiled_pts_path, exist_ok=True)
            [save_voxel_tile(join(tiled_pts_path, tile_filename(t[0], ext=".npy")), t[1]) for t in tileset]

    return tileset


def sort_tiles(tileset, test_fraction: float, smallest: int):
    """Sort tileset into training, testing/validation, and small-tile subsets

    First small tiles (those with only very few points) are excluded from the list
    The remaining good tiles are randomly split into training and testing, 'test_fraction'
    fraction of the tiles selected for testing

    tileset: list of (corner, tile) tuples
    test_ratio: fraction of non-small tiles to use for testing. Probably 0.15 or 0.2
    smallest: min number of points for non-small tile
    returns: train, test, small (are lists of (corner, tile))"""

    counted = list(zip([np.prod(t[1].shape) for t in tileset], tileset))

    small = [t[1] for t in counted if t[0] <= smallest]
    notsmall = [t[1] for t in counted if t[0] > smallest]
    n_test = int(test_fraction * len(notsmall))

    test_sel = np.zeros(len(notsmall), dtype=np.uint8)
    test_sel[np.random.choice(range(len(notsmall)), size=n_test, replace=False)] = 1

    train_tiles = [t for idx, t in enumerate(notsmall) if test_sel[idx] == 0]
    test_tiles = [t for idx, t in enumerate(notsmall) if test_sel[idx] == 1]

    return train_tiles, test_tiles, small


def save_voxel_tile(path, pts):
    pts_u8 = np.array(pts).astype("uint8")
    with open(path, "wb") as tile_file:
        np.save(tile_file, pts_u8)


def save_tiles(out_dir: str, suffix: str, train: list[np.array], test: list[np.array], small: list[np.array]):
    """Save the files as .ply files, with one tile per file. This is lossless."""

    def save_set(dirname, tiles):
        if len(tiles) == 0:
            return
        set_dir = join(out_dir, dirname)
        makedirs(set_dir, exist_ok=True)
        [save_voxel_tile(join(set_dir, tile_filename(t[0], suffix)), t[1]) for t in tiles]

    save_set("train", train)
    save_set("test", test)
    save_set("small", small)


def tile_file(path_in: str, out_dir: str, tile_sz: int, vox_sz: float, transforms: list[np.array], test_fraction: float, smallest: int, clean: bool, save_intermediate: bool):
    """Process a single .laz/.las/.ply/.pcd fil"""
    pc = laz_to_points(path_in, clean, save_intermediate)
    rot_axis = np.argmin(pc.get_max_bound() - pc.get_min_bound())
    for tx in transforms:
        tiled_pts_path = tiles_folder_name(path_in, vox_sz, tile_sz, tx)

        pc_tx = copy.deepcopy(pc)

        rot_xyz = [0, 0, 0]
        rot_xyz[rot_axis] = tx[3]
        rot_mat = o3d.geometry.get_rotation_matrix_from_xyz(rot_xyz)

        pc_tx.rotate(rot_mat, pc_tx.get_center())
        pc_tx.translate(tx[:3])

        tileset = points_to_tiles(np.asarray(pc_tx.points), tiled_pts_path, save_intermediate, vox_sz, tile_sz)
        if isinstance(test_fraction, float):
            train, test, small = sort_tiles(tileset, test_fraction, smallest)
            suffix = f"{tx[0]}_{tx[1]}_{tx[2]}_{tx[3]}_{basename(path_in)}"
            save_tiles(out_dir, suffix, train, test, small)
        elif isinstance(test_fraction, str) and test_fraction == "train" or test_fraction == "test":
            t = True if test_fraction == "train" else False
            suffix = f"{tx[0]}_{tx[1]}_{tx[2]}_{tx[3]}_{basename(path_in)}"
            save_tiles(out_dir, suffix, tileset if t else [], [] if t else tileset, [])


def produce_dataset(inputs: str, out_dir: str, tile_sz: int, vox_sz: float, transforms: np.array, smallest: int, test_fraction: float, clean: bool, save_intermediate: bool):
    laz_files_done = []

    for f in inputs:
        tile_file(f, out_dir, tile_sz, vox_sz, transforms, test_fraction, smallest, clean, save_intermediate)
        laz_files_done.append(f)
    meta = copy.deepcopy(DEFAULT_METADATA)
    meta["files_in"] = laz_files_done
    meta["grid_dim"] = tile_sz
    meta["vox_size"] = vox_sz
    meta["transforms"] = transforms
    util.write_file(join(out_dir, "meta.toml"), meta, overwrite_ok=True)
