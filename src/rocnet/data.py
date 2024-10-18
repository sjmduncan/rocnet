"""Utility functions for parsing various input data for use with the model"""

from os.path import basename, exists, join, splitext

import numpy as np
import open3d as o3d
import torch

from rocnet.octree import Octree


def load_model_as_voxel_grid(file_path, grid_dim):
    """Load a model, scale it to fit a cube of size grid_dim, translate it so it's in the grid_dim cube in the positive octant, and return it as an open3d voxelgrid"""
    if not exists(file_path):
        raise FileNotFoundError(f"File doesn't exists: {file_path}")
    _, ext = splitext(file_path)
    if ext in [".off", ".obj", ".stl"]:
        model = o3d.io.read_triangle_mesh(file_path)
    elif ext in [".xyz", ".ply", ".pcd"]:
        model = o3d.io.read_point_cloud(file_path)
    else:
        raise Exception("Path should point to stl, obj, off, xyz, pcd, or ply file")
    scale = (grid_dim - 1) / np.max(model.get_max_bound() - model.get_min_bound())
    model.scale(scale, center=model.get_center())

    model.translate(-model.get_min_bound())

    model_middle = (model.get_max_bound() - model.get_min_bound()) / 2
    grid_middle = [grid_dim / 2, grid_dim / 2, grid_dim / 2]
    model.translate(grid_middle - model_middle)

    if isinstance(model, o3d.geometry.PointCloud):
        return o3d.geometry.VoxelGrid.create_from_point_cloud(model, voxel_size=1)
    elif isinstance(model, o3d.geometry.TriangleMesh):
        return o3d.geometry.VoxelGrid.create_from_triangle_mesh(model, voxel_size=1)


def tile_filename(bottom_left: np._typing.ArrayLike, suffix: str = "", ext: str = ".npy") -> str:
    """Create a filename of the form x_y_z_{suffix}{ext} where x,y,z are coords of bottom_left"""
    return f"{bottom_left[0]}_{bottom_left[1]}_{bottom_left[2]}_{suffix}{ext}"


def parse_tile_filename(file_path: str) -> np.array:
    """parse the file name of the form x_y_z_* to retrieve the coord. Raises ValueError if filename can't be parsed"""
    try:
        return np.float64(splitext(basename(file_path))[0].split("_")[:3])
    except ValueError:
        raise ValueError(f"file_path could not be parsed to 3D vector: {file_path}")


def pc_to_tiles(pts: np.array, vox_size: float, tile_grid_dim: int):
    """Quantise and tile a pointcloud to uint8 tile occpancy grids

    Quantise and deduplicate a pointcloud to vox_size, divide the result into cube-shaped tiles
    of size tile_grid_dim in voxels.

    pts: array of points
    vox_size: voxel size, vox_index = pts[idx] // vox_size
    tile_grid_dim: size of each tile in voxels, must be 64, 128, or 256
    shift: shift: 3D shift vector to apply to the voxel grid before tiling, array of int
    returns: list(corners, tiles)
              corners - grid indices of the bottom-left corners of the tiles (array, dtype=int)
              tiles - indices of occupied voxels for this tile (array dtype=uint8)
    """
    assert tile_grid_dim in [64, 128, 256]
    grid_pts = np.unique((pts // vox_size).astype(int), axis=0)
    tile_stack_idx = (grid_pts[:, :2] // tile_grid_dim).astype(int)
    tile_stack_corners = np.unique(tile_stack_idx, axis=0)
    tiles = [grid_pts[np.all(tile_stack_idx == corner, axis=1)] for corner in tile_stack_corners]

    tile_bottoms = [np.min(t[:, 2]) for t in tiles]
    corners_grid = [np.concatenate([tile_grid_dim * z[0], [z[1]]]) for z in zip(tile_stack_corners, tile_bottoms)]
    tiles = [z[0] - z[1] for z in zip(tiles, corners_grid)]
    corners_world = [vox_size * c for c in corners_grid]
    ## FIXME: deal with tall tiles properly.
    tiles = [t[t[:, 2] < tile_grid_dim] for t in tiles]

    pmax = np.max([np.max(t) for t in tiles])
    pmin = np.min([np.min(t) for t in tiles])
    assert pmax < tile_grid_dim and pmin >= 0

    return list(zip(corners_world, [t.astype("uint8") for t in tiles]))


def tiles_to_pc(tileset, vox_size: float) -> np.array:
    """Recover the full pointcloud from a list of tile corners and uint8 tile occupancy grids

    tileset: list(corners, tiles), corners is array of int, tiles is array of uint8
    returns: quantisd and deduplicated pointcloud
    """
    grid_pts = np.concatenate([t[0].astype(int) + t[1] for t in tileset])
    return grid_pts * vox_size


def load_tile_as_pt_array(file_path, grid_dim, scale: int = None):
    """Load a pre-quantised tile created by tiler.ipynb, check that it fits within grid_dim. Returns the bottom-left corner (meters), and the grid indices of the occupied voxels in the tile"""
    if not exists(file_path):
        raise FileNotFoundError(f"File doesn't exists: {file_path}")
    pts = np.load(file_path, allow_pickle=True)
    if scale is not None:
        pts = np.unique((pts * scale) // 1, axis=0).astype("int")
    return pts[pts[:, 2] < grid_dim]


def load_tile_as_pt_array_with_pos(file_path, grid_dim, vox_size, scale: int = None):
    """Load a pre-quantised tile created by tiler.ipynb, check that it fits within grid_dim, scale by vox_size, recover world-pos of bottom-left corner, and translate the points to their world position before returning"""
    bl = parse_tile_filename(file_path)
    pts = load_tile_as_pt_array(file_path, grid_dim, scale)
    return bl + pts.astype("float64") * vox_size


def load_as_voxelgrid(file_path, grid_dim, scale: int = None):
    """load a mesh, pointcloud, or tile, and return it as an open3d voxelgrid"""
    if splitext(file_path)[1] == ".npy":
        pts = load_tile_as_pt_array(file_path, grid_dim, scale)
        pointcloud = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pts))
        return o3d.geometry.VoxelGrid.create_from_point_cloud(pointcloud, voxel_size=1)
    return load_model_as_voxel_grid(file_path, grid_dim)


def load_as_pointcloud(file_path, grid_dim, scale: int = None):
    """load a mesh, pointcloud, or tile, and return it as an open3d pointcloud"""
    if splitext(file_path)[1] == ".npy":
        pts = 1.0 * load_tile_as_pt_array(file_path, grid_dim, scale)
    else:
        voxels = load_model_as_voxel_grid(file_path, grid_dim)
        pts = np.array([voxels.get_voxel_center_coordinate(p.grid_index) for p in voxels.get_voxels()])

    return o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pts))


def load_as_occupancy(file_path, grid_dim, as_tensor=True, scale: int = None):
    """Load a mesh, pointcloud, or tile, and return it as a"""
    if splitext(file_path)[1] == ".npy":
        pts = load_tile_as_pt_array(file_path, grid_dim, scale)
    else:
        pts = np.asarray(load_as_pointcloud(file_path, grid_dim).points)
    occupancy = np.zeros([grid_dim, grid_dim, grid_dim], dtype=np.int8)
    occupancy[pts[:, 0], pts[:, 1], pts[:, 2]] = 1
    if as_tensor:
        return torch.from_numpy(occupancy)
    else:
        return occupancy


def occupancy_to_voxelgrid(occupancy):
    """Convert an occupancy grid back to an open3D voxelgrid"""
    pts_arr = np.nonzero(occupancy)
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(pts_arr)
    return o3d.geometry.VoxelGrid.create_from_point_cloud(pointcloud, voxel_size=1)


def save(path, pts):
    """Save a point cloud"""
    pc2 = o3d.geometry.PointCloud()
    pc2.points = o3d.utility.Vector3dVector(np.asarray(pts))
    o3d.io.write_point_cloud(path, pc2)


def occupancy_to_features(occupancy_grid: torch.Tensor, leaf_size):
    """
    Return depth-first post-ordered list of node features and node types
    """
    node_features = []

    def get_node_and_type(subgrid):
        """Recursively get node feature and node subtype for all non-empty nodes"""
        subsize = subgrid.shape[0]
        if subsize < leaf_size:
            raise ValueError("Grid dim must be bigger than leaf size")
        if torch.all(subgrid != 0):
            node_features.append(subgrid)
            return torch.tensor([Octree.NodeType.LEAF_FULL.value])
        if torch.all(subgrid == 0):
            # Empty leaves might not be at the max depth, and so subgrid size might
            # not be leaf_size^3. So use torch.zeros instead.
            node_features.append(torch.zeros([leaf_size, leaf_size, leaf_size]))
            return torch.tensor([Octree.NodeType.LEAF_EMPTY.value])
        if subsize == leaf_size:
            node_features.append(subgrid)
            return torch.tensor([Octree.NodeType.LEAF_MIX.value])
        mgs = subsize // 2
        return torch.cat(
            [
                get_node_and_type(subgrid[:mgs, :mgs, :mgs]),
                get_node_and_type(subgrid[mgs:, :mgs, :mgs]),
                get_node_and_type(subgrid[:mgs, mgs:, :mgs]),
                get_node_and_type(subgrid[mgs:, mgs:, :mgs]),
                get_node_and_type(subgrid[:mgs, :mgs, mgs:]),
                get_node_and_type(subgrid[mgs:, :mgs, mgs:]),
                get_node_and_type(subgrid[:mgs, mgs:, mgs:]),
                get_node_and_type(subgrid[mgs:, mgs:, mgs:]),
                torch.tensor([Octree.NodeType.NON_LEAF.value]),
            ],
            axis=0,
        )

    node_types = get_node_and_type(occupancy_grid)
    return torch.stack(node_features).type(torch.float32), node_types


def get_mixed_leaves(occupancy_grid, leaf_size):
    """
    Return depth-first post-ordered list of node features and node types
    """
    node_features = []

    def get_node_and_type(subgrid):
        """Recursively get node feature and node subtype for all non-empty nodes"""
        subsize = subgrid.shape[0]
        if subsize < leaf_size:
            raise ValueError("Grid dim must be bigger than leaf size")
        if torch.all(subgrid != 0):
            # node_features.append(subgrid)
            return torch.tensor([Octree.NodeType.LEAF_FULL.value])
        if torch.all(subgrid == 0):
            # Empty leaves might not be at the max depth, and so subgrid size might
            # not be leaf_size^3. So use torch.zeros instead.
            # node_features.append(torch.zeros([leaf_size, leaf_size, leaf_size]))
            return torch.tensor([Octree.NodeType.LEAF_EMPTY.value])
        if subsize == leaf_size:
            node_features.append(subgrid.to(torch.float32).unsqueeze(0).unsqueeze(0))
            return torch.tensor([Octree.NodeType.LEAF_MIX.value])
        mgs = subsize // 2
        return torch.cat(
            [
                get_node_and_type(subgrid[:mgs, :mgs, :mgs]),
                get_node_and_type(subgrid[mgs:, :mgs, :mgs]),
                get_node_and_type(subgrid[:mgs, mgs:, :mgs]),
                get_node_and_type(subgrid[mgs:, mgs:, :mgs]),
                get_node_and_type(subgrid[:mgs, :mgs, mgs:]),
                get_node_and_type(subgrid[mgs:, :mgs, mgs:]),
                get_node_and_type(subgrid[:mgs, mgs:, mgs:]),
                get_node_and_type(subgrid[mgs:, mgs:, mgs:]),
                torch.tensor([Octree.NodeType.NON_LEAF.value]),
            ],
            axis=0,
        )

    _ = get_node_and_type(occupancy_grid)
    return node_features


def features_to_occupancy(node_features, node_types, subgrid_grid_dim):
    """
    Decode a depth-first post-ordered list of node features and node types
    return an occupancy grid of size grid_dim^3
    """
    ogrid = torch.zeros([subgrid_grid_dim, subgrid_grid_dim, subgrid_grid_dim])
    if node_types[-1] == Octree.NodeType.LEAF_FULL.value:
        ogrid = 1
        node_types = node_types[:-1]
        node_features = node_features[:-1, :, :, :]
        return ogrid, node_types, node_features

    elif node_types[-1] == Octree.NodeType.LEAF_EMPTY.value:
        ogrid = 0
        node_types = node_types[:-1]
        node_features = node_features[:-1, :, :, :]
        return ogrid, node_types, node_features

    elif node_types[-1] == Octree.NodeType.LEAF_MIX.value:
        ogrid = node_features[-1, :, :, :]
        node_types = node_types[:-1]
        node_features = node_features[:-1, :, :, :]
        return ogrid, node_types, node_features
    else:
        node_types = node_types[:-1]
        mgs = subgrid_grid_dim // 2
        c1, node_types, node_features = features_to_occupancy(node_features, node_types, mgs)  # 0
        c2, node_types, node_features = features_to_occupancy(node_features, node_types, mgs)  # 1
        c3, node_types, node_features = features_to_occupancy(node_features, node_types, mgs)  # 2
        c4, node_types, node_features = features_to_occupancy(node_features, node_types, mgs)  # 3
        c5, node_types, node_features = features_to_occupancy(node_features, node_types, mgs)  # 4
        c6, node_types, node_features = features_to_occupancy(node_features, node_types, mgs)  # 5
        c7, node_types, node_features = features_to_occupancy(node_features, node_types, mgs)  # 6
        c8, node_types, node_features = features_to_occupancy(node_features, node_types, mgs)  # 7

        ogrid[:mgs, :mgs, :mgs] = c8
        ogrid[mgs:, :mgs, :mgs] = c7
        ogrid[:mgs, mgs:, :mgs] = c6
        ogrid[mgs:, mgs:, :mgs] = c5
        ogrid[:mgs, :mgs, mgs:] = c4
        ogrid[mgs:, :mgs, mgs:] = c3
        ogrid[:mgs, mgs:, mgs:] = c2
        ogrid[mgs:, mgs:, mgs:] = c1

    return ogrid, node_types, node_features


def voxelise(file_in: str, out_dir: str, grid_dim: int, dt: str = "uint8"):
    """Convert a model to a list of occupied points, save the result as a numpy array"""
    file_out = join(out_dir, basename(file_in) + ".npy")
    if exists(file_out):
        raise FileExistsError(f"Output file exist: {file_out}")

    voxels = load_model_as_voxel_grid(file_in, grid_dim)
    if min(voxels.origin) < -1:
        raise ValueError(f"Min voxel bounds can't be negative here. Min bound is {min(voxels.origin)}")
    vo = np.floor(voxels.origin)
    vo[vo < 0] = 0
    pts = [v.grid_index + vo for v in voxels.get_voxels()]
    pts_u8 = np.array(pts).astype(dt)
    with open(file_out, "wb") as out_file:
        np.save(out_file, pts_u8)


def print_tree_postord(node_features, node_types, where):
    """Print the tree structure represented by the node types and features"""
    print(f"tree_postord _ {where}")
    fi = 0
    for nt in node_types:
        if nt.item() != Octree.NodeType.NON_LEAF.value:
            print(Octree.Node.node_render[nt.item()], torch.sum(node_features[fi]).item())
            fi += 1
        else:
            print(Octree.Node.node_render[nt.item()])
    print(f"tree_postord ^ {where}\n")
