import logging
import sys
from enum import Enum

import numpy as np
import torch

logger = logging.getLogger(__name__)
log_handler_stdout = logging.StreamHandler(sys.stdout)
logger.addHandler(log_handler_stdout)


def points_to_features(occupied_indices, grid_dim: int, leaf_dim: int, channels: int = 0):
    """Encode a list of occupied indices to a list of node features/types, allocating memory only for occupied leaves rather than all grid_dim^3 voxels

    occupied_indices: numpy array of indices of occupied voxels
    grid_dim: edge length of the whole grid, if this is smaller than the bounds of the occupied indices then
    leaf_dim: edge length of the octree leaf feature

    returns: List of octrees, each in the form of a post-ordered list of node types and leaf features.
             The number of octrees returned is determined by how many cubes of size grid_dim are required to contain the occupied indices
    """
    bl = np.min(occupied_indices[:, :3], axis=0)

    leaf_features = []

    def get_feature_and_type(indices, origin, dim):
        """Recursively compute node types, and node feature if relevant, at all octree levels"""
        if dim < leaf_dim:
            raise ValueError("sub_dim should always be larger than leaf_dim")
        tr = origin + np.array([dim, dim, dim])
        octant_indices = indices[np.all(list(zip(np.all(indices[:, :3] >= origin, axis=1), np.all(indices[:, :3] < tr, axis=1))), axis=1)]
        if len(octant_indices) == 0:
            leaf_features.append(torch.zeros([channels + 1, leaf_dim, leaf_dim, leaf_dim]))
            return torch.tensor([Octree.NodeType.LEAF_EMPTY.value])
        elif dim == leaf_dim:
            local_indices = octant_indices[:, :3] - origin
            occupancy = torch.zeros([channels + 1, leaf_dim, leaf_dim, leaf_dim])
            occupancy[:, local_indices[:, 0], local_indices[:, 1], local_indices[:, 2]] = torch.cat([torch.ones(local_indices.shape[0]).unsqueeze(1), torch.Tensor(octant_indices[:, 3:])], axis=1).T
            leaf_features.append(occupancy)
            return torch.tensor([Octree.NodeType.LEAF_MIX.value])
        else:
            octant_dim = dim // 2
            octant_origin = [
                np.array([0, 0, 0]),
                np.array([octant_dim, 0, 0]),
                np.array([0, octant_dim, 0]),
                np.array([octant_dim, octant_dim, 0]),
                np.array([0, 0, octant_dim]),
                np.array([octant_dim, 0, octant_dim]),
                np.array([0, octant_dim, octant_dim]),
                np.array([octant_dim, octant_dim, octant_dim]),
            ]
            return torch.cat(
                [
                    get_feature_and_type(octant_indices, origin + octant_origin[0], octant_dim),
                    get_feature_and_type(octant_indices, origin + octant_origin[1], octant_dim),
                    get_feature_and_type(octant_indices, origin + octant_origin[2], octant_dim),
                    get_feature_and_type(octant_indices, origin + octant_origin[3], octant_dim),
                    get_feature_and_type(octant_indices, origin + octant_origin[4], octant_dim),
                    get_feature_and_type(octant_indices, origin + octant_origin[5], octant_dim),
                    get_feature_and_type(octant_indices, origin + octant_origin[6], octant_dim),
                    get_feature_and_type(octant_indices, origin + octant_origin[7], octant_dim),
                    torch.tensor([Octree.NodeType.NON_LEAF.value]),
                ],
                axis=0,
            )

    node_types = get_feature_and_type(occupied_indices, bl, grid_dim)
    return torch.stack(leaf_features).type(torch.float32), node_types


def features_to_points(leaf_features, node_types, grid_dim: int, channels: int):
    """Decode a list of node features/types directly to a list of indices, allocating memory only for occupied leaves rather than all grid_dim^3 voxels"""

    def sub_features_to_points(leaf_features, node_types, dim, origin):
        if node_types[-1] == Octree.NodeType.LEAF_EMPTY.value:
            return np.array([]).reshape(-1, 3 + channels), node_types[:-1], leaf_features[:-1, :, :, :]
        elif node_types[-1] == Octree.NodeType.LEAF_MIX.value:
            pts = np.nonzero(leaf_features[-1].cpu()[0, :, :, :] > 0.5)
            pts = torch.cat([pts + origin, leaf_features[-1][1:, pts[:, 0], pts[:, 1], pts[:, 2]].T.cpu()], 1)
            return pts, node_types[:-1], leaf_features[:-1, :, :, :]
        elif node_types[-1] == Octree.NodeType.LEAF_FULL.value:
            pts = np.array(np.nonzero(np.ones([32, 32, 32]))).T + origin
            if channels > 0:
                pts = np.concatenate([pts, np.zeros([pts.shape[0], channels])], axis=1)
            return pts, node_types[:-1], leaf_features[:-1, :, :, :]
        else:
            octant_dim = dim // 2
            octant_origin = [
                np.array([0, 0, 0]),
                np.array([octant_dim, 0, 0]),
                np.array([0, octant_dim, 0]),
                np.array([octant_dim, octant_dim, 0]),
                np.array([0, 0, octant_dim]),
                np.array([octant_dim, 0, octant_dim]),
                np.array([0, octant_dim, octant_dim]),
                np.array([octant_dim, octant_dim, octant_dim]),
            ]
            octant_origin.reverse()
            pts1, node_types, leaf_features = sub_features_to_points(leaf_features, node_types[:-1], octant_dim, origin + octant_origin[0])
            pts2, node_types, leaf_features = sub_features_to_points(leaf_features, node_types, octant_dim, origin + octant_origin[1])
            pts3, node_types, leaf_features = sub_features_to_points(leaf_features, node_types, octant_dim, origin + octant_origin[2])
            pts4, node_types, leaf_features = sub_features_to_points(leaf_features, node_types, octant_dim, origin + octant_origin[3])
            pts5, node_types, leaf_features = sub_features_to_points(leaf_features, node_types, octant_dim, origin + octant_origin[4])
            pts6, node_types, leaf_features = sub_features_to_points(leaf_features, node_types, octant_dim, origin + octant_origin[5])
            pts7, node_types, leaf_features = sub_features_to_points(leaf_features, node_types, octant_dim, origin + octant_origin[6])
            pts8, node_types, leaf_features = sub_features_to_points(leaf_features, node_types, octant_dim, origin + octant_origin[7])
            pts = np.concatenate([pts1, pts2, pts3, pts4, pts5, pts6, pts7, pts8], axis=0)
        return pts, node_types, leaf_features

    occupied_indices, _, _ = sub_features_to_points(leaf_features, node_types, grid_dim, np.array([0, 0, 0]))
    return occupied_indices


class Octree(object):
    class NodeType(Enum):
        LEAF_FULL = 0  # full leaf node
        LEAF_EMPTY = 1  # empty leaf node
        LEAF_MIX = 2  # mixed leaf node
        NON_LEAF = 3  # non-leaf node

    class Node(object):
        node_render = ["F", "E", "M", "N"]

        def __init__(self, leaf_feature=None, children=[], node_type=None):
            self.leaf_feature = leaf_feature
            self.children = children
            self.node_type = torch.tensor([node_type], dtype=torch.int64)

        def is_leaf(self):
            return self.node_type != Octree.NodeType.NON_LEAF.value and self.leaf_feature is not None

        def is_empty_leaf(self):
            return self.node_type == Octree.NodeType.LEAF_EMPTY.value

        def is_non_leaf(self):
            return self.node_type == Octree.NodeType.NON_LEAF.value

        def __str__(self):
            return self.node_render[self.node_type[0]]

        def __repr__(self):
            return self.node_render[self.node_type[0]]

    def __str__(self):
        tstr = ""

        levels = [[self.root]]
        for level in range(self.max_depth):
            levels.append([])
            tstr += f"{level:3}: "
            for idx, n in enumerate(levels[level]):
                tstr += f"{n}"
                for c in n.children:
                    levels[level + 1].append(c)
                if idx > 0 and (idx % 8) == 0:
                    tstr += " "
            if len(levels[level + 1]) == 0:
                break
            else:
                tstr += ""
        return tstr

    def leaf_type_count(self):
        mixed = 0
        mixed_occupancy = 0
        empty = 0

        levels = [[self.root]]
        for level in range(self.max_depth):
            levels.append([])
            for node in levels[level]:
                for c in node.children:
                    levels[level + 1].append(c)
                if node.is_empty_leaf():
                    empty += 1
                elif node.is_leaf():
                    mixed += 1
                    mixed_occupancy += node.leaf_feature.sum() / torch.prod(torch.tensor(node.leaf_feature.shape))
            if len(levels[level + 1]) == 0:
                break
        return [empty, mixed, mixed_occupancy / mixed]

    def __repr__(self):
        """"""

    def __init__(self, node_features, node_types):
        """Construct an octree from a post-ordered list of nodes and node types"""
        self.max_depth = 6
        feature_list = [b for b in torch.split(node_features, 1, 0)]
        feature_list.reverse()
        node_list = [b.unsqueeze(0) for b in torch.split(node_types, 1, 0)]
        stack = []
        for nt in node_list:
            if nt == Octree.NodeType.NON_LEAF.value:
                children = []
                for _ in range(8):
                    children.append(stack.pop())
                stack.append(Octree.Node(children=children, node_type=nt))
            else:
                stack.append(Octree.Node(feature_list.pop(), node_type=nt))
        assert len(stack) == 1
        self.root = stack[0]
