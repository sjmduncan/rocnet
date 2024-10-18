from enum import Enum

import torch


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
        feature_list = [b.unsqueeze(0) for b in torch.split(node_features, 1, 0)]
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
