"""Recursive NN for compressing voxel grids"""

from math import log2

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from rocnet.octree import Octree


def _leaf_layer_sizes(leaf_dim: int, is_encoder: bool):
    """Get the kernel size, stride, and padding for the three convolutional layers of the leaf encoder or decoder
    leaf_dim should be one of 8, 16, 32
    is_encoder specifies whether this is the encoding or decoding layer (the sizes are reversed for the latter)
    """
    lut = {
        32: {"kernel_sizes": [4, 4, 4], "conv_stride": [2, 2, 2], "conv_padding": [1, 1, 1]},
        16: {"kernel_sizes": [1, 4, 4], "conv_stride": [1, 2, 2], "conv_padding": [0, 1, 1]},
        8: {"kernel_sizes": [2, 3, 1], "conv_stride": [2, 1, 1], "conv_padding": [2, 0, 0]},
    }

    if leaf_dim not in lut:
        raise ValueError(f"leaf_d={leaf_dim} (allowed values are 8, 16, or 32)")

    if not is_encoder:
        lut[leaf_dim]["kernel_sizes"].reverse()
        lut[leaf_dim]["conv_stride"].reverse()
        lut[leaf_dim]["conv_padding"].reverse()

    return lut[leaf_dim]["kernel_sizes"], lut[leaf_dim]["conv_stride"], lut[leaf_dim]["conv_padding"]


def _check_params(grid_dim: int, leaf_dim: int, feature_code_size: int):
    """Check whether grid_dim, leaf_dim, and feature_code_size are valid values"""
    assert grid_dim in [64, 128, 256, 512, 1024, 2048]
    assert leaf_dim in [16, 32]
    assert feature_code_size > 0 and feature_code_size <= 4096


class _RootEncoder(nn.Module):
    """Encode the Octree root node"""

    def __init__(self, feature_code_size: int = 200, node_channels: int = 64):
        super().__init__()
        self.conv = nn.Conv3d(node_channels, feature_code_size, kernel_size=4, stride=1)
        self.tanh = nn.ELU()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data = nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))

    def forward(self, input):
        output = self.tanh(self.conv(input))
        output = output.view(-1, output.size()[1])
        return output


class _LeafEncoder(nn.Module):
    """This encodes non-empty leaf nodes"""

    def __init__(self, leaf_dim: int, voxel_channels: int = 0, node_channels: int = 64):
        super().__init__()

        self._layer_ks, self._layer_ss, self._layer_ps = _leaf_layer_sizes(leaf_dim, True)

        self.conv1 = nn.Conv3d(voxel_channels + 1, 16, kernel_size=self._layer_ks[0], stride=self._layer_ss[0], padding=self._layer_ps[0], bias=False)
        self.bn1 = nn.BatchNorm3d(16, track_running_stats=False)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=self._layer_ks[1], stride=self._layer_ss[1], padding=self._layer_ps[1], bias=False)
        self.bn2 = nn.BatchNorm3d(32, track_running_stats=False)
        self.conv3 = nn.Conv3d(32, node_channels, kernel_size=self._layer_ks[2], stride=self._layer_ss[2], padding=self._layer_ps[2], bias=False)
        self.bn3 = nn.BatchNorm3d(node_channels, track_running_stats=False)
        self.tanh = nn.ELU()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data = nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))

    def forward(self, leaf_input):
        leaf_vector = self.conv1(leaf_input)
        leaf_vector = self.bn1(leaf_vector)
        leaf_vector = self.tanh(leaf_vector)

        leaf_vector = self.conv2(leaf_vector)
        leaf_vector = self.bn2(leaf_vector)
        leaf_vector = self.tanh(leaf_vector)

        leaf_vector = self.conv3(leaf_vector)
        leaf_vector = self.bn3(leaf_vector)
        leaf_vector = self.tanh(leaf_vector)

        return leaf_vector


class _LeafEncoderEmpty(nn.Module):
    """This encodes empty leaf nodes"""

    def __init__(self, node_channels: int):
        super().__init__()
        self.node_channels = node_channels

    def forward(self, leaf_feature):
        leaf_vector = Variable(torch.zeros(leaf_feature.size()[0], self.node_channels, 4, 4, 4, device="cuda"))
        return leaf_vector


class _NodeEncoder(nn.Module):
    """Encode internal nodes"""

    def __init__(self, node_channels: int, node_channels_inernal: int):
        super().__init__()
        self.child_conv = [nn.Conv3d(node_channels, node_channels_inernal, kernel_size=1, stride=1, bias=False) for _ in range(8)]
        [self.add_module(f"child{idx+1}", mod) for idx, mod in enumerate(self.child_conv)]
        self.child_bn = [nn.BatchNorm3d(node_channels_inernal, track_running_stats=False) for _ in range(8)]
        [self.add_module(f"bn{idx+1}", mod) for idx, mod in enumerate(self.child_bn)]

        self.bn11 = nn.BatchNorm3d(node_channels_inernal, track_running_stats=False)
        self.second = nn.Conv3d(node_channels_inernal, node_channels, kernel_size=3, stride=1, padding=1)
        self.bn12 = nn.BatchNorm3d(node_channels, track_running_stats=False)

        self.tanh = nn.ELU()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data = nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))

    def forward(self, child_features):
        output = sum([bn(conv(ii)) for (bn, conv, ii) in zip(self.child_bn, self.child_conv, child_features)])
        output = self.bn11(output)

        output = self.tanh(output)
        output = self.second(output)
        if len(output.size()) == 1:
            output = output.unsqueeze(0)
        output = self.bn12(output)
        output = self.tanh(output)

        return output


class Encoder(nn.Module):
    """Module to encode all the nodes in the  tree"""

    def __init__(self, cfg: dict):
        _check_params(cfg.grid_dim, cfg.leaf_dim, cfg.feature_code_size)
        super().__init__()
        self.leaf_encoder = _LeafEncoder(cfg.leaf_dim, cfg.voxel_channels, cfg.node_channels)
        self.leaf_encoder_empty = _LeafEncoderEmpty(cfg.node_channels)
        self.n_levels = int(log2(cfg.grid_dim / cfg.leaf_dim))
        self.node_encoders = [_NodeEncoder(cfg.node_channels, cfg.node_channels_internal) for _ in range(self.n_levels)]
        [self.add_module(f"NodeEncoder{idx+1}", mod) for idx, mod in enumerate(self.node_encoders)]
        if cfg.has_root_encoder:
            self.root_encoder = _RootEncoder(cfg.feature_code_size, cfg.node_channels)
        self.has_root_encoder = cfg.has_root_encoder
        self.leaf_dim = cfg.leaf_dim
        self.grid_dim = cfg.grid_dim

    def _tf_encode_leaf(self, leaf):
        """Boilerplate to make things work with torchfold"""
        return self.leaf_encoder(leaf)

    def _tf_encode_empty_leaf(self, leaf):
        """Boilerplate to make things work with torchfold"""
        return self.leaf_encoder_empty(leaf)

    def _tf_encode_node_lvl0(self, c1, c2, c3, c4, c5, c6, c7, c8):
        """Boilerplate to make things work with torchfold"""
        return self.node_encoders[0]([c1, c2, c3, c4, c5, c6, c7, c8])

    def _tf_encode_node_lvl1(self, c1, c2, c3, c4, c5, c6, c7, c8):
        """Boilerplate to make things work with torchfold"""
        return self.node_encoders[1]([c1, c2, c3, c4, c5, c6, c7, c8])

    def _tf_encode_node_lvl2(self, c1, c2, c3, c4, c5, c6, c7, c8):
        """Boilerplate to make things work with torchfold"""
        return self.node_encoders[2]([c1, c2, c3, c4, c5, c6, c7, c8])

    def _tf_encode_node_lvl4(self, c1, c2, c3, c4, c5, c6, c7, c8):
        """Boilerplate to make things work with torchfold"""
        return self.node_encoders[4]([c1, c2, c3, c4, c5, c6, c7, c8])

    def _tf_encode_node_lvl5(self, c1, c2, c3, c4, c5, c6, c7, c8):
        """Boilerplate to make things work with torchfold"""
        return self.node_encoders[5]([c1, c2, c3, c4, c5, c6, c7, c8])

    def _tf_encode_root(self, feature):
        """Boilerplate to make things work with torchfold"""
        return self.root_encoder(feature)

    def forward(self, leaf_features, node_types):
        """Convert the provided points to an Octree, encode the octree and return the latent vector"""
        tree = Octree(leaf_features, node_types)
        return self.encode_tree(tree)

    def encode_tree(self, tree: Octree):
        def encode_node(node, level):
            if node.is_leaf():
                if node.is_empty_leaf():
                    return self.leaf_encoder_empty(node.leaf_feature.cuda())
                else:
                    return self.leaf_encoder(node.leaf_feature.cuda())
            elif node.is_non_leaf():
                children = [encode_node(c, level + 1) for c in node.children]
                return self.node_encoders[level](children)

        encoding = encode_node(tree.root, 0)
        if self.has_root_encoder:
            root_code = self.root_encoder(encoding)
        else:
            root_code = encoding.reshape([1, -1])
        return root_code


class _RootDecoder(nn.Module):
    """Decode a randomly sampled noise into a feature vector"""

    def __init__(self, feature_code_size: int, node_channels: int = 64):
        super().__init__()

        self.deconv1 = nn.ConvTranspose3d(feature_code_size, node_channels, kernel_size=4, stride=1)
        self.tanh = nn.ELU()

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose3d):
                m.weight.data = nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))

    def forward(self, input_feature):
        output = input_feature.view(-1, input_feature.size()[1], 1, 1, 1)
        output = self.deconv1(output)
        output = self.tanh(output)
        return output


class _NodeClassifier(nn.Module):
    def __init__(self, feature_code_size: int, classifier_hidden_size: int, node_channels: int):
        super().__init__()

        self.conv1 = nn.Conv3d(node_channels, feature_code_size, kernel_size=4, stride=1)
        self.mlp1 = nn.Linear(feature_code_size, classifier_hidden_size)
        self.tanh = nn.ELU()
        self.mlp2 = nn.Linear(classifier_hidden_size, 4)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose3d):
                m.weight.data = nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))

    def forward(self, input_feature):
        output = self.conv1(input_feature)
        output = self.mlp1(output.view(-1, output.size()[1]))
        output = self.tanh(output)
        output = self.mlp2(output)
        return output


class _NodeDecoder(nn.Module):
    """Decode an input (parent) feature into a left-child and a right-child feature"""

    def __init__(self, node_channels: int, node_channels_internal: int):
        super().__init__()
        self.mlp = nn.ConvTranspose3d(node_channels, node_channels_internal, kernel_size=3, stride=1, padding=1, bias=False)
        self.child_mlp = [nn.ConvTranspose3d(node_channels_internal, node_channels, kernel_size=1, stride=1, bias=False) for _ in range(8)]
        [self.add_module(f"mlp_child{idx+1}", mod) for idx, mod in enumerate(self.child_mlp)]
        self.child_bn = [nn.BatchNorm3d(node_channels, track_running_stats=False) for _ in range(8)]
        [self.add_module(f"bn{idx+1}", mod) for idx, mod in enumerate(self.child_bn)]

        self.tanh = nn.Tanh()
        self.tanh = nn.ELU()
        self.bn = nn.BatchNorm3d(node_channels_internal, track_running_stats=False)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose3d):
                m.weight.data = nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))

    def forward(self, parent_feature):
        vector = self.tanh(self.bn(self.mlp(parent_feature)))

        return [self.tanh(bn(mlp(vector))) for (bn, mlp) in zip(self.child_bn, self.child_mlp)]


class _LeafDecoder(nn.Module):
    def __init__(self, leaf_dim: int, voxel_channels: int = 0, node_channels: int = 64):
        super().__init__()

        self._layer_ks, self._layer_ss, self._layer_ps = _leaf_layer_sizes(leaf_dim, False)

        self.deconv2 = nn.ConvTranspose3d(node_channels, 32, kernel_size=self._layer_ks[0], stride=self._layer_ss[0], padding=self._layer_ps[0], bias=False)
        self.bn2 = nn.BatchNorm3d(32, track_running_stats=False)
        self.deconv3 = nn.ConvTranspose3d(32, 16, kernel_size=self._layer_ks[1], stride=self._layer_ss[1], padding=self._layer_ps[1], bias=False)
        self.bn3 = nn.BatchNorm3d(16, track_running_stats=False)
        self.deconv4 = nn.ConvTranspose3d(16, voxel_channels + 1, kernel_size=self._layer_ks[2], stride=self._layer_ss[2], padding=self._layer_ps[2], bias=False)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.ELU()

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose3d):
                m.weight.data = nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))

    def forward(self, leaf_input):
        leaf_vector = self.deconv2(leaf_input)
        leaf_vector = self.bn2(leaf_vector)
        leaf_vector = self.tanh(leaf_vector)

        leaf_vector = self.deconv3(leaf_vector)
        leaf_vector = self.bn3(leaf_vector)
        leaf_vector = self.tanh(leaf_vector)

        leaf_vector = self.deconv4(leaf_vector)
        leaf_vector = self.sigmoid(leaf_vector)
        leaf_vector = torch.clamp(leaf_vector, min=1e-7, max=1 - 1e-7)

        return leaf_vector


class Decoder(nn.Module):
    """Module to decode all the nodes in the  tree"""

    def __init__(self, cfg: dict):
        """Init RocNet octree decoder

        leaf_dim: Edge length of the cube of voxels which makes up a leaf node
        grid_dim: Edge length of the voxel grid which RocNet operates on
        voxel_channels: Number of data channels plus one. i.e. Occupancy-only is 0+1=1, R-G-B colour is 3+1=4
        feature_code_size: Size of the final output vector
        classifier_hidden_size: ???

        """
        _check_params(cfg.grid_dim, cfg.leaf_dim, cfg.feature_code_size)
        super().__init__()
        self.cfg = cfg
        self.leaf_decoder = _LeafDecoder(cfg.leaf_dim, cfg.voxel_channels, cfg.node_channels)
        self.n_levels = int(log2(cfg.grid_dim / cfg.leaf_dim))
        self.node_decoders = [_NodeDecoder(cfg.node_channels, cfg.node_channels_internal) for _ in range(self.n_levels)]
        [self.add_module(f"NodeDecoder{idx+1}", mod) for idx, mod in enumerate(self.node_decoders)]
        if cfg.has_root_encoder:
            self.root_decoder = _RootDecoder(cfg.feature_code_size, cfg.node_channels)
        self.has_root_decoder = cfg.has_root_encoder
        self.node_classifier = _NodeClassifier(cfg.feature_code_size, cfg.classifier_hidden_size, cfg.node_channels)
        self.leaf_dim = cfg.leaf_dim
        self.grid_dim = cfg.grid_dim

        self.classify_creloss = nn.CrossEntropyLoss()
        self.recon_loss = self._recon_loss_occupancy if cfg.voxel_channels == 0 else self._recon_loss_color
        self.label_scale = cfg.loss_params.label_scale
        self.recon_scale = 1.0 / (self.leaf_dim * self.leaf_dim)
        self.cre_occupied_factor = cfg.loss_params.recon_cre_gamma * cfg.loss_params.recon_cre_scale
        self.cre_empty_factor = (1 - cfg.loss_params.recon_cre_gamma) * cfg.loss_params.recon_cre_scale

    def forward(self, latent_vector):
        """Decode the latent vector and return an array of points"""
        return self.decode_tree(latent_vector)

    def decode_tree(self, latent_vector):
        if self.has_root_decoder:
            decode = self.root_decoder(latent_vector)
            stack = [decode]
        else:
            stack = [latent_vector.reshape(self.cfg.node_channels, 4, 4, 4).unsqueeze(0)]
        leaf_features = []
        node_types = []
        depth = [1]

        while len(stack) > 0:
            f = stack.pop()
            d = depth.pop()
            label_prob = self.node_classifier(f)
            _, label = torch.max(label_prob, 1)
            label = label.data.cpu().numpy()

            if label == 3 and d > len(self.node_decoders):
                print("Leaf classifier got it wrong for a leaf node. Giving you a fully occupied leaf instead")
                leaf_features.append(torch.ones([self.cfg.voxel_channels + 1, self.leaf_dim, self.leaf_dim, self.leaf_dim]).unsqueeze(0))
                label = np.array([0])
            elif label == 3:  # NON-LEAF
                children = self.node_decoders[d - 1](f)
                children.reverse()
                stack = stack + children
                depth = depth + [d + 1 for _ in range(8)]
            else:  # LEAF
                reBox = self.leaf_decoder(f)
                leaf_features.append(reBox)

            node_types.append(label)
        node_types_sorted_tensor = np.squeeze(np.flip(np.array(node_types)))
        features_tensor = torch.flip(torch.cat(leaf_features, 0), [0])
        return features_tensor, node_types_sorted_tensor

    def _recon_loss_occupancy(self, leaf_est, leaf_gt):
        """Binary cross-entropy loss similar to that formulated by Brock2016

        self.cre_gamma corresponds to the gamma factor in Brock2016, however target/observed values remain on the [0..1] range.
        self.cre_scale normalizes the loss  value to the number of voxels in a leaf block.

        The Liu2020 implementation originally scaled the 'occupied' half by 5 and then scaled the total score by 0.001.
        For the same relative occupied/unoccupied weighting use cre_gamma=0.83 and recon_scale=6
        """
        occ_loss = torch.cat([torch.sum(-((gt.mul(self.cre_occupied_factor).mul(torch.log(est))).add((1 - gt).mul(self.cre_empty_factor).mul(torch.log(1 - est))))).mul(self.recon_scale).unsqueeze(0) for est, gt in zip(leaf_est[0, :1], leaf_gt[0, :1])], 0)
        return occ_loss

    def _recon_loss_color(self, leaf_est, leaf_gt):
        """Binary cross-entropy loss similar to that formulated by Brock2016

        self.cre_gamma corresponds to the gamma factor in Brock2016, however target/observed values remain on the [0..1] range.
        self.cre_scale normalizes the loss  value to the number of voxels in a leaf block.

        The Liu2020 implementation originally scaled the 'occupied' half by 5 and then scaled the total score by 0.001.
        For the same relative occupied/unoccupied weighting use cre_gamma=0.83 and recon_scale=6
        """
        m = np.nonzero(leaf_gt[0, 0])
        attr_loss = 10 * torch.sum(torch.sum(torch.abs(leaf_gt[0, 1:, m[:, 0], m[:, 1], m[:, 2]] - leaf_est[0, 1:, m[:, 0], m[:, 1], m[:, 2]]))) / m.shape[0]
        occ_loss = torch.cat([torch.sum(-((gt.mul(self.cre_occupied_factor).mul(torch.log(est))).add((1 - gt).mul(self.cre_empty_factor).mul(torch.log(1 - est))))).mul(self.recon_scale).unsqueeze(0) for est, gt in zip(leaf_est[0, :1], leaf_gt[0, :1])], 0)
        return occ_loss + attr_loss

    def _classify_loss_fn(self, class_est, class_gt):
        loss = torch.cat([self.classify_creloss(l_est.unsqueeze(0), l_gt.unsqueeze(0)).unsqueeze(0).mul(self.label_scale) for l_est, l_gt in zip(class_est, class_gt)], 0)
        return loss

    def decode_loss(self, feature_code, expected_tree):
        """Decodes the feature to an octree and computes the loss function.

        Recursively accumulates the classifier and reconstruction losses for the whole tree.

        model: rocnet model to use for decoding
        feature: feature vector to decode
        tree: expected octree structure
        returns: [reconstruction_loss, node_classifier_loss] (a tensor with two elements)"""

        def decode_node_leaf(tgt, est, l):
            label_prob = self.node_classifier(est)
            label_loss = self._classify_loss_fn(label_prob, tgt.node_type.cuda())
            if tgt.is_leaf():
                if tgt.is_empty_leaf():
                    return torch.tensor([0.0], device="cuda", requires_grad=False), label_loss
                else:
                    fea = self.leaf_decoder(est)
                    recon_loss = self.recon_loss(fea, tgt.leaf_feature.cuda())
                    return recon_loss, label_loss
            else:  # Non-leaf node
                children = self.node_decoders[l](est)
                child_losses = [decode_node_leaf(tgt_child, est_child, l + 1) for (tgt_child, est_child) in zip(tgt.children, children)]

                child_recon_loss = sum([loss[0] for loss in child_losses])
                child_label_loss = sum([loss[1] for loss in child_losses])

                return child_recon_loss, child_label_loss + label_loss

        if self.has_root_decoder:
            feature = self.root_decoder(feature_code)
        else:
            feature = feature_code.reshape(self.cfg.node_channels, 4, 4, 4).unsqueeze(0)
        recon_loss, label_loss = decode_node_leaf(expected_tree.root, feature, 0)
        return torch.cat([recon_loss, label_loss], 0)

    def _tf_decode_leaf(self, feature):
        """Boilerplate to make things work with torchfold"""
        return self.leaf_decoder(feature)

    def _tf_decode_empty_leaf(self):
        """Boilerplate to make things work with torchfold"""
        return torch.tensor([0.0], device="cuda", requires_grad=False)

    def _tf_decode_node_lvl0(self, feature):
        """Boilerplate to make things work with torchfold"""
        return self.node_decoders[0](feature)

    def _tf_decode_node_lvl1(self, feature):
        """Boilerplate to make things work with torchfold"""
        return self.node_decoders[1](feature)

    def _tf_decode_node_lvl2(self, feature):
        """Boilerplate to make things work with torchfold"""
        return self.node_decoders[2](feature)

    def _tf_decode_node_lvl3(self, feature):
        """Boilerplate to make things work with torchfold"""
        return self.node_decoders[3](feature)

    def _tf_decode_node_lvl4(self, feature):
        """Boilerplate to make things work with torchfold"""
        return self.node_decoders[4](feature)

    def _tf_decode_node_lvl5(self, feature):
        """Boilerplate to make things work with torchfold"""
        return self.node_decoders[5](feature)

    def _tf_decode_root(self, code):
        """Boilerplate to make things work with torchfold"""
        return self.root_decoder(code)

    def _tf_classify_node(self, feature):
        """Boilerplate to make things work with torchfold"""
        return self.node_classifier(feature)

    def _tf_vec_sum_2(self, v1, v2):
        """Boilerplate to make things work with torchfold"""
        return sum([v1, v2])

    def _tf_vec_sum_8(self, c1, c2, c3, c4, c5, c6, c7, c8):
        """Boilerplate to make things work with torchfold"""
        return sum([c1, c2, c3, c4, c5, c6, c7, c8])
