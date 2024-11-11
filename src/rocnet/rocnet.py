"""Wrapper functions for encoding/decoding from/to point clouds"""

import copy
import logging
import os.path as pth
import sys

import torch

from rocnet.octree import Octree, features_to_points, points_to_features
import rocnet.utils as utils
from rocnet import model as model

logger = logging.getLogger(__name__)
log_handler_stdout = logging.StreamHandler(sys.stdout)
logger.addHandler(log_handler_stdout)

DEFAULT_CONFIG = {
    "note": "",
    "feature_code_size": 80,
    "voxel_channels": 1,
    "classifier_hidden_size": 200,
    "node_channels": 64,
    "node_channels_internal": 128,
    "grid_dim": 256,
    "leaf_dim": 32,
    # Do not train or use the RootEncoder/RootDecoder modules.
    # If this is false then 'feature_code_size' is not used
    "has_root_encoder": True,
    "loss_params": {
        "recon_cre_gamma": 0.8333,
        "recon_cre_scale": 6,
        "label_scale": 10,
    },
}


class RocNet:
    """Transcoder for a particular implementation of RocNet"""

    def __init__(self, model_or_cfg, use_CUDA: bool = True):
        """Init either a new model from a config dict, or load one from a .pth file

        model: either a dict that conforms to rocnet.DEFAULT_CONFIG, or a path . a .pth file containing full model config + weights"""
        if use_CUDA:
            if not torch.cuda.is_available():
                raise Exception("Can't enable CUDA because CUDA isn't available")
            torch.cuda.init()
        self.cuda = use_CUDA

        if isinstance(model_or_cfg, dict):
            self.cfg = model_or_cfg
            self.encoder = model.Encoder(self.cfg)
            self.decoder = model.Decoder(self.cfg)
        elif isinstance(model_or_cfg, str) and pth.splitext(model_or_cfg)[1] == ".pth":
            self.model_path = model_or_cfg
            self.load(model_or_cfg)
        else:
            raise ValueError("RocNet init requires a model config dict or a path ot an existing .pth file")

        if use_CUDA:
            self.encoder.cuda()
            self.decoder.cuda()

        if self.cfg.grid_dim <= self.cfg.leaf_dim:
            logger.error(f"leaf_dim={self.cfg.leaf_dim} should be a power of two and less than grid_dim={self.cfg.grid_dim}")

    def compress_points(self, pointcloud):
        leaf_features, node_types = points_to_features(pointcloud, self.cfg.grid_dim, self.cfg.leaf_dim)
        tree = Octree(leaf_features, node_types)
        return self.encoder.encode_tree(tree)

    def uncompress_points(self, vector):
        leaf_features, node_types = self.decoder.decode_tree(torch.Tensor(vector.reshape(1, self.decoder.cfg.feature_code_size)).cuda())
        points = features_to_points(leaf_features, node_types, self.cfg.grid_dim, self.cfg.leaf_dim)
        return points

    def train(self):
        self.encoder.train()
        self.decoder.train()

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()

    def snapshot_state(self, metadata: dict):
        self.mod_state_snapshot = {
            "metadata": metadata,
            "config": self.cfg,
            "encoder": copy.deepcopy(self.encoder.state_dict()),
            "decoder": copy.deepcopy(self.decoder.state_dict()),
        }

    def load(self, model_path: str):
        """Load pre-trained model from 'path'"""
        mod_state = torch.load(model_path, weights_only=False)
        self.cfg = mod_state["config"]
        self.cfg = utils.deep_dict_merge(self.cfg, DEFAULT_CONFIG, override=False)
        if "metadata" in mod_state.keys():
            self.metadata = mod_state["metadata"]
        else:
            self.metadata = {}
        self.encoder = model.Encoder(self.cfg)
        self.decoder = model.Decoder(self.cfg)
        self.encoder.load_state_dict(mod_state["encoder"])
        self.decoder.load_state_dict(mod_state["decoder"])

    def save(self, model_path: str, metadata: dict, save_prev_snapshot: bool = False, best_so_far: bool = False):
        """Save the encoder & decoder model weights to '{base_path}_encoder.pth' and '${base_path}_decoder.pth' respectively"""
        if save_prev_snapshot:
            mod_state = self.mod_state_snapshot
        else:
            mod_state = {"metadata": metadata, "config": self.cfg, "encoder": self.encoder.state_dict(), "decoder": self.decoder.state_dict()}
        torch.save(mod_state, model_path)
        if best_so_far:
            torch.save(mod_state, f"{pth.join(pth.split(model_path)[0],'model')}.pth")
