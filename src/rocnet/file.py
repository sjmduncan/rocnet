import copy
import logging
import os
import struct
import sys
from io import BufferedReader, BufferedWriter
from os.path import exists
from pathlib import Path

import numpy as np

from rocnet.rocnet import RocNet

logger = logging.getLogger(__name__)
log_handler_stdout = logging.StreamHandler(sys.stdout)
logger.addHandler(log_handler_stdout)


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


class RocNetFile:
    """A class for encoding and decoding large rocnet-compressed point-cloud files"""

    def __init__(self, model_base_path: str = None):
        """Load a RocNet model for encoding/decoding pointclouds.

        model_base_path: '{model-folder}/model_{epochs}' or None if the encoder weights are bundled with the file
        """
        self.decode_bundled_only = model_base_path is None
        self.file_ext = ".rnt"
        self._model = RocNet(model_base_path)
        self._expected_file_headers = {
            # Encoder/decoder metadata
            "signature": "rnt1",
            "version": 1,
            "data_start": 36,
            # Per-file metadata
            "vox_size": 1.0,
            "grid_dim": self._model.cfg.grid_dim,
            "code_size": self._model.cfg.feature_code_size,
            "lzma": 0,
            "model_weights": 0,
            "num_tiles": 0,
        }

    def encode(self, path_out: str, pts: np.array, vox_size: float, bundle_decoder: bool = False, overwrite: bool = False, use_lzma: bool = False):
        """Encode the numpy array of points + attributes and write the result to a file

        path_out: file to write the encoded pointcloud, extension should be .rnt
        pts: numpy array with N points optionally with A attributes each, expected shape=(N, (3 + A))
        vox_size: voxel size to use when quantizing the point cloud
        bundel_decoder: set to true to include model config and decoder weights in the file header
        overwrite: set to true to overwrite existing files
        use_lzma: set to true to use LZMA compression after RocNet encoding"""

        if self.decode_bundled_only:
            raise ValueError("No encoder specified, can only decode files which include decoder weights")

        def _write_tile(tile, file: BufferedWriter):
            """Write the next tile origin + feature code to the file, return the number of bytes written"""
            origin_bytes = b"".join([struct.pack("f", v) for v in tile[0]])
            nw = file.write(origin_bytes)
            tile_bytes = b"".join([struct.pack("f", v) for v in tile[1].reshape(-1)])
            nw += file.write(tile_bytes)
            return nw

        if exists(path_out):
            if overwrite:
                logger.warning(f"Overwriting existing file: {path_out}")
                Path.unlink(path_out)
            else:
                raise FileExistsError(f"File already exists: {path_out}")

        if os.path.splitext(path_out)[1] != self.file_ext:
            logger.warning(f"Non-standard file extension '{os.path.splitext(path_out)[1]}', expected '{self.file_ext}'")

        tileset = pc_to_tiles(pts, vox_size, self._model.cfg.grid_dim)

        codes = [self._model.compress_points(t[1]).cpu() for t in tileset]

        origins = [t[0] for t in tileset]
        with open(path_out, "xb") as f:
            file_headers = copy.deepcopy(self._expected_file_headers)
            file_headers["vox_size"] = float(vox_size)
            file_headers["lzma"] = bool(use_lzma)
            file_headers["num_tiles"] = len(codes)
            file_headers["model_weights"] = bundle_decoder
            self._write_file_header(f, file_headers)
            if bundle_decoder:
                mod_config = self._model.cfg
                mod_weights = self._model.decoder.state_dict()
                # write model config
                # write model weights

                pass
            [_write_tile(t, f) for t in zip(origins, codes)]

    def decode(self, path_in: str):
        """Read and decode a rocnet-encoded pointcloud and return a numpy array of points

        path_in: rocnet-encoded file to read, extension should be .rnt
        returns: numpy array with N points
        """

        def read_tile(file: BufferedReader, code_size: int) -> tuple[np.array, np.array]:
            """Read the next tile from the file reader, return the origin and the featuer code as a tuple of numpy arrays"""
            origin = np.array([struct.unpack("f", file.read(4)) for _ in range(3)]).reshape(3)
            code = np.array([struct.unpack("f", file.read(4)) for _ in range(code_size)]).reshape([code_size])
            return origin, code

        if not exists(path_in):
            raise FileNotFoundError(f"File does not exist: {path_in}")

        with open(path_in, "rb") as f:
            file_headers = self._read_file_headers(f, self._expected_file_headers)
            if self.decode_bundled_only:
                if not file_headers["model_weights"]:
                    raise ValueError("No base model provided and file does not contain decoder weights.")
            if file_headers["model_weights"]:
                logger.info("Loading model config and weights from file")
            errs = []
            if file_headers["grid_dim"] != self._expected_file_headers["grid_dim"]:
                errs.append(f"File tile size ({file_headers['grid_dim']}) does not match model grid_dim ({self._expected_file_headers['grid_dim']})")
            if file_headers["code_size"] != self._expected_file_headers["code_size"]:
                errs.append(f"File code size ({file_headers['code_size']}) does not match model code size ({self._expected_file_headers['code_size']})")
            if len(errs) > 0:
                [logger.error(e) for e in errs]
                raise ValueError(f"Mode can't decode this file: {errs}")
            tiles_data = [read_tile(f, file_headers["code_size"]) for _ in range(file_headers["num_tiles"])]
            tiles = [(self._model.uncompress_points(np.expand_dims(t[1], 0)) + t[0]) for t in tiles_data]
            return np.concatenate(tiles)

    def examine(self, path_in: str):
        """Print file size and decoded file headers"""
        with open(path_in, "rb") as f:
            try:
                file_headers = self._read_file_headers(f, self._expected_file_headers)
            except ValueError:
                logger.error("Bad file headers")
                return

            errs = []
            if file_headers["grid_dim"] != self._expected_file_headers["grid_dim"]:
                errs.append(f"tile size ({file_headers['grid_dim']}) does not match model grid_dim ({self._expected_file_headers['grid_dim']})")
            if file_headers["code_size"] != self._expected_file_headers["code_size"]:
                errs.append(f"code size ({file_headers['code_size']}) does not match model code size ({self._expected_file_headers['code_size']})")
            if len(errs) > 0:
                [logger.error(f"Decoder mismatch: {e}") for e in errs]
        return file_headers

    def _val_to_bytes(self, val: any):
        """Convert the primitive type (incl str) to a bytearray with fixed size for int/float types. Return the byte array and its length"""
        if isinstance(val, str):
            val_b = val.encode()
        elif isinstance(val, float):
            val_b = struct.pack("f", val)
        elif isinstance(val, int):
            val_b = struct.pack("i", val)
        else:
            raise TypeError(f"Can't write file header of type {val.type()}")
        return len(val_b), val_b

    def _write_model_to_file(self, f: BufferedWriter, model: RocNet):
        """Write model config and model weights to the file, writing from the current buffer position."""

    def _read_model_from_file(self, f: BufferedWriter) -> RocNet:
        """Load model config and model weights from the file, reading from the current buffer possition."""

    def _write_file_header(self, f: BufferedWriter, file_headers: dict):
        """Write the file header data, writing from the start of the buffer"""

        f.seek(0)
        hdr_bytes = [self._val_to_bytes(file_headers[v]) for v in file_headers]
        file_headers["data_start"] = sum([hb[0] for hb in hdr_bytes])
        hdr_bytes = [self._val_to_bytes(file_headers[v]) for v in file_headers]
        for hb in hdr_bytes:
            f.write(struct.pack("i", hb[0]))
            f.write(hb[1])
        return f.tell()

    def _val_from_bytes(self, val_b: bytearray, expected: any):
        """Convert from bytes to the expected primitive type (incl str)"""
        if isinstance(expected, str):
            return val_b.decode()
        elif isinstance(expected, float):
            return struct.unpack("f", val_b)[0]
        elif isinstance(expected, int):
            return struct.unpack("i", val_b)[0]
        else:
            raise TypeError(f"Can't write file header of type {expected.type()}")

    def _read_file_headers(self, f: BufferedReader, expected_headers: dict):
        """Parse the header of a rocnet file, reading from the start of the buffer"""

        f.seek(0)
        hdrs = {}
        for h in expected_headers.keys():
            val_len = struct.unpack("i", f.read(4))[0]
            val_b = f.read(val_len)
            hdrs[h] = self._val_from_bytes(val_b, expected_headers[h])
        errs = []
        if hdrs["version"] != expected_headers["version"]:
            errs.append(f"File/decoder version mismatch: version = {hdrs['version']} != {expected_headers['version']}")
        if hdrs["signature"] != expected_headers["signature"]:
            errs.append(f"Corrupt file: signature = {hdrs['signature']} != {expected_headers['signature']}")
        if hdrs["data_start"] != expected_headers["data_start"]:
            errs.append(f"Corrupt file: data_start = {hdrs['data_start']} != {expected_headers['data_start']}")
        if hdrs["vox_size"] <= 0:
            errs.append(f"Corrupt file: vox_size = {hdrs['vox_size']}, should be positive and non-zero")
        if hdrs["lzma"] < 0 or hdrs["lzma"] > 1:
            errs.append(f"Corrupt file: lzma = {hdrs['lzma']}, should be 0 or 1")
        if hdrs["num_tiles"] <= 0 or hdrs["num_tiles"] <= 0:
            errs.append(f"Corrupt file: num_tiles = {hdrs['num_tiles']}, should be positive and non-zero")

        if len(errs) > 0:
            [logger.error(e) for e in errs]
            raise ValueError(f"Bad file header: {errs}")

        return hdrs
