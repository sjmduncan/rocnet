import copy
import logging
import sys
from datetime import datetime
from os.path import join, split

import numpy as np
import torch
from numpy import array, average, concatenate, loadtxt, savetxt

from rocnet.dataset import Dataset
from rocnet.rocnet import RocNet
from rocnet.train_torchfold import batch_loss_fold
from rocnet.utils import td_to_txt

logger = logging.getLogger(__name__)
log_handler_stdout = logging.StreamHandler(sys.stdout)
logger.addHandler(log_handler_stdout)
TIME_FMT = "%Y-%m-%d_%H.%M.%S"
START_TIME = datetime.now().strftime(TIME_FMT)

DEFAULT_CONFIG = {
    "note": "",
    "dataset_path": "../rocnet.data/test",
    "max_samples": -1,
    "max_epochs": 300,  # Default from liu2022
    "batch_size": 50,  # Default from liu2022
    "verbose": False,
    "use_torchfold": False,
    "validation": {  # Validation runs on the test dataset
        "stop_threshold": 5,
        "min_stop_epochs": 30,
    },
    "learning": {
        "warmup_epochs": 1,
        "lr_warmup": 0.0001,
        "lr_init": 0.0015,
        "lr_min": 0.0001,
        "exp_gamma": 0.96,
        "exp_epoch": 10,
    },
    "snapshot": {
        "enabled": False,
        "interval": 50,
    },
    "profile": {
        "enabled": False,
        "enable_trace": False,
    },
}


def check_training_cfg(cfg):
    if not cfg.model.has_root_encoder:
        if cfg.model.node_channels * 64 != cfg.model.feature_code_size:
            raise ValueError(f"Bad config: has_root_encoder=False, so feature_code_size ({cfg.model.feature_code_size}) should be 64*node_channels (64*{cfg.model.node_channels}={cfg.model.node_channels * 64})")


class Trainer:
    def __init__(self, run_dir: str, cfg: dict, dataset: Dataset, valid_dataset: Dataset):
        """"""
        model = RocNet(cfg.model)
        self.model = model
        self.cfg = cfg
        self.valid_dataset = valid_dataset
        self.dataset = dataset
        self.out_dir = run_dir
        self.loss_per_epoch = array([]).reshape(0, 8)
        dataset.load(model.cfg.grid_dim, model.cfg.leaf_dim)
        valid_dataset.load(model.cfg.grid_dim, model.cfg.leaf_dim)

        model.train()
        self.encoder_opt = torch.optim.Adam(model.encoder.parameters(), cfg.learning.lr_init)
        self.encoder_sched_warmup = torch.optim.lr_scheduler.LinearLR(self.encoder_opt, start_factor=cfg.learning.lr_warmup / cfg.learning.lr_init, total_iters=cfg.learning.warmup_epochs)
        self.encoder_sched_exp = torch.optim.lr_scheduler.ExponentialLR(self.encoder_opt, gamma=cfg.learning.exp_gamma)
        self.decoder_opt = torch.optim.Adam(model.decoder.parameters(), cfg.learning.lr_init)
        self.decoder_sched_warmup = torch.optim.lr_scheduler.LinearLR(self.decoder_opt, start_factor=cfg.learning.lr_warmup / cfg.learning.lr_init, total_iters=cfg.learning.warmup_epochs)
        self.decoder_sched_exp = torch.optim.lr_scheduler.ExponentialLR(self.decoder_opt, gamma=cfg.learning.exp_gamma)

    def save_snapshot(self, base_path: str, might_be_final: bool, metadata: dict):
        """Save a snapshot of the model, loss files, and optimiser/scheduler states"""
        self.model.save(f"{base_path}.pth", metadata, save_prev_snapshot=False, best_so_far=might_be_final)
        torch.save(
            {
                "decoder_lr_scheduler": self.decoder_sched_exp.state_dict(),
                "encoder_lr_scheduler": self.encoder_sched_exp.state_dict(),
                "decoder_optimiser": self.decoder_opt.state_dict(),
                "encoder_optimiser": self.encoder_opt.state_dict(),
            },
            f"{base_path}_training.pth",
        )
        savetxt(f"{base_path}_loss.csv", self.loss_per_epoch)

    def load_snapshot(self, base_path: str):
        """Load a snapshot of the model, loss files, and optimiser/scheduler states"""
        state_dict = torch.load(f"{base_path}_training.pth")
        self.decoder_sched_exp.load_state_dict(state_dict["decoder_lr_scheduler"])
        self.encoder_sched_exp.load_state_dict(state_dict["encoder_lr_scheduler"])
        self.decoder_opt.load_state_dict(state_dict["decoder_optimiser"])
        self.encoder_opt.load_state_dict(state_dict["encoder_optimiser"])
        self.loss_per_epoch = loadtxt(f"{base_path}_loss.csv")

    def save_snapshot_prev(self, base_path: str, dicts: dict, metadata: dict):
        self.model.save(f"{base_path}.pth", metadata, save_prev_snapshot=True, best_so_far=True)
        torch.save(dicts, f"{base_path}_training.pth")
        savetxt(f"{base_path}_loss.csv", self.loss_per_epoch)

    def train(self, epoch_callback: callable):
        def collator(batch):
            return batch

        def batch_loss(batch):
            if self.cfg.use_torchfold:
                return batch_loss_fold(self.model, batch)

            encoded = [self.model.encoder.encode_tree(tree) for tree in batch]
            batch_losses = torch.cat([self.model.decoder.decode_loss(code, expected_tree) for (code, expected_tree) in zip(encoded, batch)], 0).reshape(-1, 2)

            full_losses = torch.cat([batch_losses, batch_losses.sum(axis=1).unsqueeze(1)], axis=1)
            avg_loss = torch.mean(full_losses, axis=0)
            return full_losses, avg_loss

        logger.info(f"Training for {self.cfg.max_epochs} epochs")
        loss_log_suffix = ["R", "L", "T", "R", "L", "T"]
        done = False
        logger.info(f"start: {datetime.now()}")
        last_was_min = False
        this_is_min = False
        last_epoch_time = datetime.now()
        for epoch in range(self.cfg.max_epochs):
            logger.info(f"{epoch+1:4}/{self.cfg.max_epochs:<5} LR={self.encoder_opt.param_groups[0]['lr']:<12.10f} ({split(split(self.out_dir)[0])[1]})")
            #################### Training
            train_losses = []
            train_iter = torch.utils.data.DataLoader(self.dataset, batch_size=self.cfg.batch_size, shuffle=True, collate_fn=collator, pin_memory=self.model.cuda, pin_memory_device="cuda")
            dicts = {
                "decoder_lr_scheduler": copy.deepcopy(self.decoder_sched_exp.state_dict()),
                "encoder_lr_scheduler": copy.deepcopy(self.encoder_sched_exp.state_dict()),
                "decoder_optimiser": copy.deepcopy(self.decoder_opt.state_dict()),
                "encoder_optimiser": copy.deepcopy(self.encoder_opt.state_dict()),
            }

            for batch in train_iter:
                full_losses, avg_loss = batch_loss(batch)
                self.encoder_opt.zero_grad()
                self.decoder_opt.zero_grad()
                avg_loss[-1].backward()
                self.encoder_opt.step()
                self.decoder_opt.step()

                train_losses = train_losses + full_losses.cpu().tolist()
            #################### Validation
            self.model.eval()
            with torch.no_grad():
                valid_losses = []
                valid_iter = torch.utils.data.DataLoader(self.valid_dataset, batch_size=self.cfg.batch_size, shuffle=True, collate_fn=collator, pin_memory=self.model.cuda, pin_memory_device="cuda")
                for batch in valid_iter:
                    full_losses, avg_loss = batch_loss(batch)
                    valid_losses = valid_losses + full_losses.cpu().tolist()
            self.model.train()
            do_snapshot = (self.cfg.snapshot.enabled and (epoch + 1) % self.cfg.snapshot.interval == 0) or (epoch + 1) == self.cfg.max_epochs
            stopping = epoch_callback(epoch, self.cfg.max_epochs, train_losses, valid_losses, do_snapshot)

            epoch_loss = concatenate([[epoch], [self.encoder_opt.param_groups[0]["lr"]], average(train_losses, axis=0), average(valid_losses, axis=0)])

            train_logtxt = [f"{n[0]:9.4f}{n[1]}" for n in zip(list(epoch_loss[2:5]), loss_log_suffix[:3])]
            valid_logtxt = [f"{n[0]:9.4f}{n[1]}" for n in zip(list(epoch_loss[5:8]), loss_log_suffix[:3])]
            logger.info(f"{epoch+1:4} train {' '.join(train_logtxt)}")
            logger.info(f"{epoch+1:4} valid {' '.join(valid_logtxt)}")
            this_is_min = False
            if epoch > 0:
                min_loss = np.min(self.loss_per_epoch[:, 2:], axis=0)
                loss_diff_abs = min_loss - epoch_loss[2:]
                loss_diff_last = self.loss_per_epoch[-1, 2:] - epoch_loss[2:]
                train_diff_logtxt = [f"{n[0]:9.4f}{n[1]}" for n in zip(list(np.concatenate([loss_diff_abs[:3], loss_diff_last[:3]])), loss_log_suffix)]
                valid_diff_logtxt = [f"{n[0]:9.4f}{n[1]}" for n in zip(list(np.concatenate([loss_diff_abs[3:], loss_diff_last[3:]])), loss_log_suffix)]
                logger.info(f"{epoch+1:4} train {' '.join(train_diff_logtxt)}")
                logger.info(f"{epoch+1:4} valid {' '.join(valid_diff_logtxt)}")
                done = (self.cfg.validation.stop_threshold > 0) and (epoch > self.cfg.validation.min_stop_epochs) and (loss_diff_abs[-1] < -self.cfg.validation.stop_threshold)
                this_is_min = loss_diff_abs[-1] > 0
            elif self.cfg.profile.enabled:
                torch.cuda.memory._dump_snapshot(join(self.out_dir, "cumem2_first_epoch.pickle"))
            self.loss_per_epoch = concatenate([self.loss_per_epoch, epoch_loss.reshape(1, -1)])

            snapshot_meta = {"epoch": epoch, "loss": epoch_loss[2:8], "lr": self.encoder_opt.param_groups[0]["lr"], "min_loss": this_is_min}

            if do_snapshot or done or stopping:
                logger.info(f"saving {'final model' if (done or stopping) else 'snapshot'} at {epoch + 1} epochs")
                base_path = join(self.out_dir, f"model_{epoch+1}")
                self.save_snapshot(base_path, this_is_min, snapshot_meta)
            epoch_time = datetime.now()
            epoch_diff = epoch_time - last_epoch_time
            remainin_diff = td_to_txt((self.cfg.max_epochs - epoch) * epoch_diff)
            epoch_duration = td_to_txt(epoch_diff)
            logger.info(f"{epoch+1:4}        {epoch_duration.h:02}:{epoch_duration.m:02}:{epoch_duration.s:02} {remainin_diff.h:02}:{remainin_diff.m:02}:{remainin_diff.s:02} ({epoch_time})")
            last_epoch_time = epoch_time
            if done or stopping:
                return True

            #################### Adjust learning rate for next epoch
            if epoch <= self.cfg.learning.warmup_epochs:
                self.encoder_sched_warmup.step()
                self.decoder_sched_warmup.step()
            elif (epoch + 1) >= self.cfg.learning.exp_epoch - 1 and self.encoder_opt.param_groups[0]["lr"] > self.cfg.learning.lr_min:
                self.encoder_sched_exp.step()
                self.decoder_sched_exp.step()
                if self.encoder_opt.param_groups[0]["lr"] < self.cfg.learning.lr_min:
                    for param_group in self.encoder_opt.param_groups:
                        param_group["lr"] = self.cfg.learning.lr_min
                    for param_group in self.decoder_opt.param_groups:
                        param_group["lr"] = self.cfg.learning.lr_min

            #################### If loss went up during this epoch, snapshot the previous epoch as potential best-case
            if last_was_min and not this_is_min and not do_snapshot:
                logger.info(f"{epoch+1:4}  saving previous epoch weights as potential abs min")
                base_path = join(self.out_dir, f"model_{epoch}")
                self.save_snapshot_prev(base_path, dicts, snapshot_meta)
            last_was_min = this_is_min
            self.model.snapshot_state(snapshot_meta)
            logger.info("")
