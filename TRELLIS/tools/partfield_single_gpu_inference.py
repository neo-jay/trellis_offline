"""Official PartField Model/config with Lightning's single-device prediction.

No model math/weights changed. The upstream demo forces DDP even on one GPU;
this entry avoids an unnecessary NCCL runtime on a single-device deployment.
"""
import sys,random
sys.path.insert(0,'/root/autodl-tmp/PartField')
import numpy as np
import torch
from lightning.pytorch import Trainer,seed_everything
from partfield.config import default_argument_parser,setup
from partfield.model_trainer_pvcnn_only_demo import Model

if __name__=='__main__':
 args=default_argument_parser().parse_args();cfg=setup(args,freeze=False)
 seed_everything(cfg.seed);torch.manual_seed(0);random.seed(0);np.random.seed(0)
 trainer=Trainer(accelerator='gpu',devices=1,strategy='auto',precision='16-mixed',
                 logger=False,enable_checkpointing=False)
 trainer.predict(Model(cfg),ckpt_path=cfg.continue_ckpt)
