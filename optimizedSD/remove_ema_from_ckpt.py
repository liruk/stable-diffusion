"""Remove ema data in checkpoint to reduce file size."""
import sys
import os
import torch

ori_ckpt_path = sys.argv[1]
new_ckpt_path = ori_ckpt_path +".no-ema.ckpt"

ckpt = torch.load(ori_ckpt_path, map_location=(lambda storage, loc: storage))
for k in list(ckpt['state_dict']):
    if 'model_ema' in k:
        del ckpt['state_dict'][k]

torch.save(ckpt, new_ckpt_path)