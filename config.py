from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN

_C = CN()

_C.data_dir = ''
_C.resume_dir = ''
_C.num_workers = 8
_C.image_size = 224
_C.batch_size = 32
_C.mode = "train"
_C.epochs = 1

# cudnn related params
_C.cudnn = CN()
_C.cudnn.benchmark = True
_C.cudnn.deterministic = False
_C.cudnn.enabled = True

#HRNet
_C.HRNet = CN()
_C.HRNet.extra = CN(new_allowed=True)

#CapsNet
_C.CapsNet = CN()
_C.CapsNet.num_routing = 2
_C.CapsNet.seq_routing = False
_C.CapsNet.dp = 0.5
_C.CapsNet.extra = CN(new_allowed=True)

#train
_C.train = CN()
_C.train.lr_factor = 0.1
_C.train.lr_step = [30,60,90]
_C.train.lr = 0.05
_C.train.wd = 0.0001
_C.train.momentum = 0.9
_C.train.nesterov = True

def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
