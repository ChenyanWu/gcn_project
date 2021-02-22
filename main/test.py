import os
import argparse
import torch
import shutil
import __init_path
import logging
from core.config import update_config, cfg

logger = logging.getLogger()
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
logging.getLogger('').addHandler(console)

parser = argparse.ArgumentParser(description='Test Pose2Mesh')

parser.add_argument('opts',help="Modify config options using the command-line",default=None,nargs=argparse.REMAINDER)
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--cfg', type=str, help='experiment configure file name')
parser.add_argument('--debug', action='store_true', help='reduce dataset items')
parser.add_argument('--gpu', type=str, default='0,', help='assign multi-gpus by comma concat')

args = parser.parse_args()
if args.cfg:
    update_config(args.cfg)
logger.info(cfg)
torch.manual_seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
print("Work on GPU: ", os.environ['CUDA_VISIBLE_DEVICES'])

from core.base import Tester, Tester_mupo

if cfg.DATASET.test_list[0] == 'MuPoTS':
    tester = Tester_mupo(args, load_dir=cfg.TEST.weight_path)
else:
    tester = Tester(args, load_dir=cfg.TEST.weight_path)  # if not args.debug else None

print("===> Start testing...")
tester.test(0)

