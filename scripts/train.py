"""
This file runs the main training/val loop
"""




import os
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=2
os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=2
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=2

import torch
torch.set_num_threads(2)

os.environ["CUDA_VISIBLE_DEVICES"]= "4,5,6"

import json
import sys
import pprint


sys.path.append(".")
sys.path.append("..")

from options.train_options import TrainOptions
from training.coach_invert import Coach


def main():
	opts = TrainOptions().parse()
	#if os.path.exists(opts.exp_dir):
	#	raise Exception('Oops... {} already exists'.format(opts.exp_dir))
	os.makedirs(opts.exp_dir, exist_ok=True)

	opts_dict = vars(opts)
	pprint.pprint(opts_dict)
	with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
		json.dump(opts_dict, f, indent=4, sort_keys=True)

	coach = Coach(opts)
	coach.train()


if __name__ == '__main__':
	main()
