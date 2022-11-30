import os
from os.path import join
from argparse import ArgumentParser
from pprint import pprint
import torch

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything

from src.hierarchical_el import *
from src.environment import *
from src.rl import *


def load_weights(model, model_path):
	state_dict = torch.load(model_path)
	print("strictly loading...", end="")
	try:
		model.load_state_dict(state_dict)
		print("success!")
		return
	except Exception: print('fail!')
	print("unstrictly loading...", end="")
	try:
		model.load_state_dict(state_dict, strict=False)
		print("success!")
		return
	except Exception: print('fail!')
	print("success!")

if __name__ == "__main__":
	parser = ArgumentParser()

	parser.add_argument("--model", type=str, default="rl")
	parser.add_argument("--res_dir", type=str, default="./results/")
	parser.add_argument("--mention_filename", type=str, default="./data/mentions.json")
	parser.add_argument("--md_strategy", type=int, default=2)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--linking_mode", type=str, default="match")
	parser.add_argument("--append", action="store_true")
	parser.add_argument("--mode", type=int, default=1)

	parser = HierarchicalEL.add_model_specific_args(parser)

	args, _ = parser.parse_known_args()
	pprint(args.__dict__)

	seed_everything(seed=args.seed)

	model_class = HELActorCritic
	Environment = EnvironmentSUM
	if args.mode == 0:
		RLTrainer = DRLTrainer
	else: 
		RLTrainer = DRLTrainerAFM

	trainer = RLTrainer(device="cuda")
	model = model_class(**vars(args)).cuda()
	if args.linking_mode == "beam":
		model.generate_global_trie()

	test_env = Environment(
		tokenizer=model.tokenizer,
		data_path=model.hparams.test_data_path,
		max_length=model.hparams.max_length,
		max_length_span=model.hparams.max_length_span,
		metids=model.metids, 
		batch_size=1,
		test=True,
	)
	model.hparams.threshold = -3.2

	fields = ["micro_f1_sdm", "md_micro_f1_sdm", "macro_f1_sdm", 
		"md_macro_f1_sdm", "micro_prec_sdm", "micro_rec_sdm"]
	with open(join(args.res_dir, "%s.%s.txt"%(args.model, args.linking_mode)), "a" if args.append else "w", encoding="utf-8") as fout: 
		rs = "model\t%s\n"%"\t".join(fields)
		print(rs, end="")
		if not args.append:
			fout.write(rs)
		model_dir = join("./models", args.model, "checkpoints")
		print("Testing", args.model)
		for mfn in sorted(os.listdir(model_dir), key=lambda t: -os.stat(join(model_dir, t)).st_mtime):
			load_weights(model, join(model_dir, mfn))
			model = model.eval()
			# model.hparams.test_with_beam_search = True
			# model.hparams.test_with_beam_search_no_candidates = False

			ret = trainer.run_test(model, 
				test_env, 
				max_steps=300, 
				linking_mode=args.linking_mode, 
				md_threshold=model.hparams.threshold,
				test_sdm=model.hparams.threshold,
			)
			rs = "%s\t%s\n"%(mfn, "\t".join(str(ret[f]) for f in fields))
			print(rs, end="")
			fout.write(rs)
			fout.flush()
		