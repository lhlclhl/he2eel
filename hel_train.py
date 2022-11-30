import os
from argparse import ArgumentParser
from pprint import pprint
import torch

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything

from src.hierarchical_el import HierarchicalEL, HELActorCritic
from src.environment import Environment, EnvironmentMiR, EnvironmentSUM
from src.data.dataset_el import DatasetEL
from src.rl import *

def load_pretrained_weights(model, pretrain_path):
	state_dict = torch.load(pretrain_path)
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
	print("reform loading...", end="")
	msd = model.state_dict()
	for k in [
		"encoder.embeddings.word_embeddings.weight",
		"entity_linking.lstm.embeddings.weight",
		"entity_linking.lstm.lm_head.bias",
		"entity_linking.lstm.lm_head.decoder.weight",
		"entity_linking.lstm.lm_head.decoder.bias",
	]:
		state_dict[k] = state_dict[k][:len(msd[k])]
	for k in [
		"encoder.embeddings.new_word_embeddings.weight"
	]:
		state_dict[k] = msd[k]
	model.load_state_dict(
		state_dict,
		strict=False,
	)
	print("success!")

if __name__ == "__main__":
	parser = ArgumentParser()

	parser.add_argument("--dirpath", type=str, default="models")
	parser.add_argument("--model_save_path", type=str, default="models/rl")
	parser.add_argument("--pretrained", type=str, default=None)
	parser.add_argument("--save_top_k", type=int, default=10)
	parser.add_argument("--every_n_train_steps", type=int, default=None)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--n_epochs", type=int, default=50)
	parser.add_argument("--critic_update", type=float, default=0.5)
	parser.add_argument("--actor_update", type=float, default=0.1)
	parser.add_argument("--rl_epsilon", type=float, default=0.1)
	parser.add_argument("--rl_batch", type=int, default=4)
	parser.add_argument("--md_strategy", type=int, default=2)
	parser.add_argument("--md_threshold", type=float, default=0)
	parser.add_argument("--test_mdt", type=float, default=None)
	parser.add_argument("--st", action="store_true") 
	parser.add_argument("--mode", type=int, default=1)
	parser.add_argument("--fix_controller", action="store_true")
	parser.add_argument("--append", action="store_true")
	parser.add_argument("--disable_st", action="store_true")
	parser.add_argument("--val_every_n_episodes", type=int, default=None)
	parser.add_argument("--linking_mode", type=str, default="beam")

	parser = HierarchicalEL.add_model_specific_args(parser)
	parser = Trainer.add_argparse_args(parser)

	args, _ = parser.parse_known_args()
	pprint(args.__dict__)

	seed_everything(seed=args.seed)

	if args.st:
		args.mentions_filename = "./data/mentions.json"
		model_class = HELActorCritic
		Environment = EnvironmentSUM
		if args.mode == 0:
			RLTrainer = DRLTrainer
		else: 
			RLTrainer = DRLTrainerAFM
		model = model_class(**vars(args))
		if args.pretrained is None: pass
		elif args.pretrained.endswith(".torch"):
			print("loading", args.pretrained)
			load_pretrained_weights(model, args.pretrained)
		else:
			if os.path.isdir(args.pretrained):
				modfn = sorted([(fn, [
					float(param.split("=")[1]) 
					for param in fn.split("-") 
					if param.split("=")[0]=="micro_f1"
				][0]) for fn in os.listdir(args.pretrained)], key=lambda x:-x[1])[0][0]
					
				model_path = os.path.join(args.pretrained, modfn)
			else: model_path = args.pretrained
			
			model = model_class.load_from_checkpoint(model_path, 
				mentions_filename = "./data/mentions.json",#"./data/mentions_new.json",
				#entities_filename = "./data/entities.json"
			).eval()

		if args.linking_mode != "match":
			model.generate_global_trie()
		# model.hparams.threshold = -2
		# model.hparams.test_with_beam_search = True
		# model.hparams.test_with_beam_search_no_candidates = True

		env = Environment(
			tokenizer=model.tokenizer,
			data_path=model.hparams.train_data_path,
			max_length=model.hparams.max_length_train,
			max_length_span=model.hparams.max_length_span,
			metids=model.metids, 
			batch_size=args.batch_size,
		)

		trainer = RLTrainer(device="cuda")
		run_func = trainer.run_training
		run_func(
			model, env, 
			model_dir=args.model_save_path, 
			linking_mode=args.linking_mode,
			epsilon=args.rl_epsilon,
			md_strategy=args.md_strategy,
			val_env=Environment(
				tokenizer=model.tokenizer,
				data_path=model.hparams.dev_data_path,
				max_length=model.hparams.max_length,
				max_length_span=model.hparams.max_length_span,
				metids=model.metids, 
				batch_size=1,
				test=True,
				shuffle=False,
			),
			append=args.append,
			monitors={"micro_f1": True, "md_micro_f1": True} if args.disable_st else {"micro_f1": True, "micro_f1_sdm": True},
			val_every_n_episodes=args.val_every_n_episodes,
			fix_controller=args.fix_controller,
			training_bs=args.rl_batch,
			md_threshold=args.md_threshold,
			test_mdt=args.test_mdt,
			critic_update=args.critic_update,
			actor_update=args.actor_update,
			n_epochs=args.n_epochs,
			test_sdm=not args.disable_st,
		)
		#RLTrainer().run_interactive_test(model, env)
	else:
		logger = TensorBoardLogger(args.dirpath, name=None)
		callbacks = [
			ModelCheckpoint(
				mode="max",
				monitor="micro_f1",
				dirpath=os.path.join(logger.log_dir, "checkpoints"),
				save_top_k=args.save_top_k,
				filename="model-{epoch:02d}-{micro_f1:.4f}-{md_micro_f1:.4f}" if args.disable_st else "model-{epoch:02d}-{micro_f1:.4f}-{ed_micro_f1:.4f}",
				every_n_train_steps=args.every_n_train_steps,
				#val_check_interval = args.every_n_train_steps/776xxx
			),
			LearningRateMonitor(
				logging_interval="step",
			),
			EarlyStopping(
				monitor="micro_f1", mode="max", patience=50
			)
		]

		trainer = Trainer.from_argparse_args(args, logger=logger, callbacks=callbacks, gpus=1)

		if args.pretrained is None:
			model = HierarchicalEL(**vars(args))
		elif "checkpoints" in args.pretrained:
			print("loading pretrained model", args.pretrained)
			model = HierarchicalEL.load_from_checkpoint(args.pretrained, **vars(args))
			torch.save(model.state_dict(), "./models/pretrained_wiki_tw_e1.torch")
		else: 
			model = HierarchicalEL(**vars(args))
			load_pretrained_weights(model, args.pretrained)

		trainer.fit(model)
