
import torch, random, jsonlines, numpy as np, math
from torch.nn.utils.rnn import pad_sequence
from .data.dataset_el import get_negative_entity
from .data.data_utils import ment2ent
HARD_MAX_LEN = 4096

class Environment(): # as dataset for RL

	def __init__(self, tokenizer, data_path, max_length, max_length_span, metids, test=False, \
	batch_size=2, shuffle=True, mention_reward_factor=0.5, additional_data_path=None, add_ratio=None):
		self.tokenizer = tokenizer

		with jsonlines.open(data_path) as f:
			self.data = list(f)
		self.src_offset_mappings = [None for _ in self.data]

		self.max_length = max_length
		self.max_length_span = max_length_span
		self.test = test

		self.metids = metids
		self.indices = list(range(len(self.data)))
		self.shuffle = shuffle
		self.batch_size = batch_size
		self.mrf = mention_reward_factor
		self.orig_n = self._n_samples = len(self.data)
		
		if additional_data_path and add_ratio:
			with jsonlines.open(additional_data_path) as f:
				self.data += list(f)
			self.add_size = min(int(add_ratio*self.orig_n), len(self.data)-self.orig_n)
			self._n_samples += self.add_size
			
			self.src_offset_mappings += [None] * len(self.data)
		else:
			self.add_size = None

		self.initialize()

	@property
	def n_batches(self):
		return math.ceil(self._n_samples/self.batch_size)
	
	@property
	def n_samples(self):
		return self._n_samples

	# initialize an epoch
	def initialize(self):
		self.data_pointer = 0
		
		if self.add_size:
			sampled_add_ids = random.sample(list(range(self.orig_n, len(self.data))), self.add_size)
			self.indices = list(range(self.orig_n)) + sampled_add_ids

		if self.shuffle: random.shuffle(self.indices)

	# extract a batch of states (to store in replay memory)
	def extract(self, batch):
		seqlens = batch["src_attention_mask"].sum(-1).int().flatten().tolist()
		states = []
		for i in range(len(seqlens)):
			states.append((
				batch["src_input_ids"][i][:seqlens[i]],
				batch["detection_mask"][i],
				batch["original_index"][i],
				batch["src_data_ids"][i],
			) if not batch["terminal_flags"][i] else None)
		return states

	# restore a batch of states (terminal state skipped)
	def restore(self, states): 
		src_input_ids = []
		src_attention_mask = []
		detection_mask = [] 
		original_index = [] 
		src_data_ids = []

		for state in states:
			if state is not None:
				input_ids, dmask, oindex, data_id = state
				src_input_ids.append(input_ids)
				src_attention_mask.append(torch.ones(len(input_ids)))
				detection_mask.append(dmask)
				original_index.append(oindex)
				src_data_ids.append(data_id)

		batch = {
			"src_input_ids": pad_sequence(src_input_ids, batch_first=True), 
			"src_attention_mask": pad_sequence(src_attention_mask, batch_first=True), 
			"src_offset_mapping": [self.src_offset_mappings[i] for i in src_data_ids],
			"detection_mask": detection_mask,
			"original_index": original_index,
			"terminal_flags": [0] * len(src_data_ids),
			"src_data_ids": src_data_ids,
			"in_domain_flags": [i < self.orig_n for i in src_data_ids],
			"raw": [self.data[i] for i in src_data_ids],
			#"candidates": self.get_candidates(src_data_ids),
		}
		batch["candidates"] = self.get_all_candidates(batch, src_data_ids)
		return batch

	def get_candidates(self, src_data_ids):
		# construct candidate entity list for a batch of data: key(idx, start, end) -> List[tokenized entities]
		candidates = {}
		for i, data_id in enumerate(src_data_ids):
			for (s, e, g), c in zip(self.data[data_id]["anchors"], self.data[data_id]["candidates"]):
				if c: candidates[i, s, e] = self.tokenizer(c)["input_ids"]
		return candidates

	def get_all_candidates(self, batch, src_data_ids):
		# construct candidate entity list for a batch of data: key(idx, start, end) -> List[tokenized entities]
		candidates = {}
		for i, data_id in enumerate(src_data_ids):
			for (s, e, g), c in zip(self.data[data_id]["anchors"], self.data[data_id]["candidates"]):
				if c: candidates[i, s, e] = self.tokenizer(c)["input_ids"]
		
		seqlens = batch["src_attention_mask"].sum(-1).int().tolist()
		original_index = batch.get("original_index")
		for i in range(len(batch["raw"])):
			if original_index: oi = original_index[i]
			else: oi = list(range(seqlens[i]))

			for s in range(1, seqlens[i]):

				start = batch["src_offset_mapping"][i][oi[s]][0].item()
				for e in range(s, min(seqlens[i], s+self.max_length_span)):
					
					end = batch["src_offset_mapping"][i][oi[e]][1].item()
					
					cands = ment2ent.get(batch["raw"][i]["input"][start:end])
					
					if (i, s, e) not in candidates and cands:
						candidates[i, s, e] = self.tokenizer(cands)["input_ids"]

		return candidates

	# get a batch of data samples as inital states
	def reset(self, data=None):
		if data is None:
			if self.data_pointer >= self._n_samples:
				self.initialize()

			src_data_ids = [self.indices[i]
				for i in range(self.data_pointer, self.data_pointer+self.batch_size) 
				if i < len(self.indices)
			]
			batch = [self.data[i] for i in src_data_ids]

			self.data_pointer += self.batch_size
		else:
			src_data_ids = [d.pop("src_data_id") for d in data]
			batch = data
			
		batch = self.collate_fn(batch)

		terminal_flags = []
		self.metrics = []
		self.targets = []
		for i in range(len(batch["raw"])):
			terminal_flags.append(0)
			self.metrics.append([0, 0, 0, 0]) # truth positives (detection), # truth positives (linking), # predictions, # targets
			self.targets.append({})
			for s, e, t in batch["raw"][i]["anchors"]:
				self.targets[i][s, e] = t
				self.metrics[i][3] += 1
			if src_data_ids[i] is not None:
				self.src_offset_mappings[src_data_ids[i]] = batch["src_offset_mapping"][i]
			
		detection_mask = []; original_index = []
		for seqlen in batch["src_attention_mask"].sum(-1).flatten().tolist():
			detection_mask.append([1 for _ in range(seqlen)])
			original_index.append(list(range(seqlen)))

		return {
			**batch,
			"detection_mask": detection_mask,
			"original_index": original_index,
			"terminal_flags": terminal_flags,
			"src_data_ids": src_data_ids,
			"in_domain_flags": [i < self.orig_n for i in src_data_ids],
			#"candidates": self.get_candidates(src_data_ids),
			"candidates": self.get_all_candidates(batch, src_data_ids) if data is None else None,
		}

	def calc_f1_var(self, n_tp_d, n_tp_l, n_predictions, n_targets):
		n_truth_positives = self.mrf * n_tp_d + (1-self.mrf) * n_tp_l
		return (3 * n_truth_positives - n_predictions) / (n_predictions + n_targets + 1e-9)

	def calc_f1(self, n_tp_d, n_tp_l, n_predictions, n_targets): # as macro f1
		n_truth_positives = self.mrf * n_tp_d + (1-self.mrf) * n_tp_l
		return 2 * n_truth_positives / (n_predictions + n_targets + 1e-9)

	def reward(self, i, s, e, ent):
		if s == e == 0: return 0
		previous_f1 = self.calc_f1(*self.metrics[i])
		self.metrics[i][0] += int((s, e) in self.targets[i])
		self.metrics[i][1] += int(ent == self.targets[i].get((s, e)))
		# if (s, e) in self.targets[i] and ent != self.targets[i].get((s, e)):
		# 	print("prediction(%s)!=target(%s)"%(ent, self.targets[i].get((s, e))))
		self.metrics[i][2] += 1
		current_f1 = self.calc_f1(*self.metrics[i])
		return current_f1 - previous_f1 # return the difference of f1 with previous state as immediate reward

	# execute (a batch of) action to (a batch of) state and return (a batch of) new state
	def step(self, batch, action): 
		start, end, entity_tokens = action
		new_src_input_ids = []
		new_src_attention_mask = []
		new_detection_mask = [] # texts can be detect
		new_original_index = [] # current text idx to original text idx
		rewards = []
		
		seqlens = batch["src_attention_mask"].sum(-1).int().tolist()
		detection_mask = batch["detection_mask"]
		original_index = batch["original_index"]
		for i, seq_ids in enumerate(batch["src_input_ids"]):
			assert start[0][i] == i and end[0][i] == i, "start: %s, end: %s" % (start, end)
			if start[1][i] == end[1][i] == 0: 
				batch["terminal_flags"][i] = 1

			if batch["terminal_flags"][i]:
				new_src_input_ids.append(seq_ids)
				new_src_attention_mask.append(batch["src_attention_mask"][i])
				new_detection_mask.append([1] + [0] * (len(detection_mask[i])-1)) # special mask for terminal state
				new_original_index.append(original_index[i])
				rewards.append(0)
				continue

			etk = entity_tokens[i][0].cpu()
			non_special = [int(t) not in self.tokenizer.all_special_ids for t in etk]
			etk = etk[non_special]
			ent_str = self.tokenizer.decode(etk)
			org_s, org_e = original_index[i][start[1][i]], original_index[i][end[1][i]]

			rewards.append(self.reward(i, org_s, org_e, ent_str))
			seq_ids = seq_ids.cpu()[:seqlens[i]]
			ofs1, ofs2 = start[1][i], end[1][i]+1
			pre_context, mention, post_context = seq_ids.split([ofs1, ofs2-ofs1, seqlens[i]-ofs2])
			new_seq_ids = torch.cat([
				pre_context,
				torch.tensor(self.metids[:1]),
				mention,
				torch.tensor(self.metids[1:3]),
				etk,
				torch.tensor(self.metids[3:]),
				post_context,
			], dim=0)
			new_src_input_ids.append(new_seq_ids)
			new_src_attention_mask.append(torch.ones(len(new_seq_ids)))
			new_detection_mask.append(
				detection_mask[i][:ofs1] +
				[0] * (4 + ofs2-ofs1 + len(etk))
				+ detection_mask[i][ofs2:]
			)
			new_original_index.append(
				original_index[i][:ofs1] +
				[-1] + original_index[i][ofs1:ofs2] 
				+ [-1] * (len(etk)+3)
				+ original_index[i][ofs2:]
			)
			if len(new_src_input_ids[-1]) >= HARD_MAX_LEN:
				new_src_input_ids[-1] = new_src_input_ids[-1][:HARD_MAX_LEN]
				new_src_attention_mask[-1] = new_src_attention_mask[-1][:HARD_MAX_LEN]
				new_detection_mask[-1] = new_detection_mask[-1][:HARD_MAX_LEN]
				new_original_index[-1] = new_original_index[-1][:HARD_MAX_LEN]

		new_src_input_ids = pad_sequence(new_src_input_ids, batch_first=True).to(batch["src_input_ids"].device)
		new_src_attention_mask = pad_sequence(new_src_attention_mask, batch_first=True).to(batch["src_attention_mask"].device)
		
		batch.update([
			("src_input_ids", new_src_input_ids),
			("src_attention_mask", new_src_attention_mask),
			("detection_mask", new_detection_mask),
			("original_index", new_original_index),
		])

		return batch, rewards

	# execute one actions to a state in a batch
	def step_one(self, batch, idx, start, end, entity_tokens):
		new_src_input_ids = []
		new_src_attention_mask = []
		new_detection_mask = [] # texts can be detect
		new_original_index = [] # current text idx to original text idx
		
		seqlens = batch["src_attention_mask"].sum(-1).int().tolist()
		detection_mask = batch["detection_mask"]
		original_index = batch["original_index"]
		for i, seq_ids in enumerate(batch["src_input_ids"]):
			if idx != i:
				new_src_input_ids.append(seq_ids)
				new_src_attention_mask.append(batch["src_attention_mask"][i])
				new_detection_mask.append(detection_mask[i])
				new_original_index.append(original_index[i])
				continue

			etk = entity_tokens.cpu()
			non_special = [int(t) not in self.tokenizer.all_special_ids for t in etk]
			etk = etk[non_special]
			ent_str = self.tokenizer.decode(etk)
			org_s, org_e = original_index[i][start], original_index[i][end]

			reward = self.reward(i, org_s, org_e, ent_str)
			seq_ids = seq_ids.cpu()[:seqlens[i]]
			ofs1, ofs2 = start, end+1
			pre_context, mention, post_context = seq_ids.split([ofs1, ofs2-ofs1, seqlens[i]-ofs2])
			new_seq_ids = torch.cat([
				pre_context,
				torch.tensor(self.metids[:1]),
				mention,
				torch.tensor(self.metids[1:3]),
				etk,
				torch.tensor(self.metids[3:]),
				post_context,
			], dim=0)
			new_src_input_ids.append(new_seq_ids)
			new_src_attention_mask.append(torch.ones(len(new_seq_ids)))
			new_detection_mask.append(
				detection_mask[i][:ofs1] +
				[0] * (4 + ofs2-ofs1 + len(etk))
				+ detection_mask[i][ofs2:]
			)
			new_original_index.append(
				original_index[i][:ofs1] +
				[-1] + original_index[i][ofs1:ofs2] 
				+ [-1] * (len(etk)+3)
				+ original_index[i][ofs2:]
			)

		new_src_input_ids = pad_sequence(new_src_input_ids, batch_first=True).to(batch["src_input_ids"].device)
		new_src_attention_mask = pad_sequence(new_src_attention_mask, batch_first=True).to(batch["src_attention_mask"].device)
		
		batch.update([
			("src_input_ids", new_src_input_ids),
			("src_attention_mask", new_src_attention_mask),
			("detection_mask", new_detection_mask),
			("original_index", new_original_index),
		])

		return batch, reward
	
	def collate_fn(self, batch):
		batch = {
			**{
				f"src_{k}": v
				for k, v in self.tokenizer(
					[b["input"] for b in batch],
					return_tensors="pt",
					padding=True,
					max_length=self.max_length,
					truncation=True,
					return_offsets_mapping=True,
				).items()
			},
			"raw": batch,
			"target_offsets_start": (
				[
					i
					for i, b in enumerate(batch)
					for a in b["anchors"]
					if a[1] < self.max_length and a[1] - a[0] < self.max_length_span
				],
				[
					a[0]
					for i, b in enumerate(batch)
					for a in b["anchors"]
					if a[1] < self.max_length and a[1] - a[0] < self.max_length_span
				],
			),
			"target_offsets_end": (
				[
					i
					for i, b in enumerate(batch)
					for a in b["anchors"]
					if a[1] < self.max_length and a[1] - a[0] < self.max_length_span
				],
				[
					a[1]
					for i, b in enumerate(batch)
					for a in b["anchors"]
					if a[1] < self.max_length and a[1] - a[0] < self.max_length_span
				],
			),
		}

		if not self.test:

			negatives = [
				np.random.choice([e for e in cands if e != a[2]])
				if len([e for e in cands if e != a[2]]) > 0
				else get_negative_entity()
				for b in batch["raw"]
				for a, cands in zip(b["anchors"], b["candidates"])
				if a[1] < self.max_length and a[1] - a[0] < self.max_length_span
			]

			targets = [
				a[2]
				for b in batch["raw"]
				for a in b["anchors"]
				if a[1] < self.max_length and a[1] - a[0] < self.max_length_span
			]

			assert len(targets) == len(negatives)

			batch_upd = {
				**(
					{
						f"trg_{k}": v
						for k, v in self.tokenizer(
							targets,
							return_tensors="pt",
							padding=True,
							max_length=self.max_length,
							truncation=True,
						).items()
					}
					if not self.test
					else {}
				),
				**(
					{
						f"neg_{k}": v
						for k, v in self.tokenizer(
							[e for e in negatives if e],
							return_tensors="pt",
							padding=True,
							max_length=self.max_length,
							truncation=True,
						).items()
					}
					if not self.test
					else {}
				),
				"neg_mask": torch.tensor([e is not None for e in negatives]),
			}

			batch = {**batch, **batch_upd}
		
		return batch
	
	def add_targets(self, batch, device):
		batch = {		
			**batch,
			"target_offsets_start": (
				[
					i
					for i, b in enumerate(batch['raw'])
					for a in b["anchors"]
					if a[1] < self.max_length and a[1] - a[0] < self.max_length_span
				],
				[
					a[0]
					for i, b in enumerate(batch['raw'])
					for a in b["anchors"]
					if a[1] < self.max_length and a[1] - a[0] < self.max_length_span
				],
			),
			"target_offsets_end": (
				[
					i
					for i, b in enumerate(batch['raw'])
					for a in b["anchors"]
					if a[1] < self.max_length and a[1] - a[0] < self.max_length_span
				],
				[
					a[1]
					for i, b in enumerate(batch['raw'])
					for a in b["anchors"]
					if a[1] < self.max_length and a[1] - a[0] < self.max_length_span
				],
			),
		}

		negatives = [
			np.random.choice([e for e in cands if e != a[2]])
			if len([e for e in cands if e != a[2]]) > 0
			else get_negative_entity()
			for b in batch["raw"]
			for a, cands in zip(b["anchors"], b["candidates"])
			if a[1] < self.max_length and a[1] - a[0] < self.max_length_span
		]

		targets = [
			a[2]
			for b in batch["raw"]
			for a in b["anchors"]
			if a[1] < self.max_length and a[1] - a[0] < self.max_length_span
		]

		assert len(targets) == len(negatives)

		batch_upd = {
			**(
				{
					f"trg_{k}": v.to(device)
					for k, v in self.tokenizer(
						targets,
						return_tensors="pt",
						padding=True,
						max_length=self.max_length,
						truncation=True,
					).items()
				}
				if not self.test
				else {}
			),
			**(
				{
					f"neg_{k}": v.to(device)
					for k, v in self.tokenizer(
						[e for e in negatives if e],
						return_tensors="pt",
						padding=True,
						max_length=self.max_length,
						truncation=True,
					).items()
				}
				if not self.test
				else {}
			),
			"neg_mask": torch.tensor([e is not None for e in negatives]),
		}
		return {**batch, **batch_upd}

class EnvironmentMiR(Environment): # Micro Reward
	def calc_f1(self, n_tp_d, n_tp_l, n_predictions, n_targets): # as micro f1
		n_truth_positives = self.mrf * n_tp_d + (1-self.mrf) * n_tp_l
		return 2 * n_truth_positives / (n_predictions + n_targets + 1e-9) * n_targets

class EnvironmentSUM(Environment): # Reward with simple sum of each step
	def calc_f1(self, n_tp_d, n_tp_l, n_predictions, n_targets): # as micro f1
		n_truth_positives = self.mrf * n_tp_d + (1-self.mrf) * n_tp_l
		return 2 * n_truth_positives - n_predictions

if __name__ == "__main__":
	pass