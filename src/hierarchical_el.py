import torch, random
from collections import defaultdict
from transformers import (
	AutoTokenizer,
	LongformerForMaskedLM,
)
from argparse import ArgumentParser

from .model.efficient_el import EfficientEL
from .model.high_level_detection import *
from .model.low_level_linking import *
from .utils import (
	MacroF1,
	MacroPrecision,
	MacroRecall,
	MicroF1,
	MicroPrecision,
	MicroRecall,
	LongformerForMaskedLM1
)
from torch.nn.utils.rnn import pad_sequence

class HierarchicalEL(EfficientEL): # sequential decision making modeling, no meta-controller
	@staticmethod
	def add_model_specific_args(parent_parser):
		parser = EfficientEL.add_model_specific_args(parent_parser)
		for add_hparam, default_val in HierarchicalEL.additional_hparams():
			if type(default_val) == bool:
				parser.add_argument("--%s"%add_hparam, action="store_true")
			else:
				parser.add_argument("--%s"%add_hparam, type=type(default_val), default=default_val)
		return parser
	@staticmethod
	def additional_hparams(): return [
		("detection_model", "doubleend2"),
		("linking_model", "default"),
		("train_word_embeddings", False),
		("lstm_batch_size", 512),
		("lm_weight", 1.),
		("im_reward_pos", 1.),
		("im_reward_neg", -1.),
		("additional_data_path", ""),
		("additional_ratio", 0.),
	]
	def __init__(self, *args, **kwargs):
		super(EfficientEL, self).__init__()
		self.save_hyperparameters()

		for add_hparam, default_val in HierarchicalEL.additional_hparams():
			if not hasattr(self.hparams, add_hparam):
				setattr(self.hparams, add_hparam, default_val)

		if self.hparams.detection_model == "onlyend":
			detection_class = MentionDetectionOnlyEnd 
		elif self.hparams.detection_model == "doubleend2":
			detection_class = MentionDetectionDoubleEnd2
		else:
			raise Exception("Unknown detection model: %s"%self.hparams.detection_model)

		if self.hparams.linking_model == "default":
			linking_class = EntityLinking
		elif self.hparams.linking_model == "orl":
			linking_class = EntityLinkingORL
		elif self.hparams.linking_model == "is":
			linking_class = EntityLinkingIS
		elif self.hparams.linking_model == "adv":
			linking_class = AdversarialEntityLinking
		else:
			raise Exception("Unknown linking model: %s"%self.hparams.linking_training)
		
		print("linking model:", linking_class.__name__)

		self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name)

		longformer = LongformerForMaskedLM1.from_pretrained(
			self.hparams.model_name,
			num_hidden_layers=8,
			attention_window=[128] * 8,
		)
		self.encoder = longformer.longformer

		metags = ["<mstart>", "<mend>", "<estart>", "<eend>"]
		# if self.tokenizer.add_tokens(metags) > 0:
		# 	longformer.resize_token_embeddings(len(self.tokenizer))
		if self.encoder.embeddings.new_word_embeddings is None:
			if self.tokenizer.add_tokens(metags) <= 0:
				self.encoder.resize_token_embeddings(len(self.tokenizer)-len(metags))
			self.encoder.add_new_word_embeddings(len(metags))
		self.metids = self.tokenizer.convert_tokens_to_ids(metags)

		self.encoder.embeddings.word_embeddings.weight.requires_grad_(self.hparams.train_word_embeddings)

		self.entity_detection = detection_class(
			self.hparams.max_length_span,
			self.hparams.dropout,
			mentions_filename=self.hparams.mentions_filename,
		)

		self.entity_linking = linking_class(
			self.tokenizer.bos_token_id,
			self.tokenizer.pad_token_id,
			self.tokenizer.eos_token_id,
			self.encoder.embeddings.word_embeddings,
			longformer.lm_head,
			self.hparams.dropout,
		)

		self.micro_f1 = MicroF1()
		self.micro_prec = MicroPrecision()
		self.micro_rec = MicroRecall()

		self.macro_f1 = MacroF1()
		self.macro_prec = MacroPrecision()
		self.macro_rec = MacroRecall()

		self.ed_micro_f1 = MicroF1()
		self.ed_micro_prec = MicroPrecision()
		self.ed_micro_rec = MicroRecall()

		self.ed_macro_f1 = MacroF1()
		self.ed_macro_prec = MacroPrecision()
		self.ed_macro_rec = MacroRecall()

		self.metrics = {
			"micro_f1": self.micro_f1,
			"micro_prec": self.micro_prec,
			"macro_rec": self.macro_rec,
			"macro_f1": self.macro_f1,
			"macro_prec": self.macro_prec,
			"micro_rec": self.micro_rec,
			"ed_micro_f1": self.ed_micro_f1,
			"ed_micro_prec": self.ed_micro_prec,
			"ed_micro_rec": self.ed_micro_rec,
			"ed_macro_f1": self.ed_macro_f1,
			"ed_macro_prec": self.ed_macro_prec,
			"ed_macro_rec": self.ed_macro_rec,
		}

	def _tokens_scores_to_spans(self, batch, start, end, tokens, scores_el, scores_ed=None):
		if scores_ed is None: return super()._tokens_scores_to_spans(batch, start, end, tokens, scores_el)

		spans = [
			[
				[
					s,
					e,
					list(
						zip(
							self.tokenizer.batch_decode(t, skip_special_tokens=True),
							d.tolist(),
						)
					),
				]
				for s, e, t, l, d in zip(
					start[start[:, 0] == i][:, 1].tolist(),
					end[end[:, 0] == i][:, 1].tolist(),
					tokens[start[:, 0] == i],
					scores_el[start[:, 0] == i],
					scores_ed[start[:, 0] == i].unsqueeze(-1),
				)
			]
			for i in range(len(batch["src_input_ids"]))
		]

		for i in range(len(spans)):
			spans[i].sort(key=lambda x:x[2][0][1]) # rank spans by score, maximum last
			new_spans = []
			while spans[i]:
				x = spans[i].pop() # pop out the maximum mention
				overlapped = False
				for y in new_spans:
					if (x[1]-y[0]) * (y[1]-x[0]) >= 0: # overlap checking
						overlapped = True
						break
				if not overlapped: new_spans.append(x)
			new_spans.sort(key=lambda x:x[0]) # restore the span order
			spans[i] = new_spans

		return spans

	def encode(self, batch):
		device = self.encoder.device
		return self.encoder(
			input_ids=batch["src_input_ids"].to(device), 
			attention_mask=batch["src_attention_mask"].to(device)
		).last_hidden_state
		
	def linking_via_beam_search(self, batch, hidden_states, return_dict=False):
		candidates = batch.get("candidates", {})
		# beam search forward, with conditional candidates
		batch_trie_dict = []
		for (i, s), (_, e) in zip(zip(*batch["offsets_start"]), zip(*batch["offsets_end"])):
			s, e = batch["original_index"][i][s], batch["original_index"][i][e]
			cands = candidates.get((i, s, e))
			if cands:
				trie_dict = defaultdict(set)
				for c in cands:
					for i in range(1, len(c)):
						trie_dict[tuple(c[:i])].add(c[i])
				batch_trie_dict.append({k: list(v) for k, v in trie_dict.items()})
			else:
				batch_trie_dict.append(self.global_trie)

		#print("Before beam search: LSTMModule(%s): "%id(self.entity_linking.lstm), self.entity_linking.lstm.training)
		return self.entity_linking.forward_beam_search(
			batch, hidden_states,  batch_trie_dict, return_dict=return_dict, lm_weight=self.hparams.lm_weight
		)
		#print("After beam search: LSTMModule(%s): "%id(self.entity_linking.lstm), self.entity_linking.lstm.training)

	def linking_via_candidate_matching(self, batch, hidden_states, return_dict=False):
		candidates = batch.get("candidates", {})
		empty_cands = self.tokenizer(["NIL"])["input_ids"]
		all_candidates = []
		for (i, s), (_, e) in zip(zip(*batch["offsets_start"]), zip(*batch["offsets_end"])):
			s, e = batch["original_index"][i][s], batch["original_index"][i][e]
			all_candidates.append(candidates.get((i, s, e), empty_cands))

		batch_candidates = []
		split_candidates = []
		scores_clf = scores_lm = None
		tokens = []
		for i, cands in enumerate(all_candidates):
			batch_candidates.append(cands)

			if i+1 < len(all_candidates) and sum(len(c) for c in batch_candidates) \
			+ len(all_candidates[i+1]) < self.hparams.lstm_batch_size: continue

			# for k, v in self.tokenizer(
			# 	[c for candidates in batch_candidates for c in candidates],
			# 	return_tensors="pt",
			# 	padding=True,
			# ).items():
			# 	batch[f"cand_{k}"] = v.to(self.device)
			cand_input_ids = [torch.tensor(c) for candidates in batch_candidates for c in candidates]
			batch["cand_input_ids"] = pad_sequence(cand_input_ids, batch_first=True).to(self.device)
			batch["cand_attention_mask"] = (batch["cand_input_ids"] != 0).int()

			batch["offsets_candidates"] = [
				len(split_candidates) + i
				for i, candidates in enumerate(batch_candidates)
				for _ in range(len(candidates))
			]
			split_candidates += [
				len(candidates) for candidates in batch_candidates
			]
			
			lm_scores, classifier_scores = self.entity_linking.forward_all_targets(
				batch, hidden_states
			)
				
			#elscore = (lm_scores + classifier_scores).detach().cpu()

			# tokens: (mention x candidate, seq)
			tokens += [t for t in batch["cand_input_ids"].cpu()]
			# scores: (mention x candidate, )
			#if scores_el is None: scores_el = elscore
			if scores_lm is None: scores_lm, scores_clf = lm_scores.detach().cpu(), classifier_scores.detach().cpu()
			else:
				scores_lm = torch.cat([
					scores_lm,
					lm_scores.detach().cpu()
				], dim=0)
				scores_clf = torch.cat([
					scores_clf,
					classifier_scores.detach().cpu()
				], dim=0)
			

			# classifier_scores = torch.cat(
			# 	[
			# 		e.log_softmax(-1)
			# 		for e in classifier_scores.split(split_candidates)
			# 	]
			# )

			batch_candidates = []
		scores_el = self.hparams.lm_weight * scores_lm + scores_clf
		
		# for tokens (mention x candidate, seq) -> (mention, candidate, cseq): padding twice
		tokens = torch.nn.utils.rnn.pad_sequence(
			tokens, batch_first=True, padding_value=self.tokenizer.pad_token_id
		)

		tokens = [
			t[s.argsort(descending=True)]
			for s, t in zip(
				scores_el.split(split_candidates),
				tokens.split(split_candidates),
			)
		]

		tokens = torch.nn.utils.rnn.pad_sequence(
			tokens, batch_first=True, padding_value=self.tokenizer.pad_token_id
		)

		# for scores (mention x candidate, ) -> (mention, candidate, ): padding once
		if return_dict:
			scores_el = []
			scores_lm_sorted = []
			scores_clf_sorted = []
			for lm, clf in zip(
				scores_lm.split(split_candidates),
				scores_clf.split(split_candidates),
			):
				es = (lm+clf).sort(descending=True)
				scores_el.append(es.values)
				scores_lm_sorted.append(lm[es.indices])
				scores_clf_sorted.append(clf[es.indices])
			scores_el = torch.nn.utils.rnn.pad_sequence(scores_el, batch_first=True, padding_value=-float("inf"))
			scores_lm = torch.nn.utils.rnn.pad_sequence(scores_lm_sorted, batch_first=True, padding_value=-float("inf"))
			scores_clf = torch.nn.utils.rnn.pad_sequence(scores_clf_sorted, batch_first=True, padding_value=-float("inf"))
			return {
				"tokens": tokens,
				"scores": scores_el,
				"classifier_scores": scores_clf,
				"lm_scores": scores_lm,
			}
		else:
			scores_el = torch.nn.utils.rnn.pad_sequence(
				[
					e.sort(descending=True).values
					for e in scores_el.split(split_candidates)
				],
				batch_first=True,
				padding_value=-float("inf"),
			)
			return tokens, scores_el

	def select_action(self, batch, policy="greedy", epsilon=0., \
	hidden_states=None, linking_mode="beam", threshold=None):
		if hidden_states is None:
			hidden_states = self.encode(batch)

		# low-level policy: detect a mention
		(start, end, scores), (mention_scores, mention_mask) = self.entity_detection(
			batch, 
			hidden_states, 
			threshold=threshold,
		)

		mention_logits = mention_scores + (mention_mask-1) * 1e9

		if policy=="greedy":
			if random.random() > epsilon:
				goals = mention_logits.argmax(-1)
			else:
				goals = torch.multinomial(mention_mask, 1).squeeze(-1)
		else:
			mention_probs = mention_logits.softmax(-1)
			goals = torch.multinomial(mention_probs, 1).squeeze(-1)

		# low-level policy: link a mention
		offsets = torch.tensor([0] + batch["mention_counts"][:-1]).to(goals.device).cumsum(0) # cumsum counts to offsets
		indices = goals + offsets
		batch["offsets_start"] = start[indices].T.tolist()
		batch["offsets_end"] = end[indices].T.tolist()

		if linking_mode == "beam":
			tokens, scores_el = self.linking_via_beam_search(batch, hidden_states)
		else:
			tokens, scores_el = self.linking_via_candidate_matching(batch, hidden_states)
		
		del batch["mention_start"], batch["mention_end"], batch["mention_counts"]

		return {
			"actions": (batch["offsets_start"], batch["offsets_end"], tokens),
			"goals": goals.cpu().detach(),
		}

	def get_current_index(self, original_index):
		current_index = []
		for i, oi in enumerate(original_index):
			ci = []
			for j, idx in enumerate(oi):
				if idx >= 0:
					assert idx == len(ci), "bad original index: %d in %s" % (idx, oi)
					ci.append(j)
			current_index.append(ci)
		return current_index

	def training_step(self, batch, batch_idx=None):
		try:
			hidden_states = self.encoder(
				input_ids=batch["src_input_ids"], attention_mask=batch["src_attention_mask"]
			).last_hidden_state

			loss_start, loss_end = self.entity_detection.forward_loss(batch, hidden_states)

			loss_generation, loss_classifier = self.entity_linking.forward_loss(
				batch, hidden_states, epsilon=self.hparams.epsilon
			)
		except Exception:
			import traceback
			traceback.print_exc()
			loss_start = loss_end = loss_generation = loss_classifier = 0

		self.log("loss_s", loss_start, on_step=True, on_epoch=False, prog_bar=True)
		self.log("loss_e", loss_end, on_step=True, on_epoch=False, prog_bar=True)
		self.log("loss_g", loss_generation, on_step=True, on_epoch=False, prog_bar=True)
		self.log("loss_c", loss_classifier, on_step=True, on_epoch=False, prog_bar=True)

		return {"loss": loss_start + loss_end + loss_generation + loss_classifier}

	def inner_training_step(self, batch, strategy=0):
		# mapping target_offsets to offsets based on original_index
		current_index = self.get_current_index(batch["original_index"])
		offsets_start, offsets_end = ([], []), ([], [])
		target_mask = []
		for i, ofs, _, ofe in zip(*batch["target_offsets_start"], *batch["target_offsets_end"]):
			cofs, cofe = current_index[i][ofs], current_index[i][ofe]
			check = all(batch["detection_mask"][i][j] for j in range(cofs, cofe+1))
			if check:
				offsets_start[0].append(i)
				offsets_start[1].append(cofs)
				offsets_end[0].append(i)
				offsets_end[1].append(cofe)
			target_mask.append(check)
		# mask targets 
		saved = {}
		for k in batch:
			if k.startswith("trg_") or k.startswith("neg_"):
				saved[k] = batch[k]
				batch[k] = batch[k][target_mask]
		batch["offsets_start"] = offsets_start
		batch["offsets_end"] = offsets_end
		
		hidden_states = self.encode(batch)

		loss_start, loss_end = self.entity_detection.forward_loss(batch, hidden_states, strategy=strategy)

		if offsets_start[0]:
			loss_generation, loss_classifier = self.entity_linking.forward_loss(
				batch, hidden_states, epsilon=self.hparams.epsilon
			)
		else: loss_generation = loss_classifier = 0

		while saved:
			k, v = saved.popitem()
			batch[k] = v

		return loss_start + loss_end + loss_generation + loss_classifier, hidden_states
	
	def freeze_encoder(self):
		for param in self.encoder.parameters():
			param.requires_grad = False

	def span_filtering(self, batch, start, end, scores_ed):
		spans = [
			[
				[
					s,
					e,
					d,
				]
				for s, e, d in zip(
					start[start[:, 0] == i][:, 1].tolist(),
					end[end[:, 0] == i][:, 1].tolist(),
					scores_ed[start[:, 0] == i].tolist(),
				)
			]
			for i in range(len(batch["src_input_ids"]))
		]

		new_index, new_start, new_end, new_scores = [], [], [], []
		for i in range(len(spans)):
			spans[i].sort(key=lambda x:x[2]) # rank spans by score, maximum last
			new_spans = []
			while spans[i]:
				x = spans[i].pop() # pop out the maximum mention
				overlapped = False
				for y in new_spans:
					if (x[1]-y[0]) * (y[1]-x[0]) >= 0: # overlap checking
						overlapped = True
						break
				if not overlapped: new_spans.append(x)
			new_spans.sort(key=lambda x:x[0]) # restore the span order
			for s, e, d in new_spans:
				new_index.append(i)
				new_start.append(s)
				new_end.append(e)
				new_scores.append(d)
		return [new_index, new_start], [new_index, new_end], torch.tensor(new_scores, device=scores_ed.device)

	def forward_all_targets(self, batch, hidden_states=None, threshold=None, return_dict=False): # pipeline prediction with the discriminator scores
		# encoding
		if hidden_states is None:
			hidden_states = self.encode(batch)

		# mention detection
		(start, end, scores_ed), (_, _) = self.entity_detection.forward_hard(
			batch, 
			hidden_states, 
			threshold=self.hparams.threshold,
		)
		if start.shape[0] == 0: return [[] for i in range(len(batch["src_input_ids"]))]
		offsets_start, offsets_end, scores_ed = self.span_filtering(batch, start, end, scores_ed)
		start, end = torch.tensor(offsets_start).T, torch.tensor(offsets_end).T
		batch["offsets_start"], batch["offsets_end"] = offsets_start, offsets_end

		# entity linking
		try:
			el_res = self.linking_via_candidate_matching(batch, hidden_states, return_dict=True)
			tokens = el_res["tokens"]
			scores_el = el_res["scores"]
		except:
			if not self.training:
				print("error on generation")
				import traceback; traceback.print_exc()

		# return as spans
		try:
			spans = [
				[
					[
						s,
						e,
						list(
							zip(
								self.tokenizer.batch_decode(t, skip_special_tokens=True),
								l.tolist(),
							)
						),
					]
					for s, e, t, l in zip(
						start[start[:, 0] == i][:, 1].tolist(),
						end[end[:, 0] == i][:, 1].tolist(),
						tokens[start[:, 0] == i],
						scores_el[start[:, 0] == i],
					)
				]
				for i in range(len(batch["src_input_ids"]))
			]
		except:
			if not self.training:
				print("error on _tokens_scores_to_spans")
				import traceback; traceback.print_exc()

			spans = [[[0, 0, [("NIL", 0)]]] for i in range(len(batch["src_input_ids"]))]

		if return_dict:
			return {
				"spans": spans,
				"start": start,
				"end": end,
				"scores_ed": scores_ed,
				"scores_el": scores_el,
			}
		else:
			return spans

	def forward_all_targets_ed(self, batch, hidden_states=None, threshold=None, return_dict=False): # pipeline prediction with the discriminator scores
		# encoding
		if hidden_states is None:
			hidden_states = self.encode(batch)

		# mention detection
		(start, end, scores_ed), (_, _) = self.entity_detection.forward_hard(
			batch, 
			hidden_states, 
			threshold=self.hparams.threshold,
		)
		if start.shape[0] == 0: return [[] for i in range(len(batch["src_input_ids"]))]

		batch["offsets_start"], batch["offsets_end"] = start.T.tolist(), end.T.tolist()

		# entity linking
		try:
			el_res = self.linking_via_candidate_matching(batch, hidden_states, return_dict=True)
			tokens = el_res["tokens"]
			scores_el = el_res["scores"]
		except:
			if not self.training:
				print("error on generation")
				import traceback; traceback.print_exc()

		# return as spans
		spans = [
			[
				[
					s,
					e,
					list(
						zip(
							self.tokenizer.batch_decode(t, skip_special_tokens=True),
							l.tolist(),
						)
					),
				]
				for s, e, t, l, d in zip(
					start[start[:, 0] == i][:, 1].tolist(),
					end[end[:, 0] == i][:, 1].tolist(),
					tokens[start[:, 0] == i],
					scores_el[start[:, 0] == i],
					scores_ed[start[:, 0] == i].unsqueeze(-1),
				)
			]
			for i in range(len(batch["src_input_ids"]))
		]

		for i in range(len(spans)):
			spans[i].sort(key=lambda x:x[2][0][1]) # rank spans by score, maximum last
			new_spans = []
			while spans[i]:
				x = spans[i].pop() # pop out the maximum mention
				overlapped = False
				for y in new_spans:
					if (x[1]-y[0]) * (y[1]-x[0]) >= 0: # overlap checking
						overlapped = True
						break
				if not overlapped: new_spans.append(x)
			new_spans.sort(key=lambda x:x[0]) # restore the span order
			spans[i] = new_spans

		if return_dict:
			return {
				"spans": spans,
				"start": start,
				"end": end,
				"scores_ed": scores_ed,
				"scores_el": scores_el,
			}
		else:
			return spans

class HELActorCritic(HierarchicalEL):
	def select_action(self, batch, policy="greedy", epsilon=0., \
	hidden_states=None, linking_mode="beam", threshold=None):
		if hidden_states is None:
			hidden_states = self.encode(batch)

		# high-level policy: detect a mention
		(start, end, _), (mention_scores, mention_mask) = self.entity_detection(
			batch, 
			hidden_states, 
			threshold=threshold,
		)

		mention_counts = batch["mention_counts"]
		mention_logits = mention_scores + (mention_mask-1) * 1e9

		if policy=="greedy":
			if random.random() > epsilon:
				goals = mention_logits.argmax(-1)
			else:
				goals = torch.multinomial(mention_mask, 1).squeeze(-1)
		else:
			
			mention_probs = mention_logits.softmax(-1)
			
			try:
				goals = torch.multinomial(mention_probs, 1).squeeze(-1)
			except Exception:
				print("multinomial error")
				goals = torch.multinomial(mention_mask, 1).squeeze(-1)

		# low-level policy: link a mention
		offsets = torch.tensor([0] + mention_counts[:-1]).to(goals.device).cumsum(0) # cumsum counts to offsets
		indices = goals + offsets
		batch["offsets_start"] = start[indices].T.tolist()
		batch["offsets_end"] = end[indices].T.tolist()

		if linking_mode == "beam":
			tokens, scores_el = self.linking_via_beam_search(batch, hidden_states)
		else:
			tokens, scores_el = self.linking_via_candidate_matching(batch, hidden_states)
		
		del batch["mention_start"], batch["mention_end"], batch["mention_counts"]

		return {
			"actions": (batch["offsets_start"], batch["offsets_end"], tokens),
		}

	def forward_V(self, batch, hidden_states=None, threshold=None, return_dict=False): # pipeline prediction with the discriminator scores
		# encoding
		if hidden_states is None:
			hidden_states = self.encode(batch)

		# mention detection
		(start, end, scores_ed), (_, _) = self.entity_detection.forward_hard(
			batch, 
			hidden_states, 
			threshold=self.hparams.threshold,
		)
		for k in ["mention_start", "mention_end", "mention_counts"]:
			if k in batch: del batch[k]
		if start.shape[0] == 0: 
			return torch.tensor([[0.] for i in range(len(batch["src_input_ids"]))], device=self.device)

		offsets_start, offsets_end, scores_ed = self.span_filtering(batch, start, end, scores_ed)
		start, end = torch.tensor(offsets_start).T, torch.tensor(offsets_end).T
		batch["offsets_start"], batch["offsets_end"] = offsets_start, offsets_end

		# entity linking
		el_res = self.linking_via_candidate_matching(batch, hidden_states, return_dict=True)
		logits = el_res["classifier_scores"][:,0]
		probs = torch.sigmoid(logits).to(self.device)

		# predict value
		pos = self.hparams.im_reward_pos
		neg = self.hparams.im_reward_neg
		values = torch.tensor([ 
			(probs[start[:, 0] == i] * (pos-neg) + neg).sum()
			for i in range(len(batch["src_input_ids"]))
		], device=probs.device)

		return values.unsqueeze(-1)
	
	def forward_P(self, batch, selected_mentions, hidden_states=None, threshold=None): # forward mention detection logits
		if hidden_states is None:
			hidden_states = self.encode(batch)

		# high-level policy: detect a mention
		(start, end, _), (mention_scores, mention_mask) = self.entity_detection(
			batch, 
			hidden_states, 
			threshold=threshold,
		)
		for k in ["mention_start", "mention_end", "mention_counts"]:
			if k in batch: del batch[k]
		mention_logits = mention_scores + (mention_mask-1) * 1e9

		# identify the offset of `selected_mentions` in `start` and `end`
		action_index = []
		for i, (s, e) in enumerate(selected_mentions):
			identifier = (start[:, 0] == i) & (start[:, 1] == s) & (end[:, 1] == e)
			id_in_flat = identifier.int().argmax().detach()
			num_prevs = (start[:, 0] < i).sum()
			action_index.append(id_in_flat-num_prevs) # id in previous

		return mention_logits, torch.tensor(action_index, dtype=torch.long, device=mention_logits.device)

if __name__ == "__main__":
	pass