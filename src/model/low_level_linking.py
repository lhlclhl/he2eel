import torch, json, random, copy
from torch.nn.utils.rnn import pad_sequence
from torch.distributions import Categorical

from .entity_linking import EntityLinkingLSTM, beam_search, LSTM
from ..utils import label_smoothed_nll_loss

class EntityLinking(EntityLinkingLSTM):

	def forward_all_targets(self, batch, hidden_states):
		(
			decoder_hidden,
			decoder_context,
			decoder_append,
		) = self._get_hidden_context_append_vectors(batch, hidden_states)

		all_contexts, lm_scores = self.lstm.forward_all_targets(
			batch,
			decoder_hidden[batch["offsets_candidates"]],
			decoder_context[batch["offsets_candidates"]],
			decoder_append[batch["offsets_candidates"]],
		)

		classifier_scores = self.classifier(
			torch.cat(
				(
					decoder_append[batch["offsets_candidates"]],
					all_contexts,
				),
				dim=-1,
			)
		).squeeze(-1)
		
		return lm_scores, classifier_scores

	def forward_beam_search(
		self, batch, hidden_states, batch_trie_dict=None, beams=5, alpha=1, max_len=15, return_dict=False, lm_weight=1.
	):
		(
			decoder_hidden,
			decoder_context,
			decoder_append,
		) = self._get_hidden_context_append_vectors(batch, hidden_states)

		tokens, lm_scores, all_contexts = beam_search(
			self.lstm,
			self.lstm.lm_head.decoder.out_features,
			(decoder_hidden, decoder_context, decoder_append),
			self.bos_token_id,
			self.eos_token_id,
			self.pad_token_id,
			beam_width=beams,
			alpha=alpha,
			max_len=max_len,
			batch_trie_dict=batch_trie_dict,
		)

		classifier_scores = self.classifier(
			torch.cat(
				(
					decoder_append.unsqueeze(1).repeat(1, beams, 1),
					all_contexts,
				),
				dim=-1,
			)
		).squeeze(-1)

		classifier_scores[lm_scores == -float("inf")] = -float("inf")
		classifier_scores = classifier_scores.log_softmax(-1)

		scores = (classifier_scores + lm_weight * lm_scores).sort(-1, descending=True)
		tokens = tokens[
			torch.arange(scores.indices.shape[0], device=tokens.device)
			.unsqueeze(-1)
			.repeat(1, beams),
			scores.indices,
		]
		if return_dict:
			classifier_scores = classifier_scores[
				torch.arange(scores.indices.shape[0], device=classifier_scores.device)
				.unsqueeze(-1)
				.repeat(1, beams),
				scores.indices,
			]
			lm_scores = lm_scores[
				torch.arange(scores.indices.shape[0], device=lm_scores.device)
				.unsqueeze(-1)
				.repeat(1, beams),
				scores.indices,
			]
			scores = scores.values
			return {
				"tokens": tokens,
				"scores": scores,
				"classifier_scores": classifier_scores,
				"lm_scores": lm_scores,
			}
		else:
			scores = scores.values

			return tokens, scores

	def forward_loss(self, batch, hidden_states, epsilon=0): # binary discriminator

		lprobs_lm, logits_classifier = self.forward(batch, hidden_states)

		loss_generation, _ = label_smoothed_nll_loss(
			lprobs_lm,
			batch["trg_input_ids"][:, 1:],
			epsilon=epsilon,
			ignore_index=self.pad_token_id,
		)
		loss_generation = loss_generation / batch["trg_attention_mask"][:, 1:].sum()

		loss_classifier = torch.nn.functional.binary_cross_entropy_with_logits(
			logits_classifier,
			torch.stack([
				torch.ones(logits_classifier.shape[0], dtype=torch.float, device=logits_classifier.device),
				torch.zeros(logits_classifier.shape[0], dtype=torch.float, device=logits_classifier.device),
			], dim=-1),
		)

		return loss_generation, loss_classifier

class EntityLinkingORL(EntityLinking):
	def __init__(
		self, bos_token_id, pad_token_id, eos_token_id, embeddings, lm_head, dropout=0
	):
		super().__init__(bos_token_id, pad_token_id, eos_token_id, embeddings, lm_head, dropout=0)
	
		self.behavior_head = copy.deepcopy(lm_head) # additional head for behavior policy

	def forward(self, batch, hidden_states, prob_mask):
		(
			decoder_hidden,
			decoder_context,
			decoder_append,
		) = self._get_hidden_context_append_vectors(batch, hidden_states)

		all_hiddens, all_contexts_positive = self.lstm._roll(
				batch["trg_input_ids"][:, :-1],
				batch["trg_attention_mask"][:, 1:],
				decoder_hidden,
				decoder_context,
				decoder_append,
				return_lprob=False,
			)

		all_hiddens_negative, all_contexts_negative = self.lstm._roll(
			batch["neg_input_ids"][:, :-1],
			batch["neg_attention_mask"][:, 1:],
			decoder_hidden[batch["neg_mask"]],
			decoder_context[batch["neg_mask"]],
			decoder_append[batch["neg_mask"]],
			return_lprob=False,
		)

		all_hiddens *= prob_mask
		all_hiddens += (1-prob_mask[batch["neg_mask"]]) * all_hiddens_negative
		del all_hiddens_negative

		lprobs = self.lstm.lm_head(all_hiddens).log_softmax(-1)
		lprobs_behavior = self.behavior_head(all_hiddens).log_softmax(-1)

		logits_classifier = self.classifier(
			torch.cat(
				(
					decoder_append[batch["neg_mask"]].unsqueeze(1).repeat(1, 2, 1),
					torch.stack(
						(
							all_contexts_positive[batch["neg_mask"]],
							all_contexts_negative,
						),
						dim=1,
					),
				),
				dim=-1,
			)
		).squeeze(-1)
		
		return lprobs, lprobs_behavior, logits_classifier

	def forward_loss(self, batch, hidden_states, epsilon=0, delta=-0.001): 
		# convert negative reward delta to the sample probability of negative candidates
		sth = -delta/(1-delta)
		pn_mask = (~batch["neg_mask"] | (torch.rand(batch["neg_mask"].size()) >= sth)).to(batch["trg_input_ids"].device).float().unsqueeze(-1)

		tin, tam = batch["trg_input_ids"], batch["trg_attention_mask"]
		nin, nam = batch["neg_input_ids"], batch["neg_attention_mask"]

		if tin.shape[1] > nin.shape[1]:
			ldiff = tin.shape[1] - nin.shape[1]
			batch["neg_input_ids"] = torch.cat([
				nin,
				torch.full((nin.shape[0], ldiff), self.pad_token_id).to(nin.device)
			], dim=1)
			batch["neg_attention_mask"] = torch.cat([
				nam,
				torch.full((nam.shape[0], ldiff), 0).to(nam.device)
			], dim=1)
		elif tin.shape[1] < nin.shape[1]:
			ldiff = nin.shape[1] - tin.shape[1]
			batch["trg_input_ids"] = torch.cat([
				tin,
				torch.full((tin.shape[0], ldiff), self.pad_token_id).to(tin.device)
			], dim=1)
			batch["trg_attention_mask"] = torch.cat([
				tam,
				torch.full((tam.shape[0], ldiff), 0).to(tam.device)
			], dim=1)

		gen_inputs = (batch["trg_input_ids"] * pn_mask + (1-pn_mask) * batch["neg_input_ids"]).long()
		gen_mask = batch["trg_attention_mask"] * pn_mask + (1-pn_mask) * batch["neg_attention_mask"]

		lprobs_policy, lprobs_behavior, logits_classifier = self.forward(
			batch, hidden_states, pn_mask.unsqueeze(-1)
		)

		# calculate importance sampling weights
		target = gen_inputs[:, 1:]
		if target.dim() == lprobs_policy.dim() - 1:
			target = target.unsqueeze(-1)
		with torch.no_grad():
			lpi_theta = lprobs_policy.gather(dim=-1, index=target)
			lpi_beta = lprobs_behavior.gather(dim=-1, index=target)
			weights = torch.exp(lpi_theta-lpi_beta)

		# ORL objective: cross-entropy loss * reward * importance_sampling_weights
		loss_generation = label_smoothed_nll_loss(
			lprobs_policy,
			gen_inputs[:, 1:],
			epsilon=epsilon,
			ignore_index=self.pad_token_id,
			reduce=False
		)[0]
		loss_generation = loss_generation * (2*pn_mask-1).unsqueeze(-1) * weights
		loss_generation = loss_generation.sum() / gen_mask[:, 1:].sum()

		# behavior net objective: cross-entropy loss
		loss_generation_behavior = label_smoothed_nll_loss(
			lprobs_behavior,
			gen_inputs[:, 1:],
			epsilon=epsilon,
			ignore_index=self.pad_token_id,
		)[0]
		loss_generation_behavior = loss_generation_behavior
		loss_generation_behavior = loss_generation_behavior.sum() / gen_mask[:, 1:].sum()

		loss_classifier = torch.nn.functional.binary_cross_entropy_with_logits(
			logits_classifier,
			torch.stack([
				torch.ones(logits_classifier.shape[0], dtype=torch.float, device=logits_classifier.device),
				torch.zeros(logits_classifier.shape[0], dtype=torch.float, device=logits_classifier.device),
			], dim=-1),
		)

		batch["trg_input_ids"], batch["trg_attention_mask"] = tin, tam
		batch["neg_input_ids"], batch["neg_attention_mask"] = nin, nam

		return loss_generation+loss_generation_behavior, loss_classifier
	