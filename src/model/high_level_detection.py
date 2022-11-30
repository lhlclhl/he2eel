import torch, json
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict

class MentionDetectionOnlyEnd(torch.nn.Module):
	def __init__(self, max_length_span, dropout=0, mentions_filename=None):
		super().__init__()

		self.max_length_span = max_length_span
		self.classifier_start = torch.nn.Sequential(
			torch.nn.LayerNorm(768),
			torch.nn.Dropout(dropout),
			torch.nn.Linear(768, 128),
			torch.nn.ReLU(),
			torch.nn.LayerNorm(128),
			torch.nn.Dropout(dropout),
			torch.nn.Linear(128, 1),
		)
		self.classifier_end = torch.nn.Sequential(
			torch.nn.LayerNorm(768 * 2),
			torch.nn.Dropout(dropout),
			torch.nn.Linear(768 * 2, 128),
			torch.nn.ReLU(),
			torch.nn.LayerNorm(128),
			torch.nn.Dropout(dropout),
			torch.nn.Linear(128, 1),
		)

		self.mentions = None
		if mentions_filename:
			with open(mentions_filename) as f:
				self.mentions = set(json.load(f))

	def get_all_mentions(self, batch, mode="auto"):
		if "mention_start" not in batch and "mention_end" not in batch:     
			mention_start = ([], []); mention_end = ([], [])
			mention_counts = [] 
			seqlens = batch["src_attention_mask"].sum(-1).int().tolist()
			detection_mask = batch.get("detection_mask")
			original_index = batch.get("original_index")
			for i in range(len(batch["raw"])):
				if original_index: oi = original_index[i]
				else: oi = list(range(seqlens[i]))

				counts = 0
				for s in range(seqlens[i]):
					if detection_mask and detection_mask[i][s] == 0: continue
					if s == 0: continue

					start = batch["src_offset_mapping"][i][oi[s]][0].item()
					for e in range(s, min(seqlens[i], s+self.max_length_span)):
						if detection_mask and detection_mask[i][e] == 0: break
						
						end = batch["src_offset_mapping"][i][oi[e]][1].item()
						if not self.mentions or \
						batch["raw"][i]["input"][start:end] in self.mentions:
							mention_start[0].append(i)
							mention_start[1].append(s)
							mention_end[0].append(i)
							mention_end[1].append(e)
							counts += 1

				mention_counts.append(counts)
			batch["mention_start"] = mention_start
			batch["mention_end"] = mention_end
			batch["mention_counts"] = mention_counts
		return batch["mention_start"], batch["mention_end"], batch["mention_counts"]

	def forward(self, batch, hidden_states):
		mention_start, mention_end, mention_counts = self.get_all_mentions(batch)

		classifier_end_input = torch.nn.functional.pad(
			hidden_states, (0, 0, 0, self.max_length_span - 1)
		)

		classifier_end_input = torch.cat(
			(
				hidden_states[mention_start],
				hidden_states[mention_end],
			),
			dim=1,
		)

		logits_classifier_end = self.classifier_end(classifier_end_input).squeeze(-1)

		device = logits_classifier_end.device
		start = torch.tensor(mention_start, device=device).permute(1, 0)
		end = torch.tensor(mention_end, device=device).permute(1, 0)

		# construct mention scores matrix
		mention_scores = pad_sequence(logits_classifier_end.split(mention_counts), batch_first=True)
		mention_mask = pad_sequence([torch.ones(c).to(device) for c in mention_counts], batch_first=True)


		return (start, end, logits_classifier_end), (
			mention_scores,
			mention_mask,
		)       
	
	def forward_hard(self, batch, hidden_states, threshold=0):
		mention_start, mention_end, mention_counts = self.get_all_mentions(batch, mode="hard")

		classifier_end_input = torch.nn.functional.pad(
			hidden_states, (0, 0, 0, self.max_length_span - 1)
		)

		classifier_end_input = torch.cat(
			(
				hidden_states[mention_start],
				hidden_states[mention_end],
			),
			dim=1,
		)

		logits_classifier_end = self.classifier_end(classifier_end_input).squeeze(-1)

		device = logits_classifier_end.device
		start = torch.tensor(mention_start, device=device).permute(1, 0)
		end = torch.tensor(mention_end, device=device).permute(1, 0)

		valid_mentions = logits_classifier_end > threshold
		logits_classifier_end = logits_classifier_end[valid_mentions]
		start = start[valid_mentions]
		end = end[valid_mentions]

		return (start, end, logits_classifier_end), (
			None,
			None,
		)       

	def forward_loss(self, batch, hidden_states):
		(start, end, logits_classifier_end), (
			mention_scores,
			mention_mask,
		) = self.forward(batch, hidden_states)

		label_set = set(zip(batch["offsets_start"][0], batch["offsets_start"][1], batch["offsets_end"][1]))
		labels = torch.tensor([
			int(men in label_set) 
			for men in zip(batch["mention_start"][0], batch["mention_start"][1], batch["mention_end"][1])
		], device=logits_classifier_end.device, dtype=torch.float)

		loss_end = torch.nn.functional.binary_cross_entropy_with_logits(
			logits_classifier_end,
			labels,
		)

		return 0, loss_end

class MentionDetectionDoubleEnd2(MentionDetectionOnlyEnd):
	def __init__(self, max_length_span, dropout=0, mentions_filename=None):
		super(MentionDetectionOnlyEnd, self).__init__()

		self.max_length_span = max_length_span
		self.classifier_start = torch.nn.Sequential( # value network
			torch.nn.LayerNorm(768 * 2),
			torch.nn.Dropout(dropout),
			torch.nn.Linear(768 * 2, 128),
			torch.nn.ReLU(),
			torch.nn.LayerNorm(128),
			torch.nn.Dropout(dropout),
			torch.nn.Linear(128, 1),
		)
		self.classifier_end = torch.nn.Sequential( # policy network
			torch.nn.LayerNorm(768 * 2),
			torch.nn.Dropout(dropout),
			torch.nn.Linear(768 * 2, 128),
			torch.nn.ReLU(),
			torch.nn.LayerNorm(128),
			torch.nn.Dropout(dropout),
			torch.nn.Linear(128, 1),
		)

		self.mentions = None
		if mentions_filename:
			with open(mentions_filename) as f:
				self.mentions = set(json.load(f))
		
	def forward(self, batch, hidden_states, head="policy", threshold=None):
		mention_start, mention_end, mention_counts = self.get_all_mentions(batch)

		head_input = torch.nn.functional.pad(
			hidden_states, (0, 0, 0, self.max_length_span - 1)
		)

		head_input = torch.cat(
			(
				hidden_states[mention_start],
				hidden_states[mention_end],
			),
			dim=1,
		)

		if head == "policy":
			logits = self.classifier_end(head_input).squeeze(-1)
		else:
			logits = self.classifier_start(head_input).squeeze(-1)
		device = logits.device
		
		if threshold is not None: # if the threshold is set, hard coding the score of span (0, 0) to the threshold value
			mask = torch.tensor([s==e==0 for s, e in zip(mention_start[1], mention_end[1])], device=device).float()
			logits = logits * (1-mask) + mask * threshold
		
		start = torch.tensor(mention_start, device=device).permute(1, 0)
		end = torch.tensor(mention_end, device=device).permute(1, 0)

		# construct mention scores matrix
		mention_scores = pad_sequence(logits.split(mention_counts), batch_first=True)
		mention_mask = pad_sequence([torch.ones(c).to(device) for c in mention_counts], batch_first=True)


		return (start, end, logits), (
			mention_scores,
			mention_mask,
		)  
	
	def get_all_mentions(self, batch, mode="auto"):
		if "mention_start" not in batch and "mention_end" not in batch:     
			mention_start = ([], []); mention_end = ([], [])
			mention_counts = [] 
			seqlens = batch["src_attention_mask"].sum(-1).int().tolist()
			detection_mask = batch.get("detection_mask")
			original_index = batch.get("original_index")
			for i in range(len(batch["raw"])):
				if original_index: oi = original_index[i]
				else: oi = list(range(seqlens[i]))

				counts = 0
				for s in range(seqlens[i]):
					if detection_mask and detection_mask[i][s] == 0: continue
					if mode=="hard" and s == 0: continue

					start = batch["src_offset_mapping"][i][oi[s]][0].item()
					for e in range(s, min(seqlens[i], s+self.max_length_span)):
						if detection_mask and detection_mask[i][e] == 0: break
						
						end = batch["src_offset_mapping"][i][oi[e]][1].item()
						if not self.mentions or s == e == 0 or\
						batch["raw"][i]["input"][start:end] in self.mentions:
							mention_start[0].append(i)
							mention_start[1].append(s)
							mention_end[0].append(i)
							mention_end[1].append(e)
							counts += 1
						if s == 0: break 
				mention_counts.append(counts)
			batch["mention_start"] = mention_start
			batch["mention_end"] = mention_end
			batch["mention_counts"] = mention_counts
		return batch["mention_start"], batch["mention_end"], batch["mention_counts"]

	def forward_loss(self, batch, hidden_states, strategy=0):
		(start, end, logits_classifier_end), (
			mention_scores,
			mention_mask,
		) = self.forward(batch, hidden_states)

		label_set = set(zip(batch["offsets_start"][0], batch["offsets_start"][1], batch["offsets_end"][1]))

		if strategy == 0:
			end_of_episode_targets = defaultdict(lambda:1) 
			for i, _, _ in label_set: end_of_episode_targets[i] = 0.
			labels = torch.tensor([
				end_of_episode_targets[men[0]] 
				if men[1] == men[2] == 0 else 
				int(men in label_set) 
				for men in zip(batch["mention_start"][0], batch["mention_start"][1], batch["mention_end"][1])
			], device=logits_classifier_end.device, dtype=torch.float)
		else:
			labels = torch.tensor([
				0.5
				if men[1] == men[2] == 0 else 
				int(men in label_set) 
				for men in zip(batch["mention_start"][0], batch["mention_start"][1], batch["mention_end"][1])
			], device=logits_classifier_end.device, dtype=torch.float)

		if strategy == 2:
			non_special = [not s==e==0 for (s, e) in zip(batch["mention_start"][1], batch["mention_end"][1])]
			logits_classifier_end = logits_classifier_end[non_special]
			labels = labels[non_special]

		loss_end = torch.nn.functional.binary_cross_entropy_with_logits(
			logits_classifier_end,
			labels,
		)

		return 0, loss_end