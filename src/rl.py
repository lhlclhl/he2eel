import torch, time, sys, os, random, copy, json, math
from torch.nn.utils.rnn import pad_sequence
from os.path import join, exists
from collections import deque, namedtuple, Counter, defaultdict

model_checker = None

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
class ReplayMemory():
	def __init__(self, capacity=10000, batch_size=8): 
		self.memory = deque([], maxlen=capacity)
	def __len__(self):
		return len(self.memory)
	def push(self, state, action, next_state, reward):
		for i in range(len(reward)):
			if state[i] is not None:
				self.memory.append(Transition(
					state[i],
					action[i],
					next_state[i],
					reward[i],
				))
	def sample(self, batch_size): 
		return random.sample(self.memory, batch_size)

class DRLTrainer():
	def __init__(self, device):
		self.device = device

	def save_model(self, model, path): 
		torch.save(model.state_dict(), path)

	def load_model(self, model, path): 
		model.load_state_dict(torch.load(path))

	@torch.no_grad()
	def run_test(self, model, env, max_steps=300, linking_mode="beam", md_threshold=None, test_sdm=True):
		for _, v in model.metrics.items(): v.reset()
		model.train(False)
		mmetrics = [
			"n_truth_positves_detection",
			"n_truth_positves_linking",
			"n_predictions",
			"n_targets",
		]
		sdm_metrics = dict({k:1e-9 for k in mmetrics}, **{
			"n_samples": 1e-9,
			"macro_f1": 0,
			"md_macro_f1": 0,
			"rewards": 0,
		})

		env.initialize()
		t0 = time.time()
		total_steps = 0
		#env.indices = env.indices[19:] + env.indices[:19]
		while env.data_pointer < len(env.data):
			state = batch = env.reset()
			self.convert_device(state)
			model.validation_step(batch)
			del batch["mention_start"], batch["mention_end"], batch["mention_counts"]

			cum_reward = 0
			if test_sdm:
				for step in range(max_steps):
					if all(state["terminal_flags"]): break
					# select action
					sa_ret = model.select_action(state, 
						policy = "greedy", 
						epsilon = 0, 
						linking_mode = linking_mode,
						threshold = md_threshold,
					)
					action = sa_ret["actions"] # action details
					total_steps += sum([e!=0 for e in action[1][1]])

					# transition
					next_state, reward = env.step(state, action)
					cum_reward += sum(reward)

					state = next_state

			for stats in env.metrics: # stats of metrics for each sample in the batch
				for mi, mmetric in enumerate(mmetrics):
					sdm_metrics[mmetric] += stats[mi]
				sdm_metrics["macro_f1"] += 2 * stats[1] / (stats[2] + stats[3] + 1e-9)
				sdm_metrics["md_macro_f1"] += 2 * stats[0] / (stats[2] + stats[3] + 1e-9)
				sdm_metrics["n_samples"] += 1
			sdm_metrics["rewards"] += cum_reward
			
			ts = time.time()-t0
			sys.stdout.write("\rTesting: %6.2f%% | %d/%d, %d<%d secs, %.3f it/s, avg_steps=%.2f, avg_reward=%.4f    "%(
				env.data_pointer / len(env.data) * 100,
				env.data_pointer,
				len(env.data),
				ts,
				ts / env.data_pointer * (len(env.data)-env.data_pointer),
				env.data_pointer / ts,
				total_steps / sdm_metrics["n_samples"],
				sdm_metrics["rewards"] / sdm_metrics["n_samples"],
			))
			sys.stdout.flush()
		
		return {
			"micro_f1_sdm": 2 * sdm_metrics["n_truth_positves_linking"] / (sdm_metrics["n_predictions"]+sdm_metrics["n_targets"]),
			"macro_f1_sdm": sdm_metrics["macro_f1"] / sdm_metrics["n_samples"],
			"micro_prec_sdm": sdm_metrics["n_truth_positves_linking"] / sdm_metrics["n_predictions"],
			"micro_rec_sdm": sdm_metrics["n_truth_positves_linking"] / sdm_metrics["n_targets"],
			"micro_f1": model.micro_f1.compute().item(),
			"macro_f1": model.macro_f1.compute().item(),
			"micro_prec": model.micro_prec.compute().item(),
			"micro_rec": model.micro_rec.compute().item(),
			"md_micro_f1_sdm": 2 * sdm_metrics["n_truth_positves_detection"] / (sdm_metrics["n_predictions"]+sdm_metrics["n_targets"]),
			"md_macro_f1_sdm": sdm_metrics["md_macro_f1"] / sdm_metrics["n_samples"],
			"md_micro_prec_sdm": sdm_metrics["n_truth_positves_detection"] / sdm_metrics["n_predictions"],
			"md_micro_rec_sdm": sdm_metrics["n_truth_positves_detection"] / sdm_metrics["n_targets"],
			"md_micro_f1": model.ed_micro_f1.compute().item(),
			"md_macro_f1": model.ed_macro_f1.compute().item(),
			"md_micro_prec": model.ed_micro_prec.compute().item(),
			"md_micro_rec": model.ed_micro_rec.compute().item(),
			"avg_reward": sdm_metrics["rewards"] / sdm_metrics["n_samples"],
		}

	def convert_device(self, state):
		for k in state:
			if k.endswith("input_ids") or k.endswith("attention_mask"):
				state[k] = state[k].to(self.device)

	def get_banner(self, results, fields):
		return ",".join(["%s=%.4f"%(f, results[f])for f in fields if f in results])
	
	def remove_bad(self, dirn, monitors, top=10):
		fns = []
		top_cps = {k:[] for k in monitors}
		for fn in sorted(os.listdir(dirn)):
			banner, _ = fn.rsplit(".", 1)
			_, banner = banner.split(",", 1)
			for kv in banner.split(","):
				k, v = kv.split("=")
				if k in top_cps:
					top_cps[k].append((float(v), len(fns)))
			fns.append(fn)
		cps_to_keep = set()
		for k, flist in top_cps.items():
			flist.sort(reverse=monitors[k])
			for _, cp in flist[:top]: cps_to_keep.add(cp)
		for i in range(len(fns)):
			if i not in cps_to_keep:
				os.remove(join(dirn, fns[i]))

	def validate_and_save(self, model, val_env, res, res_str, fout, model_dir, \
	linking_mode, md_threshold, val_metrics, **kwargs):
		banner = ""
		monitors = kwargs.get("monitors")
		test_sdm = kwargs.get("test_sdm", True)

		sys.stdout.write("\r%s, saving first before running validation ..."%res_str)
		sys.stdout.flush()

		ckpt_dir = join(model_dir, "checkpoints")
		if not exists(ckpt_dir): os.makedirs(ckpt_dir)
		tmp_path = join(ckpt_dir, "epoch=%.4f.torch"%(res[0]))
		self.save_model(model, tmp_path)

		ret = self.run_test(model, val_env, linking_mode=linking_mode, md_threshold=md_threshold, test_sdm=test_sdm)
		for k in val_metrics:
			res.append(ret[k])
			res_str += ", %s=%.4f"%(k, ret[k])
		if monitors: banner = self.get_banner(ret, monitors)

		sys.stdout.write("\r%s\n"%res_str)
		sys.stdout.flush()
		fout.write("%s\n"%"\t".join(("%.4f"%r if type(r)==float else str(r)) for r in res))
		fout.flush()

		os.rename(tmp_path, join(ckpt_dir, "epoch=%.4f,%s.torch"%(
			res[0], banner
		)))
		if banner and monitors:
			self.remove_bad(ckpt_dir, monitors)

	def resume(self, model_dir, model):
		start_epoch, last_ckpt_path = 0, None
		ckpt_dir = join(model_dir, "checkpoints")
		if exists(ckpt_dir): 
			files = [join(ckpt_dir, fn) for fn in os.listdir(ckpt_dir) if fn.startswith("epoch=")]
			if files:
				last_ckpt_path = sorted(files, key=lambda t: -os.stat(t).st_mtime)[0]
				if "," in last_ckpt_path:
					epoch = last_ckpt_path.split(",", 1)[0]
				else: epoch = last_ckpt_path.rsplit(".", 1)[0]
				_, epoch = epoch.split("=", 1)
				start_epoch = math.ceil(float(epoch))
		print("resuming previous training session: start_epoch=%s, %s"%(start_epoch, last_ckpt_path))
		if last_ckpt_path is not None:
			self.load_model(model, last_ckpt_path)
		return start_epoch

	def forward_RL_loss(self, memory, batch_size, policy_net, target_net, env, gamma):
		if len(memory) < batch_size * 10: 
			return None
	
		transitions = memory.sample(batch_size)
		# Transpose the batch.
		batch = Transition(*zip(*transitions))
		reward_batch = torch.tensor(batch.reward, device=self.device).unsqueeze(-1)
		# print("reward_batch", reward_batch.shape)

		# Compute a mask of non-final states and concatenate the batch elements
		# (a final state would've been the one after which simulation ended)
		next_state_values = torch.zeros((batch_size, 1), device=self.device)
		non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), \
			device=self.device, dtype=torch.bool)
		if non_final_mask.any():
			non_final_next_states = env.restore([s for s in batch.next_state if s is not None])
			# for s, ns in zip(batch.state, batch.next_state): check_state_consistency(env, s, ns, entrance="sample")
			# print("next_state_values", next_state_values.shape)
			with torch.no_grad(): # TODO: to use target net
				policy_net.eval()
				next_values = policy_net.forward_V(non_final_next_states)
				policy_net.train()
				# print("next_values", next_values.shape)
				next_state_values[non_final_mask] = next_values.detach()
		# Compute the expected Q values
		expected_state_action_values = next_state_values * gamma + reward_batch
		# print("expected_state_action_values", expected_state_action_values.shape)

		state_batch = env.restore(batch.state)
		action_batch = torch.tensor(batch.action, device=self.device)

		# the policy net for update
		hidden_states = policy_net.encode(state_batch)
		# calculate the advantage as the update weights
		with torch.no_grad(): 
			policy_net.eval()
			try:
				state_values = policy_net.forward_V(state_batch, hidden_states=hidden_states).detach()
			except Exception as e:
				print("forward_V error:", e)
				policy_net.train()
				return None
			policy_net.train()
			# print("state_values", state_values.shape)
		advantage = expected_state_action_values-state_values

		state_action_logits, action_index = policy_net.forward_P(
			state_batch, 
			action_batch,
			hidden_states = hidden_states,
		)
		# print("state_action_logits", state_action_logits.shape)

		# Compute weighted cross_entropy loss
		# advantage /= torch.abs(state_values) # norm
		nll_loss = torch.nn.functional.cross_entropy(
			state_action_logits,
			action_index,
			reduction="none"
		)

		return nll_loss.matmul(advantage)  / nll_loss.shape[0]
		
	def run_training(self, model, env, model_dir="models/rl", n_epochs=50, gamma=0.99, training_bs=4, actor_update=0.1, \
	target_update=100, linking_mode="beam", epsilon=0.1, max_steps=300, val_env=None, append=False, critic_update=0.5, \
	md_strategy=0, md_threshold=0, fix_controller=False, val_every_n_episodes=None, test_mdt=None, test_sdm=True, **kwargs): 
		if md_strategy < 2: md_threshold = test_mdt = None # md_threshold is only working in strategy 2
		if test_mdt is None: test_mdt = md_threshold

		if not exists(model_dir):
			os.makedirs(model_dir)
			with open(join(model_dir, "cmd.txt"), "w") as fout: fout.write(" ".join(sys.argv))

		if append: start_epoch = self.resume(model_dir, model)
		else: start_epoch = 0

		policy_net = model 
		target_net = None #copy.deepcopy(policy_net)

		if self.device=="cuda":
			policy_net = policy_net.cuda()
			# target_net = target_net.cuda()

		memory = ReplayMemory()
		sub_optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-5)
		meta_optimizer = torch.optim.RMSprop([
			{'params': policy_net.entity_detection.parameters(), 'lr': 1e-5}
		])

		fout = open(join(model_dir, "record.txt"), "a" if append else "w", encoding="utf-8")
		train_metrics = ["Epoch", "time", "train_avg_steps", "train_avg_reward"]
		val_metrics = ["avg_reward", "micro_f1", "md_micro_f1", "micro_f1_sdm", "md_micro_f1_sdm"]
		if not append:
			fout.write("%s\n"%"\t".join(train_metrics + val_metrics))

		if val_env is not None and not append:
			self.validate_and_save(policy_net, val_env, [start_epoch, "-", "-", "-"], "Start up validation", \
				fout, model_dir, linking_mode, test_mdt, val_metrics, test_sdm=test_sdm, **kwargs)
		fout.flush()

		for epoch in range(start_epoch, n_epochs):
			total_reward = 0
			total_steps = 0
			n_samples = 0
			n_batches = env.n_batches
			t0 = time.time()
			# total_consist = pp_consist = bp_consist = 0
			
			for episode in range(n_batches):
				state = env.reset()
				self.convert_device(state)
				cum_reward = 0
				n_samples += len(state["terminal_flags"])

				for step in range(max_steps):
					# train low-level controllers
					hidden_states = None
					rd = random.random()
					
					# optimize the subtask-based critic
					if step == 0 or rd < critic_update:
						try:	
							policy_net.train()
							loss, hidden_states = policy_net.inner_training_step(state, strategy=md_strategy)
							sub_optimizer.zero_grad(set_to_none=True)
							loss.backward()
							for param in policy_net.parameters():
								if param.grad is not None:
									param.grad.data.clamp_(-1, 1)
							sub_optimizer.step()
						except Exception as e:
							import traceback; traceback.print_exc()

					if all(state["terminal_flags"]): break
		
					# optimize the actor
					if rd < actor_update:
						# select action
						with torch.no_grad():
							sa_ret = policy_net.select_action(state, policy="stochastic", epsilon=epsilon, \
								hidden_states=hidden_states, linking_mode=linking_mode, threshold=md_threshold)
						del hidden_states
						action = sa_ret["actions"] # action details: offsets_start, offsets_end, entity_tokens
						goal = list(zip(action[0][1], action[1][1])) # goal: the high-level action, i.e. spans

						state_info = env.extract(state)
						
						total_steps += sum([e!=0 for e in action[1][1]])

						# transition
						next_state, reward = env.step(state, action)
						cum_reward += sum(reward)

						next_state_info = env.extract(next_state)
						memory.push(state_info, goal, next_state_info, reward)
						state = next_state
						
						meta_loss = self.forward_RL_loss(memory, training_bs, \
							policy_net, target_net, env, gamma)	
						
						if meta_loss is not None:
							policy_net.train()
							meta_optimizer.zero_grad()
							meta_loss.backward()
							for param in policy_net.parameters():
								if param.grad is not None:
									param.grad.data.clamp_(-1, 1)
							meta_optimizer.step()
					else: 
						# select action(s) from ground truth
						total_steps += sum([1-t for t in state["terminal_flags"]])
						
						current_index = policy_net.get_current_index(state["original_index"])
						actions = [[] for _ in current_index]
						for i, ofs, _, ofe, etks in zip(
							*state["target_offsets_start"], 
							*state["target_offsets_end"],
							state["trg_input_ids"],
						):
							cofs, cofe = current_index[i][ofs], current_index[i][ofe]
							if all(state["detection_mask"][i][j] for j in range(cofs, cofe+1)):
								actions[i].append((cofs, cofe, etks))
						
						# trajectory progress alignment in a batch
						ratio_to_execute = max(1/len(a) if a else 1 for a in actions)
						n_actions_remain_to_execute = [math.ceil(len(a)*ratio_to_execute) for a in actions]
						
						if sum(n_actions_remain_to_execute) == 0: break

						while sum(n_actions_remain_to_execute):
							current_index = policy_net.get_current_index(state["original_index"])
							actions = [[] for _ in current_index]
							for i, ofs, _, ofe, etks in zip(
								*state["target_offsets_start"], 
								*state["target_offsets_end"],
								state["trg_input_ids"],
							):
								cofs, cofe = current_index[i][ofs], current_index[i][ofe]
								if all(state["detection_mask"][i][j] for j in range(cofs, cofe+1)):
									actions[i].append((cofs, cofe, etks))
							
							# transition
							for i in range(len(actions)):
								if not actions[i]:
									n_actions_remain_to_execute[i] = 0
								elif n_actions_remain_to_execute[i] > 0:
									start, end, ent = random.sample(actions[i], 1)[0]
									state, reward = env.step_one(state, i, start, end, ent)
									cum_reward += reward
									n_actions_remain_to_execute[i] -= 1

						for k in ["mention_start", "mention_end", "mention_counts"]:
							if k in state: del state[k]
				
				total_reward += cum_reward
				ts = time.time()-t0
				sys.stdout.write("\rEpoch %d: %.2f%% | %d/%d, %d<%d secs, %.3f it/s, avg_steps=%.2f, avg_reward=%.4f"%(
					epoch,
					(episode+1) / n_batches * 100,
					episode+1,
					n_batches,
					ts,
					ts / (episode+1) * (n_batches-episode-1),
					(episode+1) / ts,
					total_steps / n_samples,
					total_reward / n_samples,
				))
				sys.stdout.flush()

				if target_net and ((episode+1) % target_update == 0 or episode+1 == n_batches):
					target_net.load_state_dict(policy_net.state_dict())

				if episode+1 == n_batches or (val_every_n_episodes is not None \
				and (episode+1) % val_every_n_episodes == 0):
					res = [epoch+episode/n_batches, time.time()-t0, total_steps/n_samples, total_reward/n_samples]
					res_str = "Epoch %.4f: duration=%ds, avg_steps=%.2f, train_avg_reward=%.4f"%tuple(res)

					self.validate_and_save(policy_net, val_env, res, res_str, fout, model_dir, \
						linking_mode, test_mdt, val_metrics, test_sdm=test_sdm, **kwargs)

		fout.close()

class DRLTrainerAFM(DRLTrainer): # all from memory
	def fit_from_memory(self, memory, batch_size, policy_net, target_net, env, gamma, \
	train_actor, train_critic, meta_optimizer, sub_optimizer, md_strategy):
		if len(memory) < batch_size * 10: 
			return None
	
		transitions = memory.sample(batch_size)
		# Transpose the batch.
		batch = Transition(*zip(*transitions))
		
		state_batch = env.restore(batch.state)
		
		if train_actor:
			reward_batch = torch.tensor(batch.reward, device=self.device).unsqueeze(-1)
			# print("reward_batch", reward_batch.shape)

			# Compute a mask of non-final states and concatenate the batch elements
			# (a final state would've been the one after which simulation ended)
			next_state_values = torch.zeros((batch_size, 1), device=self.device)
			non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), \
				device=self.device, dtype=torch.bool)
			if non_final_mask.any():
				non_final_next_states = env.restore([s for s in batch.next_state if s is not None])
				# for s, ns in zip(batch.state, batch.next_state): check_state_consistency(env, s, ns, entrance="sample")
				# print("next_state_values", next_state_values.shape)
				with torch.no_grad(): # TODO: to use target net
					policy_net.eval()
					next_values = policy_net.forward_V(non_final_next_states)
					policy_net.train()
					# print("next_values", next_values.shape)
					next_state_values[non_final_mask] = next_values.detach()
			# Compute the expected Q values
			expected_state_action_values = next_state_values * gamma + reward_batch
			# print("expected_state_action_values", expected_state_action_values.shape)

			action_batch = torch.tensor(batch.action, device=self.device)
			# the policy net for update
			hidden_states = policy_net.encode(state_batch)
			# calculate the advantage as the update weights
			with torch.no_grad(): 
				policy_net.eval()
				try:
					state_values = policy_net.forward_V(state_batch, hidden_states=hidden_states).detach()
				except Exception as e:
					print("forward_V error:", e)
					policy_net.train()
					return None
				policy_net.train()
				# print("state_values", state_values.shape)
			advantage = expected_state_action_values-state_values

			state_action_logits, action_index = policy_net.forward_P(
				state_batch, 
				action_batch,
				hidden_states = hidden_states,
			)
			# print("state_action_logits", state_action_logits.shape)

			# Compute weighted cross_entropy loss
			# advantage /= torch.abs(state_values) # norm
			nll_loss = torch.nn.functional.cross_entropy(
				state_action_logits,
				action_index,
				reduction="none"
			)
			meta_loss = nll_loss.matmul(advantage)  / nll_loss.shape[0]

			meta_optimizer.zero_grad()
			meta_loss.backward()
			for param in policy_net.parameters():
				if param.grad is not None:
					param.grad.data.clamp_(-1, 1)
			meta_optimizer.step()

		if train_critic:
			try:	
				state_batch = env.add_targets(state_batch, device=self.device)
				policy_net.train()
				loss, hidden_states = policy_net.inner_training_step(state_batch, strategy=md_strategy)
				sub_optimizer.zero_grad(set_to_none=True)
				loss.backward()
				for param in policy_net.parameters():
					if param.grad is not None:
						param.grad.data.clamp_(-1, 1)
				sub_optimizer.step()
			except Exception as e:
				print("critic update error:", e)
				import traceback; traceback.print_exc()

	def run_training(self, model, env, model_dir="models/rl", n_epochs=50, gamma=0.99, training_bs=4, actor_update=0.1, \
	target_update=100, linking_mode="beam", epsilon=0.1, max_steps=300, val_env=None, append=False, critic_update=0.5, \
	md_strategy=0, md_threshold=0, fix_controller=False, val_every_n_episodes=None, test_mdt=None, **kwargs): 
		if md_strategy < 2: md_threshold = test_mdt = None # md_threshold is only working in strategy 2
		if test_mdt is None: test_mdt = md_threshold

		if not exists(model_dir):
			os.makedirs(model_dir)
			with open(join(model_dir, "cmd.txt"), "w") as fout: fout.write(" ".join(sys.argv))

		if append: start_epoch = self.resume(model_dir, model)
		else: start_epoch = 0

		policy_net = model 
		target_net = None #copy.deepcopy(policy_net)

		if self.device=="cuda":
			policy_net = policy_net.cuda()
			# target_net = target_net.cuda()

		memory = ReplayMemory()
		sub_optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-5)
		meta_optimizer = torch.optim.RMSprop([
			{'params': policy_net.entity_detection.parameters(), 'lr': 1e-5}
		])

		fout = open(join(model_dir, "record.txt"), "a" if append else "w", encoding="utf-8")
		train_metrics = ["Epoch", "time", "train_avg_steps", "train_avg_reward"]
		val_metrics = ["avg_reward", "micro_f1", "md_micro_f1", "micro_f1_sdm", "md_micro_f1_sdm"]
		if not append:
			fout.write("%s\n"%"\t".join(train_metrics + val_metrics))

		if val_env is not None and not append:
			self.validate_and_save(policy_net, val_env, [start_epoch, "-", "-", "-"], "Start up validation", \
				fout, model_dir, linking_mode, test_mdt, val_metrics, **kwargs)
		fout.flush()

		for epoch in range(start_epoch, n_epochs):
			total_reward = 0
			total_steps = 0
			n_samples = 0
			n_batches = env.n_batches
			t0 = time.time()
			# total_consist = pp_consist = bp_consist = 0
			
			for episode in range(n_batches):
				state = env.reset()
				self.convert_device(state)
				cum_reward = 0
				n_samples += len(state["terminal_flags"])

				for step in range(max_steps):
					rd = random.random()
					train_critic = rd < critic_update
					train_actor = rd + actor_update > 1 
					
					if all(state["terminal_flags"]): break
		
					if actor_update > 0 and (step == 0 or rd < actor_update): # select based on actor policy
						# select action
						with torch.no_grad():
							sa_ret = policy_net.select_action(state, policy="stochastic", epsilon=epsilon, \
								linking_mode=linking_mode, threshold=md_threshold)
						action = sa_ret["actions"] # action details: offsets_start, offsets_end, entity_tokens
						goal = list(zip(action[0][1], action[1][1])) # goal: the high-level action, i.e. spans

						state_info = env.extract(state)
						
						total_steps += sum([e!=0 for e in action[1][1]])

						# transition
						next_state, reward = env.step(state, action)
						cum_reward += sum(reward)

						next_state_info = env.extract(next_state)
						memory.push(state_info, goal, next_state_info, reward)
						state = next_state
					else: # select action(s) based on ground truth to alleviate error propagation of actor policy
						if actor_update == 0 and (step == 0 or rd + critic_update > 1): 
							# no actor update, only need the current state info
							state_info = env.extract(state)
							empty_info = [None] * len(state_info)
							memory.push(state_info, empty_info, empty_info, empty_info)

						total_steps += sum([1-t for t in state["terminal_flags"]])
						
						current_index = policy_net.get_current_index(state["original_index"])
						actions = [[] for _ in current_index]
						for i, ofs, _, ofe, etks in zip(
							*state["target_offsets_start"], 
							*state["target_offsets_end"],
							state["trg_input_ids"],
						):
							cofs, cofe = current_index[i][ofs], current_index[i][ofe]
							if all(state["detection_mask"][i][j] for j in range(cofs, cofe+1)):
								actions[i].append((cofs, cofe, etks))
						
						# trajectory progress alignment in a batch
						ratio_to_execute = max(1/len(a) if a else 1 for a in actions)
						n_actions_remain_to_execute = [math.ceil(len(a)*ratio_to_execute) for a in actions]
						
						if sum(n_actions_remain_to_execute) == 0: break

						while sum(n_actions_remain_to_execute):
							current_index = policy_net.get_current_index(state["original_index"])
							actions = [[] for _ in current_index]
							for i, ofs, _, ofe, etks in zip(
								*state["target_offsets_start"], 
								*state["target_offsets_end"],
								state["trg_input_ids"],
							):
								cofs, cofe = current_index[i][ofs], current_index[i][ofe]
								if all(state["detection_mask"][i][j] for j in range(cofs, cofe+1)):
									actions[i].append((cofs, cofe, etks))
							
							# transition
							for i in range(len(actions)):
								if not actions[i]:
									n_actions_remain_to_execute[i] = 0
								elif n_actions_remain_to_execute[i] > 0:
									start, end, ent = random.sample(actions[i], 1)[0]
									state, reward = env.step_one(state, i, start, end, ent)
									cum_reward += reward
									n_actions_remain_to_execute[i] -= 1

						for k in ["mention_start", "mention_end", "mention_counts"]:
							if k in state: del state[k]

					# optimize the actor/critic with the samples from memory
					if train_critic or train_actor:
						self.fit_from_memory(memory, training_bs, policy_net, target_net, \
							env, gamma, train_actor, train_critic, \
							meta_optimizer, sub_optimizer, md_strategy)

				total_reward += cum_reward
				ts = time.time()-t0
				sys.stdout.write("\rEpoch %d: %.2f%% | %d/%d, %d<%d secs, %.3f it/s, avg_steps=%.2f, avg_reward=%.4f"%(
					epoch,
					(episode+1) / n_batches * 100,
					episode+1,
					n_batches,
					ts,
					ts / (episode+1) * (n_batches-episode-1),
					(episode+1) / ts,
					total_steps / n_samples,
					total_reward / n_samples,
				))
				sys.stdout.flush()

				if target_net and ((episode+1) % target_update == 0 or episode+1 == n_batches):
					target_net.load_state_dict(policy_net.state_dict())

				if episode+1 == n_batches or (val_every_n_episodes is not None \
				and (episode+1) % val_every_n_episodes == 0):
					res = [epoch+episode/n_batches, time.time()-t0, total_steps/n_samples, total_reward/n_samples]
					res_str = "Epoch %.4f: duration=%ds, avg_steps=%.2f, train_avg_reward=%.4f"%tuple(res)

					self.validate_and_save(policy_net, val_env, res, res_str, fout, model_dir, \
						linking_mode, test_mdt, val_metrics, **kwargs)

		fout.close()

if __name__ == "__main__":
	pass