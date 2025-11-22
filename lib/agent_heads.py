import torch
from torch import nn

from lib.utils import LossRMSNormalizer


class AgentHeads(nn.Module):
    def __init__(self, d_model, num_tasks,
                 policy_out_dim, reward_bins, value_bins,
                 mtp_horizon=8):
        super().__init__()
        self.task_embed = nn.Embedding(num_tasks, d_model)
        self.mtp_horizon = mtp_horizon

        self.loss_norm = LossRMSNormalizer()

        # Policy, reward, value as small MLP heads on agent tokens
        hidden = d_model

        self.policy_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.SiLU(),
                nn.Linear(hidden, policy_out_dim),
            )
            for _ in range(mtp_horizon + 1)
        ])

        self.reward_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.SiLU(),
                nn.Linear(hidden, reward_bins),
            )
            for _ in range(mtp_horizon + 1)
        ])

        self.value_head = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, value_bins),
        )

    def insert_task_tokens(self, seq, task_ids):
        """
        seq: (B,T,S,D) from dynamics transformer
        task_ids: (B,) or (B,T)
        returns extended_seq, agent_tokens
        """
        B, T, S, D = seq.shape
        if task_ids.dim() == 1:
            task_ids = task_ids[:, None].expand(B, T)

        task_vec = self.task_embed(task_ids)  # (B,T,D)
        agent_tok = task_vec.unsqueeze(2)  # (B,T,1,D)
        seq_ext = torch.cat([seq, agent_tok], dim=2)
        return seq_ext, agent_tok

    def behavior_cloning_and_reward_loss(self, agent_tokens, actions, rewards):
        """
        Implements Eq. (9) inside the module.
        agent_tokens: (B,T,D)
        actions:      (B,T,...)  -> will be projected to logits indices
        rewards:      (B,T) or similar (discretized to reward_bins)
        """
        # Hier: Annahme, dass actions bereits als Indizes für eine
        # diskrete Policy vorliegen (z.B. Keyboard + Mouse bucketing).
        B, T, D = agent_tokens.shape
        L = self.mtp_horizon

        ce_policy = 0.0
        ce_reward = 0.0
        count = 0

        for n in range(L + 1):
            # Mask positions am Ende, an denen t+n >= T
            valid = (torch.arange(T, device=actions.device) + n) < T
            if not valid.any():
                continue
            idx = torch.nonzero(valid).flatten()
            h_n = agent_tokens[:, idx, :]  # (B, valid_T, D)
            h_n = h_n.reshape(-1, D)

            target_a = actions[:, idx + n].reshape(-1)
            target_r = rewards[:, idx + n].reshape(-1)

            logits_a = self.policy_heads[n](h_n)
            logits_r = self.reward_heads[n](h_n)

            ce_policy += nn.functional.cross_entropy(
                logits_a, target_a, reduction="mean"
            )
            ce_reward += nn.functional.cross_entropy(
                logits_r, target_r, reduction="mean"
            )
            count += 1

        if count > 0:
            ce_policy = ce_policy / count
            ce_reward = ce_reward / count

        ce_policy = self.loss_norm("bc_policy", ce_policy)
        ce_reward = self.loss_norm("bc_reward", ce_reward)

        return ce_policy + ce_reward

    def value_loss(self, agent_tokens, lambda_returns):
        """
        Eq. (10): TD(lambda) via cross-entropy on two-hot bins (vereinfachend).
        """
        h = agent_tokens.reshape(-1, agent_tokens.size(-1))
        logits_v = self.value_head(h)  # (B*T, value_bins)
        targets = lambda_returns.reshape(-1)  # assumed already binned
        loss = nn.functional.cross_entropy(logits_v, targets, reduction="mean")
        loss = self.loss_norm("value", loss)
        return loss

    def pmpo_policy_loss(self, agent_tokens, actions, advantages,
                         prior_logits, alpha=0.5, beta=0.3):
        """
        Eq. (11): PMPO objective.
        actions: (B,T) indices
        advantages: (B,T) advantages from lambda returns
        prior_logits: (B*T, A) logits of frozen behavioral prior
        """
        B, T, D = agent_tokens.shape
        h = agent_tokens.reshape(-1, D)
        logits = self.policy_heads[0](h)  # use horizon 0 for RL

        actions_flat = actions.reshape(-1)
        adv_flat = advantages.reshape(-1)

        log_probs = nn.functional.log_softmax(logits, dim=-1)
        lp_taken = log_probs[torch.arange(logits.size(0)), actions_flat]

        mask_pos = adv_flat >= 0
        mask_neg = adv_flat < 0

        lp_pos = lp_taken[mask_pos]
        lp_neg = lp_taken[mask_neg]

        loss = 0.0
        if lp_neg.numel() > 0:
            loss += (1 - alpha) * (-lp_neg.mean())
        if lp_pos.numel() > 0:
            loss += alpha * (-lp_pos.mean())

        # Behavioral prior KL
        prior_log_probs = nn.functional.log_softmax(prior_logits, dim=-1)
        kl = torch.sum(
            torch.exp(log_probs) * (log_probs - prior_log_probs),
            dim=-1
        ).mean()
        loss += beta * kl

        loss = self.loss_norm("pmpo_policy", loss)
        return loss
