import torch
from torch import nn
from lib.tokenizer import CausalTokenizer
from lib.dynamics import ShortcutDynamics
from lib.agent_heads import AgentHeads


class Dreamer4Agent(nn.Module):
    def __init__(self, tokenizer_cfg, dynamics_cfg, agent_cfg):
        super().__init__()
        self.tokenizer = CausalTokenizer(**tokenizer_cfg)
        self.dynamics = ShortcutDynamics(**dynamics_cfg)
        self.agent_heads = AgentHeads(**agent_cfg)

    # -------- Phase 1: World model pretraining ----------

    def tokenizer_step(self, batch):
        """
        batch: dict with 'video'
        """
        out = self.tokenizer(batch["video"], targets=batch["video"])
        return out["loss"], out

    def dynamics_step(self, batch):
        """
        batch: dict with 'video', 'actions', 'tau', 'd'
        """
        with torch.no_grad():
            z_clean = self.tokenizer.encode(batch["video"])
        out = self.dynamics(
            z_clean=z_clean,
            a=batch["actions"],
            tau=batch["tau"],
            d=batch["d"],
        )
        return out["loss"], out

    # -------- Phase 2: Agent finetuning (BC + reward) -----

    def agent_bc_step(self, batch):
        """
        batch: 'video', 'actions', 'rewards', 'task_ids'
        """
        with torch.no_grad():
            z_clean = self.tokenizer.encode(batch["video"])
        # For finetuning, world model may still be trained with shortcut loss:
        dyn_out = self.dynamics(
            z_clean=z_clean,
            a=batch["actions"],
            tau=batch["tau"],
            d=batch["d"],
        )

        # Insert task tokens and run transformer one more time if nötig
        # (oder du nutzt hidden states direkt aus Dynamics-Backbone).
        seq = dyn_out["z_pred"]  # (B,T,S_z,D)
        seq_ext, agent_tokens = self.agent_heads.insert_task_tokens(
            seq, batch["task_ids"]
        )
        # Hier vereinfachend: agent_tokens = letzter Token
        agent_tokens = seq_ext[:, :, -1, :]  # (B,T,D)

        bc_loss = self.agent_heads.behavior_cloning_and_reward_loss(
            agent_tokens, batch["actions_discrete"], batch["rewards_binned"]
        )

        total_loss = dyn_out["loss"] + bc_loss
        return total_loss, {"dyn_loss": dyn_out["loss"], "bc_loss": bc_loss}

    # -------- Phase 3: Imagination training ---------------

    @torch.no_grad()
    def imagine_rollout(self, batch, horizon):
        """
        Start imagined trajectories from dataset contexts.
        Returns imagined_states dict with z, actions, rewards, values.
        """
        video = batch["video"]
        actions = batch["actions"]
        task_ids = batch["task_ids"]

        z_context = self.tokenizer.encode(video)
        # In der Praxis startest du nach einigen Kontextframes.
        z_t = z_context[:, -1:]  # (B,1,S_z,D)
        imagined = {
            "z": [], "a": [], "r": [], "v": [], "task_ids": task_ids
        }

        for t in range(horizon):
            # Build dynamics input, sample actions with current policy head usw.
            # Hier skizziere ich nur die API.
            # ...
            pass

        # Stack to tensors
        return imagined

    def rl_step(self, imagined, prior_logits):
        """
        Compute PMPO + value loss on imagined trajectories.
        """
        agent_tokens = imagined["agent_tokens"]
        actions = imagined["a"]
        lambda_returns = imagined["lambda_returns"]
        advantages = imagined["advantages"]

        val_loss = self.agent_heads.value_loss(agent_tokens, lambda_returns)
        pol_loss = self.agent_heads.pmpo_policy_loss(
            agent_tokens, actions, advantages, prior_logits
        )
        total = val_loss + pol_loss
        return total, {"val_loss": val_loss, "pol_loss": pol_loss}
