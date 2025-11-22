# training/phase1_tokenizer.py
import torch


def train_tokenizer_step(agent, batch, optimizer, scaler=None):
    """
    Single optimization step for tokenizer pretraining.
    """
    optimizer.zero_grad(set_to_none=True)
    loss, outputs = agent.tokenizer_step(batch)

    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    return {"loss": loss.detach()}


# training/phase1_dynamics.py
def train_dynamics_step(agent, batch, optimizer, scaler=None):
    optimizer.zero_grad(set_to_none=True)
    loss, outputs = agent.dynamics_step(batch)

    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    return {"loss": loss.detach()}


# training/phase2_agent.py
def finetune_agent_step(agent, batch, optimizer, scaler=None):
    optimizer.zero_grad(set_to_none=True)
    loss, logs = agent.agent_bc_step(batch)

    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    logs["loss"] = loss.detach()
    return logs


# training/phase3_imagination.py
def imagination_training_step(agent, batch, optimizer, horizon,
                              prior_logits_fn, scaler=None):
    """
    prior_logits_fn: function that returns logits of behavioral prior
                     given states/actions.
    """
    # 1) Rollout inside world model
    imagined = agent.imagine_rollout(batch, horizon)

    # 2) Get prior logits for KL term
    prior_logits = prior_logits_fn(imagined)

    # 3) RL losses
    optimizer.zero_grad(set_to_none=True)
    loss, logs = agent.rl_step(imagined, prior_logits)

    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    logs["loss"] = loss.detach()
    return logs
