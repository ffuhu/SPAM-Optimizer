import math
from functools import partial

import torch
from torch.optim.lr_scheduler import LambdaLR
import transformers

# to time functions
import time
from functools import wraps


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'\t[{func.__name__} took {total_time:.4f} s.]')
        return result
    return timeit_wrapper


def get_scheculer(
    optimizer,
    *,
    scheduler_type,
    num_training_steps,
    warmup_steps,
    min_lr_ratio,
    cycle_length=None,
    restart_warmup_steps=None,
    adjust_step=0,
    last_epoch=-1,
):
    if adjust_step != 0 and scheduler_type != "cosine_restarts":
        raise ValueError("adjust_step is only supported for cosine_restarts scheduler")

    if scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )
    if scheduler_type == "cosine":
        return get_cyclical_cosine_schedule_with_min_lr(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            cycle_length=cycle_length,
            min_lr_ratio=min_lr_ratio,
            last_epoch=last_epoch,
        )
    if scheduler_type == "cosine_restarts":
        assert restart_warmup_steps is not None, "restart_warmup_steps must be specified for cosine_restarts scheduler"
        return get_cosine_schedule_with_multiple_warmups(
            optimizer,
            num_training_steps=num_training_steps,
            first_warmup_steps=warmup_steps,
            restart_warmup_steps=restart_warmup_steps,
            restart_every=cycle_length,
            min_lr_ratio=min_lr_ratio,
            last_epoch=last_epoch,
            adjust_step=adjust_step,
        )

    raise NotImplementedError(f"Scheduler {scheduler_type} is not implemented")


def get_cyclical_cosine_schedule_with_min_lr(optimizer, num_warmup_steps, num_training_steps, cycle_length, min_lr_ratio=0.1, last_epoch=-1):
    assert cycle_length is not None or num_training_steps is not None, "You must specify either cycle_length or num_training_steps"
    
    if cycle_length is None:
        cycle_length = num_training_steps

    if num_training_steps % cycle_length != 0:
        raise ValueError(f"num_training_steps ({num_training_steps}) must be divisible by cycle_length ({cycle_length})")

    lr_lambda = partial(
        _get_cyclical_cosine_schedule_with_min_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        cycle_length=cycle_length,
        min_lr_ratio=min_lr_ratio,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_multiple_warmups(
    optimizer,
    *,
    num_training_steps,
    first_warmup_steps,
    restart_warmup_steps,
    restart_every,
    min_lr_ratio=0.1,
    adjust_step=0,
    last_epoch=-1,
):
    if restart_every is None:
        raise ValueError("restart_every must be specified for cosine_restarts scheduler")

    if num_training_steps % restart_every != 0:
        raise ValueError(f"num_training_steps ({num_training_steps}) must be divisible by restart_every ({restart_every})")

    lr_lambda = partial(
        _get_cosine_schedule_with_multiple_warmups_lambda,
        num_training_steps=num_training_steps,
        first_warmup_steps=first_warmup_steps,
        restart_warmup_steps=restart_warmup_steps,
        restart_every=restart_every,
        min_lr_ratio=min_lr_ratio,
        adjust_step=adjust_step,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


@torch.no_grad()
def random_pruning(tensor, prune_ratio):
    """
    Performs random pruning dimensionality reduction.
    Only reduces the inner dimensionality, does not affect the shape of the tensor
    """
    random_pruning_mask = torch.rand_like(tensor) > prune_ratio
    tensor = tensor * random_pruning_mask
    return tensor


@torch.no_grad()
def magnitude_pruning(tensor, prune_ratio):
    """
    Performs magnitude pruning dimensionality reduction.
    Only reduces the inner dimensionality, does not affect the shape of the tensor
    """
    tensor_magnitude = torch.abs(tensor)
    threshold = torch.quantile(tensor_magnitude.flatten().to(dtype=torch.float32), prune_ratio).to(dtype=tensor.dtype)

    mask = tensor_magnitude > threshold
    tensor = tensor * mask.to(dtype=tensor.dtype)
    return tensor


def _get_cyclical_cosine_schedule_with_min_lr_lambda(current_step, *, num_warmup_steps, cycle_length, min_lr_ratio):
    assert 0 < min_lr_ratio <= 1.0, "min_lr_ratio must be in (0,1]"

    # compute where we are in the current cycle
    cycle_step = current_step % cycle_length

    if cycle_step < num_warmup_steps:
        if current_step != cycle_step:
            if cycle_step < 2:
                return 1e-7
        return float(cycle_step) / float(max(1, num_warmup_steps))

    progress = float(cycle_step - num_warmup_steps) / float(max(1, cycle_length - num_warmup_steps))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay


def _get_cosine_schedule_with_multiple_warmups_lambda(
    current_step,
    *,
    num_training_steps,
    first_warmup_steps,
    restart_warmup_steps,
    restart_every,
    min_lr_ratio,
    adjust_step,
):
    """
    Args:
        adjust_step: useful when continuing training from a warmed up checkpoint,
            it allows to sync the resets by reducing the number of steps
            after the first warmup and before the first reset.
            Thus, your ReLoRA resets can be synced with the optimizer resets.
    """
    assert 0 < min_lr_ratio <= 1.0, "min_lr_ratio must be in (0,1]"
    assert restart_every > 0, "restart_every must be positive"
    assert adjust_step + first_warmup_steps < num_training_steps, "warmup + adjust_step is more than full training steps"
    assert adjust_step + first_warmup_steps < restart_every, "the first reset will happen before the warmup is done"

    if current_step < first_warmup_steps:
        return float(current_step) / float(max(1, first_warmup_steps))

    _current_step = current_step + adjust_step

    restart_step = _current_step % restart_every
    restart_number = _current_step // restart_every

    if restart_step < restart_warmup_steps:
        # get expected lr multipler at the end of the warmup
        end_of_warmup_progress = (
            float(restart_number * restart_every) /
            float(max(1, num_training_steps - first_warmup_steps))
        )

        _cosine_decay = 0.5 * (1.0 + math.cos(math.pi * end_of_warmup_progress))
        warmup_lr_multiplier = min_lr_ratio + (1.0 - min_lr_ratio) * _cosine_decay
    
        return float(restart_step) / float(max(1, restart_warmup_steps)) * warmup_lr_multiplier

    progress = float(_current_step - first_warmup_steps) / float(max(1, num_training_steps - first_warmup_steps))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay


def collate_fn(batch_list):
    batch = {
        "input_ids": torch.stack([torch.Tensor(example["input_ids"]).long() for example in batch_list]),
        "attention_mask": torch.stack([torch.Tensor(example["attention_mask"]).long() for example in batch_list]),
    }
    return batch


def batch_fn(dataset, batch_size):
    batch = []
    for example in dataset:
        batch.append(example)
        if len(batch) == batch_size:
            batch = collate_fn(batch)
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch


def max_train_tokens_to_number(max_train_tokens):
    if type(max_train_tokens) is int:
        return max_train_tokens
    if max_train_tokens.endswith("M"):
        return int(float(max_train_tokens.rstrip("M")) * 1_000_000)
    elif max_train_tokens.endswith("B"):
        return int(float(max_train_tokens.rstrip("B")) * 1_000_000_000)
    else:
        return int(max_train_tokens)


# compute gradient norm of the whole model
def compute_grad_norm(model):

    # Calculate global gradient norm
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().flatten())

    local_norm = torch.norm(torch.cat(grads), 2)

    return local_norm

# compute gradient norm of the whole model
def compute_grad_norm_by_param_name(model):

    # Calculate global gradient norm
    grads = []
    grads_embed = []
    grads_scalar = []
    grads_head = []
    grads_hidden_matrix = []
    for n, p in model.named_parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().flatten())
            if "embed" in n:
                grads_embed.append(grads[-1])
            if p.ndim < 2:
                grads_scalar.append(grads[-1])
            if p.ndim >= 2 and "embed" not in n and "lm_head" not in n:
                grads_hidden_matrix.append(grads[-1])
            if "lm_head" in n:
                grads_head.append(grads[-1])

    local_norm = torch.norm(torch.cat(grads), 2)
    local_embed = torch.norm(torch.cat(grads_embed), 2)
    local_scalar = torch.norm(torch.cat(grads_scalar), 2)
    local_head = torch.norm(torch.cat(grads_head), 2)
    local_hidden_matrix = torch.norm(torch.cat(grads_hidden_matrix), 2)

    return local_norm, (local_embed, local_scalar, local_head, local_hidden_matrix)


# detect gradient spikes as in SPAM
def detect_grad_spikes(optimizers, threshold=50):

    if not isinstance(optimizers, list):
        optimizers = [optimizers]

    gradient_spikes = [0]

    for optimizer in optimizers:

        for group in optimizer.param_groups:

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = optimizer.state[p]

                # If EMA states not yet initialized, cannot detect spikes
                if "exp_avg_sq" not in state:
                    break

                # Threshold-based gradient masking
                mask = (grad ** 2) > (threshold * state["exp_avg_sq"])
                if mask.sum() > 0:
                    idx_max_grad = torch.argmax(torch.abs(grad[mask]))
                    max_grad = grad[mask][idx_max_grad]
                    gradient_spikes.append(max_grad)

    gradient_spike = max(gradient_spikes)

    return gradient_spike


# detect gradient spikes as in SPAM
# @timeit ~ 0.007 segs, slightly faster than `log_max_grads_by_param_name`
def log_max_grads(optimizers):

    if not isinstance(optimizers, list):
        optimizers = [optimizers]

    gradient_max = [0]

    for optimizer in optimizers:

        for group in optimizer.param_groups:

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Log the maximum (neg or pos) gradient
                idx_max_grad = torch.argmax(torch.abs(p.grad.view(-1)))
                max_grad = p.grad.view(-1)[idx_max_grad]
                gradient_max.append(max_grad)

    gradient_max = torch.tensor(gradient_max)
    idx_max_grad = torch.argmax(torch.abs(gradient_max))
    gradient_spike = gradient_max[idx_max_grad]

    return gradient_spike

# compute gradient norm of the whole model
# @timeit # ~ 0.0093, slightly slower than `log_max_grads`
def log_max_grads_by_param_name(model):

    # Calculate global gradient norm
    grads = []
    grads_embed = []
    grads_scalar = []
    grads_head = []
    grads_hidden_matrix = []
    for n, p in model.named_parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().flatten())
            if "embed" in n:
                grads_embed.append(grads[-1])
            if p.ndim < 2:
                grads_scalar.append(grads[-1])
            if p.ndim >= 2 and "embed" not in n and "lm_head" not in n:
                grads_hidden_matrix.append(grads[-1])
            if "lm_head" in n:
                grads_head.append(grads[-1])

    grads = torch.cat(grads)
    idx_max_grads = torch.argmax(torch.abs(grads))
    local_norm = grads[idx_max_grads]

    grads_embed = torch.cat(grads_embed)
    idx_max_grads_embed = torch.argmax(torch.abs(grads_embed))
    local_embed = grads[idx_max_grads_embed]

    grads_scalar = torch.cat(grads_scalar)
    idx_max_grads_scalar = torch.argmax(torch.abs(grads_scalar))
    local_scalar = grads[idx_max_grads_scalar]

    grads_head = torch.cat(grads_head)
    idx_max_grads_head = torch.argmax(torch.abs(grads_head))
    local_head = grads[idx_max_grads_head]

    grads_hidden_matrix = torch.cat(grads_hidden_matrix)
    idx_max_grads_hidden_matrix = torch.argmax(torch.abs(grads_hidden_matrix))
    local_hidden_matrix = grads[idx_max_grads_hidden_matrix]

    return local_norm, (local_embed, local_scalar, local_head, local_hidden_matrix)