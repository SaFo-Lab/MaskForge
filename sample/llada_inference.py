from __future__ import annotations
import math, json, os, time
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from jinja2 import Template
import torch
from termcolor import cprint
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from .llada.modeling_llada import LLaDAModelLM
import multiprocessing as mp

from omegaconf import DictConfig, ListConfig, OmegaConf


def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    noise = (- torch.log(noise)) ** temperature
    return logits.exp() / noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


# ──────────────────────────── return type ────────────────────────────────
@dataclass
class DiffusionOutput:
    sequences: torch.Tensor  # final result  (B, L_total)  (GPU)
    history: List[torch.Tensor]  # all intermediate x (CPU)
    nfe: int


@torch.no_grad()
def generate_with_prefix_cache(
    model,
    prompt,
    template_ids,
    editable_mask,
    steps,
    gen_length,
    block_length,
    temperature,
    target,
    mask_id,
    further_horizon,
    use_cache,
    unmask_threshold,
) -> DiffusionOutput:
    """
    Args:
        prompt:        (B, L0)
        template_ids:  (B, Lt)   tokenized template region
        editable_mask: (B, Lt)   True means this position can be denoised/unmasked
    """

    cgws = further_horizon
    device = prompt.device

    B, L0 = prompt.shape
    B2, Lt = template_ids.shape

    assert B == B2, "prompt and template_ids batch size mismatch"
    assert editable_mask.shape == template_ids.shape, \
        "editable_mask must have same shape as template_ids"

    # normalize template length to gen_length
    if Lt < gen_length:
        pad_len = gen_length - Lt

        template_pad = torch.full(
            (B, pad_len),
            mask_id,
            dtype=template_ids.dtype,
            device=template_ids.device,
        )
        editable_pad = torch.ones(
            (B, pad_len),
            dtype=torch.bool,
            device=editable_mask.device,
        )

        template_ids = torch.cat([template_ids, template_pad], dim=1)
        editable_mask = torch.cat([editable_mask, editable_pad], dim=1)

    elif Lt > gen_length:
        template_ids = template_ids[:, :gen_length]
        editable_mask = editable_mask[:, :gen_length]

    assert template_ids.shape == (B, gen_length), \
        f"template_ids shape {template_ids.shape} != {(B, gen_length)}"
    assert editable_mask.shape == (B, gen_length), \
        f"editable_mask shape {editable_mask.shape} != {(B, gen_length)}"

    # full sequence = [prompt | template]
    x = torch.empty((B, L0 + gen_length), dtype=torch.long, device=device)
    x[:, :L0] = prompt
    x[:, L0:] = template_ids

    max_length = L0 + gen_length

    # full editable mask on whole sequence
    editable_mask_full = torch.zeros((B, max_length), dtype=torch.bool, device=device)
    editable_mask_full[:, L0:] = editable_mask

    # fixed token positions in the template region
    fixed_mask_full = torch.zeros((B, max_length), dtype=torch.bool, device=device)
    fixed_mask_full[:, L0:] = ~editable_mask

    # save fixed values so we can restore them after each update
    fixed_token_values = x.clone()

    assert gen_length % block_length == 0, \
        f"gen_length {gen_length} must be divisible by block_length {block_length}"

    num_blocks = gen_length // block_length
    base, rem = divmod(steps, num_blocks)
    steps_per_block = [base + (i < rem) for i in range(num_blocks)]

    nfe = 0
    hist: List[torch.Tensor] = []

    for blk in range(num_blocks):
        s = L0 + blk * block_length
        e = L0 + (blk + 1) * block_length

        if cgws is not None:
            window_end = min(e + cgws, max_length)
            window_slice = slice(s, window_end)

        cur_steps = steps_per_block[blk]

        # only masked + editable tokens in current block are candidates
        cur_block_mask = (x[:, s:e] == mask_id) & editable_mask_full[:, s:e]
        num_transfer = get_num_transfer_tokens(cur_block_mask, cur_steps)

        # first forward for this block
        if use_cache:
            out = model(x, use_cache=True)
            pkv = out.past_key_values
            pkv = tuple(
                tuple(t[:, :, :s] for t in layer) for layer in pkv
            )
        else:
            out = model(x, use_cache=False)

        # first update: only allow positions up to e
        mask_all = (x == mask_id) & editable_mask_full
        mask_all[:, e:] = 0

        x0, tr_idx = get_transfer_index(
            out.logits,
            temperature,
            target,
            mask_all,
            x,
            num_transfer[:, 0],
            unmask_threshold,
        )
        x[tr_idx] = x0[tr_idx]

        # restore fixed tokens
        x[fixed_mask_full] = fixed_token_values[fixed_mask_full]

        hist.append(x.clone().cpu())
        nfe += 1

        i = 1
        while True:
            # stop if no editable masked tokens remain in current block
            if ((x[:, s:e] == mask_id) & editable_mask_full[:, s:e]).sum() == 0:
                break

            # safety: respect step budget for this block
            if i >= cur_steps:
                break

            nfe += 1

            if cgws is not None:
                local_x = x[:, window_slice]
                local_editable = editable_mask_full[:, window_slice]
                mask_blk = (local_x == mask_id) & local_editable
                mask_blk[:, block_length:] = 0
            else:
                local_x = x[:, s:]
                local_editable = editable_mask_full[:, s:]
                mask_blk = (local_x == mask_id) & local_editable
                mask_blk[:, block_length:] = 0

            if use_cache:
                if cgws is not None:
                    logits = model(
                        x[:, window_slice],
                        past_key_values=pkv,
                        use_cache=True,
                    ).logits

                    x0, tr_idx = get_transfer_index(
                        logits,
                        temperature,
                        target,
                        mask_blk,
                        x[:, window_slice],
                        num_transfer[:, i],
                        unmask_threshold,
                    )
                    x[:, window_slice][tr_idx] = x0[tr_idx]
                else:
                    logits = model(
                        x[:, s:],
                        past_key_values=pkv,
                        use_cache=True,
                    ).logits

                    x0, tr_idx = get_transfer_index(
                        logits,
                        temperature,
                        target,
                        mask_blk,
                        x[:, s:],
                        num_transfer[:, i],
                        unmask_threshold,
                    )
                    x[:, s:][tr_idx] = x0[tr_idx]
            else:
                logits = model(x, use_cache=False).logits[:, s:]

                x0, tr_idx = get_transfer_index(
                    logits,
                    temperature,
                    target,
                    mask_blk,
                    x[:, s:],
                    num_transfer[:, i],
                    unmask_threshold,
                )
                x[:, s:][tr_idx] = x0[tr_idx]

            # restore fixed tokens
            x[fixed_mask_full] = fixed_token_values[fixed_mask_full]

            hist.append(x.clone().cpu())
            i += 1

    return DiffusionOutput(sequences=x, history=hist, nfe=nfe)


def get_transfer_index(logits, temperature, target, mask_index, x, num_transfer_tokens, threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

    if target == 'confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
    elif target == 'margin_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        top2 = torch.topk(p, 2, dim=-1).values  # (b, l, 2)
        x0_p = top2[..., 0] - top2[..., 1]  # Δ(top1, top2)
    elif target == 'neg_entropy':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = -torch.sum(p * torch.log(p + 1e-10), dim=-1)  # –entropy
    elif target == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(target)

    x0 = torch.where(mask_index, x0, x)

    if threshold is not None:
        selected = mask_index & (x0_p >= threshold)  # (B, T)

        has_mask = mask_index.any(dim=-1)  # (B,)
        none_sel = (~selected.any(dim=-1)) & has_mask  # (B,)
        if none_sel.any():
            masked_scores = x0_p.masked_fill(~mask_index, float("-inf"))
            best_idx = masked_scores.argmax(dim=-1)  # (B,)
            rows = torch.nonzero(none_sel, as_tuple=False).squeeze(-1)
            selected[rows, best_idx[rows]] = True

        return x0, selected

    confidence = x0_p.masked_fill(~mask_index, float("-inf"))
    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    for j in range(confidence.shape[0]):
        k = int(num_transfer_tokens[j].item() if torch.is_tensor(num_transfer_tokens[j]) else num_transfer_tokens[j])
        if k <= 0:
            continue
        _, sel = torch.topk(confidence[j], k=k)
        transfer_index[j, sel] = True
    return x0, transfer_index


import random


def random_select(data_list, random_k):
    data_list = random.sample(data_list, random_k)
    return data_list


# obtain prompt
def get_prompt(data_i):
    return Template(system_prompts).render(problem=data_i["question"])

def get_template(template):
    template = template.replace("<mask>", "<|mdm_mask|>" * 50)
    return template


def extract_final_boxed_answer(s: str):
    tag = r'\boxed{'
    start = s.rfind(tag)  # last \boxed{
    if start == -1:
        return "Can not extract the answer!"

    i = start + len(tag)
    depth = 1  # we are already inside one '{'
    buf = []

    while i < len(s) and depth:
        ch = s[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:  # matching '}' for the opening \boxed{
                break
        buf.append(ch)
        i += 1

    return ''.join(buf) if depth == 0 else "Can not extract the answer!"


def denoise_step_map(history, mask_id: int, sample_idx: int = 0):
    L = history[0].shape[1]
    step_map = torch.zeros(L, dtype=torch.long)
    prev = torch.full((L,), mask_id, dtype=torch.long)

    for t, snap in enumerate(history, start=0):
        cur = snap[sample_idx]
        changed = (prev == mask_id) & (cur != mask_id)
        step_map[changed] = t
        prev = cur
        if (step_map == 0).sum() == 0:
            break
    return step_map


from tqdm import tqdm

def get_new_prompt(system_prompts: str, question: str):
    return Template(system_prompts).render(problem=question)

def LLaDA_Inference(
    model_gpu,
    tokenizer_gpu,
    rank,
    prompts,
    templates,
    batch_size,
    config,
):
    if isinstance(prompts, str):
        prompts = [prompts]
    if isinstance(templates, str):
        templates = [templates]

    assert len(prompts) == len(templates), "prompts and templates must have the same length"

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    mask_id = tokenizer_gpu.encode('<|mdm_mask|>')[0]

    if config.rollout.use_cache == False:
        config.rollout.further_horizon = None

    if config.rollout.remasking_strategy == "low_confidence_static":
        unmask_threshold = None
    else:
        unmask_threshold = config.rollout.dynamic_threshold

    all_texts = []
    all_step_maps = []

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]
        batch_templates = templates[start:start + batch_size]

        enc = tokenizer_gpu(
            batch_prompts,
            padding=True,
            return_tensors="pt",
            padding_side="left",
        )

        template_enc = tokenizer_gpu(
            batch_templates,
            padding=True,
            return_tensors="pt",
            padding_side="right",
        )

        input_ids = enc["input_ids"].to(device)
        template_ids = template_enc["input_ids"].to(device)
        editable_mask = (template_ids == mask_id)

        out = generate_with_prefix_cache(
            model=model_gpu,
            prompt=input_ids,
            template_ids=template_ids,
            editable_mask=editable_mask,
            steps=config.rollout.steps,
            gen_length=config.rollout.max_gen_length,
            block_length=config.rollout.block_size,
            temperature=config.rollout.temperature,
            target=config.rollout.target,
            mask_id=mask_id,
            further_horizon=config.rollout.further_horizon,
            use_cache=config.rollout.use_cache,
            unmask_threshold=unmask_threshold,
        )

        sequences = out.sequences.detach().cpu()

        seq_ids = sequences[:, input_ids.shape[1]:].tolist()
        texts = tokenizer_gpu.batch_decode(
            seq_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )

        for i in range(len(texts)):
            step_map_tensor = denoise_step_map(
                out.history,
                mask_id=mask_id,
                sample_idx=i,
            )
            step_map = step_map_tensor[input_ids.shape[1]:].tolist()

            all_texts.append(texts[i])
            all_step_maps.append(step_map)

        del enc, input_ids, template_ids, editable_mask, out, sequences, seq_ids, texts
        torch.cuda.empty_cache()

    return all_texts, all_step_maps