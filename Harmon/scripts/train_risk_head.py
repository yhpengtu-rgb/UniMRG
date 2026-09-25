#!/usr/bin/env python3
"""Train the trajectory-evidence RiskHead (spec §5.4'.3) and produce the
risk-audit Go/No-Go report (spec §5.5').

Pipeline:
1. Load ``trajectory_evidence.pt`` produced by
   ``scripts/collect_trajectory_evidence.py``.
2. Flatten per-round records into per-token rows
   ``(hidden, confidence, entropy, js_div, stable, committed_history,
      remask, block_commit_corr, state, wrong_commit)``.
3. Split: 80% train / 20% internal validation (early-stop + hyper-parameter
   selection). The risk-audit (512) split is run separately by the same
   script with ``--mode audit`` against an already-trained head.
4. Train RiskHead with BCE, ≤200K params (checked), AdamW, early-stop on
   validation Brier.
5. Output Go/No-Go report with the §5.5' thresholds:
   - AUROC ≥ 0.75
   - Brier ≤ 0.15
   - ECE ≤ 0.10
   - risk-coverage AUC > 0.10 above random commit
   - committed vs uncommitted KS > 0.05

Usage:
    # Fit mode: train on risk-fit trajectory data
    python scripts/train_risk_head.py fit \\
        --trajectory /nvmedata/xiexu/data/uni/guard_trajectory_v1/trajectory_evidence.pt \\
        --output-dir /nvmedata/xiexu/data/uni/guard_risk_v1 \\
        --epochs 50 --lr 1e-3 --batch-size 1024

    # Audit mode: run the §5.5' Go/No-Go on the risk-audit split
    python scripts/train_risk_head.py audit \\
        --risk-audit /nvmedata/xiexu/data/uni/guard_exclusions/risk-audit.json \\
        --checkpoint /home/xiexu/code/UniMRG/Harmon/work_dirs/UniMRG_dllm_lora_opt/iter_20000.pth \\
        --config configs/examples/UniMRG_dllm_lora_stable_infer.py \\
        --risk-head /nvmedata/xiexu/data/uni/guard_risk_v1/risk_head.pt \\
        --output-dir /nvmedata/xiexu/data/uni/guard_risk_v1/audit
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

HARMON_ROOT = Path(__file__).resolve().parents[1]
if str(HARMON_ROOT) not in sys.path:
    sys.path.insert(0, str(HARMON_ROOT))

from src.models.dllm.guard.risk_control import RiskHead


def configure_reproducibility(seed: int) -> dict:
    """Seed every RNG used by RiskHead fitting and require deterministic ops."""
    seed = int(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    return {
        'seed': seed,
        'pythonhashseed': str(seed),
        'cublas_workspace_config': ':4096:8',
        'torch_deterministic_algorithms': True,
        'cudnn_benchmark': False,
        'cudnn_deterministic': True,
        'dataloader_workers': 0,
        'dataloader_generator_seed': seed,
    }


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_trajectory_evidence(path: str) -> List[dict]:
    """Load ``trajectory_evidence.pt`` and return the per-sample records list."""
    blob = torch.load(path, map_location='cpu')
    return blob['samples']


def flatten_records(samples: List[dict]) -> Tuple[torch.Tensor, ...]:
    """Flatten per-sample, per-round records into per-token tensors.

    Returns
    -------
    tuple of tensors
        hidden, confidence, entropy, js_div, stable,
        committed_history, remask, block_commit_corr, state, wrong_commit
        ``hidden``: ``[N, hidden_size]`` (cast to float32).
        Others: ``[N]`` or ``[N, 3]``.
    """

    hidden_chunks = []
    conf_chunks = []
    ent_chunks = []
    js_chunks = []
    stable_chunks = []
    comh_chunks = []
    rem_chunks = []
    corr_chunks = []
    state_chunks = []
    y_chunks = []

    for s in samples:
        for r in s['records']:
            # Each tensor has shape [1, block, ...]; flatten to [block, ...].
            hidden_chunks.append(r['hidden'].reshape(-1, r['hidden'].shape[-1]).float())
            conf_chunks.append(r['confidence'].reshape(-1).float())
            ent_chunks.append(r['entropy'].reshape(-1).float())
            js_chunks.append(r['js_div'].reshape(-1).float())
            stable_chunks.append(r['stable'].reshape(-1).float())
            comh_chunks.append(r['committed_history'].reshape(-1).float())
            rem_chunks.append(r['remask'].reshape(-1).float())
            corr_chunks.append(r['block_commit_corr'].reshape(-1).float())
            # state is [1, 3] per round; broadcast to each token in the block.
            state = r['state'].reshape(-1).float()  # [3]
            block_len = r['hidden'].shape[1]
            state_chunks.append(state.unsqueeze(0).expand(block_len, 3))
            y_chunks.append(r['wrong_commit'].reshape(-1).float())

    hidden = torch.cat(hidden_chunks, dim=0)  # [N, H]
    confidence = torch.cat(conf_chunks, dim=0)  # [N]
    entropy = torch.cat(ent_chunks, dim=0)
    js_div = torch.cat(js_chunks, dim=0)
    stable = torch.cat(stable_chunks, dim=0)
    committed_history = torch.cat(comh_chunks, dim=0)
    remask = torch.cat(rem_chunks, dim=0)
    block_commit_corr = torch.cat(corr_chunks, dim=0)
    state = torch.cat(state_chunks, dim=0)  # [N, 3]
    wrong_commit = torch.cat(y_chunks, dim=0)  # [N]

    return (hidden, confidence, entropy, js_div, stable,
            committed_history, remask, block_commit_corr, state, wrong_commit)


# ---------------------------------------------------------------------------
# Metrics (spec §5.5')
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """ROC-AUC via the Wilcoxon-Mann-Whitney statistic."""
    scores = scores.float().cpu()
    labels = labels.float().cpu()
    pos = scores[labels > 0.5]
    neg = scores[labels <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return 0.5
    # Count pairs (i, j) with score_pos > score_neg, 0.5 for ties.
    diff = pos.unsqueeze(1) - neg.unsqueeze(0)
    auroc = (diff > 0).float().mean().item() + 0.5 * (diff == 0).float().mean().item()
    return float(auroc)


@torch.no_grad()
def compute_brier(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores = scores.float().cpu()
    labels = labels.float().cpu()
    return float(((scores - labels) ** 2).mean().item())


@torch.no_grad()
def compute_ece(scores: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    scores = scores.float().cpu()
    labels = labels.float().cpu()
    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = scores.numel()
    for i in range(n_bins):
        mask = (scores >= bin_edges[i]) & (scores < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (scores >= bin_edges[i]) & (scores <= bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf = scores[mask].mean().item()
        bin_acc = labels[mask].mean().item()
        ece += (mask.float().sum().item() / n) * abs(bin_conf - bin_acc)
    return float(ece)


@torch.no_grad()
def compute_risk_coverage_auc(
    scores: torch.Tensor, labels: torch.Tensor, n_points: int = 10
) -> float:
    """Risk-coverage AUC above the random-commit baseline.

    Tokens are sorted by ascending risk; we commit them in order, recording
    accuracy at each coverage level. The random-commit baseline has constant
    accuracy = base_acc, so its AUC over coverage [0,1] = base_acc. We report
    ``auc(model) - base_acc``.

    ``labels`` are wrong-commit indicators (1 = wrong). Correct = 1 - label.
    """
    scores = scores.float().cpu()
    labels = labels.float().cpu()
    order = torch.argsort(scores)
    labels_sorted = labels[order]
    n = labels_sorted.numel()
    # Base accuracy under random commit = fraction of correct tokens.
    base_acc = float((1.0 - labels).mean().item())
    # Correct = 1 - wrong_commit, so accuracy = 1 - label.
    correct_sorted = 1.0 - labels_sorted
    cum_acc = torch.cumsum(correct_sorted, dim=0) / torch.arange(
        1, n + 1, dtype=torch.float32
    )
    # Sample n_points coverage levels uniformly in (0, 1].
    coverages = torch.linspace(1.0 / n_points, 1.0, n_points)
    auc_model = 0.0
    prev_c = 0.0
    prev_acc = base_acc  # at coverage 0 we know nothing; assume base_acc
    for c in coverages:
        idx = max(1, int(c.item() * n)) - 1
        acc = float(cum_acc[idx].item())
        auc_model += 0.5 * (acc + prev_acc) * (c.item() - prev_c)
        prev_c = c.item()
        prev_acc = acc
    return float(auc_model - base_acc)


@torch.no_grad()
def compute_ks(
    scores_committed: torch.Tensor, scores_uncommitted: torch.Tensor
) -> float:
    """Kolmogorov-Smirnov statistic between two 1D distributions."""
    if scores_committed.numel() == 0 or scores_uncommitted.numel() == 0:
        return 0.0
    a = torch.sort(scores_committed.float().cpu()).values
    b = torch.sort(scores_uncommitted.float().cpu()).values
    all_scores = torch.cat([a, b])
    # Use the sorted unique values as evaluation points.
    eval_pts = torch.unique(torch.sort(all_scores).values)
    cdf_a = torch.searchsorted(a, eval_pts, right=True).float() / a.numel()
    cdf_b = torch.searchsorted(b, eval_pts, right=True).float() / b.numel()
    return float(torch.max(torch.abs(cdf_a - cdf_b)).item())


# ---------------------------------------------------------------------------
# Fit mode: train RiskHead on risk-fit trajectory data
# ---------------------------------------------------------------------------

def fit(args):
    reproducibility = configure_reproducibility(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading trajectory evidence from {args.trajectory}...')
    samples = load_trajectory_evidence(args.trajectory)
    print(f'  {len(samples)} samples loaded')

    (hidden, confidence, entropy, js_div, stable,
     committed_history, remask, block_commit_corr, state, wrong_commit
     ) = flatten_records(samples)

    n = hidden.shape[0]
    hidden_size = hidden.shape[1]
    pos_rate = float(wrong_commit.mean().item())
    print(f'  {n} per-token rows, hidden_size={hidden_size}, '
          f'wrong-commit rate={pos_rate:.4f}')

    # Train / validation split (80/20), shuffled.
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed))
    n_val = max(1, int(0.2 * n))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    def select(idx):
        return (hidden[idx], confidence[idx], entropy[idx], js_div[idx],
                stable[idx], committed_history[idx], remask[idx],
                block_commit_corr[idx], state[idx], wrong_commit[idx])

    train_tensors = select(train_idx)
    val_tensors = select(val_idx)

    train_ds = TensorDataset(*train_tensors)
    val_ds = TensorDataset(*val_tensors)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False
    )

    # Build RiskHead.
    head = RiskHead(hidden_size=hidden_size, state_dim=state.shape[-1],
                    proj_hidden=args.proj_hidden).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f'RiskHead params: {n_params:,} (budget 200K)')
    if n_params > 200_000:
        raise RuntimeError(
            f'RiskHead exceeds 200K param budget: {n_params}'
        )

    # Class imbalance handling.
    pos_weight = torch.tensor(
        (1.0 - pos_rate) / max(pos_rate, 1e-6),
        device=device, dtype=torch.float32,
    )
    criterion = nn.BCELoss()  # risk head already applies sigmoid
    optim = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_brier = float('inf')
    best_state = None
    patience_counter = 0

    def _head_forward(head, h, conf, ent, js, st, comh, rem, corr, st_state):
        """Call RiskHead on flat per-token batches by adding a seq dim."""
        n = h.shape[0]
        h3 = h.unsqueeze(1)  # [N, 1, hidden]
        conf3 = conf.unsqueeze(1)
        ent3 = ent.unsqueeze(1)
        js3 = js.unsqueeze(1)
        st3 = st.unsqueeze(1)
        comh3 = comh.unsqueeze(1)
        rem3 = rem.unsqueeze(1)
        corr3 = corr.unsqueeze(1)
        # state is [N, 3]; no seq dim needed (RiskHead broadcasts).
        risk = head(
            h3, conf3, ent3, js3, st3, comh3, rem3, corr3, st_state
        )  # [N, 1, 1]
        return risk.squeeze(-1).squeeze(-1)  # [N]

    def _head_logits(head, h, conf, ent, js, st, comh, rem, corr, st_state):
        """Return pre-sigmoid risk logits (for temperature fitting)."""
        h3 = h.unsqueeze(1)
        conf3 = conf.unsqueeze(1)
        ent3 = ent.unsqueeze(1)
        js3 = js.unsqueeze(1)
        st3 = st.unsqueeze(1)
        comh3 = comh.unsqueeze(1)
        rem3 = rem.unsqueeze(1)
        corr3 = corr.unsqueeze(1)
        # Mirror RiskHead.forward feature packing.
        confidence = head._ensure_3d(conf3)
        entropy = head._ensure_3d(ent3)
        js_div = head._ensure_3d(js3)
        stable = head._ensure_3d(st3)
        committed_history = head._ensure_3d(comh3)
        remask = head._ensure_3d(rem3)
        block_commit_corr = head._ensure_3d(corr3)
        batch, seq, _ = h3.shape
        state = st_state
        if state.ndim == 1:
            state = state.unsqueeze(0)
        state_exp = state.unsqueeze(1).expand(batch, seq, head.state_dim)
        risk_in = torch.cat(
            [h3, confidence, entropy, js_div, stable,
             committed_history, remask, block_commit_corr, state_exp],
            dim=-1,
        )
        return head.risk_fn(risk_in).squeeze(-1).squeeze(-1)  # [N]

    history = []
    for epoch in range(args.epochs):
        head.train()
        train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            (h, conf, ent, js, st, comh, rem, corr, st_state, y) = (
                t.to(device) for t in batch
            )
            risk = _head_forward(
                head, h, conf, ent, js, st, comh, rem, corr, st_state
            )
            # pos_weight via per-sample weight.
            weight = torch.where(
                y > 0.5, pos_weight, torch.ones_like(y)
            )
            loss = F.binary_cross_entropy(risk, y, weight=weight)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optim.step()
            train_loss += float(loss.item())
            n_batches += 1

        # Validation metrics.
        head.eval()
        all_scores = []
        all_labels = []
        for batch in val_loader:
            (h, conf, ent, js, st, comh, rem, corr, st_state, y) = (
                t.to(device) for t in batch
            )
            with torch.no_grad():
                risk = _head_forward(
                    head, h, conf, ent, js, st, comh, rem, corr, st_state
                )
            all_scores.append(risk.cpu())
            all_labels.append(y.cpu())
        scores = torch.cat(all_scores)
        labels = torch.cat(all_labels)
        auroc = compute_auroc(scores, labels)
        brier = compute_brier(scores, labels)
        ece = compute_ece(scores, labels)

        history.append({
            'epoch': epoch,
            'train_loss': train_loss / max(1, n_batches),
            'val_auroc': auroc,
            'val_brier': brier,
            'val_ece': ece,
        })
        print(f'Epoch {epoch:3d} | loss={train_loss/max(1,n_batches):.4f} '
              f'| val AUROC={auroc:.4f} | val Brier={brier:.4f} '
              f'| val ECE={ece:.4f}')

        if brier < best_brier:
            best_brier = brier
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f'Early stop at epoch {epoch} (patience={args.patience})')
                break

    # Save best head.
    if best_state is not None:
        head.load_state_dict(best_state)
    head_path = out_dir / 'risk_head.pt'

    # --- Temperature scaling for ECE calibration (Path C §5.5' fix) ---
    # Post-hoc: freeze head weights, fit scalar T on validation set to
    # minimise ECE. T>1 sharpens confidence, T<1 smooths it.
    head.eval()
    val_logits = []
    val_labels_t = []
    for batch in val_loader:
        (h, conf, ent, js, st, comh, rem, corr, st_state, y) = (
            t.to(device) for t in batch
        )
        with torch.no_grad():
            lg = _head_logits(head, h, conf, ent, js, st, comh, rem, corr, st_state)
        val_logits.append(lg.cpu())
        val_labels_t.append(y.cpu())
    val_logits = torch.cat(val_logits)
    val_labels_t = torch.cat(val_labels_t)
    fitted_t = head.fit_temperature(val_logits, val_labels_t)
    # Measure post-calibration metrics on val.
    with torch.no_grad():
        calibrated_probs = torch.sigmoid(val_logits / fitted_t)
    val_ece_pre = compute_ece(torch.sigmoid(val_logits), val_labels_t)
    val_ece_post = compute_ece(calibrated_probs, val_labels_t)
    val_brier_post = compute_brier(calibrated_probs, val_labels_t)
    print(f'Temperature fit: T={fitted_t:.4f} | '
          f'val ECE pre={val_ece_pre:.4f} -> post={val_ece_post:.4f} | '
          f'val Brier post={val_brier_post:.4f}')

    # --- Isotonic regression on top of temperature (Path C §5.5' ECE fix) ---
    # Temperature scaling is parametric (1 DOF); isotonic regression is
    # non-parametric (monotone PAV) and can correct any monotone distortion.
    # We fit it on the val set AFTER temperature scaling, so the final
    # calibration pipeline is: logit -> /T -> sigmoid -> isotonic -> p_cal.
    from src.models.dllm.guard.risk_control import IsotonicCalibrator
    iso = IsotonicCalibrator()
    iso.fit(calibrated_probs, val_labels_t)
    iso_probs = iso.transform(calibrated_probs)
    val_ece_post_iso = compute_ece(iso_probs, val_labels_t)
    val_brier_post_iso = compute_brier(iso_probs, val_labels_t)
    print(f'Isotonic fit: val ECE post-T={val_ece_post:.4f} -> '
          f'post-iso={val_ece_post_iso:.4f} | '
          f'val Brier post-iso={val_brier_post_iso:.4f}')

    torch.save({
        'state_dict': head.state_dict(),
        'hidden_size': hidden_size,
        'state_dim': state.shape[-1],
        'proj_hidden': args.proj_hidden,
        'best_val_brier': best_brier,
        'n_params': n_params,
        'temperature': float(fitted_t),
        'val_ece_pre_temperature': float(val_ece_pre),
        'val_ece_post_temperature': float(val_ece_post),
        'val_ece_post_isotonic': float(val_ece_post_iso),
        'iso_xs': iso._xs,
        'iso_ys': iso._ys,
        'seed': int(args.seed),
        'reproducibility': reproducibility,
    }, head_path)
    print(f'Saved RiskHead to {head_path} (best val Brier={best_brier:.4f}, '
          f'T={fitted_t:.4f}, iso ECE={val_ece_post_iso:.4f})')

    # Final summary.
    summary = {
        'n_samples': len(samples),
        'n_tokens': n,
        'hidden_size': hidden_size,
        'n_params': n_params,
        'pos_rate': pos_rate,
        'best_val_brier': best_brier,
        'epochs_run': len(history),
        'history': history,
        'temperature': float(fitted_t),
        'val_ece_pre_temperature': float(val_ece_pre),
        'val_ece_post_temperature': float(val_ece_post),
        'val_ece_post_isotonic': float(val_ece_post_iso),
        'iso_n_breakpoints': int(iso._xs.numel()) if iso._xs is not None else 0,
        'reproducibility': reproducibility,
    }
    with open(out_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Summary written to {out_dir / "training_summary.json"}')


# ---------------------------------------------------------------------------
# Audit mode: §5.5' Go/No-Go on the risk-audit (512) split
# ---------------------------------------------------------------------------

def audit(args):
    """Run §5.5' Go/No-Go audit on the risk-audit split.

    Uses the exact guarded ``generate_dllm`` runtime through
    ``collect_self_trajectory.collect_turn_trajectory``.  This is required
    for TRACER because the audit distribution must contain the same block
    size, schedule-aligned state, lagged RankGate and real remask features as
    inference; the older standalone simulator is retained only for legacy
    GUARD reports.
    """
    from PIL import Image
    from src.models.dllm.guard.risk_control import (
        compute_trajectory_features, transfer_schedule_to_mask_probs,
    )
    from transformers import AutoTokenizer
    from xtuner.utils import PROMPT_TEMPLATE

    # We reuse collect_trajectory_evidence.py's helpers to avoid duplication.
    sys.path.insert(0, str(HARMON_ROOT / 'scripts'))
    import collect_trajectory_evidence as cte
    import collect_self_trajectory as cst

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load risk-audit split.
    with open(args.risk_audit, 'r') as f:
        risk_audit_samples = json.load(f)
    if args.max_samples:
        risk_audit_samples = risk_audit_samples[:args.max_samples]
    print(f'Loaded {len(risk_audit_samples)} risk-audit samples')

    # Load model.
    print(f'Loading guarded model from {args.checkpoint}...')
    # New TRACER configs resolve the RiskHead from this environment variable.
    # Binding it to the audited artifact prevents the runtime router from
    # silently using a different head than the final metric computation.
    os.environ['TRACER_RISK_HEAD_PATH'] = str(args.risk_head)
    model, cfg = cst.load_guarded_model(
        args.config, args.checkpoint, device)
    print(f'Model loaded. hidden_size={model.llm.config.hidden_size}')

    # Tokenizer.
    tokenizer_path = cfg.model.llm.get(
        'pretrained_model_name_or_path',
        '/nvmedata/xiexu/data/uni/Qwen2.5-1.5B-Instruct',
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    HARMON_MASK_TOKEN = '<|dllm_mask|>'
    if HARMON_MASK_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_tokens([HARMON_MASK_TOKEN])
    mask_token_id = int(tokenizer.convert_tokens_to_ids(HARMON_MASK_TOKEN))
    eos_token_id = tokenizer.eos_token_id
    template = PROMPT_TEMPLATE['qwen_chat']

    from src.datasets.understanding.llava_datasets import MARProcessor
    image_processor = MARProcessor(image_size=512)

    # Load trained RiskHead.
    ckpt = torch.load(args.risk_head, map_location='cpu')
    head = RiskHead(
        hidden_size=ckpt['hidden_size'],
        state_dim=ckpt['state_dim'],
        proj_hidden=ckpt['proj_hidden'],
    )
    # strict=False for backwards compat with checkpoints saved before the
    # temperature buffer was added.
    missing, unexpected = head.load_state_dict(ckpt['state_dict'], strict=False)
    if missing:
        # 'temperature' is the only expected buffer; default to 1.0 (no scaling).
        head.temperature.fill_(float(ckpt.get('temperature', 1.0)))
        print(f'  Missing buffers (defaulted): {missing}')
    if unexpected:
        print(f'  Unexpected keys: {unexpected}')
    head = head.to(device).eval()
    # Load isotonic calibrator if saved.
    from src.models.dllm.guard.risk_control import IsotonicCalibrator
    iso = None
    if ckpt.get('iso_xs') is not None and ckpt.get('iso_ys') is not None:
        iso = IsotonicCalibrator()
        iso._xs = ckpt['iso_xs'].cpu()
        iso._ys = ckpt['iso_ys'].cpu()
        print(f'  Isotonic calibrator loaded ({iso._xs.numel()} breakpoints)')
    print(f'RiskHead loaded from {args.risk_head} '
          f'(val Brier={ckpt.get("best_val_brier", float("nan")):.4f}, '
          f'T={head.temperature.item():.4f})')

    # Collect trajectory evidence on the risk-audit split, per gpt turn.
    audit_records = []
    for idx, sample in enumerate(risk_audit_samples):
        if idx % 20 == 0:
            print(f'Audit sample {idx}/{len(risk_audit_samples)}...')
        try:
            image_path = os.path.join(args.image_root, sample['image'])
            if not os.path.exists(image_path):
                continue
            image = Image.open(image_path).convert('RGB')
            pixel_values = image_processor.preprocess(
                image, return_tensors='pt')['pixel_values'][0]

            input_ids_list, labels_list, gpt_turns = \
                cte.normalize_conversations(sample, tokenizer, template)
            if not gpt_turns:
                continue

            input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            labels = torch.tensor([labels_list], dtype=torch.long, device=device)
            vocab_size = model.llm.config.vocab_size
            labels[labels >= vocab_size] = -100

            z_enc = cte.encode_image(model, pixel_values, device)
            inputs_embeds, input_ids_expanded, attn_expanded = \
                cte.build_inputs_embeds(
                    model, input_ids, z_enc, attention_mask, device
                )
            n_image_tokens = z_enc.shape[1]
            shift = n_image_tokens - 1

            sample_records = []
            for turn_idx, (resp_start, resp_end) in enumerate(gpt_turns):
                if resp_end <= resp_start:
                    continue
                resp_start_exp = resp_start + shift
                prefix_embeds = inputs_embeds[:, :resp_start_exp, :]
                gt_response_ids = labels[0, resp_start:resp_end].clamp(min=0).unsqueeze(0)

                turn_records = cst.collect_turn_trajectory(
                    model,
                    prefix_embeds,
                    gt_response_ids,
                    args.block_size,
                    args.denoising_steps,
                    args.temperature,
                    mask_token_id,
                    eos_token_id,
                )
                if turn_records:
                    for r in turn_records:
                        r['turn_idx'] = turn_idx
                    sample_records.extend(turn_records)

            if sample_records:
                audit_records.append({
                    'sample_idx': idx,
                    'records': sample_records,
                })
        except Exception as e:
            print(f'  ERROR on sample {idx}: {e}')
            continue

    print(f'Audit records: {len(audit_records)} samples')

    # Flatten + score with RiskHead.
    samples_for_scoring = [{'records': r['records']} for r in audit_records]
    (hidden, confidence, entropy, js_div, stable,
     committed_history, remask, block_commit_corr, state, wrong_commit
     ) = flatten_records(samples_for_scoring)

    hidden = hidden.to(device)
    confidence = confidence.to(device)
    entropy = entropy.to(device)
    js_div = js_div.to(device)
    stable = stable.to(device)
    committed_history = committed_history.to(device)
    remask = remask.to(device)
    block_commit_corr = block_commit_corr.to(device)
    state = state.to(device)
    wrong_commit = wrong_commit  # keep on CPU

    with torch.no_grad():
        # Add seq dim for RiskHead: [N, hidden] -> [N, 1, hidden].
        scores = head(
            hidden.unsqueeze(1),
            confidence.unsqueeze(1),
            entropy.unsqueeze(1),
            js_div.unsqueeze(1),
            stable.unsqueeze(1),
            committed_history.unsqueeze(1),
            remask.unsqueeze(1),
            block_commit_corr.unsqueeze(1),
            state,
        ).squeeze(-1).squeeze(-1).cpu()

    # Apply isotonic calibration on top of temperature-scaled probabilities.
    # Calibration pipeline: logit -> /T -> sigmoid -> isotonic -> p_cal.
    if iso is not None:
        scores = iso.transform(scores)

    # Metrics.
    auroc = compute_auroc(scores, wrong_commit)
    brier = compute_brier(scores, wrong_commit)
    ece = compute_ece(scores, wrong_commit)
    rc_auc = compute_risk_coverage_auc(scores, wrong_commit)
    # KS between committed and uncommitted groups (using committed_history mask).
    committed_mask = committed_history.cpu() > 0.5
    scores_committed = scores[committed_mask]
    scores_uncommitted = scores[~committed_mask]
    ks = compute_ks(scores_committed, scores_uncommitted)

    # Go/No-Go verdict.
    thresholds = {
        'AUROC >= 0.75': auroc >= 0.75,
        'Brier <= 0.15': brier <= 0.15,
        'ECE <= 0.10': ece <= 0.10,
        'risk-coverage AUC > 0.10': rc_auc > 0.10,
        'KS > 0.05': ks > 0.05,
    }
    passed = all(thresholds.values())

    report = {
        'schema_version': 2,
        'audit_runtime': 'exact_guarded_generate_dllm',
        'block_size': int(args.block_size),
        'denoising_steps': int(
            args.denoising_steps or args.block_size),
        'seed': int(args.seed),
        'provenance': {
            'risk_head_path': str(Path(args.risk_head).resolve()),
            'risk_head_sha256': file_sha256(args.risk_head),
            'checkpoint_path': str(Path(args.checkpoint).resolve()),
            'checkpoint_sha256': file_sha256(args.checkpoint),
            'risk_audit_path': str(Path(args.risk_audit).resolve()),
            'risk_audit_sha256': file_sha256(args.risk_audit),
            'config_path': str(Path(args.config).resolve()),
            'config_sha256': file_sha256(args.config),
        },
        'n_audit_samples': len(audit_records),
        'n_tokens': int(wrong_commit.numel()),
        'wrong_commit_rate': float(wrong_commit.mean().item()),
        'metrics': {
            'AUROC': auroc,
            'Brier': brier,
            'ECE': ece,
            'risk_coverage_auc_above_random': rc_auc,
            'KS_committed_vs_uncommitted': ks,
        },
        'thresholds': {k: bool(v) for k, v in thresholds.items()},
        'go_no_go': 'PASS' if passed else 'FAIL',
    }
    print('\n=== Risk-audit Go/No-Go Report (spec §5.5\') ===')
    print(json.dumps(report, indent=2))

    out_path = out_dir / 'risk_audit_report.json'
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\nReport written to {out_path}')

    # Also save raw scores for downstream inspection.
    torch.save({
        'scores': scores,
        'wrong_commit': wrong_commit,
        'committed_history': committed_history.cpu(),
    }, out_dir / 'audit_scores.pt')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)

    p_fit = sub.add_parser('fit', help='Train RiskHead on risk-fit trajectory data')
    p_fit.add_argument('--trajectory', type=str, required=True,
                       help='Path to trajectory_evidence.pt')
    p_fit.add_argument('--output-dir', type=str, required=True)
    p_fit.add_argument('--epochs', type=int, default=50)
    p_fit.add_argument('--lr', type=float, default=1e-3)
    p_fit.add_argument('--weight-decay', type=float, default=1e-4)
    p_fit.add_argument('--batch-size', type=int, default=1024)
    p_fit.add_argument('--proj-hidden', type=int, default=128)
    p_fit.add_argument('--patience', type=int, default=5)
    p_fit.add_argument('--seed', type=int, default=42)
    p_fit.set_defaults(func=fit)

    p_audit = sub.add_parser('audit', help='Run §5.5\' Go/No-Go on risk-audit split')
    p_audit.add_argument('--risk-audit', type=str, required=True,
                         help='Path to risk-audit.json (512 samples)')
    p_audit.add_argument('--checkpoint', type=str, required=True,
                         help='Path to LoRAOpt20k checkpoint')
    p_audit.add_argument('--config', type=str, required=True,
                         help='Path to inference config')
    p_audit.add_argument('--risk-head', type=str, required=True,
                         help='Path to trained risk_head.pt')
    p_audit.add_argument('--output-dir', type=str, required=True)
    p_audit.add_argument('--gpu', type=int, default=0)
    p_audit.add_argument('--block-size', type=int, default=32)
    p_audit.add_argument('--denoising-steps', type=int, default=None)
    p_audit.add_argument('--temperature', type=float, default=0.0)
    p_audit.add_argument('--seed', type=int, default=42,
                         help='RNG seed for sampling (kept for shell parity)')
    p_audit.add_argument('--max-samples', type=int, default=None,
                         help='Limit samples for debugging')
    p_audit.add_argument('--image-root', type=str,
                          default='/nvmedata/xiexu/data/LLaVA-Instruct-150K-UniMRG/tuning_data')
    p_audit.set_defaults(func=audit)

    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
