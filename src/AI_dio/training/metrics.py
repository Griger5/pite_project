from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


def _safe_div(num: float, denom: float) -> float:
    return num / denom if denom else 0.0


def _pr_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0:
        return float("nan")
    order = np.argsort(-scores)
    y = labels[order]
    total_pos = int((y == 1).sum())
    if total_pos == 0:
        return float("nan")
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / total_pos
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return float(np.trapz(precision, recall))


@dataclass
class BinaryMetricsAccumulator:
    track_pr_auc: bool = True
    tp0: int = 0
    fn0: int = 0
    fp0: int = 0
    tn0: int = 0
    tp1: int = 0
    fn1: int = 0
    fp1: int = 0
    tn1: int = 0
    _scores: list[torch.Tensor] = field(default_factory=list)
    _labels: list[torch.Tensor] = field(default_factory=list)
    _disabled: bool = False

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        if self._disabled:
            return
        if logits.dim() != 2 or logits.size(1) != 2:
            self._disabled = True
            return
        preds = logits.argmax(dim=1)
        pred0 = preds == 0
        pred1 = preds == 1
        y0 = labels == 0
        y1 = labels == 1
        self.tp0 += (pred0 & y0).sum().item()
        self.fn0 += (pred1 & y0).sum().item()
        self.fp0 += (pred0 & y1).sum().item()
        self.tn0 += (pred1 & y1).sum().item()
        self.tp1 += (pred1 & y1).sum().item()
        self.fn1 += (pred0 & y1).sum().item()
        self.fp1 += (pred1 & y0).sum().item()
        self.tn1 += (pred0 & y0).sum().item()
        if self.track_pr_auc:
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu()
            self._scores.append(probs)
            self._labels.append(labels.detach().cpu())

    def compute(self) -> dict:
        if self._disabled:
            return {}
        recall0 = _safe_div(self.tp0, self.tp0 + self.fn0)
        precision0 = _safe_div(self.tp0, self.tp0 + self.fp0)
        f1_0 = _safe_div(2 * precision0 * recall0, precision0 + recall0)
        recall1 = _safe_div(self.tp1, self.tp1 + self.fn1)
        balanced_acc = 0.5 * (recall0 + recall1)
        if self.track_pr_auc and self._scores and self._labels:
            y_np = torch.cat(self._labels).numpy().astype(np.int64)
            s_np = torch.cat(self._scores).numpy().astype(np.float64)
            pr_auc_1 = _pr_auc(y_np, s_np)
            pr_auc_0 = _pr_auc(1 - y_np, 1 - s_np)
        else:
            pr_auc_1 = float("nan")
            pr_auc_0 = float("nan")
        return {
            "tp0": self.tp0,
            "fn0": self.fn0,
            "fp0": self.fp0,
            "tn0": self.tn0,
            "tp1": self.tp1,
            "fn1": self.fn1,
            "fp1": self.fp1,
            "tn1": self.tn1,
            "recall0": recall0,
            "precision0": precision0,
            "f1_0": f1_0,
            "recall1": recall1,
            "balanced_acc": balanced_acc,
            "pr_auc_1": pr_auc_1,
            "pr_auc_0": pr_auc_0,
        }
