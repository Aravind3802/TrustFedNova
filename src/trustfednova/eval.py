from __future__ import annotations
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn

"""The Evaluate() methoda below is used to evaluate the global model on test datset after each communcation round and then computes the accuracy and loss. It runsafter each round, computes the metrics for plotting anf then updates the lambda
value  accordingly."""
@torch.no_grad()
def Evaluate(Model: nn.Module, EvaluationLoader: DataLoader, Device: torch.device) -> Tuple[float, float]:
    Model.eval()
    TotalSamples = 0
    CorrectPredictions = 0
    TotalLoss = 0.0
    for InputBatch, TargetBatch in EvaluationLoader:
        InputBatch, TargetBatch = InputBatch.to(Device), TargetBatch.to(Device)
        Logits = Model(InputBatch)
        LossValue = F.cross_entropy(Logits, TargetBatch)
        BatchSizeValue = InputBatch.size(0)
        TotalLoss += float(LossValue.cpu().item()) * BatchSizeValue
        Predictions = Logits.argmax(dim=1)
        CorrectPredictions += int((Predictions == TargetBatch).sum().cpu().item())
        TotalSamples += int(BatchSizeValue)
    return CorrectPredictions / max(1, TotalSamples), TotalLoss / max(1, TotalSamples)



