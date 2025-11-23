from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from .utils import CloneStateDict, SubtractStateDictSafe, ToDevice

"""The class below is used configuring the client, it loads server weights, train on local client dataset, if FedProx is used it applied the proximal term, handles device movement,
collects training statistica, computres model delta, return all-client specific values for aggregation."""
@dataclass
class ClientConfig:
    BatchSize: int = 64
    LearningRate: float = 0.01
    Momentum: float = 0.9
    WeightDecay: float = 5e-4
    ProxMu: float = 0.0  # FedProx μ - proximal term's multiplier


def LocalTrain(
    Model: nn.Module,
    BaseState: Dict[str, torch.Tensor],  # w_t (float-only dict)
    DataLoaderObject: DataLoader,
    Steps: int,
    Device: torch.device,
    Config: ClientConfig,
    AlgorithmName: str,
) -> Tuple[Dict[str, torch.Tensor], int, int, float]:
    
    Model.load_state_dict(BaseState, strict=False)  # float-only dict
    Model.to(Device)
    Model.train()

    Optimizer = optim.SGD(
        Model.parameters(),
        lr=Config.LearningRate,
        momentum=Config.Momentum,
        weight_decay=Config.WeightDecay,
    )

    if AlgorithmName == "fedprox" and Config.ProxMu > 0:
        BaseParameterState = {ParameterName: ParameterValue.detach().clone() for ParameterName, ParameterValue in Model.named_parameters()}
    else:
        BaseParameterState = None

    TotalLoss = 0.0
    SamplesSeen = 0
    StepsCompleted = 0
    DataIterator = iter(DataLoaderObject)

    while StepsCompleted < Steps:
        try:
            InputBatch, TargetBatch = next(DataIterator)
        except StopIteration:
            DataIterator = iter(DataLoaderObject)
            InputBatch, TargetBatch = next(DataIterator)

        InputBatch, TargetBatch = InputBatch.to(Device), TargetBatch.to(Device)
        Optimizer.zero_grad()
        Logits = Model(InputBatch)
        LossValue = F.cross_entropy(Logits, TargetBatch)

        if BaseParameterState is not None:
            ProximalTerm = 0.0
            for ParameterName, ParameterValue in Model.named_parameters():
                BaseParameterValue = BaseParameterState.get(ParameterName)
                if BaseParameterValue is None or BaseParameterValue.shape != ParameterValue.shape:
                    continue
                ProximalTerm = ProximalTerm + (ParameterValue - BaseParameterValue.to(ParameterValue.device)).pow(2).sum()
            LossValue = LossValue + (Config.ProxMu / 2.0) * ProximalTerm

        LossValue.backward()
        Optimizer.step()

        BatchSizeValue = InputBatch.size(0)
        TotalLoss += float(LossValue.detach().cpu().item()) * BatchSizeValue
        SamplesSeen += int(BatchSizeValue)
        StepsCompleted += 1

    NewState = CloneStateDict(Model)                      
    DeltaState = SubtractStateDictSafe(BaseState, NewState)  
    AverageLoss = TotalLoss / max(1, SamplesSeen)
    return DeltaState, StepsCompleted, SamplesSeen, AverageLoss
 
