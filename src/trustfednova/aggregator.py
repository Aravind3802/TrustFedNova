from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np
import torch

from .controller import DynamicController
from .utils import (
    ZerosLikeStateDictSafe,
    ScaleStateDictSafe,
    AddStateDictInplaceSafe,
    StateDictCosine,
)

"""The aggregator class below is used to compute FedAvg updates, FedNova updates, FedProx updates, TrustFedNova updates and also the log statistics. This is the core of the server side."""

class Aggregator:
    def __init__(self, AlgorithmName: str, ControllerObject: DynamicController | None):
        self.AlgorithmName = AlgorithmName.lower()
        self.ControllerObject = ControllerObject

    def Aggregate(
        self,
        BaseState: Dict[str, torch.Tensor],
        ClientDeltas: List[Dict[str, torch.Tensor]],
        Taus: List[int],
        SampleCounts: List[int],
        CurrentValidationLoss: float | None,
    ) -> Tuple[Dict[str, torch.Tensor], float, float, float]:
        ProbabilityVector = np.array(SampleCounts, dtype=np.float64)
        ProbabilityVector = ProbabilityVector / ProbabilityVector.sum()
        TauArray = np.array(Taus, dtype=np.float64)

        FedAvgUpdate = ZerosLikeStateDictSafe(BaseState)
        for ClientIndex, DeltaState in enumerate(ClientDeltas):
            AddStateDictInplaceSafe(FedAvgUpdate, DeltaState, Alpha=float(ProbabilityVector[ClientIndex]))

        FedNovaUpdate = ZerosLikeStateDictSafe(BaseState)
        for ClientIndex, DeltaState in enumerate(ClientDeltas):
            TauValue = max(1.0, float(TauArray[ClientIndex]))
            AddStateDictInplaceSafe(FedNovaUpdate, ScaleStateDictSafe(DeltaState, float(ProbabilityVector[ClientIndex] / TauValue)))

        EpsilonValue = 1e-12
        FedAvgWeights = (ProbabilityVector * TauArray)
        FedAvgWeights = FedAvgWeights / max(EpsilonValue, FedAvgWeights.sum())
        ChiSquareValue = float(((ProbabilityVector - FedAvgWeights) ** 2 / (FedAvgWeights + EpsilonValue)).sum())
        CosineSimilarityValue = StateDictCosine(FedAvgUpdate, FedNovaUpdate)

        if self.AlgorithmName == "fedavg":
            LambdaValue = 0.0
            UpdateState = FedAvgUpdate
        elif self.AlgorithmName == "fednova":
            LambdaValue = 1.0
            UpdateState = FedNovaUpdate
        elif self.AlgorithmName == "trustfednova":
            assert self.ControllerObject is not None
            LambdaValue = self.ControllerObject.Step(
                ChiSquareValue,
                CosineSimilarityValue,
                CurrentValidationLoss if CurrentValidationLoss is not None else 0.0,
            )
            UpdateState = ScaleStateDictSafe(FedAvgUpdate, 1.0 - LambdaValue)
            AddStateDictInplaceSafe(UpdateState, FedNovaUpdate, Alpha=LambdaValue)
        elif self.AlgorithmName == "fedprox":
            LambdaValue = 0.0
            UpdateState = FedAvgUpdate
        else:
            raise ValueError(f"Unknown AlgorithmName {self.AlgorithmName}")
        return UpdateState, LambdaValue, ChiSquareValue, CosineSimilarityValue

