from __future__ import annotations
import math


""" This class below is the main contributing part to the project where the dynamica controller contols the value of lambs dynamically by monitoring ht chi^2 value, cosine similarity and the loss progress."""
class DynamicController:
    def __init__(self, LamMin: float, EmaBeta: float, AParameter: float, BParameter: float, CParameter: float, DParameter: float):
        self.LamMin = LamMin
        self.EmaBeta = EmaBeta
        self.AParameter, self.BParameter, self.CParameter, self.DParameter = AParameter, BParameter, CParameter, DParameter
        self.ExponentialMovingAverage = None
        self.PreviousLoss = None

    @staticmethod
    def _Sigmoid(InputValue: float):
        return 1.0 / (1.0 + math.exp(-InputValue))

    def Step(self, ChiSquareFedAvg: float, CosineFed: float, CurrentLoss: float) -> float:
        NoProgressFlag = 0.0
        if self.PreviousLoss is not None and CurrentLoss >= self.PreviousLoss - 1e-4:
            NoProgressFlag = 1.0
        self.PreviousLoss = CurrentLoss
        RawLambda = self._Sigmoid(
            self.AParameter * ChiSquareFedAvg
            + self.BParameter * (1.0 - max(-1.0, min(1.0, CosineFed)))
            + self.CParameter * NoProgressFlag
            + self.DParameter
        )
        if self.ExponentialMovingAverage is None:
            self.ExponentialMovingAverage = RawLambda
        else:
            self.ExponentialMovingAverage = self.EmaBeta * self.ExponentialMovingAverage + (1 - self.EmaBeta) * RawLambda
        LambdaValue = max(self.LamMin, min(1.0, self.ExponentialMovingAverage))
        return LambdaValue


