from __future__ import annotations
from dataclasses import dataclass

"""The class below defines the global simulation parameters for the server."""
@dataclass
class ServerConfig:
    Rounds: int = 30
    Clients: int = 30
    Sampled: int = 10
    MeanSteps: float = 20.0
    HeteroScale: float = 0.6
    DirichletAlpha: float = 0.5
    EvalEvery: int = 1
    LamMin: float = 0.2
    EmaBeta: float = 0.9
    AParameter: float = 8.0
    BParameter: float = 4.0
    CParameter: float = 1.0
    DParameter: float = -1.0
    NumWorkers: int = 0
    LogEvery: int = 1
    ServerLearningRate: float = 1.0


""" The class below is used to log the metrics after evry round of communication."""
@dataclass
class RoundLog:
    Round: int
    Accuracy: float
    Loss: float
    LambdaValue: float
    ChiSquare: float
    CosineSimilarity: float

