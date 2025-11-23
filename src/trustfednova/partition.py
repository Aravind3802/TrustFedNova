from __future__ import annotations
import math
import numpy as np
from typing import Dict, List, Tuple

"""The method below is used to create realistic non-IID client data distribution, the skewness in controlled by the alpha paramter and it becomes determinitic when given for a certain seed, which facilitates fair comparison between the algorithms."""
def DirichletNoniidIndices(LabelArray: np.ndarray, NumClients: int, AlphaParameter: float, Seed: int) -> List[List[int]]:
    RandomGenerator = np.random.default_rng(Seed)
    NumClasses = int(LabelArray.max()) + 1
    IndicesByClass = [np.where(LabelArray == ClassIndex)[0] for ClassIndex in range(NumClasses)]
    for ClassIndex in range(NumClasses):
        RandomGenerator.shuffle(IndicesByClass[ClassIndex])
    ClientIndices = [[] for _ in range(NumClients)]
    for ClassIndex in range(NumClasses):
        ClassCount = len(IndicesByClass[ClassIndex])
        if ClassCount == 0:
            continue
        Proportions = RandomGenerator.dirichlet(alpha=[AlphaParameter] * NumClients)
        Counts = (Proportions * ClassCount).astype(int)
        while Counts.sum() < ClassCount:
            Counts[RandomGenerator.integers(0, NumClients)] += 1
        StartIndex = 0
        for ClientIndex in range(NumClients):
            TakeCount = Counts[ClientIndex]
            ClientIndices[ClientIndex].extend(IndicesByClass[ClassIndex][StartIndex:StartIndex + TakeCount].tolist())
            StartIndex += TakeCount
    for ClientIndex in range(NumClients):
        RandomGenerator.shuffle(ClientIndices[ClientIndex])
    return ClientIndices


"""The method below is used to simulate client compute heterogenity, which means that each client will have different number of local steps."""
def DrawTauLognormal(NumClients: int, MeanSteps: float, HeterogeneityScale: float, Seed: int) -> np.ndarray:
    RandomGenerator = np.random.default_rng(Seed)
    SigmaValue = max(1e-6, HeterogeneityScale)
    MuValue = math.log(max(1.0, MeanSteps)) - 0.5 * SigmaValue * SigmaValue
    TauArray = RandomGenerator.lognormal(mean=MuValue, sigma=SigmaValue, size=NumClients)
    TauArray = np.maximum(1.0, TauArray)
    return TauArray

