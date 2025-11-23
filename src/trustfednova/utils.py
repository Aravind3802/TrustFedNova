from __future__ import annotations
import math
import random
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn

""" The SetSeed() method I used was making the experiments reproducible and makes it easy for comparision between the algorithms, which is the major outcome of this project."""
def SetSeed(Seed: int):
    random.seed(Seed)
    np.random.seed(Seed)
    torch.manual_seed(Seed)


"""The GetDevice() method is machine-specific, that is, I used it to check for Apple MPS so that PyTorch uses MPS instead of CUDA, without this when the simulator was initially run,Pytorch tried to use CUDA and crashed as
Apple does not support CUDA. This also makes the code cross-platform implementable."""
def GetDevice() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


"""The ToDevice() method used below is for moving all the tensors to same device, that is to CPU, GPU or Applw Silicon and thus making sure that the run time saying " all tensors to be on the same device is not faced"."""
def ToDevice(StateDict: Dict[str, torch.Tensor], Device: torch.device):
    return {Key: Value.to(Device) for Key, Value in StateDict.items()}



"""The _IsFloatTensor()  method is used to check if the instances in state_dict are tensors and are of the datatype float/complex, this is needed beacyse we want to compute the deltas
only for learnable parameters but not on other parameters like flags or counters."""
def _IsFloatTensor(TensorObject):
    return isinstance(TensorObject, torch.Tensor) and (TensorObject.is_floating_point() or TensorObject.is_complex())


"""The safe helper utilites are named such way as they make sure that the tensors that they perform the operations are infact floating point values and not other parameters, the functions below get the local model state, subtract the global model,
, normalie them in case of FedNova, accumulate the weights and also clone the state dict for next-round of updates."""
def SubtractStateDictSafe(StateDictA: Dict[str, torch.Tensor], StateDictB: Dict[str, torch.Tensor]):
    OutputStateDict = {}
    for Key, ValueA in StateDictA.items():
        if not _IsFloatTensor(ValueA):
            continue
        ValueB = StateDictB.get(Key, None)
        OutputStateDict[Key] = ValueA.detach().clone()
        if _IsFloatTensor(ValueB):
            OutputStateDict[Key].add_(ValueB, alpha=-1.0)
    return OutputStateDict


def AddStateDictSafe(StateDictA: Dict[str, torch.Tensor], StateDictB: Dict[str, torch.Tensor], Alpha: float = 1.0):
    OutputStateDict = {}
    AlphaFloat = float(Alpha)
    for Key, ValueA in StateDictA.items():
        if not _IsFloatTensor(ValueA):
            continue
        OutputStateDict[Key] = ValueA.detach().clone()
        ValueB = StateDictB.get(Key, None)
        if _IsFloatTensor(ValueB):
            OutputStateDict[Key].add_(ValueB, alpha=AlphaFloat)
    return OutputStateDict


def ScaleStateDictSafe(State: Dict[str, torch.Tensor], Scale: float):
    OutputStateDict = {}
    ScaleFloat = float(Scale)
    for Key, Value in State.items():
        if _IsFloatTensor(Value):
            OutputStateDict[Key] = Value.detach().clone().mul_(ScaleFloat)
    return OutputStateDict


def ZerosLikeStateDictSafe(ReferenceStateDict: Dict[str, torch.Tensor], FillValue: float = 0.0):
    OutputStateDict = {}
    FillFloat = float(FillValue)
    for Key, Value in ReferenceStateDict.items():
        if _IsFloatTensor(Value):
            OutputStateDict[Key] = Value.detach().clone().fill_(FillFloat)
    return OutputStateDict


def CloneStateDictSafe(ReferenceStateDict: Dict[str, torch.Tensor]):
    OutputStateDict = {}
    for Key, Value in ReferenceStateDict.items():
        if _IsFloatTensor(Value):
            OutputStateDict[Key] = Value.detach().clone()
    return OutputStateDict


"""The functions below is used do all the operations in-place instead of allocating the memory for same set of tensors again and again. """
def AddStateDictInplaceSafe(TargetStateDict: Dict[str, torch.Tensor], SourceStateDict: Dict[str, torch.Tensor], Alpha: float = 1.0):
    AlphaFloat = float(Alpha)
    for Key, Value in SourceStateDict.items():
        if Key in TargetStateDict and _IsFloatTensor(TargetStateDict[Key]) and _IsFloatTensor(Value):
            TargetStateDict[Key].add_(Value, alpha=AlphaFloat)


def SubStateDictInplaceSafe(TargetStateDict: Dict[str, torch.Tensor], SourceStateDict: Dict[str, torch.Tensor], Alpha: float = 1.0):
    AlphaFloat = float(Alpha)
    for Key, Value in SourceStateDict.items():
        if Key in TargetStateDict and _IsFloatTensor(TargetStateDict[Key]) and _IsFloatTensor(Value):
            TargetStateDict[Key].add_(Value, alpha=-AlphaFloat)


def ScaleStateDictInplaceSafe(TargetStateDict: Dict[str, torch.Tensor], Scale: float):
    ScaleFloat = float(Scale)
    for Key, Value in list(TargetStateDict.items()):
        if _IsFloatTensor(Value):
            Value.mul_(ScaleFloat)


def AddStateDictInplace(TargetStateDict, SourceStateDict, Alpha: float = 1.0):
    return AddStateDictInplaceSafe(TargetStateDict, SourceStateDict, Alpha)


def SubStateDictInplace(TargetStateDict, SourceStateDict, Alpha: float = 1.0):
    return SubStateDictInplaceSafe(TargetStateDict, SourceStateDict, Alpha)


def ScaleStateDict(State, Scale: float):
    return ScaleStateDictSafe(State, Scale)


def SubtractStateDict(StateDictA, StateDictB):
    return SubtractStateDictSafe(StateDictA, StateDictB)


def ZerosLikeStateDict(ReferenceStateDict):
    return ZerosLikeStateDictSafe(ReferenceStateDict)


def ZeroLike(ReferenceStateDict):
    return ZerosLikeStateDictSafe(ReferenceStateDict)


def CloneStateDict(ObjectToClone):
    
    if isinstance(ObjectToClone, nn.Module):
        return {Key: Value.detach().clone() for Key, Value in ObjectToClone.state_dict().items() if _IsFloatTensor(Value)}
    elif isinstance(ObjectToClone, dict):
        return CloneStateDictSafe(ObjectToClone)
    else:
        raise TypeError(f"CloneStateDict expected nn.Module or Dict[str, Tensor], got {type(ObjectToClone)}")

"""The method below computes the moralisation for the updates."""
def StateDictNorm(StateDictionary: Dict[str, torch.Tensor]) -> float:
    return math.sqrt(sum((Value.float().pow(2).sum().item()) for Value in StateDictionary.values() if _IsFloatTensor(Value)))

"""The function below computes the cosine-similarity between the updates """
def StateDictCosine(StateDictA: Dict[str, torch.Tensor], StateDictB: Dict[str, torch.Tensor], Epsilon: float = 1e-12) -> float:
    CommonKeys = [Key for Key in StateDictA.keys() if Key in StateDictB and _IsFloatTensor(StateDictA[Key]) and _IsFloatTensor(StateDictB[Key])]
    if not CommonKeys:
        return 1.0
    Numerator = 0.0
    DenominatorA = 0.0
    DenominatorB = 0.0
    for Key in CommonKeys:
        VectorA = StateDictA[Key].float().view(-1)
        VectorB = StateDictB[Key].float().view(-1)
        Numerator += float(torch.dot(VectorA, VectorB))
        DenominatorA += float(torch.dot(VectorA, VectorA))
        DenominatorB += float(torch.dot(VectorB, VectorB))
    if DenominatorA <= Epsilon or DenominatorB <= Epsilon:
        return 1.0
    return float(Numerator / math.sqrt(DenominatorA * DenominatorB))

