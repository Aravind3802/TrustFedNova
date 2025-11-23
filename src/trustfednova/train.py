from __future__ import annotations
import os
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .aggregator import Aggregator
from .client import ClientConfig, LocalTrain
from .controller import DynamicController
from .eval import Evaluate
from .partition import DirichletNoniidIndices, DrawTauLognormal
from .server import RoundLog, ServerConfig
from .utils import CloneStateDict, GetDevice, SetSeed, ToDevice

"""The RunExperiment() method is used to fully combine the other methoda and class instances and run the entire fully deterministic experiment for each seed value. Its does non-IID partitioning, heterogenous steps generation, local training as per the
alogirthm choice, dyanmica control of lambda, aggregation, global model update, evaluation and logging."""
def RunExperiment(
    AlgorithmName: str,
    ModelFactory,
    TrainDataset,
    TestDataset,
    ServerConfiguration: ServerConfig,
    ClientConfiguration: ClientConfig,
    SaveDirectory: str,
    Seed: int,
) -> Tuple[List[RoundLog], float]:
    SetSeed(Seed)
    Device = GetDevice()
    WorkerCount = int(ServerConfiguration.NumWorkers)
    print(f"Running {AlgorithmName} on device={Device} with num_workers={WorkerCount} ...")

    LabelArray = np.array(TrainDataset.targets)
    ClientIndices = DirichletNoniidIndices(LabelArray, ServerConfiguration.Clients, ServerConfiguration.DirichletAlpha, Seed)

    ClientLoaders = []
    for ClientIndexList in ClientIndices:
        ClientSubset = Subset(TrainDataset, indices=ClientIndexList)
        ClientLoaders.append(
            DataLoader(
                ClientSubset,
                batch_size=ClientConfiguration.BatchSize,
                shuffle=True,
                num_workers=WorkerCount,
                persistent_workers=(WorkerCount > 0),
                pin_memory=False,
            )
        )

    TestLoader = DataLoader(
        TestDataset,
        batch_size=256,
        shuffle=False,
        num_workers=WorkerCount,
        persistent_workers=(WorkerCount > 0),
        pin_memory=False,
    )

    GlobalModel = ModelFactory().to(Device)
    BaseState = CloneStateDict(GlobalModel)

    ControllerObject = None
    if AlgorithmName == "trustfednova":
        ControllerObject = DynamicController(
            LamMin=ServerConfiguration.LamMin,
            EmaBeta=ServerConfiguration.EmaBeta,
            AParameter=ServerConfiguration.AParameter,
            BParameter=ServerConfiguration.BParameter,
            CParameter=ServerConfiguration.CParameter,
            DParameter=ServerConfiguration.DParameter,
        )
    AggregatorObject = Aggregator(AlgorithmName=AlgorithmName, ControllerObject=ControllerObject)

    RoundLogs: List[RoundLog] = []
    StartTime = time.time()

    ClientSampleCounts = [len(ClientLoader.dataset) if isinstance(ClientLoader.dataset, Subset) else len(ClientLoader.dataset) for ClientLoader in ClientLoaders]

    for RoundIndex in range(1, ServerConfiguration.Rounds + 1):
        RoundRandomGenerator = np.random.default_rng(Seed + RoundIndex)
        ChosenClients = RoundRandomGenerator.choice(ServerConfiguration.Clients, size=ServerConfiguration.Sampled, replace=False)

        TauDrawnArray = DrawTauLognormal(ServerConfiguration.Sampled, ServerConfiguration.MeanSteps, ServerConfiguration.HeteroScale, Seed + 1000 + RoundIndex)
        TauList = [int(max(1, round(TauValue))) for TauValue in TauDrawnArray]

        BaseState = CloneStateDict(GlobalModel)

        MomentumBackup = ClientConfiguration.Momentum
        if AlgorithmName in ("fednova", "trustfednova"):
            ClientConfiguration.Momentum = 0.0

        DeltaList = []
        SampleCountList = []
        for ClientIndex, TauValue in zip(ChosenClients, TauList):
            ClientLoader = ClientLoaders[ClientIndex]

            ProxMuBackup = ClientConfiguration.ProxMu
            if AlgorithmName == "fedprox":
                ProxMuValue = ProxMuBackup if ProxMuBackup > 0 else 0.01
            else:
                ProxMuValue = 0.0
            ClientConfiguration.ProxMu = ProxMuValue

            DeltaState, StepsCompleted, SampleCount, AverageLoss = LocalTrain(
                Model=GlobalModel,
                BaseState=ToDevice(BaseState, Device),
                DataLoaderObject=ClientLoader,
                Steps=TauValue,
                Device=Device,
                Config=ClientConfiguration,
                AlgorithmName=AlgorithmName,
            )
            DeltaList.append(DeltaState)
            SampleCountList.append(ClientSampleCounts[ClientIndex])

        ClientConfiguration.ProxMu = 0.0
        ClientConfiguration.Momentum = MomentumBackup

        if ServerConfiguration.EvalEvery and RoundIndex % ServerConfiguration.EvalEvery == 0:
            PreUpdateAccuracy, ValidationLoss = Evaluate(GlobalModel, TestLoader, Device)
        else:
            PreUpdateAccuracy, ValidationLoss = 0.0, 0.0

        UpdateState, LambdaValue, ChiSquareValue, CosineSimilarityValue = AggregatorObject.Aggregate(BaseState, DeltaList, TauList, SampleCountList, ValidationLoss)

        ServerLearningRateFloat = float(ServerConfiguration.ServerLearningRate)
        NewState = {Key: BaseState[Key] - ServerLearningRateFloat * UpdateState[Key] for Key in BaseState}
        GlobalModel.load_state_dict(NewState, strict=False)

        AccuracyValue, LossValue = Evaluate(GlobalModel, TestLoader, Device)
        RoundLogs.append(
            RoundLog(
                Round=RoundIndex,
                Accuracy=AccuracyValue,
                Loss=LossValue,
                LambdaValue=LambdaValue,
                ChiSquare=ChiSquareValue,
                CosineSimilarity=CosineSimilarityValue,
            )
        )

        if RoundIndex % max(1, ServerConfiguration.LogEvery) == 0:
            print(
                f"[Round {RoundIndex:3d}] acc={AccuracyValue:.4f} loss={LossValue:.4f} "
                f"lam={LambdaValue:.3f} chi2={ChiSquareValue:.4f} cos={CosineSimilarityValue:.3f}"
            )

    ElapsedTime = time.time() - StartTime

    os.makedirs(SaveDirectory, exist_ok=True)
    CsvPath = os.path.join(SaveDirectory, f"{AlgorithmName}_seed{Seed}.csv")
    with open(CsvPath, "w") as CsvFile:
        CsvFile.write("round,acc,loss,lam,chi2,cos\n")
        for LogEntry in RoundLogs:
            CsvFile.write(
                f"{LogEntry.Round},{LogEntry.Accuracy:.6f},{LogEntry.Loss:.6f},"
                f"{LogEntry.LambdaValue:.6f},{LogEntry.ChiSquare:.6f},{LogEntry.CosineSimilarity:.6f}\n"
            )
    print(f"Saved logs to {CsvPath}")

    torch.save(GlobalModel.state_dict(), os.path.join(SaveDirectory, f"{AlgorithmName}_final.pt"))

    return RoundLogs, ElapsedTime

""" The PlotResults() method is used to plot the comparitive results of the simulation run between the four algorithms in consideration: FedAvg, FedProx, fedNova and TrustFedNova"""
def PlotResults(AllLogs: Dict[str, List[RoundLog]], SaveDirectory: str):
    import matplotlib.pyplot as plt


    plt.figure()
    for AlgorithmName, Logs in AllLogs.items():
        XAxisRounds = [LogEntry.Round for LogEntry in Logs]
        YAxisAccuracy = [LogEntry.Accuracy for LogEntry in Logs]
        plt.plot(XAxisRounds, YAxisAccuracy, label=AlgorithmName)
    plt.xlabel("Round")
    plt.ylabel("Test Accuracy")
    plt.title("Accuracy vs Rounds")
    plt.legend()
    plt.grid(True, alpha=0.3)
    AccuracyPlotPath = os.path.join(SaveDirectory, "accuracy.png")
    plt.savefig(AccuracyPlotPath, bbox_inches="tight")
    plt.close()
    print(f"Saved {AccuracyPlotPath}")

    if "trustfednova" in AllLogs:
        plt.figure()
        LambdaRounds = [LogEntry.Round for LogEntry in AllLogs["trustfednova"]]
        LambdaValues = [LogEntry.LambdaValue for LogEntry in AllLogs["trustfednova"]]
        plt.plot(LambdaRounds, LambdaValues)
        plt.xlabel("Round")
        plt.ylabel("lambda (λ_t)")
        plt.title("Dynamic λ_t (TrustFedNova)")
        plt.grid(True, alpha=0.3)
        LambdaPlotPath = os.path.join(SaveDirectory, "lambda_trace.png")
        plt.savefig(LambdaPlotPath, bbox_inches="tight")
        plt.close()
        print(f"Saved {LambdaPlotPath}")

    plt.figure()
    for AlgorithmName, Logs in AllLogs.items():
        ChiRounds = [LogEntry.Round for LogEntry in Logs]
        ChiValues = [LogEntry.ChiSquare for LogEntry in Logs]
        plt.plot(ChiRounds, ChiValues, label=AlgorithmName)
    plt.xlabel("Round")
    plt.ylabel("chi2_FA proxy")
    plt.title("Objective Inconsistency Proxy (χ²_FA)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    ChiPlotPath = os.path.join(SaveDirectory, "chi2_trace.png")
    plt.savefig(ChiPlotPath, bbox_inches="tight")
    plt.close()
    print(f"Saved {ChiPlotPath}")

