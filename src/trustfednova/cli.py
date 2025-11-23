from __future__ import annotations
import argparse
import os
from typing import Dict, List, Tuple

from .data import LoadCifar10
from .env_utils import *  # noqa: F401,F403  (run side-effects)
from .models import SmallCifarCnn
from .client import ClientConfig
from .server import ServerConfig, RoundLog
from .train import RunExperiment, PlotResults

def ParseArgs():
    ArgumentParser = argparse.ArgumentParser(description="TrustFedNova simulator")
    ArgumentParser.add_argument(
        "--algs",
        nargs="+",
        default=["fedavg", "fedprox", "fednova", "trustfednova"],
        choices=["fedavg", "fedprox", "fednova", "trustfednova"],
        help="Algorithms to run and compare",
    )
    ArgumentParser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10"], help="Dataset")
    ArgumentParser.add_argument("--rounds", type=int, default=30)
    ArgumentParser.add_argument("--clients", type=int, default=30)
    ArgumentParser.add_argument("--sampled", type=int, default=10, help="Clients sampled per round")
    ArgumentParser.add_argument("--batch_size", type=int, default=64)
    ArgumentParser.add_argument("--lr", type=float, default=0.01)
    ArgumentParser.add_argument("--momentum", type=float, default=0.9)
    ArgumentParser.add_argument("--weight_decay", type=float, default=5e-4)
    ArgumentParser.add_argument("--prox_mu", type=float, default=0.01, help="FedProx μ (only for fedprox)")
    ArgumentParser.add_argument("--mean_steps", type=float, default=20.0, help="Mean local steps τ_i")
    ArgumentParser.add_argument("--hetero_scale", type=float, default=0.6, help="Lognormal σ for τ heterogeneity")
    ArgumentParser.add_argument("--dirichlet_alpha", type=float, default=0.5, help="Non-IID strength; smaller=more skewed")
    ArgumentParser.add_argument("--eval_every", type=int, default=1)
    ArgumentParser.add_argument("--save_dir", type=str, default="./runs")
    ArgumentParser.add_argument("--seed", type=int, default=7)
    ArgumentParser.add_argument("--num_workers", type=int, default=0)
    ArgumentParser.add_argument("--log_every", type=int, default=1)
    ArgumentParser.add_argument(
        "--server_lr",
        type=float,
        default=1.0,
        help="Server step size applied to the aggregated update u_t",
    )
    ArgumentParser.add_argument("--lam_min", type=float, default=0.2)
    ArgumentParser.add_argument("--ema_beta", type=float, default=0.9)
    ArgumentParser.add_argument("--ctrl_a", type=float, default=8.0)
    ArgumentParser.add_argument("--ctrl_b", type=float, default=4.0)
    ArgumentParser.add_argument("--ctrl_c", type=float, default=1.0)
    ArgumentParser.add_argument("--ctrl_d", type=float, default=-1.0)
    return ArgumentParser.parse_args()


def RunFromArgs():
    ParsedArguments = ParseArgs()

    if ParsedArguments.dataset == "cifar10":
        TrainDataset, TestDataset = LoadCifar10()
        ModelFactory = lambda: SmallCifarCnn(10)
    else:
        raise NotImplementedError

    ServerConfiguration = ServerConfig(
        Rounds=ParsedArguments.rounds,
        Clients=ParsedArguments.clients,
        Sampled=ParsedArguments.sampled,
        MeanSteps=ParsedArguments.mean_steps,
        HeteroScale=ParsedArguments.hetero_scale,
        DirichletAlpha=ParsedArguments.dirichlet_alpha,
        EvalEvery=ParsedArguments.eval_every,
        LamMin=ParsedArguments.lam_min,
        EmaBeta=ParsedArguments.ema_beta,
        AParameter=ParsedArguments.ctrl_a,
        BParameter=ParsedArguments.ctrl_b,
        CParameter=ParsedArguments.ctrl_c,
        DParameter=ParsedArguments.ctrl_d,
        NumWorkers=ParsedArguments.num_workers,
        LogEvery=ParsedArguments.log_every,
        ServerLearningRate=ParsedArguments.server_lr,
    )
    ClientConfiguration = ClientConfig(
        BatchSize=ParsedArguments.batch_size,
        LearningRate=ParsedArguments.lr,
        Momentum=ParsedArguments.momentum,
        WeightDecay=ParsedArguments.weight_decay,
        ProxMu=ParsedArguments.prox_mu,
    )

    os.makedirs(ParsedArguments.save_dir, exist_ok=True)

    AllLogs: Dict[str, List[RoundLog]] = {}
    SummaryList: List[Tuple[str, float, float, float]] = []
    for AlgorithmName in ParsedArguments.algs:
        Logs, ElapsedTime = RunExperiment(
            AlgorithmName=AlgorithmName,
            ModelFactory=ModelFactory,
            TrainDataset=TrainDataset,
            TestDataset=TestDataset,
            ServerConfiguration=ServerConfiguration,
            ClientConfiguration=ClientConfiguration,
            SaveDirectory=ParsedArguments.save_dir,
            Seed=ParsedArguments.seed,
        )
        AllLogs[AlgorithmName] = Logs
        BestAccuracy = max(LogEntry.Accuracy for LogEntry in Logs)
        SummaryList.append((AlgorithmName, BestAccuracy, Logs[-1].Accuracy, ElapsedTime))

    from .train import PlotResults
    PlotResults(AllLogs, ParsedArguments.save_dir)

    print("\nSummary:")
    print("ALG\tBEST_ACC\tFINAL_ACC\tTIME(s)")
    for AlgorithmName, BestAccuracy, FinalAccuracy, ElapsedTime in SummaryList:
        print(f"{AlgorithmName}\t{BestAccuracy:.4f}\t{FinalAccuracy:.4f}\t{ElapsedTime:.1f}")

