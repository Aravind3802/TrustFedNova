**Dynamic-Trust FedNova: Adaptive Federated Learning Under Client Heterogeneity
**
**Introduction:
**
Federated Learning(FL) is an emerging distributed inference paradigm that tackles the need for data-privacy and security in distributed edge nodes during ML inference. FL enables collaborative model training between clients without sharing the raw data with each other. But FL faces several issues due to the heterogeneity of the edge devices, like memory bandwidth, varying computing power etc. One main issue with heterogeneity is the objective inconsistency issue, wherein, the difference the number of local updates done among each client may vary and thus in turn bias the convergence of the global model to an sub-optimal level, one classic example is FedAvg algorithm, wherein the algorithm implicitly biases the global model towards the clients that perform more number of global updates.

To address this issue of objective inconsistency, Wang J. et.al., have come up with FedNova, an algorithm that normalises the global updates given to the global model by normalising the client updates by the number of local steps, which in turn tackles the biased convergence of global optima which happens in the FedAvg algorithm. But despite the advantages, FedNova does not perform well when the data is highly non-IID, the updates get overdamped and the strong gradients do not have much effect on the global model as they should in the beginning. This means that the FedNova has high variance and FedAvg has high bias.

To get a better Bias-Variance tradeoff, a solution with a dynamic trust factor to make the algorithm swing between FedNova and FedAvg and the trust factor is dynamically controlled during run time and thus allowing us to measure the bias and variance and dynamically control which of the two algorithms are being used at a given time and thus help us get a better bias-variance tradeoff and a smoother convergence.

**Theoretical Motivation:
**
The core challenge addressed in this work is the objective inconsistency problem faced in heterogeneous FL environments, where clients with more local steps bias the global update direction, which leads to:
Biased convergence in FedAvg,
Overly-damped updates in FedNova during early stages under non-IIt data distribution,
And unstable behaviour when heterogeneity is high.
Dynamic-Trust FedNova is motivated to tackle the balance between bias-variance tradeoffs by switching between the two algorithms dynamically as FedNova suffers from high variance and low bias and FedAvg suffers from high bias and low variance, it uses a controller to favour each of the algorithm dynamically to attain a balance in the bias-variance characteristics. This will lead to:
Faster early convergence
Stable long-term behavior
And improved accuracy when compared to FedNova.

**Methodology:
**
We will discuss the methodology we have used to tackle the bias-variance tradeoff in this section. A synchronous federated earning setup in which a central server broadcasts the global model to a subset of participating clients in each communication round. Clients perform local Stochastic Gradient Descent updates on the then return the deltas to the server for aggregation. 

Setup: Let there be ‘m’ clients, each with a local dataset Di of size ni. At the global round ‘t’, a subset of ‘q’ clients is sampled. Each selected client receives the global model ‘xi’, performs the local SGD and returns the updates to the global model. Due to heterogeneous computer capabilities, clients perform varying numbers of local SGD steps ‘𝜏i’, thereby introducing imbalance into the training process.

**Algorithms in Consideration: 
**
We analyse the four most common federated learning Algorithms:
FedAvg: This algorithm aggregates updates proportionally to dataset size, it is an effective solution but can easily bias the convergence depending on the difference in local steps.
FedProx: This algorithm introduces a proximal term to tackle the client drifts caused but it drastically slows down the learning.
FedNova: This algorithm introduces the method of normalizing the updates based on the number of local steps taken, it is like an extension of the FedAvg.
Dynamic-Trust FedNova: This is the proposed solution to tackle the bias-variance tradeoff using a dynamic trust factor 𝜆t, It blends FedNova and FedAvg adapting to heterogeneity and non-IID data effects.

**Proposed Algorithm:
**
Dynamic-Trust fedNova introduces a scalar trust factor 𝜆t, which fluctuates between the range of [0,1] and modulates the contribution of FedAvg and FedNova each round. When the 𝜆t is 0, the algorithm works like FedAvg and when the 𝜆t is 1, the algorithm works as FedNova and the intermediate values provide a smoother curve for learning.

The role of the trust factor 𝜆t measure the quality of the aggregated update direction and it updates based on the cosine similarity between the FedNova and FedAvg updates, validation loss curvature and update magnitude stability, all the three factors together decide the value of 𝜆t which then acts as the controller for deciding to favour FedNova or FedAvg.

**Experimental Setup: 
**
Experiments have been conducted on the CIFAR-10 dataset using a simple, lightweight CNN suitable for distributed systems. The data set for each client has been split by using the Dirichlet Distribution with 𝛂 = 0.3, that reflects an highly non-IID data distributed as in a real-world scenario and the number of local steps for each client is sampled from a lognormal distribution, which helps in simulating the varied computing speeds of each client. The simulation was run for a total of 120 rounds.

Baselines:
FedAvg
FedProx (ɱ = 0.01)
FedNova
Dynamic-Trust FedNova


**For running a smoke test you can use:
**
python3 main.py \
  --algs fedavg fedprox fednova trustfednova \
  --rounds 2 \
  --clients 6 \
  --sampled 3 \
  --batch_size 32 \
  --lr 0.01 \
  --momentum 0.9 \
  --weight_decay 5e-4 \
  --prox_mu 0.01 \
  --mean_steps 5 \
  --hetero_scale 0.3 \
  --dirichlet_alpha 0.6 \
  --eval_every 1 \
  --server_lr 1.0 \
  --num_workers 0 \
  --log_every 1 \
  --lam_min 0.2 \
  --ema_beta 0.9 \
  --ctrl_a 8.0 \
  --ctrl_b 4.0 \
  --ctrl_c 1.0 \
  --ctrl_d -1.0 \
  --seed 7 \
  --save_dir ./runs_smoke
