**Post-Training for Embodied Artificial Intelligence: A Comprehensive Survey**

A curated collection of papers, methods, and benchmarks on **post-training and policy improvement for embodied AI**, emphasizing **vision-language-action (VLA) models**, robot manipulation, and related embodied policies.

> **Research question:** After pretraining, how can an embodied foundation policy acquire more reliable task skills through demonstrations, preferences, rewards, real-world interaction, simulation, and combinations of these training regimes?

**Core taxonomy:** **Training Paradigm × Learning Signal**. **Focus:** robotic manipulation and VLA post-training, with pertinent work on navigation, mobile manipulation, and dexterous control.

**Literature coverage:** 112 distinct arXiv-linked papers in this curated README (plus this expansion's ACL 2026 paper), with core and supporting/boundary records distinguished. Previous coverage: 73 distinct arXiv-linked papers, including 28 source-checked paper entries from the October 2026 sweeps (two were already listed elsewhere). An additional [79-paper screening queue](literature/screening_queue_2026-10.md) is maintained separately and is **not** counted as verified post-training literature. The manuscript [taxonomy figure](https://github.com/jindongli-Ai/2026_Survey_Post-Training-for-Embodied-AI/blob/main/figs/fig-2_Taxonomy-Tree.tex) cites 81 method studies; its [bibliography](https://github.com/jindongli-Ai/2026_Survey_Post-Training-for-Embodied-AI/blob/main/references.bib) contains 134 records including foundations, benchmarks and adjacent research.

## Contents

- [1. Scope and Definitions](#1-scope-and-definitions)
- [2. Taxonomy and Survey Outline](#2-taxonomy-and-survey-outline)
- [3. Offline Post-Training](#3-offline-post-training)
- [4. Real-World Interactive Post-Training](#4-real-world-interactive-post-training)
- [5. Simulation-Based Post-Training](#5-simulation-based-post-training)
- [6. Cross-Regime Post-Training](#6-cross-regime-post-training)
- [7. Benchmarks and Evaluation](#7-benchmarks-and-evaluation)
- [8. Open Challenges](#8-open-challenges)
- [9. Related Collections and Surveys](#9-related-collections-and-surveys)
- [October 2026 Literature Expansion](#october-2026-literature-expansion--primary-source-checked)
- [10. Contributing and Citation](#10-contributing-and-citation)

## 1. Scope and Definitions

**Post-training** refers to a training stage **after an initial pretrained policy/model exists**, intended to produce a persistently improved policy for downstream embodied tasks. It can be completed before deployment or involve multiple physical deployment and retraining cycles. This includes full-parameter fine-tuning, adapter/action-head updates, reward- and preference-driven updates, and distillation into a persistently updated policy.

| Term | Working definition |
| --- | --- |
| **Offline Post-Training** | Optimizes a pretrained policy from fixed demonstrations, preference pairs, logged transitions, or offline evaluations without collecting *new policy-conditioned rollouts as part of the optimization loop*. |
| **Real-World Interactive Post-Training** | Collects feedback or policy-dependent experience in the physical world and feeds it back into persistent policy updates; may use human intervention, preferences, critics, or rewards. |
| **Simulation-Based Post-Training** | Generates policy-dependent trajectories, training states, or feedback inside a physics simulator or learned world model, then updates a persistent policy. Fixed synthetic datasets alone are not sufficient for this classification. |
| **Cross-Regime Post-Training** | Makes the transition/co-optimization across distinct experience collection or optimization regimes a primary technical contribution (e.g., offline-to-online, simulation-to-real, experience-generation-to-distillation). |
| **Demonstration Signal** | Supervision on desired actions/trajectories, including expert demonstration, corrective action, behavior cloning, and teacher distillation. |
| **Preference Signal** | Relative comparison or desirability labeling of actions/trajectories, including DPO-style methods and learned preference models. |
| **Reward / Value Signal** | Scalar or dense returns, advantages, value estimates, verifiable outcomes, or learned critics that optimize policy quality through RL-style objectives. |

**Included:** fine-tuning of pretrained VLAs and embodied policies; offline and interactive imitation; preference optimization; offline/online RL; simulation/world-model-assisted policy optimization; and genuinely cross-regime post-training.

**Excluded from the core taxonomy:** pretraining from scratch without a downstream policy-improvement stage; generic robot-learning methods without a meaningful pretrained-policy connection; purely frozen-policy inference-time guidance or search; and methods that only improve an auxiliary verifier but never update the embodied policy. These may appear as context or related work.

**Boundary with [Test-Time Methods for Embodied AI](https://github.com/jindongli-Ai/Awesome-Test-Time-Methods-for-Embodied-AI):** this repository foregrounds a **persistent training product** (an updated policy), while the companion collection foregrounds deployment-time adaptation, steering, and scaling. Online post-training and test-time training can overlap; a paper is not forced into an artificial exclusive category.

**Classification policy:** Each core method appears in one **primary category** based on its central contribution, with optional cross-references and multiple learning-signal labels. In particular, a method using a world model *only to score a fixed dataset* remains offline; a method updating a policy *using new imagined rollouts* belongs to simulation-based training. Cross-regime methods receive their own placement only when interaction between regimes is a principal contribution. Venue tags denote **confirmed venues**; otherwise **arXiv** is used conservatively.

## 2. Taxonomy and Survey Outline

The primary axis answers **where policy-improving experience comes from and how the training loop is organized**. The second axis answers **what learning signal drives the update**.

| Primary training regime | Demonstration | Preference | Reward / Value |
| --- | --- | --- | --- |
| **Offline** | Supervised fine-tuning, behavior cloning, representation-preserving adaptation | Trajectory preference alignment, direct preference optimization | Offline RL, advantage-weighted regression, offline critic- or world-model-guided optimization |
| **Real-World Interactive** | Human correction, DAgger, teleoperation-assisted policy updates | Human-assisted action/trajectory preference feedback | Physical-robot online RL, process rewards, critic-guided improvement |
| **Simulation-Based** | Simulated expert rollouts and on-policy teacher supervision | Simulated comparisons or preference labels *(emerging; no forced standalone subsection)* | Simulator-based RL, world-model-based RL and imagined-rollout optimization |

**Cross-Regime** is represented as **training pathways across these cells**, not a fourth mutually exclusive row. For example, offline demonstration learning → simulator RL → real-world refinement. The scheme is a literature-navigation device; it does not imply that each method uses only one signal.

| Main research theme | Central question | Typical methods |
| --- | --- | --- |
| **Offline Post-Training** | What can be learned from fixed, previously collected experience? | SFT/BC, DPO, offline RL. |
| **Real-World Interactive Post-Training** | How can physical execution and human/robot feedback improve a policy? | DAgger, intervention learning, online RL. |
| **Simulation-Based Post-Training** | How can low-cost, repeatable or imagined interaction supply training experience? | Physics-simulator RL, world-model rollouts, on-policy distillation. |
| **Cross-Regime Post-Training** | How can different training stages and experience sources reinforce one another? | Offline-to-online, sim-to-real co-training, iterative data generation and distillation. |

### Proposed paper outline

1. **Introduction**
2. **Background and Foundations**
   - 2.1. Embodied Foundation Models and Policies
   - 2.2. Post-Training Formulation and Scope
   - 2.3. Learning Signals and Training Paradigms
3. **Offline Post-Training**
   - 3.1. Demonstration-Based Fine-Tuning
   - 3.2. Preference-Based Optimization
   - 3.3. Offline Reinforcement Learning
4. **Real-World Interactive Post-Training**
   - 4.1. Interactive Imitation Learning
   - 4.2. Human Feedback and Preference Optimization
   - 4.3. Online Reinforcement Learning
5. **Simulation-Based Post-Training**
   - 5.1. Synthetic Demonstration and Policy Distillation
   - 5.2. Simulator-Based Reinforcement Learning
   - 5.3. World-Model-Based Policy Optimization
6. **Cross-Regime Post-Training**
   - 6.1. Offline-to-Online Policy Optimization
   - 6.2. Simulation-to-Real Co-Training
   - 6.3. Iterative Experience Recycling and Policy Refinement
7. **Benchmarks and Evaluation**
   - 7.1. Tasks, Environments, and Benchmarks
   - 7.2. Performance and Generalization
   - 7.3. Data Efficiency and Training Cost
   - 7.4. Real-World Safety and Reliability
8. **Challenges and Future Directions**
   - 8.1. Data and Feedback Efficiency
   - 8.2. Reward Reliability and Credit Assignment
   - 8.3. Generalization and Catastrophic Forgetting
   - 8.4. Sim-to-Real Transfer and Model Exploitation
   - 8.5. Safe and Continual Post-Training
9. **Conclusion**

The outline is the current **working taxonomy (v1.0)**. Its boundaries will be revised when full-text screening reveals a better grouping; new categories should not be created merely to fill every cell of the matrix.

---

## 3. Offline Post-Training

Offline methods refine a pretrained policy using a fixed data corpus. Training can use demonstration, preference, reward, and value supervision; the key criterion is **no new policy-dependent environment rollouts inside the optimization loop**.

### 3.1. Demonstration-Based Fine-Tuning

Fine-tuning from demonstration actions/trajectories, including efficient action decoders and representation-preserving adaptation.

1. 2025_arXiv_OpenVLA-OFT_Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success.
   [[arXiv]](https://arxiv.org/abs/2502.19645)  [[Project]](https://openvla-oft.github.io/)

2. 2026_arXiv_VLA-GSE_VLA-GSE: Boosting Parameter-Efficient Fine-Tuning in VLA with Generalized and Specialized Experts.
   [[arXiv]](https://arxiv.org/abs/2605.06175)

3. 2026_arXiv_Anchor-Align_Generalizable VLA Finetuning via Representation Anchoring and Language-Action Alignment.
   [[arXiv]](https://arxiv.org/abs/2607.13429)  [[Project]](https://anchoralignvla.github.io/)

### 3.2. Preference-Based Optimization

Directly optimize action or trajectory preferences collected or constructed **before** the policy-improvement step. A world-model-based preference label does not automatically imply online world-model RL.

1. 2024_arXiv_GRAPE_GRAPE: Generalizing Robot Policy via Preference Alignment.
   [[arXiv]](https://arxiv.org/abs/2411.19309)

2. 2025_arXiv_NORA-1.5_NORA-1.5: A Vision-Language-Action Model Trained using World Model- and Action-based Preference Rewards.
   [[arXiv]](https://arxiv.org/abs/2511.14659)  [[GitHub]](https://github.com/declare-lab/nora-1.5)

3. 2026_arXiv_FlowPRO_FlowPRO: Reward-Free Reinforced Fine-Tuning of Flow-Matching VLAs via Proximalized Preference Optimization.
   [[arXiv]](https://arxiv.org/abs/2606.05468)

### 3.3. Offline Reinforcement Learning

Use logged transitions, value/advantage estimates, or model-provided scores to refine a pretrained policy without new interactive rollouts.

1. 2025_arXiv_CO-RFT_CO-RFT: Efficient Fine-Tuning of Vision-Language-Action Models through Chunked Offline Reinforcement Learning.
   [[arXiv]](https://arxiv.org/abs/2508.02219)

2. 2025_arXiv_ARFM_Balancing Signal and Variance: Adaptive Offline RL Post-Training for VLA Flow Models.
   [[arXiv]](https://arxiv.org/abs/2509.04063)

3. 2026_arXiv_AtomVLA_AtomVLA: Scalable Post-Training for Robotic Manipulation via Predictive Latent World Models.
   [[arXiv]](https://arxiv.org/abs/2603.08519)

---

## 4. Real-World Interactive Post-Training

Physical robot execution generates new states, failures, interventions and/or rewards, which are used in **persistent policy updates**. This is different from evaluating a frozen policy on a real robot.

### 4.1. Interactive Imitation Learning

Improve actions through robot rollouts, expert takeovers, intervention data, and policy-dependent corrective demonstrations.

1. 2026_arXiv_SOP_SOP: A Scalable Online Post-Training System for Vision-Language-Action Models.
   [[arXiv]](https://arxiv.org/abs/2601.03044)

2. 2026_arXiv_HandITL_Hand-in-the-Loop: Improving VLA Policies for Dexterous Manipulation via Seamless Hand-Arm Intervention.
   [[arXiv]](https://arxiv.org/abs/2605.15157)

3. 2026_arXiv_LIFT_Never Too Late for Force: Accelerating VLA Post-Training with Reactive Force Injection.
   [[arXiv]](https://arxiv.org/abs/2607.14236)

4. 2026_arXiv_HELP_HELP: Human-Efficient Large-Scale Robot Post-Training with Rollout Segmentation.
   [[arXiv]](https://arxiv.org/abs/2607.09776)

**Placement note:** SOP instantiates more than one update algorithm (interactive imitation and RL). It appears here for its physical-world fleet learning loop rather than being duplicated in Section 4.3.

### 4.2. Human Feedback and Preference Optimization

Optimize the policy using relative preference or desirability information acquired from human-robot interaction.

1. 2025_NeurIPS_APO_Human-assisted Robotic Policy Refinement via Action Preference Optimization.
   [[arXiv]](https://arxiv.org/abs/2506.07127)  [[NeurIPS]](https://proceedings.neurips.cc/paper_files/paper/2025/hash/347110fb894281e5e937f6ccd998a6eb-Abstract-Conference.html)

**Coverage note:** This remains a relatively small standalone subcategory. Methods based on interventions but trained only from fixed preference pairs can instead belong to Section 3.2; the physical interaction loop and policy-update schedule determine placement.

### 4.3. Online Reinforcement Learning

Update a pretrained VLA using physical-world policy rollouts and feedback, such as task outcomes, learned process rewards, or critic/advantage estimates.

1. 2025_arXiv_VLAC_A Vision-Language-Action-Critic Model for Robotic Real-World Reinforcement Learning.
   [[arXiv]](https://arxiv.org/abs/2509.15937)

2. 2026_arXiv_EXPO-FT_EXPO-FT: Sample-Efficient Reinforcement Learning Finetuning for Vision-Language-Action Models.
   [[arXiv]](https://arxiv.org/abs/2605.25477)

3. 2026_arXiv_Real-Time-EXPO-FT_Reinforcement Learning for Real-Time Vision-Language-Action Policies.
   [[arXiv]](https://arxiv.org/abs/2609.18207)

4. 2026_arXiv_HABC_Hierarchical Advantage Weighting for Online RL Fine-Tuning of VLAs from Sparse Episode Outcomes.
   [[arXiv]](https://arxiv.org/abs/2606.17043)

---

## 5. Simulation-Based Post-Training

Generate new experience inside a **physics simulator** or a **learned world model** and use it to improve a pretrained embodied policy. A paper's *evaluation in simulation* is not, by itself, evidence of simulation-based post-training.

### 5.1. Synthetic Demonstration and Policy Distillation

Generate policy-conditioned states and use simulated teacher signals or synthetic demonstration supervision to update the student policy.

1. 2026_arXiv_VLA-OPD_VLA-OPD: Bridging Offline SFT and Online RL for Vision-Language-Action Models via On-Policy Distillation.
   [[arXiv]](https://arxiv.org/abs/2603.26666)

**Coverage note:** On-policy distillation is supported by verified work, but this branch is less populated than simulation RL. Adjacent work that only trains a *new specialist* or constructs a pretraining dataset is listed as background rather than counted as directly equivalent VLA post-training.

### 5.2. Simulator-Based Reinforcement Learning

Use interaction in an executable physics simulator for reward-guided policy optimization; studies may additionally evaluate the resulting policy on physical hardware.

1. 2024_arXiv_FLaRe_FLaRe: Achieving Masterful and Adaptive Robot Policies with Large-Scale Reinforcement Learning Fine-Tuning.
   [[arXiv]](https://arxiv.org/abs/2409.16578)  [[Project]](https://robot-flare.github.io/)

2. 2025_arXiv_RIPT-VLA_Interactive Post-Training for Vision-Language-Action Models.
   [[arXiv]](https://arxiv.org/abs/2505.17016)

3. 2025_arXiv_VLA-RL_VLA-RL: Towards Masterful and General Robotic Manipulation with Scalable Reinforcement Learning.
   [[arXiv]](https://arxiv.org/abs/2505.18719)

4. 2026_ICLR_SimpleVLA-RL_SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning.
   [[ICLR]](https://proceedings.iclr.cc/paper_files/paper/2026/hash/cbfbcb4da14235bd69b134070898ae9d-Abstract-Conference.html)

5. 2026_arXiv_Z-1_Z-1: Efficient Reinforcement Learning for Vision-Language-Action Models.
   [[arXiv]](https://arxiv.org/abs/2606.31846)

**Boundary note:** RIPT-VLA is framed as interactive post-training; it is placed here based on its simulation-centric reported optimization setting, not simply because its experiments contain simulation.

### 5.3. World-Model-Based Policy Optimization

Replace or augment interactive environment rollouts with learned imagined dynamics. The policy must be updated using newly generated simulated experience.

1. 2026_CVPR_RehearseVLA_RehearseVLA: Simulated Post-Training for VLAs with Physically-Consistent World Model.
   [[arXiv]](https://arxiv.org/abs/2509.24948)  [[CVPR]](https://openaccess.thecvf.com/content/CVPR2026/html/Xiao_RehearseVLA_Simulated_Post-Training_for_VLAs_with_Physically-Consistent_World_Model_CVPR_2026_paper.html)  [[GitHub]](https://github.com/iSEE-Laboratory/RehearseVLA)

2. 2025_arXiv_VLA-RFT_VLA-RFT: Vision-Language-Action Reinforcement Fine-tuning with Verified Rewards in World Simulators.
   [[arXiv]](https://arxiv.org/abs/2510.00406)  [[Project]](https://vla-rft.github.io/)

3. 2026_arXiv_WoVR_WoVR: World Models as Reliable Simulators for Post-Training VLA Policies with RL.
   [[arXiv]](https://arxiv.org/abs/2602.13977)  [[Project]](https://wovr-corl.github.io/)

**Version note:** The 2025 preprint *World-Env: Leveraging World Model as a Virtual Environment for VLA Post-Training* (arXiv:2509.24948) and its 2026 CVPR version **RehearseVLA** are listed as **one research lineage**, not two independent methods.

---

## 6. Cross-Regime Post-Training

Methods that primarily contribute a **coupling, transition, or iterative co-improvement** between training regimes. Simple SFT followed by RL is **not automatically** a cross-regime contribution; otherwise this chapter would absorb much of the literature.

### 6.1. Offline-to-Online Policy Optimization

Connect fixed-data initialization or offline value learning with subsequent policy-dependent interactive optimization. Human interventions may stabilize the transition.

1. 2025_arXiv_ConRFT_ConRFT: A Reinforced Fine-tuning Method for VLA Models via Consistency Policy.
   [[arXiv]](https://arxiv.org/abs/2502.05450)

2. 2025_arXiv_GR-RL_GR-RL: Going Dexterous and Precise for Long-Horizon Robotic Manipulation.
   [[arXiv]](https://arxiv.org/abs/2512.01801)

3. 2026_arXiv_IG-RFT_IG-RFT: An Interaction-Guided RL Framework for VLA Models in Long-Horizon Robotic Manipulation.
   [[arXiv]](https://arxiv.org/abs/2602.20715)

4. 2026_arXiv_LWD_Learning While Deploying: Fleet-Scale Reinforcement Learning for Generalist Robot Policies.
   [[arXiv]](https://arxiv.org/abs/2605.00416)

**Note:** The listed methods differ in whether online interaction is conducted in the physical world or with additional auxiliary environments; consult the individual paper before comparing interaction budgets.

### 6.2. Simulation-to-Real Co-Training

Explicitly coordinate virtual experience and real-robot data to improve transferable policies, e.g., real-data anchoring during simulator RL.

1. 2026_arXiv_RL-Co_Beyond Imitation: Reinforcement Learning-Based Sim-Real Co-Training for VLA Models.
   [[arXiv]](https://arxiv.org/abs/2602.12628)

### 6.3. Iterative Experience Recycling and Policy Refinement

Repeatedly discover useful trajectories or policy failures, distill new experience back into the generalist, or co-improve a policy and its data-generation/world model.

1. 2025_arXiv_PLD_Self-Improving Vision-Language-Action Models with Data Generation via Residual RL.
   [[arXiv]](https://arxiv.org/abs/2511.00091)

2. 2026_ICML_VLAW_VLAW: Iterative Co-Improvement of Vision-Language-Action Policy and World Model.
   [[arXiv]](https://arxiv.org/abs/2602.12063)  [[ICML]](https://proceedings.mlr.press/v306/guo26i.html)  [[Project]](https://sites.google.com/view/vla-w)

3. 2026_arXiv_High-DoF-VLA_Towards High-DoF Dexterous Manipulation through VLA Post-Training.
   [[arXiv]](https://arxiv.org/abs/2609.19666)

---

## 7. Benchmarks and Evaluation

### 7.1. Representative Environments

| Area | Benchmark / platform | Typical post-training question |
| --- | --- | --- |
| Language-conditioned manipulation | [LIBERO](https://libero-project.github.io/), [RLBench](https://github.com/stepjam/RLBench) | Fine-tuning success, task transfer, policy learning. |
| Robustness under shift | [LIBERO-Plus](https://sylvestf.github.io/LIBERO-plus/), [LIBERO-Pro2.0](https://arxiv.org/abs/2610.24587) | Viewpoint, embodiment, language, state, execution and environmental shifts. |
| Simulated physical manipulation | [RoboTwin](https://robotwin-platform.github.io/), [RoboCasa](https://robocasa.ai/), [ManiSkill](https://maniskill.ai/) | Simulation RL, scalable experience collection, transfer. |
| Real-to-simulation comparison | [SimplerEnv](https://simpler-env.github.io/) | Simulation fidelity and policy-behavior consistency. |
| Long-horizon manipulation | [CALVIN](https://github.com/mees/calvin) | Long-horizon execution and compound failures. |
| Common evaluation harness | [VLA Evaluation Harness](https://github.com/allenai/vla-evaluation-harness) | Reusable evaluation across multiple simulation benchmarks. |

*These platforms are evaluation resources, not post-training algorithms. Different papers evaluate under different tasks, reward sources, reset access, and simulator configurations.*

### 7.2. Evaluation Dimensions

| Dimension | Suggested measurements |
| --- | --- |
| Task performance | Success rate, return, task progress, gain over identical pretrained/SFT starting point. |
| Generalization | Unseen tasks, objects, environments, embodiments, contact conditions, and runtime disturbances. |
| Training data efficiency | Human demonstrations, corrections, preference pairs, labels, and offline trajectory counts. |
| Interaction cost | Real robot-hours, episodes, resets, interventions, and simulator/model-generated rollouts. |
| Training compute | GPU hours, tokens, optimizer steps, action generation cost, and world-model training cost. |
| Reward and verifier reliability | Correctness/calibration, reward hacking, false completion labels, and critic drift. |
| Safety and deployment | Unsafe contacts, near misses, intervention frequency, recovery success, and rollback/stop protocols. |
| Retention | Preservation of pretrained generality and previously learned tasks after specialization. |
| Reproducibility | Fixed initial policy, same environment version, reward privileges, reset protocol, trial count, and variance. |

**Fairness principle:** simulation steps, learned-world-model steps, and real robot-hours are **different resources** and must be reported separately. Similarly, a result with access to human corrective labels cannot be interpreted as comparable to purely autonomous RL without disclosing that extra feedback.

### 7.3. Benchmark and Evaluation Papers

1. 2023_NeurIPS_LIBERO_LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning.
   [[NeurIPS]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8c3c666820ea055a77726d66fc7d447f-Abstract-Datasets_and_Benchmarks.html)  [[arXiv]](https://arxiv.org/abs/2306.03310)  [[GitHub]](https://github.com/Lifelong-Robot-Learning/LIBERO)

2. 2024_CoRL_SIMPLER_Evaluating Real-World Robot Manipulation Policies in Simulation.
   [[CoRL]](https://proceedings.mlr.press/v270/li25c.html)  [[arXiv]](https://arxiv.org/abs/2405.05941)  [[GitHub]](https://github.com/simpler-env/SimplerEnv)

3. 2025_CVPR_RoboTwin_RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins.
   [[CVPR]](https://openaccess.thecvf.com/content/CVPR2025/html/Mu_RoboTwin_Dual-Arm_Robot_Benchmark_with_Generative_Digital_Twins_CVPR_2025_paper.html)  [[arXiv]](https://arxiv.org/abs/2504.13059)  [[GitHub]](https://github.com/RoboTwin-Platform/RoboTwin)

4. 2026_CVPR_LIBERO-Plus_LIBERO-Plus: A Progressive Robustness Benchmark for Visual-Language-Action Models.
   [[CVPR]](https://openaccess.thecvf.com/content/CVPR2026/html/Fei_LIBERO-Plus_A_Progressive_Robustness_Benchmark_for_Visual-Language-Action_Models_CVPR_2026_paper.html)  [[arXiv]](https://arxiv.org/abs/2510.13626)  [[GitHub]](https://github.com/sylvestf/LIBERO-plus)

5. 2026_arXiv_LIBERO-Pro2.0_LIBERO-Pro2.0: A Benchmark for Evaluating the Robustness of Embodied Foundation Models.
   [[arXiv]](https://arxiv.org/abs/2610.24587)

### 7.4. Evaluation Protocol Priorities

- Report the **initial policy and all post-training stages**, including new data collected between updates.
- Separate **offline human data**, **online real-robot interaction**, **physics-simulation interaction**, and **learned-world-model rollouts**.
- Distinguish **the environment used for training** from the environment used for testing.
- Compare **performance versus interaction/human/compute budget**, not just final task success.
- Audit whether robustness gains are accompanied by **catastrophic forgetting** or a loss of generalist capabilities.

## 8. Open Challenges

1. **Feedback efficiency:** How to obtain high-value demonstrations, corrections, preferences, and rewards without scaling robot-hours and human labor linearly?
2. **Reward credibility and credit assignment:** How to avoid reward hacking, unreliable learned critics, sparse outcomes, and poorly attributed transition rewards?
3. **Sim-to-real reliability:** How to detect when imagined trajectories are physically implausible or exploit world-model errors?
4. **Post-training generalization:** How to specialize a generalist policy without destroying broad visual, semantic, and motor capabilities?
5. **Safe online learning:** How to improve on physical robots under irreversible failures, contact constraints, operator takeovers, and limited resets?
6. **Hybrid training design:** When should a system use SFT, preference updates, offline RL, simulator RL, and physical-robot RL, and in what sequence?
7. **Cross-paper comparability:** How to normalize interaction budgets, demonstration privileges, simulator quality, model size, and robustness evaluation?

### 8.1. Related Work on Verifiers and Data Generation

These items inform feedback reliability or data generation. They are **supporting literature**, not automatically core post-training-policy algorithms.

1. 2026_arXiv_Verifier-Survey_No Free Checker: A Survey of Verifiers for Robot Policies.
   [[arXiv]](https://arxiv.org/abs/2609.09250)  [[GitHub]](https://github.com/ZJUSCL/Awesome-Robot-Verifier)

2. 2025_arXiv_Beyond-Human-Demonstrations_Beyond Human Demonstrations: Diffusion-Based Reinforcement Learning to Generate Data for VLA Training.
   [[arXiv]](https://arxiv.org/abs/2509.19752)

3. 2025_arXiv_RPD_Refined Policy Distillation: From VLA Generalists to RL Experts.
   [[arXiv]](https://arxiv.org/abs/2503.05833)

**Scope note:** A student specialist distilled *from* a VLA generalist is a relevant neighboring mechanism, but does not necessarily update the generalist VLA itself.

---

## 9. Related Collections and Surveys

### 9.1. Community Collections

These are **curated repositories, not research papers**.

- [Awesome-VLA-Post-Training](https://github.com/AoqunJin/Awesome-VLA-Post-Training) — dedicated VLA post-training literature collection.
- [Awesome-VLA-RL](https://github.com/OpenHelix-Team/Awesome-VLA-RL) — RL-oriented VLA literature and taxonomies.
- [Awesome-Test-Time-Methods-for-Embodied-AI](https://github.com/jindongli-Ai/Awesome-Test-Time-Methods-for-Embodied-AI) — companion collection focused on deployment-time learning, steering, and scaling.

### 9.2. Related Survey Papers

1. 2025_arXiv_VLA-Post-Training-Survey_Parallels Between VLA Model Post-Training and Human Motor Learning: Progress, Challenges, and Trends.
   [[arXiv]](https://arxiv.org/abs/2506.20966)  [[GitHub]](https://github.com/AoqunJin/Awesome-VLA-Post-Training)

2. 2025_arXiv_VLA-Manipulation-Survey_Survey of Vision-Language-Action Models for Embodied Manipulation.
   [[arXiv]](https://arxiv.org/abs/2508.15201)

3. 2026_TPAMI_Efficient-VLA-Survey_A Survey on Efficient Vision-Language-Action Models.
   [[arXiv]](https://arxiv.org/abs/2510.24795)  [[GitHub]](https://github.com/YuZhaoshu/Efficient-VLAs-Survey)

4. 2026_arXiv_Driving-Post-Training-Survey_Post-Training in End-to-End Autonomous Driving.
   [[arXiv]](https://arxiv.org/abs/2607.08072)

5. 2026_arXiv_Verifier-Survey_No Free Checker: A Survey of Verifiers for Robot Policies.
   [[arXiv]](https://arxiv.org/abs/2609.09250)

The current collection emphasizes the **training regime × learning signal** view across supervised, preference-based, and reward-based policy improvement, while the surveys above take complementary ability-centric, architecture-centric, efficiency-centric, driving-centric, or verifier-centric perspectives.

## October 2026 Literature Expansion — Primary-Source Checked

This extension cross-checks candidate titles against their original arXiv abstract pages. **Core** entries are grouped by the dominant persistent policy-update mechanism. Fixed synthetic demonstrations are classified as offline data; interactive simulation requires policy-conditioned virtual rollouts. The full screening record and BibTeX keys are [available in the manuscript repository](https://github.com/jindongli-Ai/2026_Survey_Post-Training-for-Embodied-AI/blob/main/literature/verified_additions_2026-10.json).

### Offline — Demonstration Fine-Tuning

- **CLIP-RT** (2025, RSS): [Learning Language-Conditioned Robotic Policies from Natural Language Supervision](https://arxiv.org/abs/2411.00508). Offline imitation fine-tuning using natural-language motion supervision.
- **DemoGen** (2025): [DemoGen: Synthetic Demonstration Generation for Data-Efficient Visuomotor Policy Learning](https://arxiv.org/abs/2502.16932). Fixed synthetic demonstration augmentation, not interactive simulator RL.
- **InstructVLA** (2025): [InstructVLA: Vision-Language-Action Instruction Tuning from Understanding to Manipulation](https://arxiv.org/abs/2507.17520). Instruction tuning combines multimodal reasoning and action learning.
- **RoboTron-Platter** (2025): [Boosting Robotic Manipulation Generalization with Minimal Costly Data](https://arxiv.org/abs/2503.19516). Stage-divided data mixing for VLA generalization.
- **THAW-VLA** (2026): [Think Like a World Model, Act Like a VLA: Distilling World-Model Representations into Compact Robot Policies](https://arxiv.org/abs/2609.24682). Fixed-frame offline world-model feature alignment.
- **VOTE** (2025): [VOTE: Vision-Language-Action Optimization with Trajectory Ensemble Voting](https://arxiv.org/abs/2507.05116). Fine-tuning for faster action tokens; inference voting separate.
- **PALM** (2026, CVPR): [PALM: Progress-Aware Policy Learning via Affordance Reasoning for Long-Horizon Robotic Manipulation](https://arxiv.org/abs/2601.07060). Training with affordance and within-subtask progress cues.

### Offline — Preference Optimization

- **VTLA** (2025): [VTLA: Vision-Tactile-Language-Action Model with Preference Learning for Insertion Manipulation](https://arxiv.org/abs/2505.09577). Direct preference optimization for tactile VLA insertion.

### Offline — Offline RL

- **ReinboT** (2025): [ReinboT: Amplifying Robot Visual-Language Manipulation with Reinforcement Learning](https://arxiv.org/abs/2505.07395). Offline return-aware policy optimization from mixed-quality data.

### Real-World — Interactive Imitation

- **RoboCopilot** (2025): [RoboCopilot: Human-in-the-loop Interactive Imitation Learning for Robot Manipulation](https://arxiv.org/abs/2503.07771). Human-autonomy switching for interactive imitation learning.
- **VLA-Expert** (2025): [VLA Model-Expert Collaboration for Bi-directional Manipulation Learning](https://arxiv.org/abs/2503.04163). Expert-assisted collaboration and collected experience for refinement.

### Real-World — Online RL

- **Res-HIL** (2026): [Res-HIL: Human-Guided Residual Reinforcement Learning for Sample-Efficient Dexterous Manipulation](https://arxiv.org/abs/2609.30023). Human-guided online residual RL on frozen imitation priors.

### Simulation — Policy Distillation

- **RPD** (2026, IROS): [Refined Policy Distillation: From VLA Generalists to RL Experts](https://arxiv.org/abs/2503.05833). On-policy simulated RL guided by a pretrained VLA teacher. *(Already listed earlier in this collection.)*

### Simulation — Simulator RL

- **RFTF** (2025): [RFTF: Reinforcement Fine-tuning for Embodied Agents with Temporal Feedback](https://arxiv.org/abs/2505.19767). Temporal value-model rewards for embodied RL fine-tuning.
- **TGRPO** (2025): [Fine-tuning Vision-Language-Action Model via Trajectory-wise Group Relative Policy Optimization](https://arxiv.org/abs/2506.08440). Trajectory-grouped RL with dense task feedback on LIBERO.
- **Action-Chunked-PPO** (2025): [VLA Model Post-Training via Action-Chunked PPO and Self Behavior Cloning](https://arxiv.org/abs/2509.25718). PPO and dynamically collected self demonstrations on MetaWorld.
- **RL4VLA** (2025, NeurIPS): [What Can RL Bring to VLA Generalization? An Empirical Study](https://arxiv.org/abs/2505.19789). PPO fine-tuning recipe and generalization comparison.

### Simulation — World-Model Rollouts

- **WISE** (2026): [WISE: World-model-guided Imagination Scheduling for Efficient Post-training of Vision-Language-Action Models](https://arxiv.org/abs/2609.03681). Bounded and scheduled imagined rollouts to refine policy outputs.

### Cross-Regime — Offline-to-Online

- **Q-Chunking** (2025, NeurIPS): [Reinforcement Learning with Action Chunking](https://arxiv.org/abs/2507.07969). Offline-to-online temporal-difference RL in chunked action spaces.

### Supporting and Boundary Literature

- **AMF** (2024): [Active Fine-Tuning of Multi-Task Policies](https://arxiv.org/abs/2410.05026). Active expert-demonstration selection; no physical loop assumed.
- **ADC** (2025): [Adversarial Data Collection: Human-Collaborative Perturbations for Efficient and Robust Robotic Imitation Learning](https://arxiv.org/abs/2503.11646). Human-in-the-loop data construction without proof of policy-loop interaction.
- **Robot-R1** (2025): [Robot-R1: Reinforcement Learning for Enhanced Embodied Reasoning in Robotics](https://arxiv.org/abs/2506.00070). Embodied reasoning RL, not direct persistent VLA motor-policy training.

The linked papers above were inspected at the abstract level. Full-text mechanism and experiment checks remain necessary before submission. This section does not treat every paper in a broader VLA collection as a post-training policy-update method.

---

### Further 2026 Methods — Real-World RL and Virtual Experience

Additional source-screened papers emphasizing real-robot online RL, simulated policy improvement, and structured rewards. World-Env was already listed elsewhere in this repository and is cross-referenced here for its mechanism.

- **BEE** (2026): [BEE: Intervention-Adaptive Real-World Reinforcement Learning with Vision-Language-Action Models](https://arxiv.org/abs/2609.27450). Residual RL under human-correction consistency constraints.
- **StructRL** (2026): [StructRL: Online Structured Reinforcement Learning for Long-Horizon Vision-Language-Action Tasks](https://arxiv.org/abs/2609.36352). Simulated online RL with prerequisite-aware subtask rewards.
- **World-Env** (2025): [World-Env: Leveraging World Model as a Virtual Environment for VLA Post-Training](https://arxiv.org/abs/2509.24948). Video-world simulator with VLM progress and termination feedback.
- **RL-Token** (2026): [RL Token: Bootstrapping Online RL with Vision-Language-Action Models](https://arxiv.org/abs/2604.23073). Real-robot actor-critic refinement using pretrained VLA features.
- **VLA-Precision** (2026): [VLA-Precision: Asymmetric Co-Bootstrapping for Efficient Real-World Online RL of Vision-Language-Action Models](https://arxiv.org/abs/2609.04355). Intervention and preference-calibrated online RL for precise manipulation.
- **Uncertainty-Gated-RL** (2026): [Uncertainty-Gated Exploration Noise Suppresses Task Collapse in Online RL Fine-Tuning of a Flow-Matching Vision-Language-Action Policy](https://arxiv.org/abs/2609.28838). Exploration noise regulation; no gain over behavior cloning in tested budget.

**Related test-time boundary (not counted as core post-training):**

- **T2VLA** (2026): [Trust Your Instincts: Confidence-Driven Test-Time RL for Vision-Language-Action Models](https://arxiv.org/abs/2606.29892). Confidence-guided test-time policy update; cross-over with test-time methods.



## October 30, 2026 Literature Expansion — Source-Checked at Abstract Level

This increment adds **40 distinct paper records**: 16 core post-training candidates, 22 foundational/supporting studies and 2 boundary studies. Titles and links have been checked against primary arXiv landing pages or official ACL proceedings. **These are abstract-level checks, not yet full-paper mechanism validation**; supporting/boundary papers are not counted as core algorithms.

### Core Post-Training Candidates

- **OBEYED-VLA (2025)**: [Clutter-Robust Vision-Language-Action Models through Object-Centric and Geometry Grounding](https://arxiv.org/abs/2512.22519). *Offline / Demonstration.* Pretrained VLA fine-tuned on grounded single-object demonstrations.
- **ChatVLA (2025)**: [ChatVLA: Unified Multimodal Understanding and Robot Control with Vision-Language-Action Model](https://arxiv.org/abs/2502.14420). *Offline / Demonstration.* Phased alignment of multimodal understanding and robot control.
- **ChatVLA-2 (2025)**: [ChatVLA-2: Vision-Language-Action Model with Open-World Embodied Reasoning from Pretrained Knowledge](https://arxiv.org/abs/2505.21906). *Offline / Demonstration.* Two-stage reasoning and action training.
- **InSpire (2025)**: [InSpire: Vision-Language-Action Models with Intrinsic Spatial Reasoning](https://arxiv.org/abs/2505.13888). *Offline / Demonstration.* Spatial direction and action supervision.
- **VLAS (2025)**: [VLAS: Vision-Language-Action Model With Speech Instructions For Customized Robot Manipulation](https://arxiv.org/abs/2502.13508). *Offline / Demonstration.* Speech-instruction and robot-action tuning.
- **ForceVLA (2025)**: [ForceVLA: Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation](https://arxiv.org/abs/2505.22159). *Offline / Demonstration.* Force-conditioned action-model adaptation.
- **3D-CAVLA (2025)**: [3D CAVLA: Leveraging Depth and 3D Context to Generalize Vision Language Action Models for Unseen Tasks](https://arxiv.org/abs/2505.05800). *Offline / Demonstration.* Depth-augmented VLA fine-tuning.
- **CronusVLA (2025)**: [CronusVLA: Towards Efficient and Robust Manipulation via Multi-Frame Vision-Language-Action Modeling](https://arxiv.org/abs/2506.19816). *Offline / Demonstration.* Multi-frame post-training after single-frame pretraining.
- **SpatialVLA (2025)**: [SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Model](https://arxiv.org/abs/2501.15830). *Offline / Demonstration.* Adaptive action-grid fine-tuning.
- **HAMSTER (2025)**: [HAMSTER: Hierarchical Action Models For Open-World Robot Manipulation](https://arxiv.org/abs/2502.05485). *Offline / Demonstration.* VLM trajectory prediction fine-tuning for hierarchical robot control.
- **DexVLA (2025)**: [DexVLA: Vision-Language Model with Plug-In Diffusion Expert for General Robot Control](https://arxiv.org/abs/2502.05855). *Offline / Demonstration.* Embodiment alignment followed by task post-training.
- **Phoenix (2025)**: [Phoenix: A Motion-based Self-Reflection Framework for Fine-grained Robotic Action Correction](https://arxiv.org/abs/2504.14588). *Real-World / Continual.* Motion-based correction with interaction-driven lifelong learning.
- **Xiaomi-Robotics-1 (2026)**: [Xiaomi-Robotics-1: Scaling Vision-Language-Action Models with over 100K Hours of Real-World Trajectories](https://arxiv.org/abs/2607.15330). *Offline / Demonstration.* Explicit post-training for embodiment and instruction alignment.
- **LLaVA-VLA (2026)**: [Rethinking the Practicality of Vision-language-action Model: A Comprehensive Benchmark and An Improved Baseline](https://arxiv.org/abs/2602.22663). *Offline / Demonstration.* Two-stage post-training and fine-tuning on CEBench.
- **PAIR-VLA (2026)**: [What to Ignore, What to React: Visually Robust RL Fine-Tuning of VLA Models](https://arxiv.org/abs/2605.13105). *Simulation / Reward.* PPO robustness fine-tuning with visual invariance and sensitivity.
- **Spatial Traces (2025)**: [Spatial Traces: Enhancing VLA Models with Spatial-Temporal Understanding](https://arxiv.org/abs/2508.09032). *Offline / Demonstration.* Spatial-temporal adaptation with limited training data.

### Foundation and Supporting Literature

- **ManualVLA (2025)**: [ManualVLA: A Unified VLA Model for Chain-of-Thought Manual Generation and Robotic Manipulation](https://arxiv.org/abs/2512.02013). *Planning / Synthetic.* Synthetic manuals for planning-expert training.
- **OE-VLA (2025)**: [Unveiling the Potential of Vision-Language-Action Models with Open-Ended Multimodal Instructions](https://arxiv.org/abs/2505.11214). *Multimodal instructions.* Open-ended instruction formats; persistent policy update not explicit.
- **DexGraspVLA (2025)**: [DexGraspVLA: A Vision-Language-Action Framework Towards General Dexterous Grasping](https://arxiv.org/abs/2502.20900). *Hierarchical control.* Learns diffusion controller under pretrained VLM.
- **OmniVTLA (2025)**: [OmniVTLA: Vision-Tactile-Language-Action Models with Semantic-Aligned Tactile Sensing](https://arxiv.org/abs/2508.08706). *Tactile representation.* Tactile encoder training and multimodal action conditioning.
- **LLARVA (2024)**: [LLARVA: Vision-Action Instruction Tuning Enhances Robot Learning](https://arxiv.org/abs/2406.11815). *Instruction training.* Visual-trace instruction model training originally released in 2024.
- **ThinkAct (2025)**: [ThinkAct: Vision-Language-Action Reasoning via Reinforced Visual Latent Planning](https://arxiv.org/abs/2507.16815). *Embodied reasoning RL.* Trains high-level visual plans rather than direct motor-policy RL.
- **OTTER (2025)**: [OTTER: A Vision-Language-Action Model with Text-Aware Visual Feature Extraction](https://arxiv.org/abs/2503.03734). *Frozen visual encoder.* Policy learning with frozen pretrained VLM encoders.
- **Evo-0 (2025)**: [Evo-0: Vision-Language-Action Model with Implicit Spatial Understanding](https://arxiv.org/abs/2507.00416). *Geometry module.* Plug-and-play spatial features; post-training unconfirmed.
- **FLARE-Latent (2025)**: [FLARE: Robot Learning with Implicit World Modeling](https://arxiv.org/abs/2505.15659). *Latent world modeling.* Future-latent alignment during imitation policy learning.
- **DreamVLA (2025)**: [DreamVLA: A Vision-Language-Action Model Dreamed with Comprehensive World Knowledge](https://arxiv.org/abs/2507.04447). *World-knowledge model.* Joint future world prediction and action training.
- **GigaBrain-0 (2025)**: [GigaBrain-0: A World Model-Powered Vision-Language-Action Model](https://arxiv.org/abs/2510.19430). *Synthetic foundation data.* Foundation training with world-model-generated trajectories.
- **GeoVLA (2025)**: [GeoVLA: Empowering 3D Representations in Vision-Language-Action Models](https://arxiv.org/abs/2508.09071). *3D representation.* Depth and 3D action-expert architecture.
- **RoboFlamingo-Plus (2025)**: [RoboFlamingo-Plus: Fusion of Depth and RGB Perception with Vision-Language Models for Enhanced Robotic Manipulation](https://arxiv.org/abs/2503.19510). *Visual architecture.* RGB/depth feature adaptation; policy-update stage needs review.
- **OneTwoVLA (2025)**: [OneTwoVLA: A Unified Vision-Language-Action Model with Adaptive Reasoning](https://arxiv.org/abs/2505.11917). *Reasoning co-training.* Synthesized reasoning and robot-data co-training.
- **World Action Agent (2026)**: [World Action Agent: Harnessing VLMs for Robot Manipulation via World Action Rehearsal](https://arxiv.org/abs/2609.29964). *Action rehearsal.* Visual rehearsal plus trace-based smaller-VLM fine-tuning.
- **GR00T N1 (2025)**: [GR00T N1: An Open Foundation Model for Generalist Humanoid Robots](https://arxiv.org/abs/2503.14734). *Foundation policy.* Generalist VLA trained on mixed robot and human data.
- **π0.5 (2025)**: [$π_{0.5}$: a Vision-Language-Action Model with Open-World Generalization](https://arxiv.org/abs/2504.16054). *Foundation policy.* Heterogeneous co-training for open-world generalization.
- **SmolVLA (2025)**: [SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics](https://arxiv.org/abs/2506.01844). *Efficient foundation VLA.* Compact model with inexpensive training and asynchronous inference.
- **FAST (2025)**: [FAST: Efficient Action Tokenization for Vision-Language-Action Models](https://arxiv.org/abs/2501.09747). *Action representation.* DCT-based action tokenization for scalable VLA training.
- **π0.7 (2026)**: [$π_{0.7}$: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities](https://arxiv.org/abs/2604.15483). *Foundation policy.* Steerable generalist robotic foundation model.
- **BRIDGE (2025)**: [Fine-tuning Behavioral Cloning Policies with Preference-Based Reinforcement Learning](https://arxiv.org/abs/2509.26605). *Generic cross-regime RL.* ICLR 2026: offline BC and online preference RL in general control.
- **ORBIT (2026)**: [On-policy Reinforcement Fine-tuning with Offline reward for Multi-step Embodied Planning](https://aclanthology.org/2026.acl-long.1822/). *Embodied planning.* ACL 2026: fine-tunes planning model, not direct VLA motor policy.

### Boundary Literature (Not Core Post-Training)

- **VLA-Touch (2025)**: [VLA-Touch: Enhancing Vision-Language-Action Models with Dual-Level Tactile Feedback](https://arxiv.org/abs/2507.17294). *Frozen-policy enhancement.* Explicitly avoids fine-tuning base VLA.
- **Rolling-WAM (2026)**: [Rolling-WAM: World Action Models with Rolling Imagination](https://arxiv.org/abs/2609.30247). *Test-time inference.* Rolling denoising for fast planning; no persistent policy update.

## 10. Contributing and Citation

Corrections, verified publication updates, new post-training methods, benchmark results, and taxonomy suggestions are welcome through [GitHub issues](https://github.com/jindongli-Ai/Awesome-Post-Training-for-Embodied-AI/issues) or pull requests. Please include the **exact paper title**, **stable paper URL**, **initial release year**, **verified venue (if published)**, **starting pretrained policy**, **training environment**, **learning signal**, **persistent policy update**, and **reason for inclusion**.

**Suggested classification metadata:** primary regime (offline / real-world interactive / simulation-based / cross-regime); supervision signals (demonstration / preference / reward-value); data source; real-robot interaction budget; simulator/world-model dependence; and benchmark(s).

**Citation:** Please cite the original papers for their methods and link to this repository when referring to the collection. A BibTeX entry for the survey will be provided when a public preprint is available.

---
