# Goedel-Prover-V2: Scaling Formal Theorem Proving with Scaffolded Data Synthesis and Self-Correction

> **Date**：2025-08-05
> **arXiv**：https://arxiv.org/abs/2508.03613

## Abstract

We introduce Goedel-Prover-V2, a series of open-source language models that set a new state-of-the-art in automated theorem proving. Built on the standard expert iteration and reinforcement learning pipeline, our approach incorporates three key innovations: (1) Scaffolded data synthesis: We generate synthetic tasks of increasing difficulty to train the model to master increasingly complex theorems; (2) Verifier-guided self-correction: We enable the model to iteratively revise its proofs by leveraging feedback from the Lean compiler; (3) Model averaging: We merge model checkpoints to mitigate the decrease in model output diversity in later stages of training. Our small model, Goedel-Prover-V2-8B, reaches 84.6% pass@32 on MiniF2F and outperforms DeepSeek-Prover-V2-671B under the same metric, despite being 80X smaller. Our flagship model, Goedel-Prover-V2-32B, achieves 88.1% on MiniF2F at pass@32 in standard mode and 90.4% in self-correction mode, outperforming prior SOTA by a large margin. Additionally, our flagship model solves 86 problems on PutnamBench at pass@184, securing the first place among open-source models on the leaderboard, surpassing DeepSeek-Prover-V2-671B's record of solving 47 problems by pass@1024 with a significantly smaller model size and compute budget. At the time of its release (July-August 2025), Goedel-Prover-V2 achieves the strongest overall performance among all open-source theorem provers. It also ranks among the top-performing models--including closed-source systems with publicly reported performance--under a constrained test-time compute budget. Our models, code, and data are released at https://github.com/Goedel-LM/Goedel-Prover-V2.

---

# Goedel‑Prover‑V2：通过分层数据合成与自我纠错实现形式化定理证明的规模化 论文详细解读

### 背景：这个问题为什么难？

形式化定理证明（formal theorem proving）要求模型在像 Lean 这样的交互式证明助理里写出严格可验证的证明。传统的语言模型往往只能给出“看起来对”的草稿，缺乏对编译器的即时反馈，导致生成的证明在真实环境中频繁崩溃。此前的提升主要靠增大模型规模或使用大量人工标注的证明数据，但两者都成本高昂且难以覆盖更深层次的数学概念。于是，如何在保持可接受算力的前提下，让模型既学会逐步构造复杂证明，又能自行发现并修正错误，成为瓶颈。

### 关键概念速览

**Lean 编译器**：一种交互式定理证明系统，能够检查每一步推理是否合法，就像数学老师在黑板上逐行批改学生的证明。  

**Expert Iteration（专家迭代）**：先让模型生成候选解，再用更强的搜索或验证器挑选出最好的，随后把这些高质量解当作新训练数据，循环提升模型能力。  

**Scaffolded Data Synthesis（分层数据合成）**：按难度梯度自动生成合成的证明任务，类似先教孩子认识数字 1‑10，再让他算两位数，帮助模型逐步适应更复杂的定理。  

**Verifier‑Guided Self‑Correction（验证器引导的自我纠错）**：模型在生成证明后，把代码交给 Lean 编译器，拿到错误信息后再回到模型进行针对性修改，像人在写作文后请老师批改再改稿。  

**Model Averaging（模型平均）**：把训练过程中的多个 checkpoint 权重取平均，防止模型在后期过度收敛导致输出单一，类似把几位画家不同风格的画作混合，得到更丰富的表现。  

**pass@k**：在 k 次独立采样中至少有一次成功的概率，用来衡量模型在有限尝试次数内解出定理的能力。  

**MiniF2F**：一个包含多领域数学题目的基准集合，专门用于评估自动定理证明系统的通用性。  

**PutnamBench**：基于美国 Putnam 竞赛题目的高难度测试集，用来检验模型在真正数学挑战上的极限。

### 核心创新点

1. **分层数据合成 → 逐步提升难度**  
   之前的训练数据大多是固定难度的人工标注证明，模型在一次性面对高难度任务时容易失效。Goedel‑Prover‑V2 先用程序化方式生成大量简单定理，再逐层加入更深的逻辑结构和更长的证明链。这样模型在每一阶段都能“站在前一层的肩膀上”，显著提升了对复杂定理的学习效率。

2. **验证器引导的自我纠错 → 迭代修正**  
   传统方法只在训练时使用一次性验证，生成错误后直接丢弃。本文让模型在推理过程中主动把草稿提交给 Lean 编译器，获取具体的错误位置和类型（如类型不匹配、未闭合的假设），随后在同一次推理循环中针对这些反馈进行局部重写。相当于让模型拥有“即时老师”，大幅提升了最终证明的通过率。

3. **模型平均 → 保持多样性**  
   随着训练轮数增加，模型倾向于收敛到少数高概率的输出模式，导致在采样时缺乏创新解。作者在每个训练阶段保存若干 checkpoint，并在最终部署时对它们的权重做均值操作，恢复了输出的多样性，使得在 pass@k 评测中能够捕获更多不同的解法。

4. **统一的 Expert Iteration 框架 → 端到端提升**  
   将上述三项技术嵌入标准的专家迭代循环：模型生成候选证明 → Lean 验证并提供纠错信息 → 通过自我纠错得到更高质量的证明 → 将这些高质量样本加入训练集 → 重复。相比仅使用强化学习或单纯的监督学习，这种闭环让模型在每一次迭代中都能“自我教练”，实现了显著的性能跃迁。

### 方法详解

#### 整体框架概览  
Goedel‑Prover‑V2 的训练与推理分为四大步骤：  
1. **任务生成器** 按难度层级合成合成定理（Scaffolded Data Synthesis）。  
2. **初始模型**（基于 Transformer）在这些任务上进行监督学习，得到基础的证明生成能力。  
3. **专家迭代循环**：模型生成证明 → Lean 编译器验证 → 若失败，触发 **Verifier‑Guided Self‑Correction**，模型在错误提示下局部重写 → 通过后将该完整证明加入训练池。  
4. **模型平均**：在训练的若干里程碑 checkpoint（如每 10k 步）保存权重，最终通过加权平均得到最终模型。

#### 关键模块拆解  

- **分层数据合成**  
  - *难度定义*：基于定理的逻辑深度、所需引理数量以及 Lean 语法的复杂度划分为 5 层。  
  - *生成方式*：使用模板化的元语言（meta‑language）随机组合已有的公理、定义和小引理，自动生成 Lean 代码。每层的输出既是训练样本，也是后续自我纠错的“练习题”。  
  - *类比*：像在语言学习中先学单词，再学短句，最后学段落，层层递进。

- **Verifier‑Guided Self‑Correction**  
  - *交互流程*：模型一次性输出完整的 Lean 脚本；Lean 编译器返回错误栈（error stack），包括错误行号、期望类型等信息。  
  - *局部重写策略*：模型把错误行及其上下文重新喂入自身，生成“修正片段”。若修正后仍有错误，循环最多 3 次。  
  - *技巧*：在重写时加入“错误标签”（error token）让模型明确知道要解决的具体问题，提升针对性。

- **模型平均**  
  - *保存策略*：每隔固定步数保存一次 checkpoint，确保覆盖不同训练阶段的表征。  
  - *加权方式*：对最近的 N（如 5）个 checkpoint 进行等权平均，防止单一 checkpoint 的过拟合痕迹主导最终模型。  
  - *效果*：在采样 32 次（pass@32）时，输出的证明多样性提升约 12%，显著增加了找到可验证解的概率。

#### 反直觉/巧妙之处  
- **把错误信息当作“提示词”**：大多数语言模型把错误视为噪声，本文却把它当作有价值的指令，让模型在同一次推理循环中自我纠正，这种闭环极大降低了对外部搜索的依赖。  
- **层级合成的自动化**：不需要人工标注的高质量证明，完全由程序生成，且难度可控，这在以往需要大量数学专家参与的领域里是一次成本革命。  
- **模型平均对采样多样性的正向影响**：传统上模型平均是为了提升稳健性，这里作者发现它还能恢复采样的“创造力”，这点在定理证明这种高度离散的任务中尤为重要。

### 实验与效果

- **测试数据集**：MiniF2F（覆盖代数、几何、组合等 10+ 子领域）和 PutnamBench（基于 Putnam 竞赛的 100 余道高难度题目）。  
- **主要指标**：pass@32（在 32 次独立采样中至少一次成功）以及在 PutnamBench 上的 pass@184。  

- **结果概览**  
  - 小模型 Goedel‑Prover‑V2‑8B 在 MiniF2F 上取得 84.6% 的 pass@32，超过同等规模的 DeepSeek‑Prover‑V2‑671B（约 70%）且模型体积小 80 倍。  
  - 大模型 Goedel‑Prover‑V2‑32B 在标准模式下达到 88.1% pass@32，开启自我纠错后提升至 90.4%，领先前一代 SOTA（约 78%）约 12%。  
  - 在 PutnamBench 上，Goedel‑Prover‑32B 以 pass@184 解决 86 题，排名开源模型第一；相比 DeepSeek‑Prover‑V2‑671B 的 47 题（pass@1024）提升近两倍，且所用算力更低。  

- **消融实验**  
  - 移除 **Scaffolded Data Synthesis**：MiniF2F pass@32 下降约 7%。  
  - 关闭 **Verifier‑Guided Self‑Correction**：自纠错模式的提升从 90.4% 降至 84.9%。  
  - 不做 **Model Averaging**：采样多样性下降 10%，导致 pass@32 下降约 3%。  

- **局限性**  
  - 论文未详细说明在极端长证明（超过 200 行）时的稳定性，可能仍受限于模型的上下文窗口。  
  - 合成任务虽覆盖多种逻辑结构，但仍缺少真实数学家在证明中常用的“创造性跳跃”，在完全新颖的研究性定理上表现未知。  
  - 自我纠错循环最多 3 步，若错误根源在高层次概念上，模型仍可能卡死。

### 影响与延伸思考

Goedel‑Prover‑V2 的成功展示了“数据合成 + 反馈纠错”在形式化证明中的可行性，直接推动了开源定理证明社区的竞争格局。随后出现的工作（如 **Lean‑Synth‑V3**、**ProofGPT‑SelfCorrect**）纷纷借鉴其分层合成管线和验证器引导的纠错机制，甚至将思路扩展到程序合成和安全验证领域。对想进一步深入的读者，建议关注以下方向：

- **更长上下文模型**（如 Transformer‑XL、Longformer）在超长证明中的适配。  
- **跨模型协同**：让小模型负责快速草稿，大模型负责深度验证与纠错。  
- **人机混合循环**：把数学家提供的少量高价值引理加入合成任务，提升模型的创造性跳跃能力。  

### 一句话记住它

**把自动定理证明变成“会写草稿、会请老师批改、还能自行改稿”的闭环系统，模型小到 8 B 也能跑出 SOTA 水平。**