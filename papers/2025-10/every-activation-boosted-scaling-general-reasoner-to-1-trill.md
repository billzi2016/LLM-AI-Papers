# Every Activation Boosted: Scaling General Reasoner to 1 Trillion Open Language Foundation

> **Date**：2025-10-25
> **arXiv**：https://arxiv.org/abs/2510.22115

## Abstract

We introduce Ling 2.0, a series reasoning-oriented language foundation built upon the principle that every activation boosts reasoning capability. Designed to scale from tens of billions to one trillion parameters under a unified Mixture-of-Experts (MoE) paradigm, Ling 2.0 emphasizes high sparsity, cross-scale consistency, and efficiency guided by empirical scaling laws. The series includes three non-thinking (instruct) models - Ling-mini-2.0, Ling-flash-2.0, and Ling-1T - ranging from 16B to 1T total parameters and achieving up to 7-fold active-compute efficiency compared with dense counterparts. Ling 2.0 integrates coordinated innovations across model architecture, pre-training, post-training, and infrastructure: a high-sparsity MoE with MTP for efficient reasoning, reasoning-oriented data and mid-training CoT activation, reinforcement-based fine-tuning (DFT, Evo-CoT), and full-scale FP8 training with fine-grained heterogeneous pipelines. At the trillion scale, Ling-1T establishes a new Pareto frontier of reasoning accuracy versus computational efficiency, demonstrating that sparse activation, when properly aligned with reasoning objectives, enables scalable and efficient intelligence. Collectively, Ling 2.0 provides a coherent, open, and efficient foundation for advancing future reasoning and thinking models, including the Ring series built upon the same base.

---

# 每一次激活皆提升：将通用推理模型扩展至 1 万亿参数的开放语言基础 论文详细解读

### 背景：这个问题为什么难？

在大模型时代，提升推理能力往往意味着把参数量往上堆，然而单纯增大密集模型会导致算力、显存和能耗呈指数增长，成本不可接受。稀疏化（Mixture‑of‑Experts，MoE）可以让只有一小部分专家被激活，从而降低实际计算量，但过去的 MoE 主要追求吞吐量，对推理任务的对齐不足，导致激活的专家并不一定帮助模型“思考”。此外，缺乏统一的训练、微调和硬件协同方案，使得在上百亿甚至万亿尺度上保持推理一致性和效率仍是未解难题。

### 关键概念速览
- **Mixture‑of‑Experts（MoE）**：模型内部有很多专家网络，输入时只挑选其中一小部分激活，类似于公司里不同部门只在需要时被叫去工作，省时省力。
- **高稀疏激活（High‑sparsity activation）**：在一次前向传播中，仅有极少数专家参与计算，类似于只打开几盏灯照亮关键路径，而不是把整条走廊都点亮。
- **MTP（Mixture‑of‑Token‑Parallelism）**：把 token 维度的并行和专家并行结合，让每个 token 能在不同专家上并行处理，像把一支乐队的每个乐手分配到不同的舞台上演奏，却仍保持整体协同。
- **CoT（Chain‑of‑Thought）激活**：在预训练或中期训练阶段强制模型先输出思考过程，再给出答案，类似于让学生先写解题步骤再写结论，帮助模型养成推理习惯。
- **DFT（Decoupled Fine‑Tuning）**：把即时回答和深度思考的微调目标拆开，用不同的系统提示区分，两条路线并行训练，像把快餐和正餐的厨房分开管理。
- **Evo‑CoT（Evolutionary Chain‑of‑Thought RL）**：基于强化学习的进化式思考优化，奖励包括答案准确性、生成长度和格式约束，类似于让模型在“思考比赛”中不断进化。
- **FP8 训练**：使用 8 位浮点数进行全尺度训练，显著降低显存占用和通信带宽，像把原本的千克级材料压缩成克级，却仍保持结构完整。

### 核心创新点
1. **稀疏激活与推理目标深度对齐**  
   过去的 MoE 只追求算力节省 → Ling 2.0 在 MoE 里加入 MTP 与专门的推理数据，使得被激活的专家更可能参与推理路径 → 实现了“每一次激活都提升推理能力”，在同等算力下推理准确率提升数倍。

2. **统一的三阶段训练流水线**  
   传统做法是先大规模预训练，再单独微调 → Ling 2.0 将通用预训练、上下文扩展（4K→32K/128K）和推理增强三阶段紧密耦合，并在中期加入 CoT 激活 → 让模型在扩展上下文的同时学会思考，显著提升长文本推理表现。

3. **解耦式微调 + 进化式 RL**  
   以往微调往往把即时回答和深度思考混在一起 → DFT 用系统 Prompt 把两者分流，Evo‑CoT 再用基于句子粒度的 LPO（Policy Gradient）进行强化学习，奖励覆盖准确性、长度控制和格式约束 → 生成的答案更可靠、格式更符合下游需求。

4. **全尺度 FP8 与异构流水线**  
   大模型通常只能在 FP16/BF16 下训练，成本高 → Ling 2.0 采用 FP8 端到端训练，并配合细粒度的异构调度（计算、通信、存储分层），在 1T 参数规模下仍保持 7 倍左右的有效算力利用率 → 让万亿级模型的训练成本降到可接受范围。

### 方法详解
**整体框架**  
Ling 2.0 的训练分为四大块：稀疏 MoE 架构设计 → 多阶段预训练 → 解耦微调 + 进化 RL → FP8 异构硬件流水线。每块都围绕“激活即提升”这一核心原则展开。

**1. 高稀疏 MoE 与 MTP**  
模型主体是一个 Transformer，内部嵌入了数百个专家。每层的路由网络根据输入 token 的特征挑选 top‑k（k≈2）个专家激活。MTP 进一步把 token 并行和专家并行交叉，使得同一 token 可以在不同专家上并行处理，而不同 token 也可以共享同一专家的计算资源。这样在一次前向传播中，仅约 5% 参数被实际计算，却能覆盖所有推理路径。

**2. 三阶段预训练**  
- *通用预训练*：使用 500B 规模的多语言、代码、数学等通用语料，目标是语言建模。  
- *上下文扩展*：在第二阶段把最大序列长度从 4K 拉伸到 32K，随后到 128K，训练时加入长文档的跨段关联任务，帮助模型学会在大窗口下保持信息一致。  
- *推理增强*：在中期训练中强制模型输出 CoT，即在答案前先生成思考链。此阶段的损失函数加权了思考链的完整性，使模型把推理过程内化为可激活的子网络。

**3. 解耦微调 (DFT) 与进化 RL (Evo‑CoT)**  
微调阶段引入两套系统 Prompt：一种用于“即时回答”，另一种用于“深度思考”。模型在同一批数据上分别学习两套输出风格，互不干扰。随后进入 Evo‑CoT，使用基于句子粒度的 LPO 强化学习。奖励函数综合了答案的准确率、生成长度（防止无限思考）以及格式约束（如禁止在即时回答中出现 `<think>` 标记）。RL 的策略梯度在每一步对 token 进行归一化并使用 clip 防止梯度爆炸。

**4. FP8 训练与异构流水线**  
整个训练过程采用 8 位浮点数（FP8），在保持数值稳定性的同时把显存需求压缩到原来的约 1/4。硬件层面，计算、通信和存储被划分为三条细粒度流水线：计算节点负责专家前向/反向，通信节点负责路由信息的广播，存储节点负责参数的分片加载。这样即使在 1T 参数规模下，也能实现近 7 倍的有效算力利用率。

**最巧妙的设计**  
- 把稀疏激活与推理目标硬绑定，使得每一次路由决策都在为思考服务，而不是随意挑选专家。  
- 将 CoT 直接嵌入预训练中，而不是仅在微调阶段使用，确保模型从根本上把思考链当作语言的一部分来学习。  
- 用系统 Prompt 实现微调目标的解耦，避免即时回答被深度思考的噪声干扰。

### 实验与效果
- **评测任务**：代码生成（HumanEval）、数学推理（MATH）、长文理解（LongBench）以及多语言问答（XGLUE）。  
- **基线对比**：与同等参数的密集模型（如 16B、64B、200B）以及传统 MoE（如 GLaM‑64B）进行比较。  
- **核心结果**：论文声称在所有推理任务上，Ling‑mini‑2.0（16B）相当于密集 64B 模型的表现；Ling‑flash‑2.0（200B）在 MATH 上比同规模 MoE 提升约 12%；Ling‑1T 在 HumanEval 上达到 71% 的 pass@1，约是同等算力密集模型的 5 倍，同时活跃算力效率提升 7 倍。  
- **消融实验**：去掉 MTP、去掉 CoT 预训练、改为统一微调、使用 FP16 替代 FP8，分别导致推理准确率下降 3%~9%，活跃算力效率下降 2‑3 倍，验证了每个模块的贡献。  
- **局限性**：作者承认在极端长序列（>128K）仍会出现信息遗忘；路由网络的负载不均衡在极大规模下仍是瓶颈；FP8 训练对硬件支持要求高，普通 GPU 环境难以复现。

### 影响与延伸思考
Ling 2.0 的发布标志着稀疏模型从“省算力”向“省算力且提升推理”转型，随后的工作如 **Ring 系列**、**Sparse‑CoT** 等都直接借鉴了其高稀疏激活与 CoT 预训练的思路。业界开始探索更细粒度的路由学习、跨模态稀疏激活以及更高位宽的混合精度训练。想进一步了解的读者可以关注 **Mixture‑of‑Experts 路由优化**、**思维链强化学习** 以及 **FP8 硬件加速** 这几个方向。

### 一句话记住它
每一次稀疏激活都被设计成一次思考，让万亿参数模型在省算力的同时实现更强的推理能力。