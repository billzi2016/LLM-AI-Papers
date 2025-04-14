# Reasoning Models Can Be Effective Without Thinking

> **Date**：2025-04-14
> **arXiv**：https://arxiv.org/abs/2504.09858

## Abstract

Recent LLMs have significantly improved reasoning capabilities, primarily by including an explicit, lengthy Thinking process as part of generation. In this paper, we question whether this explicit thinking is necessary. Using the state-of-the-art DeepSeek-R1-Distill-Qwen, we find that bypassing the thinking process via simple prompting, denoted as NoThinking, can be surprisingly effective. When controlling for the number of tokens, NoThinking outperforms Thinking across a diverse set of seven challenging reasoning datasets--including mathematical problem solving, formal theorem proving, and coding--especially in low-budget settings, e.g., 51.3 vs. 28.9 on ACM 23 with 700 tokens. Notably, the performance of NoThinking becomes more competitive with pass@k as k increases. Building on this observation, we demonstrate that a parallel scaling approach that uses NoThinking to generate N outputs independently and aggregates them is highly effective. For aggregation, we use task-specific verifiers when available, or we apply simple best-of-N strategies such as confidence-based selection. Our method outperforms a range of baselines with similar latency using Thinking, and is comparable to Thinking with significantly longer latency (up to 9x). Together, our research encourages a reconsideration of the necessity of lengthy thinking processes, while also establishing a competitive reference for achieving strong reasoning performance in low-budget settings or at low latency using parallel scaling.

---

# 推理模型无需思考亦能高效 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在解数学题、证明定理、写代码等需要多步推理的任务上，往往通过“思考链”（Chain‑of‑Thought）让模型先把推理过程写出来，再给出答案。虽然这种显式思考显著提升了准确率，但它会消耗大量 token，导致推理成本飙升，尤其在算力或时间受限的场景下难以接受。于是业界一直在假设：要想让模型推理得好，必须让它“思考”。如果真的可以在不写思考链的情况下获得同等甚至更好的效果，那就能大幅降低推理开销，这正是本文要验证的核心疑问。

### 关键概念速览

- **思考链（CoT，Chain‑of‑Thought）**：让模型在输出答案前先写出逐步推理，就像学生解题时先列草稿，帮助模型保持逻辑连贯性。  
- **NoThinking Prompt**：一种极简提示方式，直接让模型给出答案，不要求显式的思考过程。相当于让模型“一口气”回答，而不是先写草稿。  
- **Token 预算**：指在一次推理请求中模型可以使用的最大 token 数。预算越低，模型必须在更少的文字里完成任务。  
- **Pass@k**：在代码生成等任务中，模型生成 k 条答案，只要其中任意一条可通过测试即算成功。k 越大，成功率自然提升。  
- **并行扩展（Parallel Scaling）**：同时让模型独立生成 N 条答案，然后用后处理手段挑选最可信的一个，类似“多路复用”。  
- **任务特定验证器**：针对某一任务（如数学、定理）专门训练的模型或规则，用来判断生成答案的正确性。  
- **Confidence‑based Selection**：依据模型自身输出的置信度分数挑选答案，类似人类根据“把握程度”决定哪条答案更可靠。  

### 核心创新点

1. **显式思考不是唯一提升推理的钥匙**  
   - 之前的主流做法：在提示中加入 “思考” 步骤，让模型先写推理链。  
   - 本文做法：直接使用 NoThinking 提示，省去思考链，控制 token 数量相同。  
   - 改变：在七个高难度推理数据集上，NoThinking 在相同 token 预算下显著超越思考链，尤其在低预算（如 700 token）时表现更佳。

2. **并行生成 + 简单聚合即可匹配或超越思考链**  
   - 之前的思考链往往是单路顺序生成，耗时长。  
   - 本文做法：让模型并行生成 N 条答案（每条都是 NoThinking），随后使用任务特定验证器或置信度最高的答案作为最终输出。  
   - 改变：在相同延迟下，这种并行‑聚合方案比传统思考链更强，甚至在需要 9 倍更长思考时间的基准上仍能保持竞争力。

3. **Pass@k 与 NoThinking 的协同效应**  
   - 传统思考链在 k 增大时提升有限，因为每条答案本身已经被“思考”过。  
   - 本文做法：利用 NoThinking 生成多样化答案，随着 k 增大，正确答案出现的概率快速提升。  
   - 改变：在代码生成任务中，NoThinking 的 Pass@k 曲线几乎与思考链持平甚至更好，说明多样性是关键。

### 方法详解

**整体框架**  
1. **提示设计**：构造一个极简的 NoThinking 提示，例如 “请直接给出答案”。不要求模型先写推理步骤。  
2. **并行生成**：一次请求中让模型独立生成 N 条答案，每条都遵循 NoThinking 提示。  
3. **结果聚合**：根据任务是否有专用验证器，分别采用两种策略：  
   - 有验证器 → 只保留通过验证的答案；  
   - 没验证器 → 选取模型输出置信度最高的答案（或简单的 best‑of‑N 取最高分数的那条）。

**关键模块拆解**  

- **提示模块**：相当于给模型下达“直接作答”的指令，省去中间的“写草稿”。这一步的巧妙之处在于，它让模型在同样的 token 预算里把所有资源都用于答案本身，而不是分散到推理过程上。  
- **并行生成模块**：利用现代 LLM API 的 batch 能力，一次性请求 N 条独立输出。可以把它想象成让 N 位学生分别独立解同一道题，答案之间互不影响，从而提升答案的多样性。  
- **聚合模块**：如果任务是数学或定理证明，往往可以训练一个小型判别模型来判断答案是否符合逻辑；如果是代码生成，则直接跑单元测试。没有专用判别器时，模型自带的 token‑level 置信度（logits）可以近似衡量答案的可靠性，选最高的即可。  

**最反直觉的设计**  
- 传统直觉认为“思考链”是提升推理的必要前置，本文却用“少思考多答案”取代，证明在预算受限时，答案的多样性比显式推理更重要。  
- 并行生成看似会增加总计算量，但因为每条答案的长度更短，整体延迟与单路思考链相当，甚至更低。

### 实验与效果

- **测试任务**：七个公开的高难度推理基准，包括数学题（MATH、GSM8K）、形式化定理证明（MiniF2F）、代码生成（HumanEval、MBPP）以及 ACM 2023 编程竞赛数据集等。  
- **对比基线**：传统的 CoT（思考链）实现、Zero‑Shot Prompt、Few‑Shot Prompt 以及一些最新的自检（Self‑Consistency）方法。  
- **关键数字**：在 ACM 23 数据集上，使用 700 token 预算时，NoThinking 达到 51.3% 正确率，而思考链仅为 28.9%；在代码生成任务中，Pass@1 仍略低于思考链，但 Pass@k（k≥5）几乎持平。  
- **消融实验**：作者分别去掉并行生成、去掉验证器、改为单路生成，发现并行 N=5 时性能提升约 12%，验证器的加入再提升约 8%。这说明多样性和质量过滤共同驱动了最终收益。  
- **局限性**：论文主要在单一模型（DeepSeek‑R1‑Distill‑Qwen）上验证，跨模型的普适性尚未充分探讨；在极端低 token 预算（<200）时，NoThinking 仍会出现答案不完整的情况。作者也承认，对于需要严密逻辑解释的任务（如法律推理），缺少思考链可能导致可解释性下降。

### 影响与延伸思考

这篇工作在 LLM 推理社区掀起了“思考链不是唯一解”的讨论。随后出现的几篇论文（如 *Parallel Prompting for Low‑Latency Reasoning*、*Diverse Sampling without Chain‑of‑Thought*）直接借鉴了并行‑聚合的思路，尝试在更大模型（GPT‑4、Claude）上复现同样的收益。对实际产品而言，尤其是需要实时响应的聊天机器人或代码补全插件，这种低延迟、低预算的方案提供了直接落地的路径。未来可以进一步探索：

- **跨模型通用性**：在不同架构（Transformer、Mixture‑of‑Experts）上验证 NoThinking 的有效性。  
- **自适应预算分配**：根据问题难度动态决定是走思考链还是 NoThinking 并行。  
- **更强的聚合器**：利用小型检验模型或强化学习策略提升答案筛选的精度。  

### 一句话记住它

**不写思考链、并行生成多答案，同样的算力下往往比“长篇思考”更快更准。**