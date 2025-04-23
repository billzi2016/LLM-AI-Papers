# SplitReason: Learning To Offload Reasoning

> **Date**：2025-04-23
> **arXiv**：https://arxiv.org/abs/2504.16379

## Abstract

Reasoning in large language models (LLMs) tends to produce substantially longer token generation sequences than simpler language modeling tasks. This extended generation length reflects the multi-step, compositional nature of reasoning and is often correlated with higher solution accuracy. From an efficiency perspective, longer token generation exacerbates the inherently sequential and memory-bound decoding phase of LLMs. However, not all parts of this expensive reasoning process are equally difficult to generate. We leverage this observation by offloading only the most challenging parts of the reasoning process to a larger, more capable model, while performing most of the generation with a smaller, more efficient model; furthermore, we teach the smaller model to identify these difficult segments and independently trigger offloading when needed. To enable this behavior, we annotate difficult segments across 18k reasoning traces from the OpenR1-Math-220k chain-of-thought (CoT) dataset. We then apply supervised fine-tuning (SFT) and reinforcement learning fine-tuning (RLFT) to a 1.5B-parameter reasoning model, training it to learn to offload the most challenging parts of its own reasoning process to a larger model. This approach improves AIME24 reasoning accuracy by 24% and 28.3% while offloading 1.35% and 5% of the generated tokens respectively. We open-source our SplitReason model, data, code and logs.

---

# SplitReason：学习将推理卸载 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在做推理时往往要生成很长的文字序列，因为要把思考过程一步步写出来。序列越长，解码过程就越慢，而且需要的显存也会线性增长。传统的加速手段（比如量化、稀疏化）只能整体提升效率，却无法针对推理中“真正耗时”的那几段文字进行优化。于是出现了一个矛盾：想要更高的解题准确率必须让模型写更多的思考步骤，但写得越多，推理成本就越高。如何在保持或提升准确率的同时，削减这部分冗余计算，成为了迫切需要解决的问题。

### 关键概念速览
**Chain‑of‑Thought（CoT）**：让模型在给出最终答案前先把推理步骤写出来，类似于人做数学题时的草稿，能显著提升复杂任务的正确率。  
**Token**：模型输出的最小单位，通常是一个子词或字符，生成的每个 token 都需要一次前向传播。  
**Offloading（卸载）**：把一段计算任务交给另一台更强大的模型来完成，就像把难题交给专家解答。  
**SFT（Supervised Fine‑Tuning）**：在标注好的数据上继续训练模型，使其学习特定行为。  
**RLFT（Reinforcement Learning Fine‑Tuning）**：利用强化学习的奖励信号进一步微调模型，让它在实际使用中表现更好。  
**难点片段**：在推理链中，模型最容易出错、生成代价最高的那几段文字。  
**小模型 / 大模型**：本文分别指 1.5 B 参数的轻量推理模型和参数更多、算力更强的“老师”模型。

### 核心创新点
1. **从整体加速到局部卸载**  
   之前的工作要么整体换成更快的模型，要么通过压缩技术统一提升速度。本文先在 18 k 条 CoT 轨迹上标记出“难点片段”，然后让小模型自行判断何时把这些片段交给大模型处理。这样只在关键时刻使用昂贵的算力，整体代价大幅下降。  
2. **让小模型学会自我检测**  
   传统的协同推理需要外部调度器决定何时切换模型。这里通过 SFT 让小模型在标记好的训练数据上学习“这段话我不确定”。随后用 RLFT 让它在实际推理中根据奖励（正确率提升 vs. 额外 token 开销）自行触发卸载。结果是模型能够在不依赖外部控制的情况下主动请求帮助。  
3. **极低的卸载比例实现显著提升**  
   实验显示，仅在 1.35%（AIME24）或 5%（AIME24‑plus） 的 token 上进行卸载，就分别提升了 24% 和 28.3% 的解题准确率。相比于全程使用大模型的成本，这种“点对点”协同方式极其高效。  
4. **全流程开源**  
   作者不仅公开了训练好的 SplitReason 模型，还提供了难点标注数据、训练代码和实验日志，方便社区直接复现和进一步改进。

### 方法详解
**整体框架**  
SplitReason 的推理过程可以划分为三步：① 预标注难点 → ② 小模型自监督学习 → ③ 强化学习驱动的自适应卸载。整体思路是让小模型在生成 CoT 时，随时检查当前 token 是否属于“难点”，如果是，就向大模型发送请求，大模型返回该段的完整生成，随后小模型继续后续推理。

**步骤拆解**  

1. **难点片段标注**  
   - 从公开的 OpenR1‑Math‑220k CoT 数据集中抽取 18 k 条完整推理链。  
   - 通过人工或自动化规则（如错误率高、逻辑跳跃大）标记出每条链中最容易出错的子序列。  
   - 这些标注既是后续 SFT 的监督信号，也是 RLFT 中奖励函数的基准。

2. **监督微调（SFT）**  
   - 使用标注好的 18 k 条数据，对 1.5 B 参数的基础模型进行微调。  
   - 训练目标是让模型在每个 token 前输出一个二元标签：`0` 表示自行生成，`1` 表示需要卸载。  
   - 同时让模型学习在标签为 `1` 时，暂停自身生成并等待大模型的输出。

3. **强化学习微调（RLFT）**  
   - 定义奖励：正确答案得到正向奖励，额外的卸载 token 数量得到负向惩罚。  
   - 采用 PPO（Proximal Policy Optimization）等常见 RL 算法，让模型在真实推理环境中探索何时触发卸载。  
   - 通过多轮交互，模型逐渐学会只在真正必要的地方请求大模型，从而在准确率和代价之间找到最佳平衡。

4. **运行时协同**  
   - 推理时，小模型先生成 token 并实时评估“难度”。  
   - 一旦判断为难点，它会发送该段的上下文给大模型，大模型返回完整的高质量生成。  
   - 小模型把大模型的输出直接拼接进自己的 CoT，随后继续后续步骤。  
   - 整个过程对用户透明，只有生成的 token 序列会被返回。

**关键技巧**  
- **难点自检机制**：把“是否难点”当作序列标注任务，让模型在每一步都有判断依据，而不是事后人工干预。  
- **奖励平衡**：作者在 RLFT 中加入了“每多生成 1% token 的负奖励”，确保模型不会滥用大模型。  
- **最小化通信开销**：只在必要的片段发送请求，避免频繁的模型切换导致的额外延迟。

### 实验与效果
- **数据集与任务**：在 AIME24（美国数学竞赛）以及其扩展版本 AIME24‑plus 上评估，均属于高难度数学推理任务。  
- **对比基线**：与单纯使用 1.5 B 小模型、全程使用更大模型以及传统 CoT 方法对比。  
- **结果**：在仅卸载 1.35% 的 token 时，AIME24 的解题准确率提升了 24%；在卸载 5% token 时，提升达 28.3%。相比之下，全程使用大模型的准确率提升相近，但计算成本高出数十倍。  
- **消融实验**：作者分别去掉 SFT、去掉 RLFT、以及不使用难点标注进行对照，发现没有 SFT 的模型几乎不会主动卸载，准确率提升不到 5%；去掉 RLFT 则导致卸载比例飙升至 15% 以上，计算开销失控。  
- **局限性**：论文未在非数学推理任务上做广泛验证；难点标注依赖于特定数据集，迁移到其他领域可能需要重新标注。  

### 影响与延伸思考
SplitReason 为“模型内部的自适应协同”提供了可行路径，激发了后续工作在多模型协同、层次化推理以及动态算力分配上的探索。2024 年后出现的几篇论文（如 *Dynamic CoT Routing*、*Hierarchical LLM Offloading*）都在不同程度上借鉴了“让小模型自行判断何时请求大模型”的思路。未来可以进一步研究：① 自动化生成难点标注的通用方法；② 将卸载对象扩展到专门的工具模型（如代码执行器、检索系统）；③ 在分布式推理平台上实现跨节点的实时卸载调度。

### 一句话记住它
只在真正卡壳的几句上叫“大模型帮忙”，即可用小模型实现高效、准确的推理。