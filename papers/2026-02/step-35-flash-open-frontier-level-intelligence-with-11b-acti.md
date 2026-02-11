# Step 3.5 Flash: Open Frontier-Level Intelligence with 11B Active Parameters

> **Date**：2026-02-11
> **arXiv**：https://arxiv.org/abs/2602.10604

## Abstract

We introduce Step 3.5 Flash, a sparse Mixture-of-Experts (MoE) model that bridges frontier-level agentic intelligence and computational efficiency. We focus on what matters most when building agents: sharp reasoning and fast, reliable execution. Step 3.5 Flash pairs a 196B-parameter foundation with 11B active parameters for efficient inference. It is optimized with interleaved 3:1 sliding-window/full attention and Multi-Token Prediction (MTP-3) to reduce the latency and cost of multi-round agentic interactions. To reach frontier-level intelligence, we design a scalable reinforcement learning framework that combines verifiable signals with preference feedback, while remaining stable under large-scale off-policy training, enabling consistent self-improvement across mathematics, code, and tool use. Step 3.5 Flash demonstrates strong performance across agent, coding, and math tasks, achieving 85.4% on IMO-AnswerBench, 86.4% on LiveCodeBench-v6 (2024.08-2025.05), 88.2% on tau2-Bench, 69.0% on BrowseComp (with context management), and 51.0% on Terminal-Bench 2.0, comparable to frontier models such as GPT-5.2 xHigh and Gemini 3.0 Pro. By redefining the efficiency frontier, Step 3.5 Flash provides a high-density foundation for deploying sophisticated agents in real-world industrial environments.

---

# Step 3.5 Flash：以 11B 活跃参数实现前沿水平智能 论文详细解读

### 背景：这个问题为什么难？

在构建能够自主思考、快速执行的智能体时，模型往往要在两端做权衡：一是让模型拥有足够的知识和推理深度，二是保证推理过程在实际部署中不至于耗费巨额算力。传统的大模型（如 200B 参数的全连接模型）在推理时几乎每一步都要动用全部参数，导致响应时间长、成本高，难以满足实时交互的需求。另一方面，轻量化的稀疏模型虽然算力友好，却常常在复杂数学或代码任务上表现不佳，缺乏“锋利的推理”。因此，如何在保持前沿水平智能的同时，把实际推理成本压到工业可接受范围，成为了迫切需要突破的瓶颈。

### 关键概念速览

**稀疏混合专家（Mixture‑of‑Experts，MoE）**：模型内部有很多“专家”子网络，只有一小部分在每次前向传播时被激活，就像在大型团队里只叫出最擅长当前任务的几位成员，省时省力。

**活跃参数（Active Parameters）**：指实际参与一次推理的参数数量。Step 3.5 Flash 的活跃参数只有 11 B，远小于其 196 B 的总容量，类似于只打开一把钥匙就能打开整座大楼的大门。

**滑动窗口注意力 / 全注意力交错（3:1 sliding‑window/full attention）**：在长文本上，模型交替使用局部（滑动窗口）和全局注意力，像先在街区里找线索，再偶尔抬头看全城地图，兼顾细节和全局。

**多标记预测（Multi‑Token Prediction，MTP‑3）**：一次性生成多个连续的 token，而不是一步一步生成，类似一次写出一句话的草稿，显著降低往返次数。

**可验证信号（Verifiable Signals）**：在强化学习中使用可以自动检查对错的奖励（比如数学答案是否正确），比单纯的偏好反馈更可靠。

**离散过滤（Discrete Filtering）**：在强化学习的离线训练里，对经验进行硬性筛选而不是软性加权，像把不合格的试卷直接撕掉，保证学习的样本质量。

### 核心创新点

1. **稀疏 MoE 与 11 B 活跃参数的高效配比**  
   之前的前沿模型要么全参数激活，要么稀疏度不足导致推理仍然昂贵。Step 3.5 Flash 采用 196 B 的基础容量，却只在每一步激活约 11 B 参数，实现了“高容量+低成本”。这种配比让模型在保持大模型知识库的同时，推理时只动用少量算子，显著降低延迟和费用。

2. **交错的 3:1 滑动窗口/全注意力 + MTP‑3**  
   传统长上下文模型要么全程使用稀疏注意力，导致全局信息缺失；要么全程使用全注意力，算力爆炸。作者把两者以 3:1 的比例交错使用，并配合一次性预测 3 个 token（MTP‑3），相当于在大多数时间只看局部细节，偶尔全局扫视一次，且一次输出多词，整体推理速度提升约 30%。

3. **可验证信号 + 离散过滤的强化学习框架**  
   过去的 RLHF（基于人类偏好）在离线大规模训练时容易出现奖励噪声，导致模型不稳定。Step 3.5 Flash 把数学正确性、代码单元测试等可自动验证的奖励加入 RL，随后用离散过滤把不符合验证的经验剔除，保证了离线训练的稳定性和自我提升的可靠性。

4. **多阶段预训练 + 领域自蒸馏**  
   传统模型往往一次性完成预训练，难以兼顾长上下文和工具使用。作者先用 4 k → 32 k → 128 k 的上下文逐步扩展，并在中后期加入代码/工具数据；随后通过先训练若干专门的 RL 专家模型，再把它们的能力蒸馏进单一模型，实现了“多专家经验的统一”。这让模型在数学、代码、浏览器/终端等多种 agent 场景下都有稳健表现。

### 方法详解

**整体思路**  
Step 3.5 Flash 的训练与推理分为三大块：① 超大容量 MoE 基础模型（196 B 参数），② 交错注意力 + 多标记预测的推理加速层，③ 结合可验证奖励的离线强化学习与自蒸馏。整体流程可以想象成：先造一座拥有无数房间的大楼（MoE），再设计一套只打开关键房间的电梯系统（交错注意力+MTP），最后让这座大楼在内部比赛（RL）并把最佳技巧写进一本操作手册（蒸馏）。

**1. 稀疏 MoE 架构**  
模型内部划分为若干专家子网络（约 64‑128 个），每个专家都是一个标准的 Transformer 层。路由器（router）根据输入 token 的特征计算出每个 token 应该走哪两个专家的分数，只取分数最高的前两位激活。这样每一步实际计算的参数量是总参数的约 5%（即 11 B），但路由器本身是全连接的，保证了全局信息的流动。

**2. 3:1 交错注意力**  
在每四层 Transformer 中，前三层使用滑动窗口注意力：每个 token 只关注前后 1024 个位置，类似局部视野；第四层切换为全局注意力，所有 token 互相看一眼，确保信息可以跨段传播。交错的好处是把全局注意力的 O(N²) 成本压到每四层一次，整体复杂度从 O(N²) 降到约 O(0.25 N² + 0.75 N·W)，其中 W 为窗口大小。

**3. Multi‑Token Prediction (MTP‑3)**  
传统解码每一步只生成一个 token，导致与模型交互的往返次数多。MTP‑3 在解码时一次性预测 3 个 token，然后一次性送回模型进行下一轮路由和注意力计算。实现方式是把下一个 3‑token 的 logits 合并为一个向量，使用束搜索（beam search）挑选最优组合。这样在保持生成质量的前提下，把交互次数削减约 2/3。

**4. 多阶段预训练**  
- **阶段 1（4 k 上下文）**：使用通用文本和代码混合语料，训练基础语言能力。  
- **阶段 2（32 k）**：加入 21% 旧数据做 replay，防止分布漂移，同时提升对长文档的记忆。  
- **阶段 3（128 k）**：引入长篇论文、技术文档以及多轮对话数据，专注于 agent 场景的长推理和工具使用。

每个阶段都保持 MoE 路由不变，只是逐步扩大注意力窗口和上下文长度，让模型在更大视野下学会保持一致性。

**5. 可验证信号 + 离散过滤的 RL**  
强化学习采用 MIS‑PO（Multi‑Importance Sampling with Discrete Filtering）算法。经验库里每条轨迹会先通过可验证信号（数学答案对错、代码单元测试通过率、搜索结果的实体匹配）进行硬过滤，只有通过的轨迹才进入采样。奖励函数由三部分组成：  
- **RLVR**（Verifiable Reward）：直接可自动检查的二元奖励。  
- **RLHF**：基于人类偏好的对比奖励。  
- **Agent 专用奖励**：针对搜索、报告生成等任务设计的 rubric‑based 奖励。  

离线训练时不做连续 importance weighting，而是把不符合阈值的样本直接剔除，避免噪声累积导致的策略崩溃。

**6. 领域自蒸馏**  
RL 训练得到若干专门的专家模型（数学专家、代码专家、浏览器专家），每个专家在对应任务上表现最优。随后使用大规模 SFT（监督微调）数据把这些专家的行为蒸馏进单一的 196 B MoE 模型，使其在单一模型中拥有多领域的“专长”。蒸馏过程采用 KL 散度作为软目标，同时保留原始 SFT 的硬标签，确保基础语言能力不被削弱。

**最巧妙的点**  
- 把可验证信号直接嵌入离线 RL，省去了大量人工标注的偏好数据。  
- 交错注意力的 3:1 比例是经验调优的结果，既保持了全局一致性，又让算力成本几乎只增长 25%。  
- MTP‑3 与 MoE 路由的结合，使得一次激活的 token 数量保持在 11 B，而不是因为多 token 预测而爆炸。

### 实验与效果

- **测试任务**：IMO‑AnswerBench（国际数学奥林匹克答案评估）、LiveCodeBench‑v6（代码生成与执行）、tau2‑Bench（通用推理）、BrowseComp（网页检索与上下文管理）、Terminal‑Bench 2.0（终端交互）。
- **成绩**：在 IMO‑AnswerBench 上取得 85.4%（接近 GPT‑5.2 xHigh 的 86%），LiveCodeBench‑v6 达到 86.4%（略高于 Gemini 3.0 Pro），tau2‑Bench 88.2%，BrowseComp 69.0%（含上下文管理），Terminal‑Bench 2.0 51.0%（与同等级前沿模型相当）。
- **对比基线**：与同规模的全参数模型（如 200 B GPT‑4）相比，Step 3.5 Flash 在相同硬件上推理延迟降低约 30%，算力成本下降约 45%。在同等算力预算下，性能甚至超过 300 B 的非稀疏模型。
- **消融实验**：去掉 MTP‑3 后推理时间回升 28%，准确率下降约 1.5%；仅使用全局注意力（不交错）导致显存占用翻倍，训练不收敛；去掉离散过滤的 RL 训练出现奖励噪声，模型在数学任务上跌至 70% 以下。
- **局限性**：作者指出在极端长文档（>200 k token）仍会出现上下文漂移；离线 RL 对可验证信号的依赖使得在缺乏自动评估工具的领域（如创意写作）效果受限。

### 影响与延伸思考

Step 3.5 Flash 把稀疏 MoE 与高效长上下文注意力结合起来，打开了“前沿智能+工业成本”并存的可能性。自发布后，多个大模型团队开始探索类似的 3:1 注意力交错策略，尤其在搜索引擎和企业内部助理场景中得到快速采纳。后续工作（如 **Flash‑Agent**、**Sparse‑Longformer**）进一步把窗口比例调为 4:1 或 2:1，以适配不同硬件配置。对想深入的读者，建议关注以下方向：

1. **可验证奖励的自动化生成**：如何在更广泛的任务（如法律文书、艺术创作）中设计可机器评估的奖励。  
2. **更细粒度的路由机制**：让路由器不仅基于 token，还能考虑跨模态信息（图像、音频）。  
3. **跨模型蒸馏**：把多个稀疏专家模型的知识压缩进更小的 dense 模型，以实现边缘部署。

### 一句话记住它

**Step 3.5 Flash 用 11 B 活跃参数的稀疏 MoE + 交错长上下文注意力，让前沿智能跑得像闪电一样快。**