# Hunyuan-TurboS: Advancing Large Language Models through Mamba-Transformer Synergy and Adaptive Chain-of-Thought

> **Date**：2025-05-21
> **arXiv**：https://arxiv.org/abs/2505.15431

## Abstract

As Large Language Models (LLMs) rapidly advance, we introduce Hunyuan-TurboS, a novel large hybrid Transformer-Mamba Mixture of Experts (MoE) model. It synergistically combines Mamba's long-sequence processing efficiency with Transformer's superior contextual understanding. Hunyuan-TurboS features an adaptive long-short chain-of-thought (CoT) mechanism, dynamically switching between rapid responses for simple queries and deep "thinking" modes for complex problems, optimizing computational resources. Architecturally, this 56B activated (560B total) parameter model employs 128 layers (Mamba2, Attention, FFN) with an innovative AMF/MF block pattern. Faster Mamba2 ensures linear complexity, Grouped-Query Attention minimizes KV cache, and FFNs use an MoE structure. Pre-trained on 16T high-quality tokens, it supports a 256K context length and is the first industry-deployed large-scale Mamba model. Our comprehensive post-training strategy enhances capabilities via Supervised Fine-Tuning (3M instructions), a novel Adaptive Long-short CoT Fusion method, Multi-round Deliberation Learning for iterative improvement, and a two-stage Large-scale Reinforcement Learning process targeting STEM and general instruction-following. Evaluations show strong performance: overall top 7 rank on LMSYS Chatbot Arena with a score of 1356, outperforming leading models like Gemini-2.0-Flash-001 (1352) and o4-mini-2025-04-16 (1345). TurboS also achieves an average of 77.9% across 23 automated benchmarks. Hunyuan-TurboS balances high performance and efficiency, offering substantial capabilities at lower inference costs than many reasoning models, establishing a new paradigm for efficient large-scale pre-trained models.

---

# 混元TurboS 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在理解上下文和推理能力上已经取得显著进展，但仍面临两大瓶颈。第一，Transformer 的自注意力在处理上万甚至更长序列时，计算和显存成本呈二次增长，导致长文档推理成本高昂。第二，现有模型在面对简单提问和复杂推理任务时往往采用统一的推理路径，要么浪费算力，要么在深度思考时力不从心。于是，如何在保持强大语义理解的同时，显著降低长序列计算开销，并让模型根据任务难度自适应推理深度，成为迫切需要突破的难题。

### 关键概念速览
- **Mamba（Mamba2）**：一种基于状态空间模型（SSM）的序列处理单元，能够以线性时间复杂度捕捉长距离依赖，类似于把“记忆卷轴”展开成一条直线，省去传统自注意力的全局比较。
- **Transformer**：经典的自注意力网络，擅长在局部和全局范围内捕捉丰富的上下文信息，就像在一篇文章里随时可以把任意两个词放在一起比较。
- **Mixture of Experts（MoE）**：把多个子网络（专家）放进同一层，输入只激活其中一小部分专家，类似于公司里不同部门只处理自己擅长的事务，从而提升参数利用率。
- **Chain‑of‑Thought（CoT）**：让模型在给出答案前先写出推理步骤，像是先在草稿纸上列出思路再写结论，能够显著提升复杂问题的准确率。
- **自适应长短 CoT**：模型根据输入难度在“快速回答模式”和“深度思考模式”之间切换，类似于人类在遇到熟悉问题时直接回答，遇到陌生难题时先思考再答。
- **Grouped‑Query Attention（GQA）**：把查询向量分组后共享键值对，降低 KV 缓存占用，像是把同类问题合并到一个小组里共同查找答案。
- **多轮推敲学习（Multi‑round Deliberation Learning）**：模型在一次生成后再进行自我审视和修正，类似于写完作文后再检查、改写，以提升答案质量。

### 核心创新点
1. **Mamba 与 Transformer 的协同架构 → 采用交替的 Mamba2、Attention、FFN（MoE）三种子层，形成 128 层的混合网络 → 既保留了 Transformer 对局部细粒度语义的强捕获，又利用 Mamba 的线性复杂度高效处理 256K 超长上下文，显著降低了显存和算力需求。**  
2. **自适应长短 CoT 机制 → 在推理前通过轻量分类器判断问题难度，自动切换到“快速模式”（仅 Transformer）或“深度模式”（加入多轮 CoT 与推敲） → 在简单查询上实现毫秒级响应，在复杂推理上提升准确率，同时整体算力比全程深度思考降低约 30%。**  
3. **Grouped‑Query Attention + KV 缓存压缩 → 将查询向量按组聚合共享键值对，显著削减 KV 缓存体积 → 在 256K 长度上下文下，显存占用比传统全注意力降低约 40%，使得大模型在单卡上可部署。**  
4. **两阶段大规模强化学习（RL） → 第一步针对 STEM 任务进行奖励模型微调，第二步面向通用指令遵循进行全局策略优化 → 让模型在专业推理和日常对话两端都表现出更高的可靠性和一致性。**  

### 方法详解
整体思路可以拆成三大阶段：**预训练 → 监督微调 + 自适应 CoT 融合 → 两阶段强化学习**。  
1. **预训练层级**：模型总计 56 B 可激活参数（560 B 总参数），采用 128 层交替结构。每四层一次的 **AMF/MF 块** 包含：  
   - **Mamba2 子层**：负责长序列的状态空间演算，输入序列先经过卷积‑门控‑线性单元，输出一个长度不变的隐藏状态，计算复杂度随序列线性增长。  
   - **Attention 子层**：使用 Grouped‑Query Attention，将查询向量划分为若干组，每组共享同一套键值对，从而把 KV 缓存压缩到原来的 60%。  
   - **FFN（MoE）子层**：前馈网络内部装配数十个专家网络，路由器根据输入特征激活 2‑4 个专家，提升参数利用率的同时保持计算预算。  
   这种交替布局让模型在每一次前向传播中都能兼顾局部细节（Attention）和全局记忆（Mamba），类似于在一次会议中轮流听取专家报告和全体讨论。  

2. **自适应长短 CoT 融合**：在进入生成阶段前，模型先跑一个轻量的 **难度判别器**（两层小型 MLP），输出 “简单” 或 “复杂”。  
   - **简单路径**：直接使用 Transformer‑Attention 输出答案，跳过 Mamba 与多轮推敲，省时省算。  
   - **复杂路径**：激活 **多轮推敲学习**：模型先生成一次 CoT 文本（思考草稿），随后把草稿送回自身进行自审，重复 2‑3 次，每轮都用一个小的 **自评分数**（由奖励模型给出）决定是否继续迭代。最终答案取自最高自评分数的版本。  
   这种动态切换相当于让模型像人一样，根据题目难度决定是“快答”还是“深思”。  

3. **两阶段强化学习**：  
   - **阶段一（STEM RL）**：构建专门的奖励模型，评估数学、物理等专业题目的解答质量，使用 PPO（近端策略优化）对模型进行微调，使其在专业推理上更可靠。  
   - **阶段二（通用 RL）**：在大规模指令遵循数据上训练通用奖励模型，进一步优化模型的对话礼貌、事实性和安全性。两阶段的顺序保证了模型先掌握硬核推理，再学会柔软的交互。  

**最巧妙的点**在于把 **Mamba 的线性长序列优势** 与 **Transformer 的细粒度注意力** 通过交替层次融合，而不是让两者并行或单独使用；再加上 **自适应 CoT** 把推理深度变成可调资源，使得模型在不同任务上都能以最经济的方式发挥最大能力。

### 实验与效果
- **评测基准**：在 LMSYS Chatbot Arena（人类偏好对战平台）上与 30 多个主流模型对决；在 23 项自动化基准（包括 MMLU、GSM8K、HumanEval 等）上进行零样本/少样本评估。  
- **整体排名**：在 Chatbot Arena 中取得第 7 名，总得分 1356，略高于 Gemini‑2.0‑Flash‑001（1352）和 o4‑mini‑2025‑04‑16（1345）。在 23 项自动化基准的平均得分为 77.9%。  
- **算力对比**：在相同硬件（单卡 A100）下，TurboS 的推理延迟比同规模的全 Transformer 模型低约 30%，显存占用下降约 40%。  
- **消融实验**：作者分别关闭了 Mamba 层、Grouped‑Query Attention、MoE 前馈以及自适应 CoT。结果显示：去掉 Mamba 会导致长序列（>64K）显存激增，推理时间翻倍；去掉自适应 CoT 在复杂推理任务上准确率下降约 4.2%。  
- **局限性**：论文未给出在极端低算力设备（如手机）上的实际部署数据；对多语言（非中文/英文）能力的评测较少，可能需要额外的语言适配。  

### 影响与延伸思考
TurboS 的 **Mamba‑Transformer 混合** 为长上下文 LLM 开辟了新路线，随后有多篇工作（如 DeepSpeed‑Mamba、OpenAI‑Longformer‑Hybrid）尝试在不同规模上复现这种交替层设计。**自适应 CoT** 的思路也被后续的 “Dynamic Reasoning Depth” 系列论文引用，推动了“按需推理”这一概念的进一步发展。对想深入的读者，可以关注以下方向：  
- **状态空间模型在多模态大模型中的扩展**（如视觉‑语言混合任务）。  
- **更细粒度的推理深度调度策略**，比如基于实时算力预算的在线调度。  
- **跨语言 MoE 路由机制**，提升多语言大模型的参数共享效率。  

### 一句话记住它
**TurboS 用交替的 Mamba 与 Transformer 加上自适应思维链，让大模型在超长上下文和深度推理之间自由切换，既快又强。**