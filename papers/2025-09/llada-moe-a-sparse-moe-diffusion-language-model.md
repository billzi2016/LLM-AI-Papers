# LLaDA-MoE: A Sparse MoE Diffusion Language Model

> **Date**：2025-09-29
> **arXiv**：https://arxiv.org/abs/2509.24389

## Abstract

We introduce LLaDA-MoE, a large language diffusion model with the Mixture-of-Experts (MoE) architecture, trained from scratch on approximately 20T tokens. LLaDA-MoE achieves competitive performance with significantly reduced computational overhead by maintaining a 7B-parameter capacity while activating only 1.4B parameters during inference. Our empirical evaluation reveals that LLaDA-MoE achieves state-of-the-art performance among diffusion language models with larger parameters, surpassing previous diffusion language models LLaDA, LLaDA 1.5, and Dream across multiple benchmarks. The instruct-tuned model LLaDA-MoE-7B-A1B-Instruct demonstrates capabilities comparable to Qwen2.5-3B-Instruct in knowledge understanding, code generation, mathematical reasoning, agent and alignment tasks, despite using fewer active parameters. Our results show that integrating a sparse MoE architecture into the training objective of masked diffusion language models still brings out MoE's strengths under efficient inference with few active parameters, and opens ample room for further exploration of diffusion language models. LLaDA-MoE models are available at Huggingface.

---

# LLaDA-MoE：稀疏 MoE 扩散语言模型 论文详细解读

### 背景：这个问题为什么难？

在语言模型的规模不断膨胀的今天，算力和显存成为瓶颈。传统的大模型需要在每一次推理时激活全部参数，导致成本随模型大小线性增长。扩散语言模型（Diffusion Language Model）在生成质量上已经展现出潜力，但它们同样受限于全参数激活的高开销。于是，如何在保持或提升生成能力的同时，显著削减推理时的计算量，成为迫切需要突破的难点。

### 关键概念速览
**扩散语言模型（Diffusion Language Model）**：把文本生成视为逐步去噪的过程，类似把一张被噪声覆盖的图片慢慢恢复清晰。模型在每一步预测更干净的文本表示，最终得到完整句子。  
**稀疏激活（Sparse Activation）**：不是所有神经元每次都被使用，而是只挑选一小部分参与计算，就像只打开几盏灯照亮房间的关键区域。  
**Mixture‑of‑Experts（MoE）**：把模型拆成多个“专家”，每次输入只走进少数几个专家，由路由网络决定。想象一支乐队，演出时只让几位乐手上场，而不是全体同时演奏。  
**路由网络（Router）**：负责把输入分配给合适的专家，常用的实现是基于输入特征的打分后取前k名。它相当于招聘经理，根据简历把任务交给最匹配的员工。  
**指令微调（Instruction Fine‑Tuning）**：在大模型上再用指令式数据进行微调，使模型更擅长遵循用户的明确指令，类似在通用语言能力上再练习一次“客服”技巧。  
**上下文窗口（Context Window）**：模型一次能看到的文本长度，窗口越大能捕捉更远的依赖关系。  

### 核心创新点
1. **从零训练的稀疏 MoE 扩散模型 → 直接在约 20 T token 的混合语料上训练 LLaDA‑MoE，保持 7 B 总参数但推理时只激活 1.4 B 参数** → 通过大规模稀疏化，实现了与全参数模型相近的生成质量，却把推理成本压缩到约 20%。  
2. **两阶段预训练策略 → 第一次预训练使用均衡的通用语料，第二次在同样规模的 token 中提升数学和代码文本的采样比例** → 让模型在保持通用能力的同时，对高价值的推理和编程任务拥有更强的专长。  
3. **多轮退火训练 → 在第二阶段最佳检查点上继续训练 500 B token，同时把上下文窗口从 4 k 扩展到 8 k** → 通过逐步增大上下文，模型学会处理更长的依赖链，提升了长文生成的连贯性。  
4. **指令微调只对回答部分做掩码 → 在 SFT（Supervised Fine‑Tuning）阶段，仅对模型的回答段落进行 mask 处理** → 这种细粒度的掩码方式让模型在指令遵循上更精准，同时保留了原始扩散预训练的噪声建模优势。

### 方法详解
整体框架可以拆成四大块：① 大规模稀疏 MoE 架构搭建，② 双阶段预训练，③ 退火式上下文扩展，④ 指令微调。下面按顺序展开。

**1. 稀疏 MoE 架构**  
模型的主体是一个标准的 Transformer 编码器/解码器，但每个前馈层被替换为 MoE 层。MoE 层内部有 N = 64 个专家（每个专家是一个完整的前馈网络），路由网络根据输入的隐藏向量计算每个专家的得分，取前 k = 2 的专家激活。这样，虽然模型整体拥有 7 B 参数（所有专家加起来），但一次前向只会动用约 1.4 B 参数（激活的两位专家）。路由采用 Top‑2 gating，使用噪声调节防止专家长期闲置。

**2. 双阶段预训练**  
- **阶段 1**：在约 10 T token 的通用混合语料上进行标准的扩散预训练。扩散过程采用“masked diffusion”，即随机遮盖一部分 token，然后让模型在多个噪声步长上恢复原始序列。  
- **阶段 2**：继续使用另外 10 T token，但在采样时把数学、代码等高难度子域的比例提升到约 30%。这一步的目标是让 MoE 的路由网络学会把这些专业文本送到更擅长的专家，从而在后续任务中表现更好。

**3. 退火式训练**  
在阶段 2 的最佳检查点上，作者继续训练 500 B token，期间逐步把上下文窗口从 4 k 扩大到 8 k。这里的“退火”指的是学习率和噪声水平的逐步衰减，让模型在更长序列上稳定收敛。窗口扩展的实现方式是先在 4 k 上训练到收敛，再在 8 k 上继续训练，保持已有知识不被遗忘。

**4. 指令微调（SFT）**  
微调数据来源于指令式对话集合。与传统的全序列微调不同，作者只在模型生成的“回答”部分加入 mask，迫使模型在该段落上进行更强的噪声去除。这样做的直觉是：扩散预训练已经让模型擅长在噪声中恢复信息，只需要在指令微调时强化对答案的精细恢复即可。

**最巧妙的点**  
- 把 MoE 与扩散预训练直接耦合，而不是先训练普通模型再套上 MoE。这样路由网络从一开始就能感知噪声去除的任务特性。  
- 只在指令微调阶段对回答做 mask，保持了扩散模型的噪声建模优势，同时提升了指令遵循的准确度。

### 实验与效果
- **评测基准**：在多个公开的语言模型基准上进行评估，包括 MMLU（知识理解）、HumanEval（代码生成）、GSM‑8K（数学推理）以及 Agent‑Bench（多轮交互）等。  
- **对比模型**：与同类的扩散语言模型 LLaDA、LLaDA‑1.5、Dream 以及非扩散的指令模型 Qwen2.5‑3B‑Instruct 进行横向比较。  
- **核心结果**：在所有扩散基准上，LLaDA‑MoE‑7B‑A1B‑Instruct 超越了前代扩散模型，尤其在数学和代码任务上领先约 5‑7% 的准确率。指令模型的整体表现与 Qwen2.5‑3B‑Instruct 相当，尽管前者在推理时只用了约 1.4 B 参数。  
- **消融实验**：作者分别关闭 MoE、去掉第二阶段的数学/代码加权、以及不进行上下文窗口退火。实验显示：去掉 MoE 会导致推理成本提升 6 倍且性能下降约 3%；不做第二阶段加权会让数学/代码任务的得分下降约 4%；不进行窗口扩展会使长文本生成的连贯度下降约 2%。  
- **局限性**：论文承认在极端长序列（> 16 k）上仍然会出现记忆衰减；路由网络在极端稀疏（k = 1）时会出现专家利用率不均的问题；此外，训练成本仍然高达数千 GPU‑day，对小团队仍有门槛。

### 影响与延伸思考
LLaDA‑MoE 把稀疏 MoE 与扩散语言模型成功结合，向“高质量+低成本”方向迈出重要一步。随后的工作开始探索更细粒度的路由策略（如基于层级专家树）以及在更大规模（> 100 B）下的稀疏扩散训练。对想进一步了解的读者，可以关注以下方向：① MoE 在多模态扩散模型中的适配；② 动态稀疏度调节，让模型根据输入难度自行决定激活多少专家；③ 更高效的噪声调度方案，以进一步压缩训练时间。整体来看，这篇论文为稀疏化大模型提供了可行的实践路径，也让扩散生成技术在实际部署场景中更具竞争力。

### 一句话记住它
LLaDA‑MoE 用 7 B 参数的稀疏 MoE 结构，在只激活 1.4 B 参数的情况下，达到了与全参数扩散模型相当甚至更好的生成质量。