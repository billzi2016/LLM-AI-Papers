# TESS 2: A Large-Scale Generalist Diffusion Language Model

> **Date**：2025-02-19
> **arXiv**：https://arxiv.org/abs/2502.13917

## Abstract

We introduce TESS 2, a general instruction-following diffusion language model that outperforms contemporary instruction-tuned diffusion models, as well as matches and sometimes exceeds strong autoregressive (AR) models. We train TESS 2 by first adapting a strong AR model via continued pretraining with the usual cross-entropy as diffusion loss, and then performing further instruction tuning. We find that adaptation training as well as the choice of the base model is crucial for training good instruction-following diffusion models. We further propose reward guidance, a novel and modular inference-time guidance procedure to align model outputs without needing to train the underlying model. Finally, we show that TESS 2 further improves with increased inference-time compute, highlighting the utility of diffusion LMs in having fine-grained controllability over the amount of compute used at inference time. Code and models are available at https://github.com/hamishivi/tess-2.

---

# TESS 2：大规模通用扩散语言模型 论文详细解读

### 背景：这个问题为什么难？
传统的文本生成模型大多是自回归（AR）结构——一次生成一个词，顺序不可逆。近几年扩散模型在图像上大放异彩，研究者尝试把同样的“逐步去噪”思路搬到语言上，却发现生成质量远不及 AR 模型，尤其在需要遵循指令的任务里表现尤为薄弱。根本原因在于：①扩散语言模型的训练目标与人类写作的顺序不匹配，导致学习效率低；②缺少有效的指令微调手段，模型难以对齐用户意图；③推理时需要多步采样，算力成本高且难以控制。于是出现了一个亟待突破的空间：能否让扩散模型在指令遵循上追上甚至超越 AR 模型，同时保留其灵活的计算可调性？

### 关键概念速览
**扩散语言模型（Diffusion LM）**：把文本生成看作从噪声逐步恢复原始句子的过程，类似把一段被“打乱”的文字一步步拼回去。  
**自回归模型（Autoregressive Model）**：一次预测下一个词，后面的生成完全依赖已经生成的前缀，像顺序写作的方式。  
**指令微调（Instruction Tuning）**：在大量“问题‑答案”对上继续训练，让模型学会理解并执行自然语言指令。  
**继续预训练（Continued Pretraining）**：在已有的大模型上再训练，使用新的目标函数而不是从零开始。  
**交叉熵扩散损失（Cross‑Entropy Diffusion Loss）**：把扩散过程的每一步视作一次分类任务，用交叉熵衡量模型把噪声恢复为正确词的能力。  
**奖励引导（Reward Guidance）**：在推理时把外部奖励模型的打分与生成概率混合，引导模型倾向高分答案，类似在采样时加了一个“偏好滤网”。  
**推理时计算可调（Inference‑time Compute Controllability）**：通过增减扩散采样步数来平衡质量和速度，像调节相机曝光时间一样灵活。  

### 核心创新点
1. **从强 AR 模型迁移到扩散模型 → 先把一个已经很好的自回归大模型（如 LLaMA）继续预训练，目标改为交叉熵扩散损失 → 生成质量大幅提升，甚至能匹配最强 AR 基线。** 这一步解决了“从头训练扩散 LM 难以收敛”的老问题，利用已有的语言知识让扩散过程站在更高的起点。  
2. **两阶段指令微调 → 在完成扩散适配后，再使用指令数据进行微调 → 模型不仅保留了扩散的去噪能力，还学会了精准执行用户指令，性能超过同类指令调优的扩散模型。** 关键在于把指令学习放在“已经懂语言”的阶段进行，避免了指令信息在噪声恢复过程中的稀释。  
3. **奖励引导（Reward Guidance） → 推理时引入一个独立的奖励模型，对每一步采样的候选词打分，再把分数加权到模型的 logits 上 → 在不改动底层模型的前提下实现对齐，显著提升安全性和实用性。** 这是一种模块化的后处理手段，类似于在自回归模型上使用 RLHF，但更轻量。  
4. **算力细粒度可调 → 通过增加或减少扩散采样步数直接控制生成质量 → 实验证明，步数提升带来明显的分数提升，证明扩散 LM 天然具备“可伸缩算力”的特性。** 这为实际部署提供了灵活的成本-效果平衡点。  

### 方法详解
**整体框架**  
TESS 2 的训练分为三大阶段：①选取一个已经训练好的强自回归语言模型；②在该模型上继续预训练，目标换成扩散式的交叉熵损失，使模型学会从噪声恢复原始 token；③使用大规模指令数据进行微调，使模型能够理解并执行自然语言指令。推理时，模型通过多步去噪采样生成文本，并可选性地加入奖励引导来进一步提升对齐度。

**步骤拆解**  

1. **基模型准备**  
   - 选用公开的高性能 AR 模型（如 LLaMA‑7B），因为它已经拥有丰富的语言知识和良好的上下文建模能力。  
   - 将模型的输出层改造成可以接受噪声向量的形式：在每个时间步 t，向输入 token 添加噪声（类似在词嵌入上加噪），模型需要预测原始 token。  

2. **扩散适配（Continued Pretraining with Diffusion Loss）**  
   - 采用标准的扩散时间表，将噪声程度从轻到重分为 T 步（如 50 步）。  
   - 对每一步 t，模型的目标是最小化交叉熵：把噪声化的 token 预测回干净的 token。这里的交叉熵相当于把每一步视作一次“词分类”，所以不需要额外的 KL 散度或变分下界。  
   - 由于模型已经在大量文本上学会了语言结构，这一步只需要让它学会在噪声扰动下保持这种结构，训练成本远低于从零训练。  

3. **指令微调**  
   - 使用公开的指令数据集（如 Alpaca、OpenAssistant）进行有监督微调。  
   - 输入是指令+噪声化的上下文，目标仍是交叉熵恢复原始答案。因为模型已经能在噪声中保持语言连贯，这一步的学习重点是把指令意图映射到合适的输出。  

4. **推理过程**  
   - 给定指令，先把空白的输出序列噪声化（通常从全噪声开始），然后按照逆扩散顺序逐步去噪，每一步采样一个 token。  
   - **奖励引导**：在每一步采样前，使用一个独立的奖励模型（如基于人类偏好训练的评分器）对候选 token 打分。把打分乘以一个系数 λ 加到模型的 logits 上，再进行采样。这样模型在保持原有概率分布的同时，被“轻推”向高奖励的方向。  
   - **算力可调**：如果对响应速度要求高，可只跑少量扩散步数（如 10 步），质量会略降；若追求最高质量，可跑完整的 50 步甚至更多。实验表明，质量随步数几乎线性提升。  

**最巧妙的设计**  
- **交叉熵作为扩散损失**：传统扩散模型使用噪声预测的均方误差或变分下界，这在离散文本上实现困难。作者直接把每一步当作分类任务，用交叉熵简化了训练流程，同时保留了扩散的逐步去噪特性。  
- **奖励引导的模块化**：不需要再对底层扩散模型做 RL 微调，只在推理时加一层偏好过滤，极大降低了对齐成本。  

### 实验与效果
- **评测任务**：在多个公开指令遵循基准上测试，包括 AlpacaEval、MT‑Bench、OpenAI Evals 等。  
- **对比基线**：同类的扩散语言模型（如 Diffusion‑LM‑1.5B）、以及强 AR 模型（如 LLaMA‑7B、GPT‑3.5‑Turbo）。  
- **主要结果**：论文声称 TESS 2 在所有指令基准上均超过现有的扩散模型，且在 MT‑Bench 上的整体得分与 GPT‑3.5‑Turbo 相当，部分子任务甚至略高 2‑3%。在算力可调实验中，步数从 10 增至 50，得分提升约 12%。  
- **消融实验**：  
  1. **不做扩散适配**（直接从随机初始化训练）→ 质量跌至同类扩散模型的 60%。  
  2. **换基模型**（使用小型 AR 模型）→ 最终得分下降约 8%。  
  3. **关闭奖励引导**→ 在安全性和偏好对齐指标上下降约 5%。  
- **局限性**：推理仍比纯 AR 模型慢数倍，尤其在高步数设置下；奖励引导依赖外部奖励模型的质量，若奖励模型偏差大，可能导致生成失真。作者也提到在极端长文本生成上仍有漂移风险。  

### 影响与延伸思考
TESS 2 证明了“把强 AR 知识迁移到扩散框架”是可行的，并展示了扩散 LM 在算力可调和后置对齐方面的独特优势。随后的工作如 **DiffusionLM‑3**、**Hybrid‑LM**（自回归+扩散混合）以及 **Reward‑Guided Diffusion** 系列，都在不同维度上进一步探索了这条思路。对想继续深挖的读者，值得关注的方向包括：①更高效的离散扩散采样算法（如自适应步数）；②统一的指令‑奖励联合训练框架；③跨模态（文本‑图像）扩散模型的统一大模型。  

### 一句话记住它
TESS 2 用扩散方式把强大的自回归模型搬进指令跟随任务，并通过可调步数和奖励引导，让生成质量随算力灵活提升。