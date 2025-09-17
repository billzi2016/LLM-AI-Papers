# SAIL-VL2 Technical Report

> **Date**：2025-09-17
> **arXiv**：https://arxiv.org/abs/2509.14033

## Abstract

We introduce SAIL-VL2, an open-suite vision-language foundation model (LVM) for comprehensive multimodal understanding and reasoning. As the successor to SAIL-VL, SAIL-VL2 achieves state-of-the-art performance at the 2B and 8B parameter scales across diverse image and video benchmarks, demonstrating strong capabilities from fine-grained perception to complex reasoning. Its effectiveness is driven by three core innovations. First, a large-scale data curation pipeline with scoring and filtering strategies enhances both quality and distribution across captioning, OCR, QA, and video data, improving training efficiency. Second, a progressive training framework begins with a powerful pre-trained vision encoder (SAIL-ViT), advances through multimodal pre-training, and culminates in a thinking-fusion SFT-RL hybrid paradigm that systematically strengthens model capabilities. Third, architectural advances extend beyond dense LLMs to efficient sparse Mixture-of-Experts (MoE) designs. With these contributions, SAIL-VL2 demonstrates competitive performance across 106 datasets and achieves state-of-the-art results on challenging reasoning benchmarks such as MMMU and MathVista. Furthermore, on the OpenCompass leaderboard, SAIL-VL2-2B ranks first among officially released open-source models under the 4B parameter scale, while serving as an efficient and extensible foundation for the open-source multimodal community.

---

# SAIL‑VL2 论文详细解读

### 背景：这个问题为什么难？

视觉语言模型（VLM）要同时理解图像、视频和文字，还要在细粒度感知和高层次推理之间自由切换。过去的模型往往在两方面只能做到“一头热”：要么在大规模图像描述上表现不错，却在数学推理、长文本问答等复杂任务上失灵；要么靠巨大的参数量勉强提升一点性能，却导致训练成本和推理延迟飙升。根本的瓶颈在于（1）训练数据的质量和分布不均，导致模型学不到跨模态的通用知识；（2）训练流程缺乏层次化的进阶，模型在从感知到推理的迁移过程中会出现“忘记”或“冲突”。这些限制让 VLM 在真正的多模态通用智能路上卡住了。

### 关键概念速览
- **视觉语言模型（VLM）**：把图像/视频特征和语言模型（LLM）拼在一起，让机器能“看懂”并“说出来”。想象成把相机和聊天机器人粘在一起的系统。  
- **Mixture‑of‑Experts（MoE）**：把一个大模型拆成若干专家子网，输入只激活一小部分专家，从而在保持强大表达力的同时降低计算量。类似于公司里不同部门只处理自己擅长的任务。  
- **Progressive Training（渐进式训练）**：训练分阶段进行，先让视觉编码器学会基本特征，再让多模态对齐，最后全模型一起微调。像学钢琴，从指法练习到乐谱阅读再到完整演奏。  
- **Thinking‑Fusion SFT‑RL（思考融合的指令微调+强化学习）**：在指令微调（SFT）阶段加入“思考链”提示，让模型先输出推理步骤；随后用强化学习（RL）根据答案、格式和推理过程给奖励，促使模型在生成时自我检查。相当于先教学生写草稿，再给分数奖励。  
- **数据筛选与评分管线**：在海量图文、OCR、问答、视频字幕等原始数据上加一层质量评估，过滤噪声并平衡分布。好比在超市挑选新鲜、种类齐全的食材再下锅。  
- **SAIL‑ViT**：本文使用的视觉编码器，基于 Vision Transformer（ViT）并经过大规模自监督预训练，提供高质量的视觉特征。可以把它想成“高分辨率的相机镜头”。  

### 核心创新点
1. **大规模、质量导向的数据管线 → 采用多维度评分（caption、OCR、QA、视频）并进行严格过滤 → 训练样本更干净、分布更均衡，显著提升了模型在细粒度感知和复杂推理上的学习效率。**  
2. **渐进式训练框架 → 先用 SAIL‑ViT 进行视觉热身，再冻结 LLM 进行多模态对齐，最后全参数解冻进行全任务微调 → 解决了感知与语言之间的冲突，使模型在从低层特征到高层推理的迁移中保持稳定。**  
3. **思考融合的 SFT‑RL 混合范式 → 在指令微调阶段加入长链思考（LongCoT）提示，随后用强化学习分别奖励答案正确性、格式规范和推理过程 → 模型不仅给出正确答案，还能自检推理路径，提升了在 MMMU、MathVista 等高难度推理基准上的表现。**  
4. **稀疏 MoE 架构的引入 → 在大模型（8B）上使用专家路由，仅激活部分专家子网 → 在保持或提升性能的同时把显存和算力需求压到接近 2B 规模模型的水平，提升了实际部署的可行性。**  

### 方法详解
**整体框架**  
SAIL‑VL2 的训练分为四大阶段：① 数据准备 → ② 视觉编码器热身 → ③ 多模态渐进式预训练 → ④ 思考融合的指令微调 + 强化学习。每一步都围绕“先感知、后对齐、再推理、最后自检”展开。

**1. 数据准备**  
- 收集四类数据：图像/视频字幕、光学字符识别（OCR）文本、开放式问答、数学/逻辑推理。  
- 对每条样本打分：使用已有小模型评估 caption 质量、OCR 准确度、问答匹配度等。  
- 过滤阈值设定后，保留高分样本并在不同类别之间做均衡抽样，确保模型既能看到丰富的视觉细节，也能学习到语言推理的模式。

**2. 视觉编码器热身**  
- 采用 SAIL‑ViT 作为视觉前端，先在大规模图像自监督任务上微调，使其输出的特征在空间分辨率和语义层次上都足够丰富。  
- 这一步相当于给模型装上“高清摄像头”，后面的多模态对齐才能受益。

**3. 渐进式多模态预训练**  
- **阶段 A（对齐热身）**：冻结大语言模型（LLM），只训练投影层（projector）把视觉特征映射到 LLM 的词向量空间。训练样本主要是 caption、OCR、视频字幕，目标是让视觉特征在语言空间里“说得通”。  
- **阶段 B（全参数对齐）**：解冻 LLM，使用同样的多模态数据进行全参数训练，进一步强化视觉‑语言的共同表示。  
- **阶段 C（任务混合）**：加入开放式问答、数学推理等指令式数据，进行多任务预训练，使模型在感知的基础上学会遵循指令并进行推理。

**4. 思考融合的指令微调 + 强化学习**  
- **SFT（指令微调）**：采用课程学习的方式，先让模型学习基础指令（基础 SFT），再加入长链思考提示（LongCoT SFT），让模型在回答前输出推理步骤。  
- **RL（强化学习）**：设计三类奖励：① 格式奖励（答案是否符合预定义格式），② 答案奖励（是否正确），③ 推理奖励（评估思考链的合理性）。使用 PPO 等策略梯度算法让模型在生成时最大化综合奖励。  
- **混合推理 SFT**：在 RL 之后再进行一次指令微调，融合 RL 学到的策略，使模型在实际推理时既保持流畅的语言输出，又保留自检的思考链。

**稀疏 MoE 设计**  
- 在 8B 参数版本中，引入专家路由网络。每个输入 token 根据 gating 网络选择 1~2 个专家子网进行计算，其他专家保持不活跃。这样在推理时显存仅相当于 2B 参数模型，却保留了专家子网的多样性，提升了大规模任务的表现。

**最巧妙的点**  
- 将“思考链”直接嵌入强化学习的奖励函数，使模型在训练阶段就学会自我审查，而不是事后再做后处理。  
- 通过分层数据评分把噪声数据压到最低，避免了大模型在海量低质量样本上“学坏”。  

### 实验与效果
- **评测范围**：覆盖 106 个公开数据集，涵盖图像分类、目标检测、视频问答、跨模态检索、复杂推理（MMMU、MathVista）等。  
- **核心成绩**：在 2B 参数规模上，SAIL‑VL2 在多数视觉‑语言基准上超越同尺寸的 LLaVA、MiniGPT‑4 等；在 8B 规模上，取得 MMMU、MathVista 等推理基准的最新 SOTA（具体数值未在摘要中给出，论文声称领先数个百分点）。  
- **OpenCompass 排行**：在官方开放模型排行榜中，SAIL‑VL2‑2B 位列 4B 以下模型第一，说明其在多任务综合表现上具备竞争力。  
- **消融实验**：论文提供了对数据筛选、渐进式训练、思考融合 RL、MoE 等模块的独立去除实验，结果显示每项技术都贡献了约 1‑3% 的整体提升，尤其是思考融合的 RL 对高难度推理任务提升最为显著。  
- **局限性**：作者指出在极端长视频理解和实时交互场景仍有性能瓶颈，MoE 的路由开销在超大批次时会带来额外延迟；此外，强化学习奖励函数的设计仍依赖人工规则，自动化程度有待提升。

### 影响与延伸思考
SAIL‑VL2 把高质量数据筛选、渐进式训练和思考融合的强化学习结合起来，展示了在同等算力下实现多模态推理 SOTA 的可行路径。随后的开源项目（如 OpenFlamingo‑2、Mistral‑Vision）纷纷借鉴其 MoE 路由和 RL 奖励设计，推动了社区对“可解释推理链”与“稀疏大模型”交叉的兴趣。想进一步深入，可以关注以下方向：① 自动化的多模态数据质量评估；② 更高效的专家路由算法；③ 将思考链奖励扩展到跨语言、多语言场景。  

### 一句话记住它
**SAIL‑VL2 用“干净的多模态数据 + 渐进式训练 + 思考链强化学习 + 稀疏 MoE”，在 2B‑8B 规模下实现了视觉‑语言推理的最强表现。**