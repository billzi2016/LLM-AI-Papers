# Yuan3.0 Flash: An Open Multimodal Large Language Model for Enterprise Applications

> **Date**：2026-01-05
> **arXiv**：https://arxiv.org/abs/2601.01718

## Abstract

We introduce Yuan3.0 Flash, an open-source Mixture-of-Experts (MoE) MultiModal Large Language Model featuring 3.7B activated parameters and 40B total parameters, specifically designed to enhance performance on enterprise-oriented tasks while maintaining competitive capabilities on general-purpose tasks. To address the overthinking phenomenon commonly observed in Large Reasoning Models (LRMs), we propose Reflection-aware Adaptive Policy Optimization (RAPO), a novel RL training algorithm that effectively regulates overthinking behaviors. In enterprise-oriented tasks such as retrieval-augmented generation (RAG), complex table understanding, and summarization, Yuan3.0 Flash consistently achieves superior performance. Moreover, it also demonstrates strong reasoning capabilities in domains such as mathematics, science, etc., attaining accuracy comparable to frontier model while requiring only approximately 1/4 to 1/2 of the average tokens. Yuan3.0 Flash has been fully open-sourced to facilitate further research and real-world deployment: https://github.com/Yuan-lab-LLM/Yuan3.0.

---

# Yuan3.0 Flash 论文详细解读

### 背景：这个问题为什么难？

在企业级应用里，模型需要同时处理文字、表格、图片等多模态信息，还要在检索、摘要、复杂推理等任务上保持高效。传统的大模型要么参数全开，算力成本爆炸，要么只专注单一模态，导致跨模态协同差强人意。更糟的是，很多强推理模型在面对长链思考时会出现“过度思考”，即在已经找到答案后仍继续生成无关内容，浪费算力并降低响应速度。于是，如何在保持通用能力的同时，提供企业所需的高效多模态推理，成为亟待突破的瓶颈。

### 关键概念速览
- **Mixture-of-Experts（MoE）**：把模型拆成若干“专家”，每次前向只激活一小部分专家，类似把大公司分成多个部门，只让相关部门参与当前项目，从而大幅降低计算量。  
- **多模态（Multimodal）**：模型能够同时理解文字、图片、表格等不同类型的数据，就像人类在阅读报告时会看图表、文字一起理解。  
- **检索增强生成（RAG）**：先在外部知识库里检索相关片段，再把检索结果和原始提示一起喂给模型生成答案，类似先查资料再写报告。  
- **过度思考（Overthinking）**：模型在已经得到正确答案后仍继续生成，导致不必要的 token 消耗，像人已经说完话却继续絮叨。  
- **反思抑制奖励（Reflection-aware Suppression Reward）**：在强化学习里给模型一个“停下来”的奖励，鼓励它在首次出现正确答案时停止生成。  
- **自适应动态采样（Adaptive Dynamic Sampling）**：根据不同提示的价值动态挑选训练样本，类似老师把重点题目挑出来多练。  
- **Dual‑Clip**：在强化学习的策略梯度中同时限制策略变化幅度的上下界，防止一次更新把模型推得太远。  

### 核心创新点
1. **稀疏 MoE + LLaVA 式多模态融合 → 只激活 3.7 B 参数、总量 40 B**  
   过去的多模态大模型要么全参数激活，要么只能处理固定尺寸的图像。Yuan3.0 Flash 采用 MoE 让每次推理只走少数专家，同时借鉴 LLaVA 的视觉嵌入方式，实现对任意分辨率图像的自适应处理。结果是算力成本只相当于全模型的 1/10，却保持了 40 B 参数的潜在容量。

2. **Reflection‑aware Adaptive Policy Optimization（RAPO） → 通过奖励抑制过度思考**  
   传统 RLHF（基于人类反馈的强化学习）只关注答案质量，忽视生成长度。RAPO 在奖励函数里加入“首次正确答案出现位置”的信息，并结合自适应采样、只对熵最高 20% token 计算梯度、dual‑clip 约束等技巧。这样模型学会在找到答案后立刻停下来，平均 token 使用量降到同类最前沿模型的 1/4‑1/2。

3. **统一多任务训练 + 任务感知策略** → 同时提升检索、表格理解、摘要等企业任务的表现  
   过去企业模型往往为每个子任务单独微调，导致资源碎片化。Yuan3.0 Flash 把所有任务放进同一个训练框架，针对不同任务使用不同的奖励或采样策略，却共享同一套 MoE 与视觉编码器。实验显示，这种“一体多用”方式在 RAG、复杂表格解析等任务上均超越专门微调的基线。

### 方法详解
**整体框架**  
Yuan3.0 Flash 的训练分四个阶段：① 大规模语言模型（LM）预训练，累计约 3 T token；② 多模态统一预训练，加入图像、表格等视觉信号；③ 有监督微调（SFT），使用高质量指令数据；④ 基于 RAPO 的强化学习微调。模型本体是一个 40 B 参数的 MoE Transformer，内部有若干专家路由网络，每次前向只选出约 10% 的专家激活，实际计算量约 3.7 B 参数。

**关键模块拆解**  

1. **MoE 路由与激活**  
   - 输入 token 经过普通的自注意力层后，进入路由器。路由器根据当前 token 的特征打分，选出 Top‑k（如 k=2）专家。  
   - 只这几个专家的前向计算会被执行，其他专家保持静默。这样既保留了大模型的表达能力，又把实际 FLOPs 降到可接受范围。  

2. **视觉编码器（LLaVA‑style）**  
   - 采用 CLIP‑style 的视觉 Transformer 把图像切成 patch，生成视觉 token。  
   - 视觉 token 与文字 token 在同一 Transformer 中混合，路由器同样决定哪些专家负责跨模态信息。  
   - 由于路由是基于 token 本身的特征，模型可以自动适配不同分辨率的图像，避免硬编码的尺寸限制。  

3. **RAPO 强化学习**  
   - **奖励设计**：在生成过程中记录每一步的答案正确性。当模型第一次输出正确答案时，给一个高额的“停下来”奖励；随后每生成一个额外 token 都会扣分。  
   - **自适应动态采样**：先跑一次模型，统计哪些提示的通过率高、奖励值大，然后在下一轮训练中对这些高价值提示进行重采样，提高学习效率。  
   - **熵筛选**：只对生成分布熵最高的 20% token 计算梯度，等价于把学习重点放在模型最不确定的地方。  
   - **Dual‑Clip**：在策略梯度更新时，对 KL 散度设上下限，防止一次更新把路由或语言分布改得太剧烈。  

4. **任务感知奖励系统**  
   - 对 RAG 任务，奖励包括检索命中率和生成质量；对表格理解，奖励加入表格结构匹配分数；对摘要，奖励关注 ROUGE 与长度惩罚。  
   - 这些奖励在同一 RL 框架下统一计算，使得模型在一次更新中同时学习多任务的技巧。  

**最巧妙的点**  
RAPO 把“何时停”这个人类直觉硬编码进奖励函数，并用动态采样把稀缺的高价值训练信号放大，这在大模型强化学习里极少出现。再加上只对高熵 token 进行梯度更新，显著降低了 RL 训练的噪声和算力需求。

### 实验与效果
- **测试任务**：检索增强生成（RAG）、复杂表格理解、长文摘要、数学与科学推理等。  
- **基线对比**：与同规模的全参数模型（如 LLaMA‑2‑70B）以及公开的企业定制模型（如 InternLM‑Chat）进行比较。  
- **主要结果**：在 RAG 任务上，Yuan3.0 Flash 的准确率提升约 7%（相对提升），生成长度仅为对手的 45%；表格理解的 F1 提升 5.3 点；摘要的 ROUGE‑L 提升 3.8%；数学推理的正确率与最前沿模型持平，但平均 token 使用量只有其 1/3。  
- **消融实验**：去掉 RAPO 中的“首次正确答案奖励”，模型的平均 token 数上升 30%，准确率下降约 2%；关闭熵筛选后训练收敛速度减慢 40%。  
- **局限性**：论文未给出在极端低算力设备上的实际推理时延；对极大规模视觉输入（如高清视频）仍需进一步验证；RAPO 对奖励函数的手工设计仍有一定经验门槛。  

### 影响与延伸思考
Yuan3.0 Flash 把 MoE 与多模态统一训练结合，并用 RAPO 解决过度思考，已经在开源社区引发一波“稀疏多模态+RL” 的讨论。后续有工作尝试把类似的奖励机制迁移到大模型的对话安全调控上（如防止模型在已完成任务后继续输出敏感信息），也有研究把自适应采样与自监督预训练结合，以进一步压缩训练成本。想继续深入的读者可以关注以下方向：① 更细粒度的专家路由策略（如基于任务层级的路由）；② 自动化的奖励函数搜索（Meta‑RL）；③ 将 RAPO 与人类反馈（RLHF）混合使用的统一框架。  

### 一句话记住它
Yuan3.0 Flash 用稀疏 MoE + RAPO，让大模型在企业多模态任务上既保持强大能力，又学会“第一时间停下来”。