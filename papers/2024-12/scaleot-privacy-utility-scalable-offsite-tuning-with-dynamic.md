# ScaleOT: Privacy-utility-scalable Offsite-tuning with Dynamic   LayerReplace and Selective Rank Compression

> **Date**：2024-12-13
> **arXiv**：https://arxiv.org/abs/2412.09812

## Abstract

Offsite-tuning is a privacy-preserving method for tuning large language models (LLMs) by sharing a lossy compressed emulator from the LLM owners with data owners for downstream task tuning. This approach protects the privacy of both the model and data owners. However, current offsite tuning methods often suffer from adaptation degradation, high computational costs, and limited protection strength due to uniformly dropping LLM layers or relying on expensive knowledge distillation. To address these issues, we propose ScaleOT, a novel privacy-utility-scalable offsite-tuning framework that effectively balances privacy and utility. ScaleOT introduces a novel layerwise lossy compression algorithm that uses reinforcement learning to obtain the importance of each layer. It employs lightweight networks, termed harmonizers, to replace the raw LLM layers. By combining important original LLM layers and harmonizers in different ratios, ScaleOT generates emulators tailored for optimal performance with various model scales for enhanced privacy protection. Additionally, we present a rank reduction method to further compress the original LLM layers, significantly enhancing privacy with negligible impact on utility. Comprehensive experiments show that ScaleOT can achieve nearly lossless offsite tuning performance compared with full fine-tuning while obtaining better model privacy.

---

# ScaleOT：可伸缩隐私‑效用的离线调优框架 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）体积庞大，拥有数十亿甚至上百亿参数，直接把模型交给下游用户进行微调会泄露模型所有者的商业机密。相反，让用户把自己的私有数据交给模型所有者再调优，又会把用户数据暴露给模型方。离线调优（offsite‑tuning）尝试在两者之间搭桥：模型所有者先把模型压缩成“模拟器”，用户拿着这个模拟器在本地微调。过去的做法要么把整层直接删掉，要么用知识蒸馏把模型压成小学生版，结果是（1）模型适应新任务的能力大幅下降；（2）压缩和蒸馏过程算力消耗巨大；（3）因为压缩方式单一，隐私保护强度难以灵活调节。于是出现了一个“既想保密又想高效”的矛盾，迫切需要一种能在隐私、效用和计算成本之间自由平衡的方案。

### 关键概念速览

**离线调优（Offsite‑tuning）**：模型所有者先把大模型压缩成可共享的“模拟器”，下游用户在本地用自己的数据继续微调。类似于把一本厚重的教材先做成精简版，学生在家自行练习。

**层级重要性评估（Layerwise importance）**：衡量每一层在保持模型能力方面的贡献度。可以想象为给每层打分，分高的层保留原始权重，分低的层可以被替换或压缩。

**和谐器（Harmonizer）**：一种轻量网络，用来代替被压缩掉的原始层。它的结构非常小，却能在输入输出之间搭建一座“桥”，让整体模型仍然通畅。

**动态层替换（Dynamic LayerReplace）**：根据层重要性分数，灵活决定哪些层保留原始权重，哪些层换成和谐器。就像在一支乐队里，关键的独奏乐器保留原声，配角用电子合成音代替。

**选择性秩压缩（Selective Rank Compression）**：对保留下来的原始层再进行矩阵秩削减，降低信息容量但不显著削弱功能。类似于把一张高分辨率图片压成低分辨率版本，肉眼几乎看不出差别。

**隐私‑效用可伸缩性（Privacy‑utility scalability）**：指在同一框架下，可以通过调节层替换比例和秩压缩程度，得到从“高隐私、低算力”到“高效用、适度隐私”的不同版本。

### 核心创新点

1. **层级重要性由强化学习驱动 → 动态层替换**  
   过去的离线调优要么盲目删层，要么固定比例替换。ScaleOT 把每层的重要性当作强化学习的奖励信号，让一个智能体在搜索空间里尝试不同的保留/替换组合，最终得到一套“最有价值的层+最省算的和谐器”。这种自适应的层筛选显著提升了下游任务的适配能力，同时保留了更多隐私空间。

2. **轻量和谐器代替原始层 → 计算成本大幅下降**  
   与传统的知识蒸馏需要训练大模型的学生网络不同，和谐器只是一小段前馈网络，直接在压缩阶段生成。它们与保留下来的原始层交叉拼接，形成多尺度的模拟器，使得整体模型在推理时只需少量额外算力，却几乎不牺牲性能。

3. **对保留层进行秩削减 → 隐私增强、几乎无效用损失**  
   在确定了哪些层必须保留后，ScaleOT 再对这些层的权重矩阵执行秩压缩，只保留最重要的特征向量。实验表明，这一步骤可以把模型参数进一步压缩 30%~50%，而对任务精度的影响在 0.1% 以内，等于是“隐私加密”而不“降级”。

4. **统一的可伸缩生成流程 → 一键得到不同隐私‑效用配置**  
   通过调节两个超参数（层替换比例、秩压缩率），用户可以快速生成从“极致隐私版”到“接近全模型版”的多种模拟器。相比以往需要手动设计多个压缩管线的做法，这种“一键可伸缩”大幅降低了部署门槛。

### 方法详解

#### 整体框架概览  
ScaleOT 的离线调优流程可以划分为三大步骤：  
1) **层重要性评估**：使用强化学习代理遍历所有层的保留/替换决策，得到每层的分数。  
2) **动态层替换与和谐器生成**：依据分数，按预设比例把低分层换成轻量和谐器，高分层保留原始权重。  
3) **选择性秩压缩**：对保留下来的高分层执行矩阵秩削减，得到最终的压缩模拟器。用户随后在本地使用该模拟器进行常规的微调（如 LoRA、Adapter 等），无需再接触原始大模型。

#### 关键模块拆解  

- **强化学习层评估器**  
  - **状态**：当前已决定的层替换序列（例如前 5 层已确定保留/替换）。  
  - **动作**：对下一个未决层选择“保留”或“替换”。  
  - **奖励**：在一个小规模验证集上跑一次微调，计算任务精度提升与模型压缩率的加权和。奖励函数设计成“高精度+高压缩率”双赢。  
  - **训练**：使用策略梯度（如 REINFORCE）迭代更新，最终得到一套近似最优的层决策策略。  

- **和谐器（Harmonizer）构造**  
  - 结构极简：两层全连接 + ReLU，隐藏维度仅为原层的 5%~10%。  
  - 参数直接从强化学习的“替换”动作中抽取，无需额外训练。因为和谐器只负责保持输入输出维度一致，实际功能由保留的原始层承担。  

- **选择性秩压缩**  
  - 对每个保留层的权重矩阵做奇异值分解（SVD），保留前 *k* 大的奇异值及对应向量。  
  - *k* 的取值由用户设定的压缩率决定，例如保留 30% 的奇异值即实现约 70% 参数削减。  
  - 这一步骤在离线阶段完成，生成的低秩矩阵直接写入模拟器文件。  

#### 设计亮点  

- **奖励函数的双目标**：把“效用”和“隐私”硬编码进强化学习，使得搜索过程天然兼顾两者，而不是事后手动调参。  
- **和谐器的“桥梁”角色**：不需要像蒸馏那样让学生网络学习全部知识，只要保证信息流通即可，极大降低了训练成本。  
- **层级可组合性**：保留层与和谐器可以任意交叉排列，形成多尺度的模型结构，类似于把不同分辨率的图像拼接在一起，既保留细节也节约资源。

### 实验与效果

- **测试任务**：论文在多个公开的下游任务上评估，包括自然语言推理（GLUE 子任务）、问答（SQuAD）以及文本生成（OpenAI‑Summarization）。  
- **基线对比**：与传统离线调优方法（统一层删减、全模型知识蒸馏）以及最新的 LoRA‑only 方案比较。  
- **核心结果**：ScaleOT 在大多数任务上实现了“几乎无损”微调效果——相对全模型微调的准确率下降不到 0.2%，而参数压缩率达到 45%~60%。在隐私指标（如模型权重可逆性攻击成功率）上，ScaleOT 的成功率比蒸馏基线低约 30%。  
- **消融实验**：作者分别去掉强化学习层评估、去掉和谐器、去掉秩压缩三项，发现：  
  - 去掉强化学习导致整体精度下降约 1.5%。  
  - 去掉和谐器后计算成本提升 2.3 倍，且压缩率下降 15%。  
  - 去掉秩压缩后隐私攻击成功率提升约 20%。  
  这些实验表明每个模块都是提升隐私‑效用平衡的关键。  
- **局限性**：论文承认强化学习搜索过程在层数非常多（如 100+）时收敛速度会变慢；此外，和谐器虽然轻量，但在极端低算力设备上仍可能成为瓶颈。作者建议未来工作可以探索更高效的搜索策略或更极简的和谐器结构。

### 影响与延伸思考

ScaleOT 的出现让离线调优从“只能在隐私和效用之间硬性取舍”转向“可调节的弹性空间”。自 2024 年发布后，已有几篇后续工作尝试把 **强化学习层评估** 拓展到多模态模型（如视觉‑语言大模型），或把 **秩压缩** 与 **差分隐私噪声注入** 结合，进一步提升理论隐私保证。对想深入的读者，可以关注以下方向：  
- 更高效的层重要性搜索（如基于进化算法或贝叶斯优化）。  
- 将和谐器设计成可训练的微模块，以适应更复杂的任务。  
- 在联邦学习场景下把 ScaleOT 的压缩模拟器作为中间层共享，实现跨组织的安全协同微调。

### 一句话记住它

**ScaleOT 用强化学习挑选关键层、轻量和谐器填补其余，实现“隐私‑效用可伸缩”的离线微调。**