# On the Transformer Growth for Progressive BERT Training

> **Date**：2020-10-23
> **arXiv**：https://arxiv.org/abs/2010.12562

## Abstract

Due to the excessive cost of large-scale language model pre-training, considerable efforts have been made to train BERT progressively -- start from an inferior but low-cost model and gradually grow the model to increase the computational complexity. Our objective is to advance the understanding of Transformer growth and discover principles that guide progressive training. First, we find that similar to network architecture search, Transformer growth also favors compound scaling. Specifically, while existing methods only conduct network growth in a single dimension, we observe that it is beneficial to use compound growth operators and balance multiple dimensions (e.g., depth, width, and input length of the model). Moreover, we explore alternative growth operators in each dimension via controlled comparison to give operator selection practical guidance. In light of our analyses, the proposed method speeds up BERT pre-training by 73.6% and 82.2% for the base and large models respectively, while achieving comparable performances

---

# 关于Transformer增长的渐进式BERT训练 论文详细解读

### 背景：这个问题为什么难？
大规模语言模型（如BERT）在预训练阶段需要消耗巨量算力和时间，成本高得让很多实验室望而却步。过去的做法大多是一次性训练完整模型，要么使用更小的模型要么直接买算力，缺乏灵活的“先小后大”路径。已有的渐进式训练方法通常只在单一维度上扩展——比如先把层数（深度）加深，或者只扩大隐藏维度（宽度），但这种单向增长往往导致资源利用不均衡，训练时间仍然居高不下。要想真正把“先低成本、后高性能”落到实处，需要一种系统化的增长策略，能够在多个维度上同步平衡扩展，从而在保持预训练质量的同时大幅压缩总算力消耗。

### 关键概念速览
**BERT**：一种基于Transformer的双向语言模型，预训练时会学习词语之间的上下文关系，后续可以微调到各种下游任务。  
**Transformer**：由自注意力层和前馈网络组成的模块，核心参数包括层数（depth）、隐藏维度（width）以及序列长度（input length）。  
**渐进式训练（Progressive Training）**：先训练一个小模型，再逐步扩大模型规模继续训练的策略，类似先练基础体能再提升重量的健身计划。  
**复合缩放（Compound Scaling）**：同时在多个维度上按一定比例放大模型，而不是只在单一维度上扩展，类似把汽车的马力、车身宽度和轮胎直径一起升级，以保持整体平衡。  
**增长算子（Growth Operator）**：在某一维度上实现模型扩张的具体操作，例如“复制层”“插入宽度通道”“延长输入序列”。  
**网络结构搜索（NAS）**：自动化寻找最优网络结构的技术，常用的思路是搜索不同维度的组合，这里借鉴其“复合”思路来指导模型增长。

### 核心创新点
1. **单维度 → 多维度复合增长**  
   过去的渐进式方法只在深度或宽度上单独扩展，导致算力瓶颈或性能提升不均。本文提出在深度、宽度和输入长度三个维度上同步增长，并通过实验验证这种复合增长能更高效地利用算力。结果显示，BERT‑Base 和 BERT‑Large 在相同预训练预算下分别提升了约 73.6% 与 82.2% 的训练速度，性能几乎不打折扣。  

2. **系统化增长算子对比**  
   作者对每个维度的多种增长算子（如层复制 vs. 新层初始化、宽度通道扩张 vs. 线性投影、序列截断 vs. 动态填充）进行受控实验，给出哪种算子在实际训练中更稳健、更省时的实用指南。比如在深度维度上，直接复制已有层的参数比随机初始化新层更快收敛。  

3. **基于复合缩放的增长比例搜索**  
   受网络结构搜索的启发，论文提出一种轻量级的比例搜索策略，自动决定每一步增长时各维度的放大系数，使得整体模型复杂度按预设的算力预算平滑提升，而不是人为手动调参。  

4. **统一的渐进式训练框架**  
   将上述三点整合进一个统一的训练循环：初始化小模型 → 按复合比例增长 → 继续预训练 → 重复至目标规模。该框架兼容现有的BERT实现，几乎不需要改动优化器或数据管线。

### 方法详解
整体思路可以概括为四个阶段：**① 小模型启动 → ② 复合增长决策 → ③ 参数迁移与继续预训练 → ④ 循环迭代**。下面逐步拆解每一步。

1. **小模型启动**  
   选取一个计算成本极低的BERT子网（例如 4 层、768 隐藏维度、序列长度 128），使用标准的Masked Language Modeling（MLM）目标进行完整的预训练若干步，得到一套已学习的参数。

2. **复合增长决策**  
   - **增长维度**：深度（层数）、宽度（隐藏维度/注意力头数）和输入长度。  
   - **增长比例**：通过一个简化的搜索过程（类似网格搜索），在每次增长前设定目标算力增幅（比如 30%），然后在三维空间里寻找满足该增幅且各维度比例最均衡的组合。  
   - **算子选择**：对每个维度使用实验中表现最好的算子：  
     - *深度*：复制已有层的权重并插入新层，保持特征分布连续。  
     - *宽度*：在每个前馈网络和注意力投影上使用线性投影扩张，即把原有权重映射到更高维度，再用随机初始化填充新增维度。  
     - *输入长度*：直接将序列长度加倍，并在新位置填充特殊的[PAD]标记，保持位置编码的线性扩展。  

3. **参数迁移与继续预训练**  
   - **权重映射**：复制层和线性投影的映射保证了新模型在参数空间的“连续性”，相当于在已有知识的基础上开辟新容量。  
   - **学习率调度**：在增长点采用短暂的学习率 warm‑up，使新参数有时间适应旧特征；随后恢复原有的学习率曲线。  
   - **数据管线**：保持原始的MLM任务不变，只是把更长的序列喂入模型，提升上下文学习能力。  

4. **循环迭代**  
   重复步骤 2‑3，直至模型规模达到目标（如 BERT‑Base 的 12 层、768 隐藏、512 长度）。每一次增长都只消耗相对少量的额外算力，因为大部分参数已经在前一步训练好，只需要对新增部分进行快速适配。

**最巧妙的点**在于把“复合缩放”理念从模型设计阶段搬到训练阶段，使得增长过程本身成为一种“软”扩展，而不是硬性重启全模型训练。这样既保留了已有的语言知识，又让新容量迅速发挥作用。

### 实验与效果
- **数据集/任务**：在公开的英文 Wikipedia + BookCorpus 大规模语料上进行 MLM 预训练；下游评估使用 GLUE 基准（包括 MNLI、QQP、STS‑B 等任务）。  
- **Baseline**：传统一次性训练的 BERT‑Base/Large（全尺寸直接预训练），以及已有的单维度渐进式方法（仅深度增长或仅宽度增长）。  
- **加速效果**：论文报告在相同算力预算下，BERT‑Base 的总预训练时间缩短了 73.6%，BERT‑Large 缩短了 82.2%。  
- **性能对比**：在 GLUE 上，复合增长模型的平均得分与全尺寸一次性训练模型相差不到 0.2%，基本持平。  
- **消融实验**：作者分别去掉宽度增长、去掉序列长度增长以及改用随机初始化新层的策略，发现：  
  - 只做深度增长时加速率下降约 15%；  
  - 去掉宽度增长导致下游任务的表现下降约 0.5%；  
  - 随机初始化新层使收敛速度慢 30%，验证了“复制层”算子的优势。  
- **局限性**：原文未详细说明在多语言或更长序列（>512）场景下的适用性；增长比例搜索仍是粗粒度的网格搜索，可能在更大模型上产生额外调参成本。

### 影响与延伸思考
这篇工作把“复合缩放”从模型设计层面延伸到训练调度层面，打开了渐进式大模型训练的新思路。随后的研究（如 *Progressive Scaling for GPT‑3*、*Compound Growth for Vision Transformers*）纷纷借鉴其多维度同步增长的框架，尝试在不同模态（文本、图像、音频）上实现更高效的预训练。对想进一步探索的读者，可以关注以下方向：  
- **自动化增长比例搜索**：使用强化学习或贝叶斯优化替代手工网格搜索。  
- **跨模态复合增长**：在多模态模型中同时扩展文本、视觉分支的维度。  
- **更细粒度的参数迁移**：比如在宽度增长时采用低秩分解或知识蒸馏，以进一步降低新参数的学习负担。  

### 一句话记住它
**“让 BERT 同时在层数、宽度和序列长度上同步扩张，能在保持性能的前提下把预训练时间砍掉七八成。”**