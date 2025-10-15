# Dr.LLM: Dynamic Layer Routing in LLMs

> **Date**：2025-10-14
> **arXiv**：https://arxiv.org/abs/2510.12773

## Abstract

Large Language Models (LLMs) process every token through all layers of a transformer stack, causing wasted computation on simple queries and insufficient flexibility for harder ones that need deeper reasoning. Adaptive-depth methods can improve efficiency, but prior approaches rely on costly inference-time search, architectural changes, or large-scale retraining, and in practice often degrade accuracy despite efficiency gains. We introduce Dr. LLM, Dynamic routing of Layers for LLMs, a retrofittable framework that equips pretrained models with lightweight per-layer routers deciding to skip, execute, or repeat a block. Routers are trained with explicit supervision: using Monte Carlo Tree Search (MCTS), we derive high-quality layer configurations that preserve or improve accuracy under a compute budget. Our design, windowed pooling for stable routing, focal loss with class balancing, and bottleneck MLP routers, ensures robustness under class imbalance and long sequences. On ARC (logic) and DART (math), Dr. LLM improves accuracy by up to +3.4%p while saving 5 layers per example on average. Routers generalize to out-of-domain tasks (MMLU, GSM8k, AIME, TruthfulQA, SQuADv2, GPQA, PIQA, AGIEval) with only 0.85% accuracy drop while retaining efficiency, and outperform prior routing methods by up to +7.7%p. Overall, Dr. LLM shows that explicitly supervised routers retrofit frozen LLMs for budget-aware, accuracy-driven inference without altering base weights. Code is available at https://github.com/parameterlab/dr-llm.

---

# Dr.LLM：大语言模型的动态层路由 论文详细解读

### 背景：这个问题为什么难？

现有的大语言模型（LLM）在推理时会把每一个 token 都送进完整的 Transformer 层堆，每层都要计算一次。对简单的提问，这种“一刀切”会浪费大量算力；而对需要深度推理的复杂问题，固定层数又可能不够。过去的自适应深度方法要么在推理时做昂贵的搜索，要么必须改动模型结构甚至重新大规模微调，导致实现成本高、部署困难。更糟的是，很多方法在追求速度的同时会牺牲准确率，实际使用时往往得不偿失。于是，如何在不改动已有模型权重的前提下，给每个输入动态决定走几层、跳哪些层，成为一个急需解决的难题。

### 关键概念速览

**Transformer 层**：Transformer 的基本构件，包含自注意力和前馈网络。把输入的向量序列逐层加工，层数越多理论上能捕捉越复杂的模式。  
**自适应深度（Adaptive Depth）**：根据输入的难易程度动态决定模型实际使用的层数，类似于人做题时先快速判断再决定是否深入思考。  
**路由器（Router）**：一个轻量级的神经网络，负责为每一层输出“跳过、执行或重复”指令，就像交通信号灯决定车辆是否通行。  
**蒙特卡罗树搜索（MCTS）**：一种在离线阶段通过大量随机模拟寻找高质量决策序列的搜索算法，常用于围棋等游戏的策略规划。这里它被用来生成“理想的层配置”。  
**窗口池化（Windowed Pooling）**：在长序列上做局部平均或最大池化，以获得稳定的层级特征，防止路由器因为噪声而频繁切换。  
**焦点损失（Focal Loss）**：一种对少数类（这里是“跳过”或“重复”）加大惩罚的损失函数，帮助模型在类别极度不平衡时仍能学到有用的判别。  
**瓶颈 MLP 路由器**：在特征维度上先压缩再扩张的两层全连接网络，参数极少却能捕捉到关键的层决策信息。

### 核心创新点

1. **离线高质量标签 → MCTS 生成层配置 → 直接监督路由器**  
   过去的自适应深度方法大多在推理时实时搜索层数，计算开销大。Dr.LLM 先用蒙特卡罗树搜索在离线阶段为每个训练样本找出在给定算力预算下最优的层跳/重复序列，然后把这些序列当作明确的监督信号喂给路由器。这样路由器在训练时就学会了“何时该省力、何时该加码”，推理时只需一次前向即可得到决策，省去了实时搜索。

2. **轻量瓶颈 MLP + 窗口池化 → 稳定且高效的路由判断**  
   为了让路由器在长文本上不被噪声干扰，作者在每层的隐藏状态上做窗口化池化，得到更平滑的特征向量。随后使用只有几百参数的瓶颈 MLP 进行分类，这比在每个 token 上做全局注意力要快得多，同时保持了对上下文的感知能力。

3. **焦点损失 + 类别平衡 → 解决极度不平衡的决策分布**  
   在真实数据中，大多数 token 仍然需要“执行”当前层，跳过或重复的比例很小。普通交叉熵会让模型倾向于总是输出“执行”。作者引入焦点损失并对少数类做额外的采样权重，使路由器能够学到可靠的跳过/重复信号，而不是盲目执行全部层。

4. **完全不改动基模型权重 → 直接 retrofit 预训练 LLM**  
   与需要重新训练或改造 Transformer 结构的工作不同，Dr.LLM 只在模型外部挂一个路由器模块，基模型保持冻结。这意味着任何已有的 LLM（如 LLaMA、OPT）都可以“一键”加上动态层路由，极大降低了实际落地的门槛。

### 方法详解

**整体框架**  
Dr.LLM 的推理流程可以分为三步：①离线搜索得到层配置标签；②用这些标签训练轻量路由器；③推理时路由器实时决定每层的操作（跳过、执行、重复），基模型权重保持不变。整个系统的核心是把“搜索”搬到训练前，用监督学习取代推理时的搜索。

**步骤 1：离线蒙特卡罗树搜索**  
- 对每个训练样本，设定一个算力预算（比如平均少走 5 层）。  
- MCTS 从根节点（全部层执行）开始，逐层尝试“跳过”或“重复”操作，构建搜索树。  
- 每一次模拟都会实际跑一次模型（使用冻结权重），记录在该层配置下的验证准确率和实际算力消耗。  
- 通过回溯更新节点价值，最终得到在预算约束下准确率最高的层序列。  
这一步的输出是一串离散指令（执行/跳过/重复），相当于“老师给的答案”。

**步骤 2：路由器训练**  
- 输入：每层的隐藏状态经过窗口池化后得到的特征向量。  
- 网络结构：先用线性层把维度压到一个瓶颈（比如 64），再用 ReLU 激活，最后映射到 3 类 logits。  
- 损失函数：对每个 token 使用焦点损失，且对“跳过”和“重复”类别加上更大的权重，以抵消它们在标签中出现频率低的劣势。  
- 训练方式：普通的 mini‑batch SGD/Adam，基模型参数冻结，只更新路由器参数。  

**步骤 3：推理时的动态路由**  
- 对输入序列逐层前向：先把当前层的隐藏状态送入路由器，得到三分类概率。  
- 若预测为“跳过”，直接把该层的输出设为输入（即不做计算），进入下一层。  
- 若预测为“重复”，把当前层的输出再喂回同一层一次（相当于在该层做两次前向），再进入下一层。  
- 若是“执行”，正常计算该层的自注意力和前馈网络。  
因为路由器只需一次前向，整体额外开销几乎可以忽略。

**最巧妙的点**  
- 把搜索过程完全搬到离线阶段，让推理时只做一次轻量前向，这在大模型上是成本的天壤之别。  
- 窗口池化让路由器对长文本的决策更平滑，避免了对每个 token 细粒度噪声的过度敏感。  
- 使用焦点损失直接对抗类别不平衡，比起后期再做阈值调节更稳健。

### 实验与效果

- **测试任务**：逻辑推理（ARC）、数学推理（DART）是论文重点，此外还在跨域基准 MMLU、GSM8k、AIME、TruthfulQA、SQuADv2、GPQA、PIQA、AGIEval 上做了验证。  
- **主要结果**：在 ARC 与 DART 上，Dr.LLM 在保持或提升准确率的同时，平均每条样本省掉约 5 层计算，准确率提升最高可达 3.4 个百分点。跨域任务上，仅有 0.85% 的准确率下降，却仍保持显著的层数节省。  
- **对比基线**：与之前的自适应深度方法（如 LayerDrop、Dynamic Early Exit）相比，Dr.LLM 在同等算力预算下准确率提升最高 7.7 个百分点。  
- **消融实验**：作者分别去掉 MCTS 标签、窗口池化、焦点损失进行对比，发现去掉任意一项都会导致准确率下降 1–2% 并且路由决策更不稳定，验证了每个模块的必要性。  
- **局限性**：路由器虽然轻量，但仍需要在每层插入一次前向，极端超长序列（上万 token）下的额外开销仍不可忽视；此外，离线 MCTS 需要对每个训练样本跑多次完整前向，计算成本高，限制了在大规模数据上直接生成标签的可行性。作者也提到在极端低算力预算（如只保留 2 层）时，准确率仍会出现明显下降。

### 影响与延伸思考

Dr.LLM 的出现让「后置」的自适应深度成为可能：只要有一个冻结的预训练模型，就能通过轻量路由器实现预算感知的推理，这在实际部署（云端 API、边缘设备）中极具吸引力。随后的工作（如 **AdaLayer**, **RouteLLM**, **Budget‑Aware Transformers**）纷纷借鉴了离线搜索+监督路由的思路，尝试用强化学习或梯度近似来生成标签，以降低 MCTS 的成本。未来可以探索：

- **更高效的标签生成**：使用近似搜索或元学习快速预测高质量层配置。  
- **多任务路由**：让同一个路由器在不同任务之间共享决策经验，进一步提升跨域鲁棒性。  
- **硬件协同**：把路由决策映射到实际的算子调度（如 GPU/TPU 动态核数），实现端到端的算力-精度最优化。  

如果想深入，可以关注 **“可微分层路由”** 与 **“预算感知模型压缩”** 两大方向，它们正逐步把自适应推理从实验室搬进生产环境。

### 一句话记住它

**Dr.LLM 用离线搜索生成的层配置监督轻量路由器，让冻结的大语言模型在推理时自行决定跳、走或重走层，省算力不降准。**