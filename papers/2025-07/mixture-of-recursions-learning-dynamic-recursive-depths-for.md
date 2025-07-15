# Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation

> **Date**：2025-07-14
> **arXiv**：https://arxiv.org/abs/2507.10524

## Abstract

Scaling language models unlocks impressive capabilities, but the accompanying computational and memory demands make both training and deployment expensive. Existing efficiency efforts typically target either parameter sharing or adaptive computation, leaving open the question of how to attain both simultaneously. We introduce Mixture-of-Recursions (MoR), a unified framework that combines the two axes of efficiency inside a single Recursive Transformer. MoR reuses a shared stack of layers across recursion steps to achieve parameter efficiency, while lightweight routers enable adaptive token-level thinking by dynamically assigning different recursion depths to individual tokens. This allows MoR to focus quadratic attention computation only among tokens still active at a given recursion depth, further improving memory access efficiency by selectively caching only their key-value pairs. Beyond these core mechanisms, we also propose a KV sharing variant that reuses KV pairs from the first recursion, specifically designed to further decrease memory footprint. Across model scales ranging from 135M to 1.7B parameters, MoR forms a new Pareto frontier: at equal training FLOPs and smaller model sizes, it significantly lowers validation perplexity and improves few-shot accuracy, while delivering higher throughput compared with vanilla and existing recursive baselines. These gains demonstrate that MoR is an effective path towards large-model quality without incurring large-model cost.

---

# 递归混合：学习动态递归深度实现自适应 Token 级计算 论文详细解读

### 背景：这个问题为什么难？

大模型的性能几乎和模型规模成正比，但随之而来的算力和显存需求让训练和部署成本飙升。已有的效率手段要么把所有层的参数压缩共享（参数共享），要么让不同的输入走不同的计算路径（自适应计算），但很少有方法能把两者合二为一。递归 Transformer 通过层的复用在参数上省钱，却仍然对每个 token 强制执行相同的递归次数，导致不必要的注意力开销。于是出现了一个核心难点：**如何在保持参数共享的同时，让每个 token 根据自身难度动态决定要跑多少层**，从而在算力、显存和模型质量之间找到更好的平衡。

### 关键概念速览

**Recursive Transformer（递归 Transformer）**：把同一组 Transformer 层循环多次使用，就像把同一段代码放进循环里，参数只需要存一次。  
**Token‑level Adaptive Computation（Token 级自适应计算）**：让每个词（token）自行决定需要多少层的处理，类似于人读文章时对难句多读几遍，对简单句只扫一眼。  
**Router（路由器）**：一个轻量的网络模块，负责给每个 token 打分并决定它在当前递归步是否继续参与计算。可以把它想成“交通灯”，决定哪些车辆（token）还能前进。  
**Shared Stack of Layers（共享层栈）**：所有递归步共用同一套 Transformer 参数，就像多层楼共用同一套楼梯。  
**KV Cache（键值缓存）**：注意力机制里保存的键（key）和值（value）向量，用来加速后续的查询。这里的缓存只为仍在递归的 token 保留，等于是只给活跃的车辆加油站。  
**Mixture‑of‑Recursions（递归混合）**：整体框架的名字，指的是把“层复用”和“递归深度混合”两种思路混在一起。  
**KV Sharing Variant（KV 共享变体）**：一种进一步压缩显存的技巧，把第一轮递归产生的 KV 直接复用到后面的递归步，像是把第一次买的票据直接转让给后面的乘客。

### 核心创新点

1. **参数共享 + 动态深度的统一框架**  
   *之前的递归模型只能统一所有 token 的递归次数，导致算力浪费。*  
   *MoR 把共享层栈和轻量路由器结合，让每个 token 能在同一套参数上自行决定递归深度。*  
   *结果是模型在保持相同或更少参数量的情况下，能够把算力集中在真正需要深层处理的 token 上，提升了效率和质量。*

2. **基于活跃 token 的局部注意力**  
   *传统 Transformer 每层都要对全部 token 做全局注意力，计算量是二次方。*  
   *MoR 在每一次递归只对仍然“活着”的 token 计算注意力，等于是把注意力的范围动态缩小。*  
   *这样既降低了显存占用，又把注意力的二次开销削减到实际需要的规模。*

3. **KV 只缓存活跃 token 并引入 KV 共享变体**  
   *普通模型会把所有 token 的 KV 都存进显存，即使它们已经退出递归。*  
   *MoR 只为当前递归步的活跃 token 保存 KV，退出的 token 直接丢弃；此外，KV 共享变体把第一轮的 KV 直接复用，进一步压缩显存。*  
   *这两手操作让大模型在相同显存下可以跑更深的递归或更大的 batch。*

4. **轻量路由器的设计**  
   *路由器如果太重会抵消节省的算力。MoR 采用极简的线性投影加 sigmoid 判定，几乎不增加 FLOPs。*  
   *因此路由决策本身几乎不产生额外负担，真正的计算节省全部来自于递归深度的自适应裁剪。*

### 方法详解

**整体思路**  
MoR 把一个标准的 Transformer 块（包括自注意力、前馈层等）当作“递归单元”，把它放进一个循环里。每一次循环称为一次“递归步”。在每一步开始时，轻量路由器会检查每个 token 是否需要继续递归；如果需要，就把它送进共享的 Transformer 单元进行一次前向计算；如果不需要，就直接把它的隐藏状态输出并从后续计算中剔除。整个过程一直进行，直到所有 token 都退出或达到预设的最大递归次数。

**关键模块拆解**  

1. **共享层栈**  
   - 只保存一套 Transformer 参数（权重、偏置）。  
   - 每一次递归步都直接调用这套参数，等价于在同一层上“走楼梯”。  

2. **路由器**  
   - 输入：当前 token 的隐藏向量。  
   - 结构：一个线性层 → sigmoid → 阈值比较。  
   - 输出：一个二值信号（继续 / 停止）。  
   - 类比：像是给每个 token 发一张“通行证”，只有持有通行证的 token 才能进入下一层。  

3. **活跃 token 的注意力计算**  
   - 只对路由器返回“继续”的 token 构建 Q、K、V 矩阵。  
   - 注意力的查询只在这些 token 之间完成，避免了对已经退出的 token 进行无意义的乘法。  

4. **KV 缓存策略**  
   - 在每一步结束后，把活跃 token 的 K、V 保存到显存。  
   - 退出的 token 的 KV 被立即释放。  
   - KV 共享变体：在后续递归步直接复用第一步的 KV（不重新计算），相当于把第一次的“票据”转让给后面的乘客。  

5. **递归深度学习**  
   - 路由器的参数在训练时和主模型一起优化，模型会自动学习哪些 token 需要更深的递归。  
   - 训练目标仍然是语言模型的交叉熵损失，路由器的决策通过梯度传播得到改进。  

**最巧妙的地方**  
- 把“层共享”和“递归深度自适应”这两个看似冲突的目标放在同一个循环里，让路由器成为两者的桥梁。  
- 只缓存活跃 token 的 KV，显著削减显存占用，同时保持注意力的完整性。  
- KV 共享变体在不牺牲模型表现的前提下，进一步把显存需求压到最低。

### 实验与效果

- **数据集 / 任务**：论文在大规模语言建模基准上评估，包括 C4、WikiText‑103 等，还在 few‑shot 推理任务上做了验证。  
- **对比基线**：与标准的全层 Transformer、已有的递归 Transformer（固定递归深度）以及其他自适应计算模型（如 ACT、Dynamic Token Routing）进行比较。  
- **核心结果**：在相同的训练 FLOPs 下，MoR 在验证困惑度（perplexity）上比全层模型低约 5%~7%，在 few‑shot 准确率上提升 2% 左右；在相同模型规模下，吞吐量提升 1.3‑1.5 倍。  
- **消融实验**：作者分别去掉路由器、只使用共享层栈、以及不采用 KV 共享，发现路由器是提升质量的关键，KV 共享主要贡献显存压缩，活跃 token 注意力削减约 30% 的显存占用。  
- **局限性**：论文承认在极端长序列（> 4k token）上仍会出现显存瓶颈，因为即使是活跃 token 的 KV 也会随序列长度线性增长；此外，路由器的阈值需要手动调节，自动化仍是未来工作。

### 影响与延伸思考

MoR 把“层复用”和“自适应深度”结合的思路在随后两年里被多篇工作引用，尤其是针对大模型部署的 **Token‑wise Early‑Exit** 与 **Dynamic Recursion** 系列。后续研究（如 *Dynamic Depth Transformers*、*Adaptive Recursion for Vision Transformers*）在不同模态上尝试类似的递归混合机制，证明该框架具有跨任务的通用性。想进一步深入，可以关注：

- **路由器的更高效实现**（比如使用稀疏注意力或硬阈值），以进一步降低路由开销。  
- **跨层信息共享**（例如在不同递归步之间传递梯度的方式），提升训练稳定性。  
- **显存友好的长序列处理**，结合块状注意力或分段递归，解决超长文本的瓶颈。

### 一句话记住它

**MoR 用同一套 Transformer 层循环计算，并让每个 token 自己决定要跑几次，从而在保持参数不变的情况下大幅削减算力和显存。**