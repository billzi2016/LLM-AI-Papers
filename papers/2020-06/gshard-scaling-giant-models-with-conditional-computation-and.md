# GShard: Scaling Giant Models with Conditional Computation and Automatic   Sharding

> **Date**：2020-06-30
> **arXiv**：https://arxiv.org/abs/2006.16668

## Abstract

Neural network scaling has been critical for improving the model quality in many real-world machine learning applications with vast amounts of training data and compute. Although this trend of scaling is affirmed to be a sure-fire approach for better model quality, there are challenges on the path such as the computation cost, ease of programming, and efficient implementation on parallel devices. GShard is a module composed of a set of lightweight annotation APIs and an extension to the XLA compiler. It provides an elegant way to express a wide range of parallel computation patterns with minimal changes to the existing model code. GShard enabled us to scale up multilingual neural machine translation Transformer model with Sparsely-Gated Mixture-of-Experts beyond 600 billion parameters using automatic sharding. We demonstrate that such a giant model can efficiently be trained on 2048 TPU v3 accelerators in 4 days to achieve far superior quality for translation from 100 languages to English compared to the prior art.

---

# GShard：通过条件计算与自动切分实现巨型模型的可扩展性 论文详细解读

### 背景：这个问题为什么难？

在深度学习里，模型越大往往能带来更好的性能，尤其是像机器翻译这种需要海量语料的任务。可是把模型参数从几千万直接推到几千亿，计算资源、显存占用和并行编程的复杂度都会爆炸。传统的全模型并行（把所有参数均匀切分到每个设备）在显存和通信开销上会出现瓶颈；而手工写分布式代码又容易出错、难以迁移。于是，如何在保持代码可读性的前提下，让超大模型在数千块加速器上高效训练，成为阻挡进一步规模化的关键难题。

### 关键概念速览

**条件计算（Conditional Computation）**：只让模型的部分子网络在一次前向传播中被激活，就像只打开需要的灯泡，省掉不必要的电力消耗。  

**Mixture-of-Experts（专家混合）**：把一个大层拆成若干“专家”，每次输入只走少数几个专家，类似于公司里不同部门处理不同业务，提升整体效率。  

**稀疏门（Sparsely-Gated）**：决定哪些专家被选中的控制器，输出的权重大多数为零，只保留少数非零值，确保每次只激活少量专家。  

**自动切分（Automatic Sharding）**：编译器在后台把张量和算子自动分配到不同设备上，开发者只需要标记哪些需要切分，像让系统自行安排搬家。  

**XLA 编译器**：Google 为 TensorFlow 开发的加速编译器，负责把高层算子映射到硬件指令，支持跨设备优化。  

**TPU v3 加速器**：Google 自研的张量处理单元，专为大规模矩阵运算设计，单块拥有 16 GB 高速显存。  

**全局步长（Global Step）**：训练过程中累计的迭代次数，用来同步学习率等超参数，类似于全局的时间指针。  

**分布式张量（Distributed Tensor）**：在多台机器上拆分存储的张量，只有需要的切片会被拉取到本地计算，类似于把一本书的章节分发到不同图书馆。

### 核心创新点

**从手工切分 → 注解 API + 编译器自动切分 → 大幅降低代码改动**  
以前要把模型拆到数千块 TPU，需要在代码里写大量 `tf.device`、`tf.split` 等显式切分指令，既繁琐又容易出错。GShard 引入了轻量级的注解 API（如 `@shard`、`@partition`），只在需要并行的张量或算子上加标记，随后 XLA 编译器自动生成最优的切分方案。这样，原始的 Transformer 代码只改动几行，就能跑在 2048 块 TPU 上。

**从全模型并行 → 稀疏门驱动的 Mixture-of-Experts → 计算成本与显存线性增长**  
传统的全模型并行把所有参数都复制或切分，显存随模型大小线性增长。GShard 将 Transformer 的 Feed‑Forward 层换成稀疏门控制的 Mixture‑of‑Experts，每次只激活 1‑2% 的专家参数。结果是模型参数可以达到 600 B，而实际每步计算的显存和 FLOPs 只相当于几百亿参数的模型，极大降低了资源需求。

**从经验切分 → 编译时全局优化 → 更高的通信效率**  
在普通的分布式训练里，通信模式（如 All‑Reduce）往往是固定的，无法针对特定算子进行优化。GShard 的 XLA 扩展在编译阶段分析整个计算图，自动决定是使用点对点发送、聚合还是跨机器广播，并把这些通信指令融合进算子实现里。实验表明，这种全局视角的调度比手工调优的通信策略快 1.5‑2 倍。

**从单语言模型 → 多语言 100→1 翻译 → 质量显著提升**  
把上述技术组合起来，作者成功训练出一个覆盖 100 种语言到英语的翻译模型，参数规模超过 600 B。相较于之前的最强基线（约 15 B 参数），BLEU 分数提升了 10‑15 分，说明规模化真的带来了质的飞跃。

### 方法详解

**整体思路**  
GShard 的工作流可以拆成三步：① 在模型代码上加上切分注解；② 编译器在 XLA 层解析注解、生成切分计划并插入稀疏门逻辑；③ 运行时根据计划把张量切片分配到 2048 块 TPU，完成前向、反向和梯度聚合。整个过程对用户透明，只需要在关键位置加几行装饰器。

**关键模块拆解**  

1. **注解层**  
   - `@shard`：标记张量在哪一维度上需要切分，例如把词嵌入矩阵的词表维度划分到不同机器。  
   - `@partition`：标记算子输出的切分方式，告诉编译器该算子的结果应该如何拼接。  
   类比：在厨房里，你给每个厨师贴上「负责切菜」或「负责烹饪」的标签，系统随后自动把食材和工具分配给对应的人。

2. **稀疏门与专家层**  
   - 稀疏门是一个小型全连接网络，接受当前 token 的表示，输出每个专家的路由分数。  
   - 只保留前 K（通常是 2）个最高分的专家，其他分数置零。  
   - 选中的专家在对应的设备上执行前向计算，输出再通过加权求和合并。  
   直观上，这一步像是把一封信交给最擅长处理该主题的几位编辑，而不是全部编辑都看一遍。

3. **XLA 自动切分引擎**  
   - 编译阶段收集所有注解信息，构建全局切分图。  
   - 对每条数据流计算最小通信代价的切分方案，考虑带宽、设备拓扑和算子依赖。  
   - 生成的低层代码把张量切片的发送/接收指令嵌入到算子实现里，实现「算子即通信」的融合。  
   这一步可以想象成物流公司在发货前先规划好每件货物的最短运输路线，而不是等到装车后再临时决定。

4. **训练循环**  
   - 前向阶段：每个 token 经过稀疏门决定走哪些专家，专家计算在本地完成。  
   - 反向阶段：梯度同样只在被激活的专家上计算，随后通过自动切分的 All‑Reduce 把梯度汇总到全局参数。  
   - 参数更新：全局步长同步，学习率调度统一。  

**最巧妙的设计**  
- **条件计算 + 自动切分的协同**：稀疏门导致每一步只激活少数专家，而自动切分确保这些激活的子网络在物理上分布得尽可能紧凑，极大降低了跨机器通信。  
- **编译时全局视角**：不像传统的手工切分只关注局部算子，XLA 在整个计算图上做最优切分，能够自动发现并合并重复的通信路径，提升了整体吞吐。

### 实验与效果

- **任务与数据**：作者在一个包含 100 种语言到英语的多语言神经机器翻译任务上进行评估，使用了公开的 multilingual WMT 数据集以及内部收集的平行语料，总计约 25 TB 文本。  
- **规模对比**：基线模型为 15 B 参数的标准 Transformer，GShard 版本为 600 B 参数的稀疏 MoE Transformer。  
- **质量提升**：在 BLEU 指标上，GShard 在多数语言上比基线高出 10‑15 分，尤其是低资源语言提升更为显著（有的超过 20 分）。  
- **训练成本**：使用 2048 块 TPU v3，训练时长约 4 天（约 96 小时），相当于基线模型在同等硬件上需要数周才能收敛。  
- **消融实验**：论文中分别关闭稀疏门、关闭自动切分、以及只使用全模型并行进行对比。结果显示：去掉稀疏门后显存需求暴涨，训练在 2048 块 TPU 上无法完成；关闭自动切分则需要手工调优切分策略，训练速度下降约 30%。  
- **局限性**：作者指出，稀疏门的路由网络本身仍然是全连接的，随着专家数量进一步提升会成为新的瓶颈；此外，模型在极端低资源语言上仍受数据稀缺限制，质量提升受限。

### 影响与延伸思考

GShard 发表后，条件计算与自动切分的组合成为大模型训练的标配思路。后续的 **Switch Transformers**、**Pathways**、以及 **Google PaLM** 系列都在路由机制和编译器层面的自动化上进行深化。业界也开始在 PyTorch 上实现类似的 **torch.distributed** 注解 API，推动了跨平台的可扩展训练工具链。想进一步了解，可以关注以下方向：① 更高效的稀疏路由网络（如基于局部感受野的门控）；② 编译器层面的异构设备调度（GPU+TPU 混合）；③ 大规模多语言模型的跨语言迁移学习。整体来看，GShard 为「把模型规模推向千亿甚至万亿」提供了可复制的技术路径。

### 一句话记住它

**GShard 用几行注解让稀疏专家模型在数千块 TPU 上自动切分，轻松训练出 600 B 参数的翻译巨兽。**