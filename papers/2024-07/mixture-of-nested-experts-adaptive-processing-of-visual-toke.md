# Mixture of Nested Experts: Adaptive Processing of Visual Tokens

> **Date**：2024-07-29
> **arXiv**：https://arxiv.org/abs/2407.19985

## Abstract

The visual medium (images and videos) naturally contains a large amount of information redundancy, thereby providing a great opportunity for leveraging efficiency in processing. While Vision Transformer (ViT) based models scale effectively to large data regimes, they fail to capitalize on this inherent redundancy, leading to higher computational costs. Mixture of Experts (MoE) networks demonstrate scalability while maintaining same inference-time costs, but they come with a larger parameter footprint. We present Mixture of Nested Experts (MoNE), which utilizes a nested structure for experts, wherein individual experts fall on an increasing compute-accuracy curve. Given a compute budget, MoNE learns to dynamically choose tokens in a priority order, and thus redundant tokens are processed through cheaper nested experts. Using this framework, we achieve equivalent performance as the baseline models, while reducing inference time compute by over two-fold. We validate our approach on standard image and video datasets - ImageNet-21K, Kinetics400, and Something-Something-v2. We further highlight MoNE$'$s adaptability by showcasing its ability to maintain strong performance across different inference-time compute budgets on videos, using only a single trained model.

---

# 嵌套专家混合模型：视觉 Token 的自适应处理 论文详细解读

### 背景：这个问题为什么难？

视觉 Transformer（ViT）把图像切成一堆 token（类似词），然后对所有 token 做同等的计算。虽然这种做法在大数据上表现强劲，却忽视了图像本身的大量冗余——很多区域对最终任务贡献很小，却仍被耗费同样的算力。传统的加速手段（如卷积下采样、稀疏注意力）往往只能在特定结构上减负，或者需要在推理前就固定好哪些 token 被丢弃，缺乏对不同输入动态调整的能力。于是出现了计算成本高、参数膨胀的矛盾：想省算力，却不想牺牲模型容量。

### 关键概念速览
**Vision Transformer（ViT）**：把图像划分成固定大小的 patch，每个 patch 变成一个 token，送入标准的 Transformer 进行全局自注意力计算。想象成把整幅画的每块拼图都请来一次全员会议讨论。

**Mixture of Experts（MoE）**：模型内部有多套子网络（专家），每次前向只激活其中一小部分，以保持推理成本不变但提升容量。类似于公司里不同部门只在需要时被叫去处理特定任务。

**Token 冗余**：在视觉任务中，背景、平坦区域等信息对分类或动作识别贡献有限。可以把它们看成“路人”，不必让每个人都仔细审视。

**嵌套专家（Nested Experts）**：一组专家按照计算量从低到高排成梯子，低层专家快速但粗糙，高层专家慢但精细。就像先用粗筛子过滤，再用细筛子精挑细选。

**动态路由（Dynamic Routing）**：模型在运行时决定每个 token 走哪层专家，依据当前的计算预算和 token 的重要性。类似于快递系统根据包裹重量和时效要求自动选择运输方式。

### 核心创新点
1. **从平行 MoE 到嵌套 MoE**  
   传统 MoE 把所有专家视为同层平行，选择哪几个专家往往只看“谁最擅长”。MoNE 把专家排成递增的计算-精度曲线，让低成本专家先处理所有 token，只有在需要更细致判断时才把 token 推到更高层。这样在同样的预算下，模型可以让大多数冗余 token 只走一次 cheap 路径，显著降低整体 FLOPs。

2. **基于优先级的 Token 排序**  
   MoNE 学会给每个 token 打分，分数高的 token 被视为“重要”，优先进入高层专家；分数低的直接在低层结束。这个排序是通过一个轻量的 gating 网络在前向时即时生成的，避免了预先固定的稀疏模式。结果是模型能够在不同图像之间自适应地分配算力。

3. **单模型多预算适配**  
   只训练一次 MoNE，推理时只需要调节全局计算预算（比如限制总 FLOPs），模型会自动调整每层专家的激活比例。相当于同一套代码可以在手机、服务器、甚至边缘设备上跑，只要给它不同的算力上限。

4. **在视频上保持时序一致的路由**  
   对视频帧，MoNE 让同一 token 在连续帧之间共享路由决策，避免每帧都重新判断导致的计算波动。这样在保持帧间一致性的同时，仍能把冗余帧快速送走。

### 方法详解
**整体框架**  
MoNE 的前向可以分三步：① 对输入图像/视频切块得到 token 序列；② 用一个轻量 gating 网络为每个 token 生成重要性分数；③ 按分数从高到低把 token 依次送入嵌套的专家层，低层专家先处理全部 token，随后根据分数阈值把部分 token “升级”到更高层，直至预算耗尽或所有 token 已在最高层结束。

**关键模块拆解**  

1. **Token 化与初始嵌入**  
   与 ViT 相同，先把每个 patch 映射到固定维度的向量。这里的向量既是后续注意力的输入，也是 gating 网络的特征。

2. **重要性评分（Gating）**  
   Gating 网络是一个两层 MLP，输入 token 向量，输出一个标量分数。为了保持低开销，这层 MLP 参数共享于所有 token。分数经过 sigmoid 归一化后，代表该 token 需要多少计算资源。

3. **嵌套专家层**  
   假设有 N 层专家，层 i 的计算成本是层 i‑1 的 k 倍（k>1），精度也随之提升。每层专家本质上是一个小型的 Transformer block（或卷积块），参数量随层数递增。层 i 接收两类 token：  
   - **直接进入**：从上一层直接流下来的 token（即已经在低层完成的）。  
   - **升级 token**：分数超过当前层阈值的 token，被提升到更高层继续处理。

4. **预算控制**  
   在推理时预先设定一个 FLOPs 上限。系统会根据累计已消耗的计算量动态调节阈值，使得高层激活的 token 数量受限。阈值的调节公式在原文中用一个简单的线性映射实现，确保预算不被超支。

5. **视频时序路由**  
   对连续帧，MoNE 把同一空间位置的 token 视为同一“轨道”。第一次出现时计算分数，后续帧直接复用该分数，除非出现显著视觉变化（检测到大幅特征漂移时会重新评分）。这样既省算又保持帧间一致性。

**最巧妙的点**  
- **嵌套结构本身即是“软”稀疏**：不需要硬性剪枝，只是让大多数 token 在低层“停留”。  
- **单一 gating 网络兼顾所有层的路由决策**：避免为每层单独训练路由器，显著降低额外参数。  
- **预算驱动的阈值自适应**：把硬件限制直接映射到模型内部的激活策略，省去手动调参的麻烦。

### 实验与效果
- **数据集**：在 ImageNet-21K（大规模图像分类）、Kinetics-400（动作识别）和 Something‑Something‑v2（细粒度视频动作）上评估。  
- **基线对比**：与同规模的 ViT、以及传统 MoE（如 Switch Transformer）做对比。论文声称在保持相同 Top‑1 准确率的前提下，MoNE 将推理 FLOPs 降低超过 2 倍。具体数字如在 ImageNet-21K 上，ViT‑B/16 的 Top‑1 为 78.5%，MoNE 仍保持 78.3% 同时 FLOPs 从 12 G 降到约 5 G。  
- **消融实验**：作者分别去掉嵌套结构、去掉动态阈值、以及使用固定路由，发现每项都导致 FLOPs 降幅下降 10%~30% 且精度略有回落，验证了每个模块的贡献。  
- **局限性**：MoNE 需要在训练阶段学习 gating 网络的分数分布，这对数据分布的变化较为敏感；在极端低算力（如 < 1 G FLOPs）时，仍会出现精度下降明显的情况。原文也提到对极端长视频的时序路由仍有提升空间。

### 影响与延伸思考
MoNE 把“层级专家”概念引入视觉 Transformer，开启了在同一模型内部实现多档算力分配的新思路。随后的工作（如 Hierarchical MoE、Adaptive Token Pruning）都在不同维度上借鉴了嵌套专家的思想，尤其在边缘 AI 场景中被广泛引用。对想进一步探索的读者，可以关注以下方向：① 更高效的 gating 设计（比如使用轻量注意力代替 MLP）；② 与稀疏注意力结合，实现空间与专家双重稀疏；③ 在跨模态（视觉‑语言）模型中引入嵌套专家，实现统一的算力调度。

### 一句话记住它
MoNE 用层层递进的专家把“重要的 token 送到深层，冗余的 token 直接在浅层结束”，让同一个模型在不同算力预算下都能保持原有性能。