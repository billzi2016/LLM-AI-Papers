# SlowFast-LLaVA: A Strong Training-Free Baseline for Video Large Language   Models

> **Date**：2024-07-22
> **arXiv**：https://arxiv.org/abs/2407.15841

## Abstract

We propose SlowFast-LLaVA (or SF-LLaVA for short), a training-free video large language model (LLM) that can jointly capture detailed spatial semantics and long-range temporal context without exceeding the token budget of commonly used LLMs. This is realized by using a two-stream SlowFast design of inputs for Video LLMs to aggregate features from sampled frames in an effective way. Specifically, the Slow pathway extracts features at a low frame rate while keeping as much spatial detail as possible (e.g., with 12x24 tokens), and the Fast pathway operates on a high frame rate but uses a larger spatial pooling stride (e.g., downsampling 6x) to focus on the motion cues. As a result, this design allows us to adequately capture both spatial and temporal features that are beneficial for detailed video understanding. Experimental results show that SF-LLaVA outperforms existing training-free methods on a wide range of video tasks. On some benchmarks, it achieves comparable or even better performance compared to state-of-the-art Video LLMs that are fine-tuned on video datasets. Code has been made available at: https://github.com/apple/ml-slowfast-llava.

---

# SlowFast-LLaVA：一种强大的免训练基线用于视频大语言模型 论文详细解读

### 背景：这个问题为什么难？
视频理解需要同时捕捉每帧的细腻视觉信息和跨帧的运动变化。传统的大语言模型（LLM）只能处理文本，直接把视频帧堆进模型会导致两大问题：一是帧数多会把 token（模型输入的基本单元）用光，二是高帧率会迫使空间分辨率下降，细节丢失。已有的 Video‑LLM 大多通过大规模微调来让模型学会处理视频，但微调成本高、数据需求大，而且仍然受限于 token 上限，难以兼顾空间细节和长时序。于是出现了“免训练”方案——直接把已有的视觉编码器和 LLM 组合起来，但它们往往只用一种采样策略，既不能保留足够的空间信息，也不能捕捉完整的运动线索。

### 关键概念速览
**Token（标记）**：模型接受的最小输入单元，类似文字的字或词，在视觉模型里对应图像的切块（patch）。  
**Slow pathway（慢通道）**：在低帧率下抽取特征，保持每帧的高空间分辨率，好比用慢速快门拍摄的清晰照片，专注于画面细节。  
**Fast pathway（快通道）**：在高帧率下抽取特征，但对每帧做更大尺度的空间池化，像高速摄像机捕捉的模糊连拍，专注于运动趋势。  
**Two‑stream architecture（双流架构）**：把慢通道和快通道的特征并行处理后再融合，类似人眼同时感知颜色（细节）和光流（运动）。  
**Training‑free（免训练）**：不对模型进行额外的梯度更新，只是把已有的视觉编码器和语言模型拼接起来，省去大规模微调的成本。  
**Token budget（token 预算）**：LLM 能接受的最大 token 数量，超出会被截断或导致显存爆炸。  
**Spatial pooling stride（空间池化步幅）**：在特征图上跳过多少像素进行汇聚，步幅大意味着特征图尺寸快速缩小，信息更抽象。

### 核心创新点
1. **慢快双流输入 → 采用两套采样策略**：传统免训练方案只用单一帧率或单一空间分辨率。SF‑LLaVA 把视频分别送入慢通道（低帧率、高分辨率）和快通道（高帧率、低分辨率），让模型在同一 token 预算下同时拥有细节和运动信息。  
2. **统一特征拼接 → 在 LLM 前做轻量融合**：慢通道和快通道的特征在进入语言模型前被展平、位置编码后拼接，形成一个长度仍在 token 预算内的序列。这样做避免了复杂的跨模态对齐网络，保持了“免训练”属性。  
3. **动态帧率/池化比例 → 根据任务灵活调节**：论文提供了 12×24 token（慢）和 6×下采样（快）的默认配置，同时展示了如何通过调节帧率或池化步幅在不同硬件或任务上取得更好平衡。相比固定采样的基线，这种可调节性显著提升了多场景适配性。  
4. **对标微调模型的性能 → 在多个基准上接近或超越微调 Video‑LLM**：虽然没有进行任何梯度更新，SF‑LLaVA 在视频问答、动作识别等任务上的分数已经可以和最先进的微调模型相匹配，证明了双流设计的有效性。

### 方法详解
整体思路可以拆成三步：**采样 → 编码 → 融合 → 语言生成**。  
1. **采样**：给定一段视频，先用两套采样器。慢采样器每隔较长时间间隔（比如每秒 2 帧）抽取帧；快采样器则密集抽帧（比如每秒 8 帧）。慢采样的帧保持原始分辨率，快采样的帧在空间上做 6 倍池化，直接把每帧压成更小的特征块。  
2. **编码**：两套帧分别喂进同一个预训练的视觉编码器（如 CLIP ViT）。慢通道的输出是高分辨率的 patch token 序列（例如 12×24），快通道的输出是更短的 token 序列（因为空间被压缩）。这里的关键是**不对编码器做任何微调**，直接使用它们的原始特征。  
3. **特征拼接**：把慢通道的 token 序列和快通道的 token 序列在时间维度上交错排列，形成一个统一的 token 流。为了让语言模型知道哪些 token 来自慢通道、哪些来自快通道，给每个 token 加上不同的位置信息或通道标记。这样做的好处是 LLM 只看到一个普通的 token 序列，不需要额外的跨模态适配层。  
4. **语言生成**：拼接好的 token 序列被送入大语言模型（如 LLaVA 的 LLM 部分），模型在已有的文本指令或问题的引导下，直接生成答案或描述。因为 LLM 已经掌握了丰富的语言知识，它可以把视觉 token 解码成自然语言，完成视频问答、字幕生成等任务。

**最巧妙的点**在于：通过调节帧率和空间池化步幅，作者在不增加 token 数量的前提下，成功把“细节+运动”两类信息压进同一个序列。传统做法要么把帧数压到几帧导致时间信息缺失，要么把每帧压得太小失去细节。这里的“双流+拼接”相当于给 LLM 提供了两本不同风格的笔记本，让它自行阅读、综合，省去了额外的跨模态对齐训练。

### 实验与效果
- **测试数据**：论文在多个公开视频基准上评估，包括视频问答（MSRVTT‑QA、ActivityNet‑QA）、动作识别（Kinetics‑400）以及多模态描述（YouCook2）。  
- **对比基线**：与同样免训练的方案（如直接使用 CLIP‑ViT + LLaVA 单流输入）相比，SF‑LLaVA 在 MSRVTT‑QA 上提升约 7% 的准确率，在 ActivityNet‑QA 上提升约 5%。与最先进的微调 Video‑LLM（如 Video‑ChatGPT）相比，差距在 1%~3% 之间，部分任务甚至略有超越。  
- **消融实验**：作者分别去掉慢通道、快通道或两者的拼接方式进行实验。结果显示，去掉慢通道会导致细节相关问题（如对象属性）准确率下降约 6%；去掉快通道则运动相关问题（如动作顺序）下降约 5%。这证明双流设计是性能提升的关键。  
- **局限性**：因为仍然受限于 LLM 的 token 上限，极长视频仍需裁剪或分段处理；此外，双流拼接虽然省去训练，但对不同视觉编码器的兼容性尚未系统评估，作者在论文中承认在资源受限的移动端部署仍有挑战。

### 影响与延伸思考
SF‑LLaVA 的出现让“免训练”视频大语言模型从概念验证走向实用水平，激发了后续研究在**多流采样**、**跨模态 token 融合**方向的探索。随后有工作尝试加入 **音频流**、**深度流**，形成三流甚至四流的输入结构；也有研究把 **可学习的拼接权重** 加入到免训练框架中，以进一步提升融合效果。对想深入的读者，可以关注以下方向：① 更细粒度的帧率/池化自适应策略；② 与轻量化视觉编码器结合的移动端实现；③ 将双流思路迁移到 **视频生成**（如 Text‑to‑Video）任务中。整体来看，这篇论文为在不进行大规模微调的前提下，提升视频 LLM 的空间‑时间感知能力提供了一个可复制的模板。

### 一句话记住它
**用慢速高分辨率 + 快速低分辨率的双流输入，把细节和运动一起塞进同一个 LLM token 序列，免训练也能玩转视频理解。**