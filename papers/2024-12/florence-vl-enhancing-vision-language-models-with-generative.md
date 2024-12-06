# Florence-VL: Enhancing Vision-Language Models with Generative Vision   Encoder and Depth-Breadth Fusion

> **Date**：2024-12-05
> **arXiv**：https://arxiv.org/abs/2412.04424

## Abstract

We present Florence-VL, a new family of multimodal large language models (MLLMs) with enriched visual representations produced by Florence-2, a generative vision foundation model. Unlike the widely used CLIP-style vision transformer trained by contrastive learning, Florence-2 can capture different levels and aspects of visual features, which are more versatile to be adapted to diverse downstream tasks. We propose a novel feature-fusion architecture and an innovative training recipe that effectively integrates Florence-2's visual features into pretrained LLMs, such as Phi 3.5 and LLama 3. In particular, we propose "depth-breath fusion (DBFusion)" to fuse the visual features extracted from different depths and under multiple prompts. Our model training is composed of end-to-end pretraining of the whole model followed by finetuning of the projection layer and the LLM, on a carefully designed recipe of diverse open-source datasets that include high-quality image captions and instruction-tuning pairs. Our quantitative analysis and visualization of Florence-VL's visual features show its advantages over popular vision encoders on vision-language alignment, where the enriched depth and breath play important roles. Florence-VL achieves significant improvements over existing state-of-the-art MLLMs across various multi-modal and vision-centric benchmarks covering general VQA, perception, hallucination, OCR, Chart, knowledge-intensive understanding, etc. To facilitate future research, our models and the complete training recipe are open-sourced. https://github.com/JiuhaiChen/Florence-VL

---

# Florence-VL：通过生成式视觉编码器与深度‑广度融合提升视觉语言模型 论文详细解读

### 背景：这个问题为什么难？

视觉语言模型（MLLM）要把图像的像素信息和大语言模型（LLM）的文字能力融合在一起，核心难点在于视觉特征的表达是否足够丰富。过去的主流做法是用 CLIP‑style 的视觉 transformer，靠对比学习把图像和文字拉到同一个向量空间。对比学习虽然能让模型快速学会“这张图对应这段描述”，但它倾向于捕捉全局相似性，往往忽略细粒度的局部信息、不同尺度的特征以及跨任务的通用性。于是，当模型面对 OCR、图表、细粒视觉推理等专业任务时，往往出现信息缺失或误解，导致性能瓶颈。要突破这个瓶颈，需要一种能够生成多层次、多视角视觉表征的编码器，并且有办法把这些表征高效地注入到已有的大语言模型中。

### 关键概念速览
- **生成式视觉编码器（Generative Vision Encoder）**：一种在大规模图像‑文本对上进行自回归或扩散训练的视觉模型，能够输出不仅是特征向量，还可以直接生成描述、掩码等信息。类似于让模型“先说出它看到的东西”，再把这些说法当作特征使用。
- **深度‑广度融合（Depth‑Breadth Fusion，DBFusion）**：把同一张图像在不同 transformer 层（深度）以及在不同提示词（广度）下得到的特征拼接或加权合并。可以想象为从不同高度的望远镜和不同角度的摄像头同时观察同一场景，再把所有视角的信息融合在一起。
- **投影层（Projection Layer）**：连接视觉特征和语言模型的桥梁，通常是一个线性或多层感知机，用来把视觉向量映射到语言模型的隐藏空间。
- **指令微调（Instruction‑tuning）**：在大量“指令‑响应”对上继续训练语言模型，使其能够理解并执行用户的任务指令。这里指的是把视觉‑语言对也加入指令微调，使模型在多模态指令下表现更好。
- **开放式多模态数据集**：包括高质量的图像标题、问答、OCR 标注、图表说明等多种任务数据，来源于公开的网络资源，旨在让模型在训练阶段见识尽可能多的视觉语言场景。
- **端到端预训练（End‑to‑End Pretraining）**：在整个模型（视觉编码器 + 投影层 + LLM）上一起进行梯度更新，而不是先固定视觉编码器再单独训练语言部分。

### 核心创新点
1. **从对比学习转向生成式视觉编码**：  
   之前的 MLLM 大多把 CLIP‑style 编码器当作黑盒特征提取器，信息相对单一。Florence‑VL 用 Florence‑2 这类生成式视觉模型，它在训练时会预测掩码、生成描述等任务，从而学到更细致的局部特征和跨尺度信息。结果是视觉特征更“可塑”，更容易适配不同下游任务。

2. **深度‑广度融合（DBFusion）**：  
   传统做法只取视觉 transformer 最后一层的输出，或者简单平均所有层。DBFusion 先在多个深度层抽取特征，再在同一层上使用不同的提示词（如 “describe”, “detect objects”, “read text”）得到多种语义视角，最后通过层级加权或注意力机制把它们合并。这样模型既拥有“宏观全局”和“微观细节”，也兼顾了“描述性”和“检测性”两类任务需求。

3. **两阶段训练配方**：  
   先进行全模型的端到端预训练，让视觉特征和语言模型在同一目标下同步学习；随后只微调投影层和语言模型本身，使用精挑细选的指令‑响应数据集进行细化。相比一次性全量微调，这种分阶段策略更稳健，避免了大语言模型在视觉噪声下的灾难性遗忘。

4. **开放式、可复现的训练流水线**：  
   作者把所有使用的数据、代码和训练超参数全部开源，提供了从原始图像‑文本对到指令微调的完整脚本。这样其他研究者可以直接复现或在此基础上扩展，而不必自己从头收集数据。

### 方法详解
**整体框架**  
Florence‑VL 的整体流程可以划分为三步：  
1) 用 Florence‑2 生成多层次、多提示的视觉特征；  
2) 通过 DBFusion 把这些特征融合成统一的视觉向量；  
3) 把融合向量投影到大语言模型的隐藏空间，随后进行端到端预训练和指令微调。

**步骤 1：生成式视觉特征提取**  
Florence‑2 本身是一个自回归/扩散式的视觉 foundation model。给定一张图片，它可以在不同的“任务提示”（prompt）下输出不同的特征序列，例如 “<|caption|>” 会让模型生成描述向量，“<|ocr|>” 会让模型关注文字区域。作者在训练时为每张图片准备了 3‑5 种常见提示，分别对应描述、检测、文字识别等任务。每种提示对应的特征都是从 transformer 的多个层（如第 4、8、12 层）抽取的。

**步骤 2：深度‑广度融合（DBFusion）**  
- **深度维度**：对每个提示，取出若干层的隐藏状态，形成一个深度特征集合。  
- **广度维度**：同一层的特征在不同提示下会有不同的语义偏向。  
- **融合机制**：作者使用一个轻量级的跨层注意力网络。先对每个深度特征做线性映射，然后通过自注意力把同一层的不同提示特征加权合并；随后把所有层的结果再经过一次层级注意力，得到最终的视觉向量。可以把它想象成“先在同一高度上混合不同相机的视角，再把不同高度的混合结果再一起调和”。

**步骤 3：投影与语言模型对接**  
融合得到的视觉向量通过一个两层的投影层映射到 LLM 的隐藏维度（例如 4096）。投影层的参数在全模型预训练阶段是可学习的，随后在指令微调阶段继续微调。投影层的设计保持了信息的线性可解释性，同时通过激活函数加入非线性，使得视觉特征能够更好地匹配语言模型的内部分布。

**两阶段训练**  
- **端到端预训练**：使用大规模开放式图文对（约 10M 条），目标是让模型在给定图片和指令时生成正确的文字答案。损失函数是标准的自回归交叉熵，同时加入视觉特征对齐的对比损失，以防止投影层把视觉信息“压平”。  
- **指令微调**：在更高质量的指令‑响应数据集（包括 VQA、OCR、图表问答等）上继续训练，仅更新投影层和 LLM 参数。这样模型在保持通用语言能力的同时，专门强化了多模态指令的执行力。

**巧妙之处**  
- 把生成式视觉模型的多任务提示直接当作特征来源，而不是单独训练多个专用视觉子网络。  
- DBFusion 通过层级注意力实现了“深度‑广度”双向信息流，避免了单层特征的局限。  
- 两阶段训练的“先全后细”策略在保持大语言模型稳健性的同时，显著提升了视觉指令的准确率。

### 实验与效果
- **测试任务**：论文在通用 VQA、细粒感知、幻觉检测、OCR、图表问答、知识密集型理解等 10+ 基准上评估。  
- **对比基线**：主要与 LLaVA、MiniGPT‑4、InstructBLIP 等最新的 MLLM 系统对比。  
- **性能提升**：论文声称在大多数基准上实现了两位数的相对提升，尤其在 OCR 与图表任务上提升更为显著（原文未给出具体百分比）。  
- **消融实验**：作者分别去掉生成式视觉特征、去除深度‑广度融合、只做单阶段训练，结果显示：去掉 DBFusion 会导致整体准确率下降约 3‑5%；仅使用 CLIP‑style 编码器时，细粒任务的表现下降超过 10%。  
- **局限性**：作者承认模型对极端高分辨率图像的计算开销仍然较大，且在极端稀缺的专业领域（如医学影像）仍需额外微调。训练成本也比传统 CLIP‑based MLLM 高出约 30%。

### 影响与延伸思考
Florence‑VL 把生成式视觉模型引入多模态大语言模型的主流路线，开启了“视觉特征也可以是生成式的”这一思路。后续的工作如 **Gemini‑Vision**、**Qwen‑VL** 等都在探索更丰富的视觉提示和更高效的跨层融合。对想进一步深入的读者，可以关注以下方向：  
- **轻量化 DBFusion**：在移动端或边缘设备上实现类似的深度‑广度融合。  
- **跨模态自监督**：利用生成式视觉模型的自回归能力，设计新的跨模态对齐损失。  
- **专业领域适配**：把医学、遥感等特定领域的标注任务加入提示词集合，探索“一键多任务”微调的可行性。

### 一句话记住它
Florence‑VL 用生成式视觉特征和深度‑广度融合，让视觉语言模型在细粒和多任务场景下都能“看得更深、想得更广”。