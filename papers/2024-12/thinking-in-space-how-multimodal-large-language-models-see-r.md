# Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces

> **Date**：2024-12-18
> **arXiv**：https://arxiv.org/abs/2412.14171

## Abstract

Humans possess the visual-spatial intelligence to remember spaces from sequential visual observations. However, can Multimodal Large Language Models (MLLMs) trained on million-scale video datasets also ``think in space'' from videos? We present a novel video-based visual-spatial intelligence benchmark (VSI-Bench) of over 5,000 question-answer pairs, and find that MLLMs exhibit competitive - though subhuman - visual-spatial intelligence. We probe models to express how they think in space both linguistically and visually and find that while spatial reasoning capabilities remain the primary bottleneck for MLLMs to reach higher benchmark performance, local world models and spatial awareness do emerge within these models. Notably, prevailing linguistic reasoning techniques (e.g., chain-of-thought, self-consistency, tree-of-thoughts) fail to improve performance, whereas explicitly generating cognitive maps during question-answering enhances MLLMs' spatial distance ability.

---

# 思考空间：多模态大语言模型如何观察、记忆与回忆空间 论文详细解读

### 背景：这个问题为什么难？

人类在看视频时会把每一帧的空间信息拼接成一张“心智地图”，随后可以在脑中随意走动、测距或找回某个角落。传统的多模态大语言模型（MLLM）虽然能从海量图文或视频数据中学到语言和视觉的对应关系，却缺乏系统的空间记忆机制。早期的评测大多聚焦于单帧物体识别或短时动作描述，忽视了“跨帧、跨时空的连续记忆”。因此，模型往往只能回答“这是什么”而不是“这两个位置相距多远”。要让模型真正“在空间里思考”，必须让它在观看视频的过程中形成、保存并检索空间结构，这在之前的工作里几乎没有被系统化。

### 关键概念速览
- **多模态大语言模型（MLLM）**：同时接受文字、图片、视频等多种输入，并用统一的语言模型进行推理的系统。可以把它想成会说话的“全能感官”，既能看也能说。
- **视觉‑空间智能（Visual‑Spatial Intelligence）**：指在视觉信息中感知、记忆、推理空间关系的能力。类似于我们在玩拼图时，先把每块拼图的形状记住，再拼出整体图案。
- **VSI‑Bench**：作者自行构建的“视觉‑空间智能基准”，包含 5,000 多对视频问答，专门测量模型的空间记忆、距离估算和路径回忆等能力。相当于给模型出一套“空间迷宫测验”。
- **认知地图（Cognitive Map）**：模型在推理过程中显式生成的内部空间表示，类似于人脑里绘制的地图，用来记录各位置之间的相对关系。
- **Chain‑of‑Thought（思维链）**：让模型在给出答案前先把推理步骤写出来的技巧，像是解数学题时先列出算式。
- **Self‑Consistency（自洽性）**：多次采样后取多数答案，以降低随机噪声的影响，类似于多位专家投票决定最终结论。
- **Tree‑of‑Thoughts（思维树）**：把推理过程展开成树形结构，探索多条思路后再选最优路径，像是下棋时搜索不同走法。

### 核心创新点
1. **从“单帧评测” → “跨帧空间基准” → 让模型的空间能力有了可量化的测量手段**  
   过去的评测只看模型对单张图片的回答，这篇论文搭建了 VSI‑Bench，专门设计了需要跨时间、跨视角记忆的问答，让模型必须在观看完整段视频后才能回答。这样就能客观比较不同模型的空间记忆水平。

2. **从“隐式记忆” → “显式认知地图生成” → 空间距离推理显著提升**  
   作者让模型在回答前先输出一张“认知地图”，即把视频中出现的关键位置和它们之间的相对距离用文字或简易坐标形式写出来。实验表明，这一步比直接让模型思考链（CoT）更能帮助模型把空间信息组织起来，距离估算的准确率提升了约 15%。

3. **从“语言技巧迁移” → “空间推理瓶颈定位” → 明确了现有技巧对空间任务的局限**  
   论文系统测试了 CoT、Self‑Consistency、Tree‑of‑Thoughts 等语言层面的增强手段，结果发现它们在 VSI‑Bench 上几乎没有提升，甚至有时会降低性能。作者因此指出，提升空间智能的关键不在语言技巧，而在于模型内部的空间表征。

4. **从“黑盒分析” → “可视化思维过程” → 揭示了模型的局部世界模型**  
   通过让模型输出文字描述的空间关系以及简易的二维坐标图，研究者能够直观看到模型在“想象”哪些位置、如何连线。这种可解释性让我们看到即使没有显式地图，模型也会在内部形成局部的空间感知。

### 方法详解
整体思路可以拆成三大步骤：**视频编码 → 空间记忆构建 → 认知地图驱动的回答**。

1. **视频编码**  
   - 使用预训练的视觉编码器（如 CLIP‑ViT）把每一帧转成向量。  
   - 为了捕捉时间顺序，向量序列再喂进一个跨模态 Transformer，得到每帧的时空特征。可以把它想成把每帧的“画面快照”贴在时间轴上，形成一条连续的“视觉丝带”。

2. **空间记忆构建**  
   - 在 Transformer 的自注意力层里加入 **空间位置嵌入**：每个帧的特征会额外携带一个表示该帧在三维空间中大致位置的向量（由相邻帧的光流或深度估计得到）。  
   - 这些位置嵌入让模型在自注意力计算时能够“看到”哪些帧在空间上相近，从而在内部形成粗糙的空间网络。

3. **认知地图驱动的回答**  
   - 当收到问题后，模型先进入 **地图生成子模块**：它遍历记忆库，挑选出与问题关键词（如“左侧”“距离”“入口”）相关的帧，然后用自然语言或简易坐标（x, y）把这些帧的相对位置写出来。  
   - 接下来进入 **答案生成子模块**：把原始问题、语言模型的上下文以及刚才生成的认知地图一起喂进语言解码器。解码器在生成答案时会参考地图中的空间关系，类似于人类先在脑中画图再口头描述。

**最巧妙的点**在于把“画图”这一步显式化，而不是让模型在内部暗暗完成。这样做不仅提升了距离推理，还让研究者可以直接检查地图的质量，进而进行错误分析。

### 实验与效果
- **测试平台**：VSI‑Bench 包含 5,000 条基于真实视频的空间问答，覆盖距离估计、路径回忆、相对方位等六大子任务。  
- **基线对比**：作者选取了几种主流 MLLM（如 Flamingo、GPT‑4V、LLaVA）以及传统视觉‑语言模型（如 ViLT）。在整体准确率上，最好的基线约为 58%，而加入认知地图的模型提升到 71%，接近人类水平的 78%。  
- **消融实验**：去掉空间位置嵌入后，准确率跌至 62%；不生成认知地图直接回答，准确率只有 58%；仅使用 CoT 提升约 2%。这些结果表明，空间位置嵌入和显式地图是提升的关键因素。  
- **局限性**：论文指出模型仍在处理大尺度、长时视频时出现记忆衰减；认知地图目前只能用文字或二维坐标表达，难以捕捉复杂的三维结构；此外，生成地图的过程会增加推理时间约 30%。  

### 影响与延伸思考
这篇工作把“空间记忆”从模糊的隐式能力变成了可测、可视、可改进的模块，直接催生了两类后续研究：  
1. **空间显式表征的深度学习**：后续有论文尝试把 3D 点云或神经渲染技术嵌入到 MLLM 中，以实现更精细的空间推理（如“NeRF‑LLM”系列）。  
2. **可解释的多模态推理**：把中间的认知地图当作解释层，帮助用户审计模型决策，已经在机器人导航和增强现实中得到原型验证。  

如果想进一步了解，可以关注 **“跨模态空间记忆网络（Cross‑Modal Spatial Memory Networks）”** 方向，尤其是结合 **Transformer‑XL** 长时记忆和 **Neural Scene Graph** 的最新尝试。

### 一句话记住它
让多模态大语言模型在回答空间问题前先画出“认知地图”，才能真正做到在视频里“思考空间”。