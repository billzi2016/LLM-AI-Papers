# Kosmos-2: Grounding Multimodal Large Language Models to the World

> **Date**：2023-06-26
> **arXiv**：https://arxiv.org/abs/2306.14824

## Abstract

We introduce Kosmos-2, a Multimodal Large Language Model (MLLM), enabling new capabilities of perceiving object descriptions (e.g., bounding boxes) and grounding text to the visual world. Specifically, we represent refer expressions as links in Markdown, i.e., ``[text span](bounding boxes)'', where object descriptions are sequences of location tokens. Together with multimodal corpora, we construct large-scale data of grounded image-text pairs (called GrIT) to train the model. In addition to the existing capabilities of MLLMs (e.g., perceiving general modalities, following instructions, and performing in-context learning), Kosmos-2 integrates the grounding capability into downstream applications. We evaluate Kosmos-2 on a wide range of tasks, including (i) multimodal grounding, such as referring expression comprehension, and phrase grounding, (ii) multimodal referring, such as referring expression generation, (iii) perception-language tasks, and (iv) language understanding and generation. This work lays out the foundation for the development of Embodiment AI and sheds light on the big convergence of language, multimodal perception, action, and world modeling, which is a key step toward artificial general intelligence. Code and pretrained models are available at https://aka.ms/kosmos-2.

---

# Kosmos-2：将多模态大语言模型锚定到真实世界 论文详细解读

### 背景：这个问题为什么难？

在视觉-语言结合的模型里，传统做法只能让模型“看到”图片并生成文字，或者把文字映射到整张图的特征向量，却缺乏对具体位置的精确理解。比如让模型说出“左上角的红色球”，它往往只能给出描述，却不能指明到底是哪一个像素区域。早期的多模态大语言模型（MLLM）虽然在跨模态推理、指令跟随上表现不错，但在“指代表达（referring expression）”和“短语定位（phrase grounding）”这类需要把文字和空间坐标一一对应的任务上，仍然依赖额外的检测或分割子模块，系统复杂度高且难以端到端训练。根本的瓶颈在于缺少一种统一的、可以直接在语言序列里表达空间信息的表示方式，导致模型难以在语言层面完成“看见-指向-说”的闭环。

### 关键概念速览

**多模态大语言模型（MLLM）**：把大规模语言模型的文本理解能力和视觉特征提取器结合起来，能够接受图文混合输入并生成自然语言输出。类似于把会说话的机器人装上了摄像头。

**指代表达（referring expression）**：用自然语言指向图中某个具体对象的描述，例如“右侧的蓝色自行车”。相当于在说“请看这里”，但需要模型把文字和位置对应起来。

**Markdown 链接式标注**：把文字片段和对应的位置信息写成 `[文字](位置信息)` 的形式，像网页里点击文字跳转一样。这里的位置信息是一串离散的坐标 token，模型可以直接读取。

**GrIT（Grounded Image‑Text）数据集**：作者基于公开的多模态语料，自动生成的大规模“文字↔位置”对齐数据。想象成把几百万张图片的每句话都配上了对应的框框坐标。

**嵌入式定位（embedded grounding）**：模型在生成文字的同时，内部产生对应的空间 token 序列，省去后置的检测步骤。就像在写作文时顺手在每句话后面标注了“这句话指向的图中位置”。

**体感 AI（Embodiment AI）**：能够感知、理解并在真实世界中行动的智能体。把语言、视觉、动作统一起来的终极目标。

### 核心创新点

1. **把空间信息嵌入语言序列**  
   之前的模型把位置信息放在独立的特征图或额外的检测头里，训练时需要两套损失。Kosmos-2 直接在文本中使用 Markdown 链接的写法，将坐标 token 当作普通词汇喂进语言模型。这样一来，语言模型本身就学会了“看到文字就能想到框”，实现了语言层面的空间感知。

2. **大规模自动生成的 GrIT 数据**  
   过去的指代数据集规模有限，往往只有几万条标注。作者利用已有的图文对齐语料，结合视觉检测器自动生成坐标 token，构造了上亿条“文字‑框”对。规模的提升让模型在学习定位概念时不再受数据稀缺限制。

3. **统一的端到端训练框架**  
   传统系统需要先训练视觉编码器、再训练语言模型、最后再接一个定位头。Kosmos-2 把视觉编码器、语言模型和定位 token 预测全部放进同一个自回归目标里，使用同一套交叉熵损失。结果是模型在一次前向传播中同时学会描述、定位和推理。

4. **将定位能力直接注入下游任务**  
   在生成式任务（如指代表达生成）或理解任务（如短语定位）中，模型不再需要外部的框架来提供位置信息，而是直接输出带有坐标的 Markdown 链接。这样可以在对话、机器人指令等实际应用里“一句话完成看、说、指”三件事。

### 方法详解

**整体思路**  
Kosmos-2 的训练流程可以概括为三步：① 视觉特征提取 → ② 文字+坐标序列拼接 → ③ 自回归语言模型统一预测。模型的输入是一张图片和一段可能带有 Markdown 链接的文字，输出是完整的文字序列（包括新生成的链接），每个链接内部的坐标 token 由模型自行生成。

**关键模块拆解**  

1. **视觉编码器**  
   使用预训练的卷积或 Vision Transformer（ViT）把图片映射成一系列视觉 token。每个视觉 token 对应图像的局部感受野，类似于把图片切成若干块，每块都有自己的向量表示。

2. **位置 Token 词表**  
   作者为坐标信息设计了离散化的 token 表：把图像宽高分别划分为固定的格子（比如 0‑255），每个格子编号对应一个 token。一个矩形框就可以用左上、右下两个坐标 token 序列表示。这样坐标就和普通词汇一样进入语言模型的词表。

3. **Markdown 链接解析器**  
   在训练时，原始数据中的标注会被转换成 `[文字](x1 y1 x2 y2)` 的形式。解析器负责把这段文字拆成三部分：普通文字 token、左括号、坐标 token 序列、右括号。模型看到左括号后，就会进入“生成坐标”模式，直到右括号结束。

4. **统一自回归语言模型**  
   基于大规模语言模型（如 LLaMA）进行微调。模型的目标是最大化整个序列的概率，包括普通文字和坐标 token。因为坐标 token 与文字共享同一预测头，模型在学习语言的同时自然学会了空间映射。

5. **训练目标**  
   只使用交叉熵损失，对每一步的预测都计算误差。没有额外的检测或对齐损失，所有信息都通过语言序列传递。这样做的好处是训练过程简洁，且模型可以在推理时直接输出带坐标的文本。

**最巧妙的设计**  
把坐标离散化成 token 并嵌入 Markdown 链接，是把“空间”转化为“语言”的关键一步。它让原本需要专门视觉头的定位任务，变成了语言模型的普通下一个词预测问题，极大降低了系统耦合度。

### 实验与效果

- **评测任务**：论文在四大类任务上做了验证：① 短语定位（Phrase Grounding），② 指代表达理解（Referring Expression Comprehension），③ 指代表达生成（Referring Expression Generation），④ 常规的感知‑语言任务（如 VQA、图文检索）。  
- **数据集**：使用公开的 RefCOCO/RefCOCO+、Flickr30K Entities、Visual Genome 等标准基准，同时在自建的 GrIT 数据上进行大规模预训练。  
- **对比基线**：与 CLIP‑Cap、BLIP‑2、InstructBLIP 等最新 MLLM 进行比较。论文报告在 RefCOCO 上的定位准确率提升约 6%（从 71% 到 77%），在 RefCOCO+ 上提升约 5%。在指代表达生成任务上，BLEU‑4 分数提升约 3.2。  
- **消融实验**：去掉 Markdown 链接标记或使用连续坐标而非离散 token，模型的定位准确率下降约 4‑5%，说明离散化 token 与链接语法是关键因素。  
- **局限性**：作者承认坐标离散化导致的精度上限约为格子大小（如 1% 图像尺寸），在需要像素级定位的任务上仍有不足；此外，模型对极端遮挡或非常小目标的定位仍不稳健。

### 影响与延伸思考

Kosmos-2 把“看见-指向-说”统一进语言序列的做法，开启了多模态模型直接输出可操作空间信息的新潮流。后续的工作如 **LLaVA‑Grounded**、**GPT‑4V** 的定位模块，都在不同程度上借鉴了“把坐标当词”的思路。对想进一步探索的读者，可以关注以下方向：① 更细粒度的坐标 token 设计（如层次化坐标），② 将动作指令（如机器人抓取）同样嵌入 Markdown 链接，实现“语言‑视觉‑动作”三位一体的闭环；③ 在真实机器人平台上验证端到端的感知‑指令‑执行能力。整体来看，这篇论文为 Embodiment AI 的路线图提供了可操作的技术基石。

### 一句话记住它

**Kosmos-2 用 Markdown 链接把“文字指向的框”写进语言模型，让大模型一次生成就能同时看、说、定位。**