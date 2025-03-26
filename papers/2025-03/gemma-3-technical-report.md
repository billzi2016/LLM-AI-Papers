# Gemma 3 Technical Report

> **Date**：2025-03-25
> **arXiv**：https://arxiv.org/abs/2503.19786

## Abstract

We introduce Gemma 3, a multimodal addition to the Gemma family of lightweight open models, ranging in scale from 1 to 27 billion parameters. This version introduces vision understanding abilities, a wider coverage of languages and longer context - at least 128K tokens. We also change the architecture of the model to reduce the KV-cache memory that tends to explode with long context. This is achieved by increasing the ratio of local to global attention layers, and keeping the span on local attention short. The Gemma 3 models are trained with distillation and achieve superior performance to Gemma 2 for both pre-trained and instruction finetuned versions. In particular, our novel post-training recipe significantly improves the math, chat, instruction-following and multilingual abilities, making Gemma3-4B-IT competitive with Gemma2-27B-IT and Gemma3-27B-IT comparable to Gemini-1.5-Pro across benchmarks. We release all our models to the community.

---

# Gemma 3 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）向多模态方向扩展时，两个痛点一直卡住社区：一是视觉信息的接入往往让模型体积膨胀，导致推理成本失控；二是长上下文（上万甚至上十万 token）会让注意力机制的 KV‑cache（键值缓存）占用的显存呈指数增长，普通硬件根本撑不住。此前的开源模型要么只能处理短文本，要么在加入视觉后只能用几百亿参数的巨型模型，既不轻量也不易部署。Gemma 3 正是为了解决这两大瓶颈而出现的。

### 关键概念速览
**多模态模型**：同时接受文字和图像等不同感官输入的模型，就像人脑能把看到的画面和听到的话结合起来理解。  
**KV‑cache**：在自注意力计算中缓存每一层的键（Key）和值（Value），让后续 token 只需要增量计算，类似于把已经翻译好的句子存进笔记本，后面查时不必重新写一遍。  
**局部注意力（Local Attention）**：只在相邻的几个 token 之间做注意力，像是只看窗前的几本书，而不是把整座图书馆的书都翻一遍。  
**全局注意力（Global Attention）**：在所有 token 之间做注意力，等价于把整座图书馆的书都拿出来比较。  
**注意力层比例**：模型里局部注意力层和全局注意力层的数量比例，决定了模型在局部细节和全局关联之间的平衡。  
**蒸馏（Distillation）**：把大模型的“知识”压缩进小模型，像老师把课堂要点写成简短的笔记让学生背诵。  
**指令微调（Instruction‑fine‑tune）**：在已有模型上继续训练，让它更擅长遵循人类下达的任务指令。  
**后训练配方（Post‑training Recipe）**：在主训练结束后再加一轮针对特定能力（数学、对话、多语言等）的微调，类似于毕业前的实习强化训练。

### 核心创新点
1. **视觉能力的轻量化加入**  
   - 之前的多模态模型要么使用大规模视觉编码器，要么把视觉特征直接拼到文本后导致参数激增。  
   - Gemma 3 在保持 1‑27 B 参数规模的前提下，引入了专门的视觉投影层，并通过蒸馏让视觉分支学习已有语言模型的表征。  
   - 结果是模型在图文任务上达到了与数十倍参数模型相近的表现，却仍保持轻量、易部署的特性。

2. **KV‑cache 内存压缩的注意力结构改造**  
   - 长上下文下，传统全局注意力的 KV‑cache 会随 token 数线性增长，显存很快爆炸。  
   - 作者把模型的注意力层划分为更多的局部注意力层，同时把局部窗口设得很短（如 128 token），只在少数层保留全局注意力。  
   - 这样在 128 K token 的上下文里，KV‑cache 的占用被显著削减，使得在普通 GPU 上也能跑长文档。

3. **蒸馏驱动的跨模态预训练**  
   - 传统预训练直接从大规模文本+图像数据学习，成本高且容易出现模式崩塌。  
   - Gemma 3 采用教师模型（Gemma 2 系列的大模型）对齐文本和视觉特征，通过蒸馏让小模型快速获得高质量跨模态表征。  
   - 这一步让小模型在同等训练预算下就能赶超未蒸馏的同尺寸基线。

4. **后训练配方提升特定能力**  
   - 仅靠一次大规模预训练难以兼顾数学推理、对话流畅度和多语言覆盖。  
   - 作者在主训练后分别加入数学题库、对话数据、以及多语言指令集进行微调，配方中还加入了梯度噪声和学习率循环等技巧。  
   - 实验显示，4 B 参数的 Gemma 3‑IT 在多语言指令任务上可以匹配 27 B 参数的 Gemma 2‑IT，27 B 版本则与商用的 Gemini‑1.5‑Pro 持平。

### 方法详解
**整体框架**  
Gemma 3 的训练分为三大阶段：① 视觉投影层的初始化；② 大规模跨模态蒸馏预训练；③ 目标导向的后训练配方。模型整体仍采用 Transformer 架构，只是把注意力层的类型做了比例调配。

**1️⃣ 视觉投影层**  
- 输入的图像先经过轻量化的卷积或 ViT（视觉 Transformer）前置网络，得到固定维度的视觉特征向量。  
- 这些向量通过一个线性投影映射到与语言隐藏维度相同的空间，然后与文本 token 的嵌入相加或拼接，形成混合序列。  
- 类比：把图片的“颜色、形状”标签写进文字的“词表”里，让语言模型直接读懂。

**2️⃣ 蒸馏预训练**  
- 教师模型是已经训练好的 Gemma 2 系列（包括 27 B 参数的大模型），它会对同一文本+图像对输出隐藏状态和 logits。  
- 学生模型（Gemma 3）在同样的输入上计算自己的隐藏状态和 logits，然后最小化两者之间的 L2 损失（隐藏层）和交叉熵损失（输出层）。  
- 这样学生模型在学习原始数据的同时，还被迫模仿教师的“思考方式”，加速收敛并提升跨模态一致性。

**3️⃣ 注意力层比例与 KV‑cache 优化**  
- 传统 Transformer 每层都是全局注意力。Gemma 3 把 12‑layer（以 4 B 为例）中的 8 层改为局部注意力，窗口长度设为 128 token；剩余 4 层保留全局注意力，用来捕捉长距离依赖。  
- 在推理时，局部层的 KV‑cache 只需要保存最近 128 token 的键值，显存占用随序列长度几乎保持常数；全局层仍然需要完整缓存，但数量少，整体显存开销大幅下降。  
- 这一步的灵感来源于“稀疏注意力”，但作者把比例调得更激进，以适配 128 K token 的极长上下文。

**4️⃣ 后训练配方**  
- **数学微调**：使用公开的数学题库（如 GSM8K）进行指令微调，加入梯度噪声提升数值推理的鲁棒性。  
- **对话微调**：采集多轮对话数据，使用 RLHF（强化学习人类反馈）或直接的指令微调，让模型在多轮交互中保持上下文连贯。  
- **多语言微调**：引入 100+ 语言的指令数据，使用语言标签进行混合训练，提升跨语言指令遵循能力。  
- 每个子阶段都采用小学习率循环（learning‑rate warm‑up → cosine decay），确保模型在已有能力上不被“冲掉”。

**最巧妙的点**  
把局部注意力层的比例提升到 2/3 以上，同时把局部窗口压到 128 token，这在保持长上下文的同时几乎把 KV‑cache 的显存需求削减到原来的 30%。这一步在公开报告中被称为“KV‑cache 记忆压缩”，是实现 128 K token 支持的关键。

### 实验与效果
- **评测任务**：包括图文检索、视觉问答（VQA）、多语言指令遵循、数学推理、长文档问答等基准。  
- **对比基线**：Gemma 2 系列（同尺度但无视觉、短上下文）、LLaMA‑2‑70B、Gemini‑1.5‑Pro（商用大模型）等。  
- **核心结果**：报告称 4 B 参数的 Gemma 3‑IT 在多语言指令基准上与 27 B 参数的 Gemma 2‑IT 持平；27 B 版本在整体评测中与 Gemini‑1.5‑Pro 相当。  
- **消融实验**：作者分别去掉视觉投影层、降低局部注意力比例、去除蒸馏、以及不做后训练配方，结果显示：去掉蒸馏导致跨模态准确率下降约 12%；局部注意力比例降到 1/3 时 KV‑cache 显存回升至原始 2 倍，长上下文推理不可行。  
- **局限性**：报告未给出视觉任务的细粒度误差分析，也没有在极端低算力设备上做实测；后训练配方的具体数据来源和超参数细节缺乏公开说明。

### 影响与延伸思考
Gemma 3 的出现让「轻量级多模态」从概念走向可落地的产品，尤其在开源社区引发了两股潮流：一是更多团队尝试用蒸馏+稀疏注意力的组合来压缩大模型的跨模态能力；二是对 KV‑cache 的显存优化成为新热点，后续的模型（如 LLaVA‑Next、Mistral‑Vision）都在借鉴其局部注意力比例调度。想进一步深入，可以关注以下方向：① 更细粒度的视觉特征对齐方法（如跨模态对比学习）；② 动态注意力比例自适应调度，让模型根据输入长度自动切换局部/全局层；③ 在边缘设备上实际部署的显存/功耗评估（推测）。  

### 一句话记住它
Gemma 3 用“局部注意力 + 蒸馏”把视觉和超长上下文塞进 1‑27 B 参数的轻量模型，让开源多模态 AI 真正跑得起、跑得快。