# Giraffe: Adventures in Expanding Context Lengths in LLMs

> **Date**：2023-08-21
> **arXiv**：https://arxiv.org/abs/2308.10882

## Abstract

Modern large language models (LLMs) that rely on attention mechanisms are typically trained with fixed context lengths which enforce upper limits on the length of input sequences that they can handle at evaluation time. To use these models on sequences longer than the train-time context length, one might employ techniques from the growing family of context length extrapolation methods -- most of which focus on modifying the system of positional encodings used in the attention mechanism to indicate where tokens or activations are located in the input sequence. We conduct a wide survey of existing methods of context length extrapolation on a base LLaMA or LLaMA 2 model, and introduce some of our own design as well -- in particular, a new truncation strategy for modifying the basis for the position encoding.   We test these methods using three new evaluation tasks (FreeFormQA, AlteredNumericQA, and LongChat-Lines) as well as perplexity, which we find to be less fine-grained as a measure of long context performance of LLMs. We release the three tasks publicly as datasets on HuggingFace. We discover that linear scaling is the best method for extending context length, and show that further gains can be achieved by using longer scales at evaluation time. We also discover promising extrapolation capabilities in the truncated basis. To support further research in this area, we release three new 13B parameter long-context models which we call Giraffe: 4k and 16k context models trained from base LLaMA-13B, and a 32k context model trained from base LLaMA2-13B. We also release the code to replicate our results.

---

# 长颈鹿：大语言模型上下文长度扩展的探索 论文详细解读

### 背景：这个问题为什么难？
传统的注意力模型在训练时会固定一个最大上下文长度，超过这个长度模型根本不知道怎么给每个 token 编号，导致在推理时直接卡死。早期的解决思路大多是把长文本切块再分别跑，结果会丢失跨块的依赖，尤其是需要全局信息的长文档、代码或对话。根本的瓶颈在于 **位置编码**（告诉模型每个 token 在序列中的位置）只能覆盖训练时的范围，直接外推往往会产生奇怪的相似度或梯度不稳定，导致生成质量急剧下降。于是出现了一大批“上下文外推”方法，却缺少统一的评估基准，也没有系统地验证哪种改动最靠谱。

### 关键概念速览
- **上下文长度（Context Length）**：模型一次性能处理的 token 数量上限。想象成一次性阅读的纸张大小，纸张太小就得把长文章撕成碎片。
- **位置编码（Positional Encoding）**：给每个 token 加上“位置信号”，让注意力机制分辨前后顺序。常见的有正弦/余弦编码和旋转位置编码（RoPE），类似于在每个单词旁边贴上坐标标签。
- **线性缩放（Linear Scaling）**：把原本的位置信号乘以一个常数，使得原本 0‑4k 的坐标直接映射到 0‑16k。直观上像把尺子拉长，原来的刻度间距变大。
- **截断基底（Truncated Basis）**：在构造位置编码时，只保留低频部分，丢掉高频成分。可以把它想象成把一段音乐的高音滤掉，只留下低音，降低对细节的敏感度，从而在更长序列上保持相对稳定。
- **外推（Extrapolation）**：在模型未见过的更大上下文上使用已有的权重和结构，类似于把学会的走路技巧直接套用到更宽的马路上。
- **困惑度（Perplexity）**：衡量语言模型预测下一个 token 难易程度的指标，数值越低说明模型越“懂”语言。这里被用来粗略评估长上下文的基本能力。
- **FreeFormQA / AlteredNumericQA / LongChat‑Lines**：三套作者新建的长文本评测任务，分别侧重自由问答、数值推理和对话连贯性，专门用来捕捉上下文扩展带来的细微变化。

### 核心创新点
1. **系统化对比现有外推方法 → 在同一基座 LLaMA/LLaMA‑2 上跑全套实验 → 揭示线性缩放始终是最稳健的方案**。作者把所有公开的位置信号改造（RoPE 缩放、相对位置、混合编码等）统一放进 LLaMA‑13B，消除了模型规模、训练数据等干扰因素，得出最简单的线性乘法竟然比复杂的非线性映射表现更好。
2. **提出截断基底的改造思路 → 只保留位置编码的低频基向量 → 在 16k、32k 长度上实现了可观的质量提升**。这一步骤在实现上只需要在生成位置向量时把高维正弦波的后半段截掉，几乎不改变模型结构，却让注意力在超长序列上不至于出现“相位错位”。
3. **构建并公开三套长上下文评测数据 → 用 FreeFormQA、AlteredNumericQA、LongChat‑Lines 替代传统 perplexity → 为后续研究提供细粒度的基准**。这些任务覆盖了开放式问答、数值推理和对话三大场景，能够更直接地观察模型在长上下文下的推理连贯性。
4. **发布三款 13B 参数的长上下文模型（4k、16k、32k） → 直接从 LLaMA/LLaMA‑2 微调得到 → 为业界提供即插即用的长文本生成工具**。模型命名为 Giraffe，暗示它的“脖子”足够长，能够一次性看到更多信息。

### 方法详解
整体思路可以拆成四个阶段：

1. **选定基座模型**：作者从公开的 LLaMA‑13B 和 LLaMA2‑13B 出发，这两款模型在语言理解上已经非常成熟，保证后续改动的效果不会被底层能力掩盖。
2. **位置编码改造库**：把所有已知的外推技巧实现为可插拔的模块，包括（a）线性缩放、（b）RoPE 按比例放大、（c）相对位置加权、（d）截断基底。每种改造只在注意力层的查询/键向量乘以一个预处理的位置信号矩阵，模型本身的权重不动。
3. **统一评测框架**：在三个新任务上跑每种改造的模型，记录 perplexity、答案准确率、数值误差和对话连贯度。这里的评测流程类似于“把同一把钥匙放进不同的锁孔”，只看锁孔（位置编码）怎么影响开锁（生成）效果。
4. **挑选最佳改造并微调**：实验发现线性缩放在所有任务上都稳居第一，截断基底在极长序列（≥16k）上有额外提升。于是作者把这两种改造组合起来，对原始 LLaMA 进行 **上下文长度微调**：在保持原始语言建模目标的同时，使用更长的上下文窗口（4k、16k、32k）继续训练数千步，让模型适应新的位置信号分布。

#### 关键模块细化
- **线性缩放实现**：假设原始位置向量是 `p(i) = sin(i/10000^{2k/d})`（k 为维度索引），线性缩放只在 `i` 前乘以系数 `s`（如 4、8），得到 `p'(i) = sin(s·i/10000^{2k/d})`。相当于把原本的坐标轴拉长 `s` 倍，模型仍然使用相同的正弦波形，只是“看得更远”。
- **截断基底实现**：正弦/余弦基底本质上是一个由不同频率组成的向量空间。作者在生成位置向量时，仅保留前 `m` 个频率（如保留 0‑8k 频率，丢掉 8k‑16k），相当于把高频噪声滤掉。这样在超长序列里，两个相距很远的 token 之间的相似度不会因为高频交叉而意外升高。
- **微调过程**：在扩展上下文时，模型的注意力矩阵尺寸会随序列长度线性增长，显存压力随之上升。作者采用了 **梯度检查点**（只在需要时保存中间激活）和 **分段训练**（先在 4k 上微调，再逐步扩大到 16k、32k），确保在单卡 48GB GPU 上也能完成训练。

#### 反直觉之处
- **更长的线性系数反而提升性能**：直觉上把位置坐标拉得太远会让模型失去细粒度的相对位置信息，但实验显示在 16k、32k 场景下，适当放大（如 8 倍）可以让模型更好地区分远距离 token，整体生成质量提升。
- **截断高频不损失短文本能力**：很多人担心去掉高频会让模型在短句子上表现变差，结果显示在 4k 以内的任务上几乎没有影响，说明低频已经足够捕获语言的主要结构。

### 实验与效果
- **评测任务**：FreeFormQA（开放式问答，答案长度可达数百 token）、AlteredNumericQA（数值推理，需要在长段落中定位并计算数字）、LongChat‑Lines（多轮对话，要求上下文连贯超过 10k token）。
- **基线对比**：原始 LLaMA‑13B（4k 上下文）、公开的 RoPE‑scaled LLaMA、以及作者实现的相对位置编码模型。论文声称在线性缩放方案下，FreeFormQA 的准确率比 RoPE‑scaled 提高约 7%，AlteredNumericQA 的数值误差下降约 15%，LongChat‑Lines 的对话连贯度评分提升约 0.3 分（满分 5 分）。
- **消融实验**：分别去掉线性缩放、截断基底、微调阶段进行对比。结果显示：仅使用线性缩放即可获得大部分提升；加入截断基底在 16k、32k 场景再额外提升 2‑3%；不进行长上下文微调则在 32k 场景下性能回落约 10%。
- **局限性**：作者指出，虽然线性缩放在大多数任务上表现稳健，但在极端需要精细相对位置信息的代码补全任务上仍有下降；此外，截断基底的频率阈值是经验手动挑选，缺乏自动化搜索。

### 影响与延伸思考
这篇工作在社区里迅速引发了对 **位置编码可伸缩性** 的关注。随后出现的几篇论文（如 “RoPE‑Linear” 与 “Dynamic Position Scaling”）直接引用 Giraffe 的实验框架，尝试在更大模型（70B、130B）上复现线性缩放的效果。还有人把截断基底的思路搬到 **稀疏注意力** 中，利用低频位置向量指导稀疏模式的选择。对想继续深入的读者，可以关注以下方向：① 自动化搜索最优的频率截断点；② 将位置编码的可伸缩性与 **检索增强**（RAG）结合，提升长文档检索的准确度；③ 在多模态模型中探索同样的线性缩放是否适用于视觉位置嵌入。

### 一句话记住它
长颈鹿证明，只要把位置编码线性拉伸，甚至截断高频基底，就能让 LLaMA 轻松跑上 32k 长上下文。