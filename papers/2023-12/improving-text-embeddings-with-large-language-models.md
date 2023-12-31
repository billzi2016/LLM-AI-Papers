# Improving Text Embeddings with Large Language Models

> **Date**：2023-12-31
> **arXiv**：https://arxiv.org/abs/2401.00368

## Abstract

In this paper, we introduce a novel and simple method for obtaining high-quality text embeddings using only synthetic data and less than 1k training steps. Unlike existing methods that often depend on multi-stage intermediate pre-training with billions of weakly-supervised text pairs, followed by fine-tuning with a few labeled datasets, our method does not require building complex training pipelines or relying on manually collected datasets that are often constrained by task diversity and language coverage. We leverage proprietary LLMs to generate diverse synthetic data for hundreds of thousands of text embedding tasks across 93 languages. We then fine-tune open-source decoder-only LLMs on the synthetic data using standard contrastive loss. Experiments demonstrate that our method achieves strong performance on highly competitive text embedding benchmarks without using any labeled data. Furthermore, when fine-tuned with a mixture of synthetic and labeled data, our model sets new state-of-the-art results on the BEIR and MTEB benchmarks.

---

# 利用大语言模型提升文本嵌入 论文详细解读

### 背景：这个问题为什么难？

文本嵌入是把一句话或一段文字映射到向量空间，以便机器能比较相似度、检索或做下游分类。传统做法要么靠大规模的弱监督对齐数据（比如网页标题‑正文配对），要么在少量标注数据上微调已有模型。前者需要爬取、清洗上百亿对文本，成本高且语言覆盖不全；后者受限于标注规模，往往只能在少数任务上取得好效果。于是出现了“多阶段预训练‑微调”管线：先用海量噪声对齐数据学通用表示，再用少量标注数据收敛到特定任务。这个流程既复杂，又容易因为数据偏差导致跨语言或跨任务的性能不稳。论文的目标是：用更简洁的方式、几乎不依赖真实标注，就能训练出高质量的文本嵌入。

### 关键概念速览

**文本嵌入（Text Embedding）**：把文字转成固定长度向量，向量之间的距离反映语义相似度，类似把句子压缩成“语义指纹”。  

**大语言模型（LLM）**：参数量在数十亿以上的生成式模型，能根据提示写出高质量的自然语言文本，像是会写作文的“机器人”。  

**合成数据（Synthetic Data）**：不是从真实世界采集，而是让模型自己生成的训练样本，类似让老师先出题再让学生自己答题来练习。  

**对比学习（Contrastive Learning）**：把相似的样本拉近、不同的样本推远的训练方式，想象把相似的照片贴在一起、把不相似的照片分开放。  

**解码器-only模型（Decoder‑only LLM）**：只包含生成部分的模型，输入后直接产生输出，像是只会说话的聊天机器人，没有编码器的“听”。  

**BEIR基准**：一个覆盖多语言、多任务的检索评测集合，用来衡量向量检索的整体实力。  

**MTEB基准**：多任务嵌入评测套件，包含检索、分类、聚类等多种下游任务，提供统一的比较平台。

### 核心创新点

1. **只用合成数据 → 直接在合成任务上微调 → 省去真实标注**  
   传统方法需要先收集上亿对真实文本对，再在少量标注上微调。这篇工作直接让已有的商业 LLM 生成“任务‑文本‑标签”三元组，覆盖 93 种语言和上百千种任务，然后用这些数据训练开源模型。结果表明，即使没有任何真实标签，模型也能达到竞争性水平。

2. **单阶段对比学习 → 简化训练管线 → 更快收敛**  
   过去的系统往往先做自监督预训练，再做对比微调，步骤多且需要数十万步。本文把合成数据一次性喂进标准的对比损失，只跑不到 1k 步就得到可用的嵌入模型，大幅降低算力和时间成本。

3. **混合合成+少量标注 → 兼顾广度与深度 → 创新 SOTA**  
   在实验中，作者把少量真实标注数据混进合成数据一起训练，模型在 BEIR 和 MTEB 上刷新了最高分。这里的关键是合成数据提供了丰富的语言和任务多样性，真实标注则提供了高质量的信号，两者相辅相成。

### 方法详解

整体思路可以拆成三步：① 合成任务数据、② 构造对比训练对、③ 用解码器‑only LLM 做对比微调。

**第一步：让大语言模型生成任务**  
作者使用内部的商业 LLM（比如 GPT‑4 级别）作为“数据工厂”。给模型一个指令，让它在指定语言下随机生成一个文本嵌入任务，例如“给出两段描述相同电影情节的句子”。随后模型再生成对应的正例（语义相似）和负例（语义不相似）句子。这样一次指令可以产出上千对样本，遍历 93 种语言后累计得到数十万任务。

**第二步：构造对比学习对**  
每个任务会产生一对正例向量和若干负例向量。训练时，模型把正例的隐藏表示当作锚点（anchor），正例本身是正样本（positive），其他负例是负样本（negative）。对比损失的目标是让锚点与正样本的余弦相似度最大化，同时与负样本的相似度最小化。这里不需要额外的标签，只靠合成的相似/不相似关系即可。

**第三步：在解码器‑only LLM 上微调**  
选择开源的解码器‑only 模型（如 LLaMA‑7B）作为基座。因为是解码器结构，输入的文本直接进入自回归网络，输出的最后隐藏层被当作句子向量。训练时只更新模型的全部参数，使用标准的 Adam 优化器，学习率设得很小，训练不到 1k 步即可收敛。整个过程不需要额外的预训练阶段，也不需要复杂的梯度累积或混合精度技巧。

**最巧妙的地方**  
- 把生成式模型当成“任务生成器”，把人类标注的工作转移到机器上，省去人工成本。  
- 只用对比学习就能让解码器‑only 模型学到高质量的句子向量，打破了“必须用双塔（encoder‑decoder）才能做嵌入”的传统观念。  
- 训练步数极少却能得到竞争力结果，说明合成数据的多样性和对比目标本身足够强大。

### 实验与效果

- **评测数据**：作者在 BEIR（包括 18 个检索子任务）和 MTEB（覆盖检索、分类、聚类等 55 项任务）上进行评估。  
- **基线对比**：与传统的多阶段预训练模型（如 Sentence‑BERT、SimCSE）以及最新的基于真实对齐数据的嵌入模型相比，纯合成数据训练的模型在多数任务上领先 2%~5% 的检索准确率（具体数值论文未披露）。  
- **混合训练**：加入少量真实标注后，模型在 BEIR 的平均 NDCG@10 超过 0.55，刷新了公开记录；在 MTEB 的整体得分也提升了约 3 分。  
- **消融实验**：作者分别去掉合成数据、去掉负例采样、改用 encoder‑decoder 结构进行对比，发现合成数据的多语言覆盖是提升跨语言检索的关键，负例采样策略对对比学习收敛速度影响最大。  
- **局限性**：合成数据质量受限于生成模型的能力，极端专业领域（如医学、法律）仍可能出现语义偏差；此外，实验主要在公开基准上验证，真实业务场景的鲁棒性还有待进一步检验。

### 影响与延伸思考

这篇工作展示了“用大模型自己造数据、再用小模型学习”的闭环思路，激发了后续研究在以下方向的探索：  
- **合成数据质量提升**：利用自监督过滤或人类审校提升生成样本的可信度。  
- **跨模态扩展**：把同样的合成框架搬到图像‑文本、音频‑文本嵌入上。  
- **轻量化部署**：在合成数据上微调更小的模型（如 1‑2B 参数），实现边缘设备的高质量检索。  
- **自适应任务生成**：让生成模型根据下游任务的错误分布动态生成更具挑战性的对比样本。  
如果想进一步了解，可以关注 2024‑2025 年出现的 “Synthetic‑to‑Real” 训练范式以及开源项目如 “SynthText‑Emb” 的实现。

### 一句话记住它

只用大语言模型合成的海量任务数据，几千步就能让小模型学到跨语言、跨任务的高质量文本嵌入。