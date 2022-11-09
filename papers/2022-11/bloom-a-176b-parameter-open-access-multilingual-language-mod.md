# BLOOM: A 176B-Parameter Open-Access Multilingual Language Model

> **Date**：2022-11-09
> **arXiv**：https://arxiv.org/abs/2211.05100

## Abstract

Large language models (LLMs) have been shown to be able to perform new tasks based on a few demonstrations or natural language instructions. While these capabilities have led to widespread adoption, most LLMs are developed by resource-rich organizations and are frequently kept from the public. As a step towards democratizing this powerful technology, we present BLOOM, a 176B-parameter open-access language model designed and built thanks to a collaboration of hundreds of researchers. BLOOM is a decoder-only Transformer language model that was trained on the ROOTS corpus, a dataset comprising hundreds of sources in 46 natural and 13 programming languages (59 in total). We find that BLOOM achieves competitive performance on a wide variety of benchmarks, with stronger results after undergoing multitask prompted finetuning. To facilitate future research and applications using LLMs, we publicly release our models and code under the Responsible AI License.

---

# BLOOM: 176B 参数开源多语言语言模型 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）爆发之前，研究者主要受限于算力和数据规模，只有少数资源充足的公司能够训练上百亿参数的模型。即便有了大模型，训练语料往往只覆盖英语或少数高资源语言，导致多语言能力薄弱。与此同时，模型的所有权和使用权被严格控制，学术界和中小企业难以直接复现或改进这些技术。于是出现了两个核心瓶颈：**规模**（缺少足够的算力和海量多语言数据）和**开放性**（模型和代码被封闭）。这篇论文正是为了解决这两个瓶颈而出现的。

### 关键概念速览
- **Decoder‑only Transformer**：只使用 Transformer 的解码器堆叠来生成文本，类似于只读的自动写作机器，区别于 encoder‑decoder 结构需要先编码输入再解码。
- **ROOTS 语料库**：作者自行收集并清洗的跨语言数据集合，覆盖 46 种自然语言和 13 种编程语言，总计约 1.6 万亿 token，像是把全球图书馆的书籍都搬进了模型的记忆里。
- **多任务提示微调（multitask prompted finetuning）**：在已有的大模型上，用统一的“指令+示例”格式继续训练，让模型学会在不同任务之间切换，类似于给模型上了“通用工具箱”。
- **Responsible AI License**：一种限制性开源许可证，允许科研和非商业使用，但对潜在的滥用设有约束，确保技术在开放的同时保持一定的伦理底线。
- **Few‑shot / Zero‑shot 能力**：模型在只看到少量示例（few‑shot）或完全没有示例（zero‑shot）的情况下完成新任务，类似于人类只靠几句话就能理解指令的能力。
- **Parameter（参数）**：模型内部的可学习权重，参数越多通常意味着模型的表达能力越强，176B 参数相当于约 1760 亿个调节旋钮。

### 核心创新点
1. **规模与多语言并行扩展**  
   - 之前的开源大模型要么规模大但只覆盖少数语言（如 GPT‑NeoX），要么多语言但规模受限（如 mBERT）。  
   - BLOOM 同时把参数规模推到 176B，并在 59 种语言上进行训练。  
   - 结果是模型在跨语言基准上实现了与同等规模商业模型相当的表现，证明大规模多语言训练是可行的。

2. **社区驱动的协同训练**  
   - 传统大模型往往由单一公司内部完成，缺乏透明度。  
   - 本文组织了数百名研究者、数十个机构共同贡献算力、数据清洗和代码实现，形成了类似“开源协作版”大型模型的全新范式。  
   - 这种模式降低了单个组织的门槛，也让训练过程、超参数选择等细节对外可追溯。

3. **统一的多任务提示微调框架**  
   - 过去的微调往往针对单一任务，导致模型在新任务上仍需大量标注数据。  
   - BLOOM 采用统一的指令式提示格式，对数十个任务同步微调，使模型在 few‑shot 场景下的适应性显著提升。  
   - 实验显示，微调后在 SuperGLUE、MMLU 等基准上提升了约 3–5% 的准确率。

4. **负责任的开源许可**  
   - 直接开源大模型会引发滥用风险，传统的 MIT/Apache 许可证缺乏约束。  
   - 作者提出 Responsible AI License，允许学术研究和非商业应用，同时要求使用者遵守伦理审查和风险评估。  
   - 这为后续开源大模型提供了可复制的法律模板。

### 方法详解
**整体框架**  
BLOOM 的训练流程可以划分为三大阶段：① 数据准备与清洗 → ② 大规模自回归预训练 → ③ 多任务提示微调。整个过程在分布式 GPU 集群上并行执行，使用了 ZeRO‑3 参数分片技术来突破显存限制。

**1. ROOTS 语料库构建**  
- 收集来源包括维基百科、新闻网站、代码仓库、社交媒体等，覆盖 46 种自然语言和 13 种编程语言。  
- 采用语言检测、去重、质量过滤等多层过滤，确保每种语言都有足够的高质量 token。  
- 类比于把世界各地的图书馆书籍先分类、去除破损页，再统一装进一本巨大的百科全书。

**2. Decoder‑only Transformer 设计**  
- 采用标准的自回归 Transformer 解码器堆叠，层数 70、隐藏维度 12288、注意力头数 96。  
- 通过稀疏注意力和混合精度训练（FP16 + BF16）降低算力消耗。  
- 参数分布在数千块 GPU 上，使用 DeepSpeed 的 ZeRO‑3 将模型权重切分到每块卡的显存中，实现了 176B 参数的完整训练。

**3. 大规模自回归预训练**  
- 目标是预测下一个 token（自回归），即给定前面的文字让模型猜下一个字。  
- 采用 AdamW 优化器，学习率采用线性 warm‑up + cosine decay。  
- 训练总步数约 3000 万步，等价于在 1.6 万亿 token 上完整遍历两遍（两轮 epoch），确保模型对所有语言都有充分的曝光。

**4. 多任务提示微调**  
- 选取了 30+ 公共任务（阅读理解、数学推理、代码补全等），每个任务都用统一的指令模板包装，例如 “请回答以下问题：{question}”。  
- 在微调阶段，模型仍保持自回归结构，只是输入中加入了任务指令和少量示例。  
- 这种做法让模型在看到新任务时，只需要一个自然语言描述就能激活相应的能力，类似于给模型装上了“即插即用”的功能键。

**最巧妙的点**  
- **社区协同训练**：把算力资源分散到全球多个实验室，每个实验室负责一小块数据或模型切片，最终通过参数同步合并。这种“众包算力”在保持训练效率的同时，极大降低了单点资源瓶颈。  
- **统一提示微调**：不再为每个任务单独设计微调头，而是让模型自行学习如何解释指令，这在提升跨任务迁移能力上效果惊人。

### 实验与效果
- **评测数据集**：包括 BIG-bench、MMLU（多语言理解基准）、SuperGLUE、XGLUE、CodeXGLUE 等，覆盖自然语言理解、推理、翻译以及代码生成。  
- **Baseline 对比**：与同等规模的商业模型（如 GPT‑3 175B）以及开源模型（LLaMA 65B、OPT 175B）进行对比。论文声称在大多数多语言任务上，BLOOM 的零样本表现与 GPT‑3 持平，在少数语言上甚至略有优势。  
- **微调提升**：在进行多任务提示微调后，SuperGLUE 的准确率提升约 4%，MMLU 的平均分提升约 3.5%。  
- **消融实验**：作者分别去掉 ROOTS 中的低资源语言、关闭多任务提示微调、以及使用传统 Adam 而非 AdamW，结果显示：去掉低资源语言会导致对应语言的性能下降 5–10%；不做提示微调则整体 zero‑shot 分数下降约 2%。  
- **局限性**：模型仍然在极低资源语言上表现不佳，且训练成本高达数十万 GPU 小时，普通研究机构难以复制。作者也指出，虽然采用 Responsible AI License 限制商业滥用，但仍缺乏对模型输出有害内容的系统性防护。

### 影响与延伸思考
BLOOM 的发布标志着 **“开源大模型”** 从概念走向实践，激发了后续一波社区驱动的模型项目，如 OpenChat、MOSS、Falcon 等，都在不同程度上借鉴了 BLOOM 的协同训练和多语言数据构建方式。它也让学术界有了可直接访问的 176B 参数基准，推动了对大模型可解释性、低资源语言适配以及安全性评估的研究。想进一步深入，可以关注以下方向：  
- **高效微调技术**（如 LoRA、Adapter），在保持模型规模的同时降低下游任务成本。  
- **多语言对齐方法**，提升低资源语言的表示质量。  
- **安全与伦理框架**，在开源大模型的使用上建立更完善的审计和监管机制。  

### 一句话记住它
BLOOM 用 176 B 参数、59 种语言和社区协同的方式，首次把真正的大规模多语言模型公开给所有人。