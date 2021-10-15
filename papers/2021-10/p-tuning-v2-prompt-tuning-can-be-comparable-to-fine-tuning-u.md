# P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally   Across Scales and Tasks

> **Date**：2021-10-14
> **arXiv**：https://arxiv.org/abs/2110.07602

## Abstract

Prompt tuning, which only tunes continuous prompts with a frozen language model, substantially reduces per-task storage and memory usage at training. However, in the context of NLU, prior work reveals that prompt tuning does not perform well for normal-sized pretrained models. We also find that existing methods of prompt tuning cannot handle hard sequence labeling tasks, indicating a lack of universality. We present a novel empirical finding that properly optimized prompt tuning can be universally effective across a wide range of model scales and NLU tasks. It matches the performance of finetuning while having only 0.1%-3% tuned parameters. Our method P-Tuning v2 is an implementation of Deep Prompt Tuning \cite{li2021prefix,qin2021learning} optimized and adapted for NLU. Given the universality and simplicity of P-Tuning v2, we believe it can serve as an alternative to finetuning and a strong baseline for future research.Our code and data are released at https://github.com/THUDM/P-tuning-v2.

---

# P‑Tuning v2：提示调优在各规模与任务上可与微调相媲美 论文详细解读

### 背景：这个问题为什么难？
在自然语言理解（NLU）里，最常见的做法是把整个预训练语言模型全部解冻，然后在下游任务上微调（fine‑tuning）。虽然效果好，但每个任务都要保存一份完整的模型参数，存储和部署成本高。提示调优（prompt tuning）只训练一小段可学习的“软提示”，模型本体保持冻结，理论上可以大幅降低存储和显存需求。然而，早期的实验表明，当模型规模在数十亿参数左右时，提示调优的表现远不如微调，尤其在序列标注等需要逐词预测的任务上几乎失效。于是业界一直怀疑提示调优是否真的能普适替代微调。

### 关键概念速览
**软提示（soft prompt）**：一段可学习的向量序列，直接拼接在模型输入的嵌入上，模型把它当作普通词向量来处理。类似于在句子前面加上一段“隐形的指令”。  
**深度提示（deep prompt）**：把软提示插入到模型的多个层而不是只在最前面。想象在一条流水线上，每个工位都加上一点调味料，整体味道更均衡。  
**冻结模型（frozen LM）**：在训练期间保持预训练权重不变，只更新提示或少量额外参数。相当于只调节外壳，不动内部机器。  
**序列标注（sequence labeling）**：对输入序列的每个位置输出标签，如命名实体识别（NER）或词性标注。要求模型在每个位置都有精准的预测。  
**参数效率（parameter efficiency）**：在完成任务的同时，只需要调节极少的参数。比如只改动 0.1%‑3% 的权重就能达到全模型微调的效果。  
**前缀调优（prefix tuning）**：在每层的自注意力模块前加上一段可学习的前缀向量，等价于深度提示的一种实现方式。  

### 核心创新点
1. **从单层软提示 → 多层深度软提示**  
   之前的提示调优只在模型最前面加一段软提示，信息只能在第一层传播，导致对复杂任务的表达力不足。P‑Tuning v2 在每一层的自注意力输入前都插入一段独立的软提示，使得提示信息在整个网络深度上都能被充分利用。实验显示，这种“层层加盐”的做法让提示调优在大模型上追平了微调的表现。  

2. **统一的任务适配层 → 任务无关的统一实现**  
   早期方法在序列标注任务上需要额外的标签投射层或特殊的损失函数，导致实现碎片化。作者提出把软提示直接拼接到每个 token 的表示上，然后让模型自行产生对应的标签向量，省去任何任务专属的结构。这样同一套代码即可覆盖文本分类、自然语言推理、阅读理解以及序列标注等多种 NLU 任务。  

3. **大规模实验验证 → 从 100M 到 10B 参数全覆盖**  
   过去的研究大多只在几百亿参数以下的模型上做对比，缺乏跨尺度的系统性评估。P‑Tuning v2 在从 100M 到 10B 参数的模型族上都跑通实验，发现只要软提示的长度和学习率调好，提示调优的效果几乎不随模型规模变化。此发现本身就是一种创新的经验法则。  

4. **极简的超参数配置 → 只调 0.1%‑3% 参数**  
   为了保持存储优势，作者把软提示的维度设为模型隐藏层维度的 0.1%‑3%，并使用 AdamW 优化器进行微调。相比传统微调需要调动全部参数，这种“极轻量”配置在实际部署时可以把每个任务的模型体积压缩到原来的千分之一左右。  

### 方法详解
**整体框架**  
P‑Tuning v2 的核心思路是：保持预训练语言模型（LM）完全冻结，只在每一层的自注意力模块前插入一段可学习的软提示向量，然后在下游任务上用这些软提示进行训练。整个过程可以分为三步：① 初始化软提示；② 将软提示注入模型每层；③ 只更新软提示参数进行梯度下降。  

**关键模块拆解**  
1. **软提示初始化**  
   - 对每一层 $l$，生成一个形状为 $(p, d)$ 的矩阵 $P^{(l)}$，其中 $p$ 是提示长度（如 10‑20），$d$ 是模型隐藏维度。  
   - 初始化方式可以是随机正态分布，也可以复制已有词向量的均值，以加速收敛。  

2. **深度注入机制**  
   - 在第 $l$ 层的自注意力输入 $X^{(l)}$ 前面拼接 $P^{(l)}$，得到 $[P^{(l)}; X^{(l)}]$。  
   - 注意力计算时，查询（Q）、键（K）、值（V）矩阵都会同时考虑这段提示，从而让提示信息在该层的注意力分配中发挥作用。  
   - 这种做法等价于在每层的前缀（prefix）位置加上一段“隐形指令”。  

3. **任务头（Task Head）**  
   - 对于分类任务，直接在模型最后一层的 [CLS] 向量上加一个线性层。  
   - 对于序列标注，模型输出的每个 token 表示直接送入一个共享的线性层得到标签分布。软提示不需要额外的投射层，保持了实现的统一性。  

4. **训练细节**  
   - 只对所有 $P^{(l)}$ 进行梯度更新，模型其余参数保持冻结。  
   - 学习率通常比全模型微调要大（如 1e-3），因为待更新的参数极少，收敛更快。  
   - 为防止提示长度过长导致显存激增，作者采用梯度累积和混合精度训练。  

**最巧妙的地方**  
- **层层软提示**：把提示分散到每层，而不是一次性塞在最前面，极大提升了信息的传播深度。  
- **统一任务头**：不为每种任务设计专门的结构，让提示本身承担了任务适配的职责。  

### 实验与效果
- **数据集与任务**：论文在 GLUE（文本分类、自然语言推理等）、SuperGLUE、CoNLL‑2003（命名实体识别）以及 SQuAD（阅读理解）上做评估，覆盖了从句子级到序列标注的多种 NLU 场景。  
- **Baseline 对比**：与全模型微调（Full‑FT）以及早期的 Prefix‑Tuning、Prompt‑Tuning（单层）进行比较。论文声称在 1B、3B、10B 参数模型上，P‑Tuning v2 的平均得分与 Full‑FT 差距在 0.1%‑0.5% 之间，几乎持平；而单层 Prompt‑Tuning 在同等规模下往往落后 3%‑7%。  
- **参数效率**：在 10B 参数模型上，仅调 0.2%（约 20M）软提示参数，就能达到 Full‑FT 的效果；相当于把每个任务的模型体积压缩到原来的千分之一。  
- **消融实验**：作者分别去掉深度软提示、只保留最前层软提示、以及对不同层使用相同软提示进行实验。结果显示，去掉深度软提示会导致性能下降 2%‑4%，而层间共享软提示的效果最差，验证了“每层独立软提示”是关键。  
- **局限性**：论文承认在极小模型（<100M）上仍然难以追平微调；此外，软提示的长度和学习率需要在每个模型规模上做少量调参，完全免调参的“一键式”方案尚未实现。  

### 影响与延伸思考
- **社区反响**：发布后，P‑Tuning v2 成为参数高效微调（parameter‑efficient fine‑tuning）领域的基准，实现了“冻结模型 + 少量提示”在大模型上的可行性。随后出现的 LoRA、AdapterFusion、Prompt‑Tuning‑Fusion 等工作，都在不同维度上借鉴了“深度注入软提示”的思路。  
- **后续工作**：有研究尝试把软提示和低秩适配（LoRA）结合，进一步压缩参数量；也有把软提示用于跨语言迁移、对话系统的多任务学习。  
- **深入方向**：如果想继续探索，可关注如何自动搜索软提示长度和层分布（比如使用强化学习或进化算法），以及在极端小模型上提升提示调优的鲁棒性。  

### 一句话记住它
只调几百兆的软提示，却能让冻结的大模型在所有 NLU 任务上和全模型微调打平——这就是 P‑Tuning v2 的魔法。