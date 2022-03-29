# Training Compute-Optimal Large Language Models

> **Date**：2022-03-29
> **arXiv**：https://arxiv.org/abs/2203.15556

## Abstract

We investigate the optimal model size and number of tokens for training a transformer language model under a given compute budget. We find that current large language models are significantly undertrained, a consequence of the recent focus on scaling language models whilst keeping the amount of training data constant. By training over 400 language models ranging from 70 million to over 16 billion parameters on 5 to 500 billion tokens, we find that for compute-optimal training, the model size and the number of training tokens should be scaled equally: for every doubling of model size the number of training tokens should also be doubled. We test this hypothesis by training a predicted compute-optimal model, Chinchilla, that uses the same compute budget as Gopher but with 70B parameters and 4$\times$ more more data. Chinchilla uniformly and significantly outperforms Gopher (280B), GPT-3 (175B), Jurassic-1 (178B), and Megatron-Turing NLG (530B) on a large range of downstream evaluation tasks. This also means that Chinchilla uses substantially less compute for fine-tuning and inference, greatly facilitating downstream usage. As a highlight, Chinchilla reaches a state-of-the-art average accuracy of 67.5% on the MMLU benchmark, greater than a 7% improvement over Gopher.

---

# 计算最优大语言模型的训练 论文详细解读

### 背景：这个问题为什么难？
在大语言模型（LLM）飞速扩张的过去几年里，研究者们几乎只关注把模型参数往上堆，训练数据量却保持不变。这样做的直觉是“更大就更好”，但实际结果是很多模型在给定的算力预算下并没有真正发挥潜力——它们的训练步数太少，导致学习不充分。缺乏系统的“算力‑模型‑数据”三者平衡原则，使得我们无法判断到底应该把算力花在更大模型上，还是在更多数据上，甚至两者该怎么配比。

### 关键概念速览
**算力预算（Compute budget）**：指在一次完整训练过程中可使用的总 FLOPs（浮点运算次数），相当于你买的“机器时间”。  
**模型规模（Model size）**：模型的参数数量，常用“B”表示十亿级别。参数越多，模型的表达能力越强。  
**训练 token 数（Training tokens）**：模型在训练时看到的词或子词的总数，类似于学生阅读的总页数。  
**Scaling law（尺度律）**：经验公式，描述模型性能随参数、数据量、算力的变化趋势，就像物理学里质量和能量的关系。  
**Chinchilla**：本文提出的算力最优模型，70 B 参数、使用 4 倍于 Gopher 的数据量。  
**MMLU（Massive Multitask Language Understanding）**：一个覆盖 57 项学科的综合评测，用来衡量模型的通用知识和推理能力。  
**微调（Fine‑tuning）**：在大模型基础上，用少量特定任务数据再训练，使模型在该任务上表现更好。  
**推理成本（Inference cost）**：模型在实际使用时每次生成答案所需的算力，直接影响部署费用。

### 核心创新点
1. **从经验尺度律到算力最优配比**  
   *之前的做法*：只增大模型参数，保持训练数据不变，导致算力使用不均衡。  
   *本文的做法*：系统训练 400 条不同规模‑数据组合的模型，发现算力最优时模型规模和训练 token 数必须同步增长——模型翻倍，数据也翻倍。  
   *带来的改变*：提供了一个简单的“等比增长”法则，让研究者在给定算力下直接算出最合适的模型大小和数据量。

2. **验证法则并打造 Chinchilla**  
   *之前的做法*：大模型往往在同等算力下使用更少的数据（如 Gopher 280 B 只用了约 300 B token）。  
   *本文的做法*：按照新法则，用与 Gopher 相同的算力训练了 70 B 参数的 Chinchilla，数据量提升 4 倍。  
   *带来的改变*：在同等算力下，Chinchilla 在所有评测上显著超越更大模型，证明“更大不一定更好”，关键是数据量要匹配。

3. **展示算力节省的二次效应**  
   *之前的做法*：大模型在微调和推理时消耗巨额算力，成本高昂。  
   *本文的做法*：因为 Chinchilla 更小，却更强，微调和推理所需的 FLOPs 大幅下降。  
   *带来的改变*：实际部署成本降低，使得高性能模型更易被企业和研究者使用。

### 方法详解
整体思路可以拆成三步：**（1）大规模实验搜集经验数据 →（2）拟合算力‑规模‑数据的尺度律 →（3）依据律式构造算力最优模型**。

**步骤 1：系统化训练实验**  
作者在同一硬件平台上训练了 400 个 transformer 语言模型，参数从 7 0 M 到 1 6 B 不等，每个模型分别在 5 B、50 B、500 B token 等不同规模的数据上训练。这样做的目的是在同一算力预算下观察“模型太小、数据太多”与“模型太大、数据太少”两极的表现差异。

**步骤 2：拟合尺度律**  
把每次实验的算力消耗、模型参数数、训练 token 数以及最终的验证损失（或下游任务分数）放进回归模型，得到经验公式：  
- 损失随参数数的对数下降率 ≈ 损失随 token 数的对数下降率。  
- 在固定算力下，最小损失出现的点满足“模型规模 ∝ 训练 token 数”。  
直白来说，就是把算力想象成一块面粉，做大饼（大模型）还是做薄饼（小模型）都要把面粉分配到面团（参数）和烤箱时间（数据）上，最好的配比是两者等比例增长。

**步骤 3：构造算力最优模型 Chinchilla**  
已知算力预算（与 Gopher 相同），按照律式算出最优参数数约为 70 B，同时对应的 token 数约为 4×Gopher 使用的量。于是作者在同样的硬件上训练了 Chinchilla，保持其它超参数（学习率、批大小、模型深度等）与 Gopher 相近，只是把数据量扩大到约 1.2 T token。

**关键细节**  
- **数据质量控制**：虽然 token 数翻了四倍，但作者仍然使用高质量的网页、书籍、代码等混合语料，避免噪声数据稀释学习效果。  
- **训练调度**：为保证算力相同，Chinchilla 的训练步数比 Gopher 多四倍，但每步的 FLOPs 少，因为模型更小。  
- **最反直觉的点**：很多人直觉认为“更大模型一定更好”，实验却显示在同等算力下，适度缩小模型、增加数据能显著提升性能。

### 实验与效果
- **评测任务**：包括 MMLU、BIG-bench、TruthfulQA、Winograd 等多项语言理解和推理基准。  
- **对比基线**：Gopher（280 B）、GPT‑3（175 B）、Jurassic‑1（178 B）以及 Megatron‑Turing NLG（530 B）。  
- **核心结果**：Chinchilla 在 MMLU 上取得 67.5% 的平均准确率，比 Gopher 高出约 7%。在其他任务上也普遍领先，且领先幅度在 5%–15% 之间。  
- **算力节省**：因为参数只有 70 B，微调和推理的 FLOPs 比 Gopher 低约 4 倍，实际部署成本大幅下降。  
- **消融实验**：作者分别固定模型大小只增数据、固定数据只增模型，发现两者单独提升都不如同步增长的效果，验证了等比增长法则的必要性。  
- **局限性**：论文主要在英文语料上实验，未评估多语言或特定领域数据的影响；此外，算力预算仍然是巨大的资源，普通研究团队难以复现。

### 影响与延伸思考
这篇工作在发布后迅速成为 LLM 训练的“黄金法则”，很多后续大模型（如 DeepMind 的 Gemini、OpenAI 的后续模型）都在公开报告中提到遵循“模型‑数据等比增长”。随后出现的研究进一步细化了尺度律，例如加入模型结构（稀疏化、Mixture‑of‑Experts）或硬件加速的因素，尝试在更低算力下仍保持最优配比。想深入了解的读者可以关注以下方向：  
- **算力‑效率的硬件‑软件协同**（如张量并行、混合精度训练）。  
- **多语言或跨模态尺度律**（文本+图像、文本+代码）。  
- **自适应数据采样**：在等比增长框架下，如何挑选最有信息量的 token，以进一步提升算力利用率（推测）。

### 一句话记住它
在固定算力下，**模型规模和训练数据量必须同步翻倍**，这才是打造最强大语言模型的关键。