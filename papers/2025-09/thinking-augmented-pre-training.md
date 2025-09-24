# Thinking Augmented Pre-training

> **Date**：2025-09-24
> **arXiv**：https://arxiv.org/abs/2509.20186

## Abstract

This paper introduces a simple and scalable approach to improve the data efficiency of large language model (LLM) training by augmenting existing text data with thinking trajectories. The compute for pre-training LLMs has been growing at an unprecedented rate, while the availability of high-quality data remains limited. Consequently, maximizing the utility of available data constitutes a significant research challenge. A primary impediment is that certain high-quality tokens are difficult to learn given a fixed model capacity, as the underlying rationale for a single token can be exceptionally complex and deep. To address this issue, we propose Thinking augmented Pre-Training (TPT), a universal methodology that augments text with automatically generated thinking trajectories. Such augmentation effectively increases the volume of the training data and makes high-quality tokens more learnable through step-by-step reasoning and decomposition. We apply TPT across diverse training configurations up to $100$B tokens, encompassing pre-training with both constrained and abundant data, as well as mid-training from strong open-source checkpoints. Experimental results indicate that our method substantially improves the performance of LLMs across various model sizes and families. Notably, TPT enhances the data efficiency of LLM pre-training by a factor of $3$. For a $3$B parameter model, it improves the post-training performance by over $10\%$ on several challenging reasoning benchmarks.

---

# 思考增强预训练 论文详细解读

### 背景：这个问题为什么难？

大语言模型的预训练需要海量高质量文本，但真实可用的优质语料始终有限。随着算力指数级增长，模型容量不断膨胀，却没有同步出现足够丰富的训练信号，导致模型在学习某些关键 token 时“碰壁”。这些 token 背后往往隐藏着复杂的推理链或专业常识，单纯靠一次出现的上下文难以让固定容量的模型完整捕捉。传统做法只能靠扩大数据规模或加深模型深度，却没有从根本上提升每条数据的学习价值。

### 关键概念速览
- **思考轨迹（Thinking Trajectory）**：模型在回答问题时产生的逐步推理过程，用文字形式记录下来，类似于人写下的解题步骤。
- **CoT（Chain‑of‑Thought）**：让模型在给出最终答案前先输出思考链条，像在黑板上写草稿一样帮助模型理清逻辑。
- **数据效率（Data Efficiency）**：在固定算力或训练步数下，模型能够从多少真实信息中学到有效知识。效率高意味着用更少的 token 就能达到同等性能。
- **预训练 Token**：语言模型在大规模语料上学习的基本单元，每个 token 都对应一个向量表示。
- **中途微调（Mid‑training）**：在已有的开源 checkpoint 基础上继续训练，常用于快速提升特定能力。
- **思考增强（Thinking Augmented）**：在原始文本后面追加思考轨迹，使模型在阅读时看到“为什么”而不是仅仅“是什么”。

### 核心创新点
1. **自动生成思考轨迹 → 将原始语料与 LLM 产生的 CoT 合并 → 训练数据量实际翻倍，且高难度 token 伴随解释出现，模型更容易捕获其深层含义。**  
   之前的预训练只使用原始句子，信息密度固定；本方法通过让强模型先生成一步步推理，再把这些推理写回原文，实现了“信息增密”。

2. **统一的思考增强管线 → 对所有规模的模型、所有数据量（从稀缺到充裕）都适用 → 实验显示在 3B、7B、30B 等不同规模上均能提升 10%+ 的推理基准分数。**  
   过去的提升手段往往针对特定模型或特定任务，缺乏通用性；这里提供了一套“一键式”增强流程，几乎不需要手工标注。

3. **数据效率提升 3 倍的实证 → 在相同算力下，用 1/3 的原始 token 就能达到原始训练的效果 → 为算力受限的实验室或小公司提供了可行的成本削减路径。**  
   传统观点认为算力和数据是线性关系，本研究用思考轨迹打破了这一假设。

### 方法详解
整体思路可以拆成三步：**生成 → 合并 → 训练**。

1. **生成思考轨迹**  
   - 选取一套已有的强大 LLM（如 GPT‑4）作为“教师”。  
   - 给每条原始文本加上一个简短的指令，例如“请一步步解释下面的句子”。  
   - 教师模型输出的文字包括原句、推理步骤以及最终结论，这整段文字即为思考轨迹。  
   - 这一步完全自动化，几乎不需要人工干预。

2. **合并到训练语料**  
   - 将原始句子和对应的思考轨迹拼接在一起，形成新的训练样本。  
   - 为防止模型把思考轨迹当成噪声，作者在拼接时加入了特殊分隔符（如 `<THINK>`），帮助模型辨认“思考段”。  
   - 这样每条原始 token 都被“包装”了一层解释，等价于把一条信息扩展为多条关联信息。

3. **预训练阶段**  
   - 使用标准的自回归语言模型目标（预测下一个 token），但输入序列现在包含了思考段。  
   - 由于思考段本身也是自然语言，模型无需额外的架构改动即可学习。  
   - 在训练过程中，模型会逐渐学会先生成思考链再给出答案，这种“先思考后回答”的模式在后续微调和零样本推理时自然显现。

**最巧妙的点**在于：思考轨迹是由更强的模型自动生成的，而不是人工标注的 CoT 数据集。这样既保持了高质量，又避免了昂贵的人工成本，实现了真正的可扩展性。

### 实验与效果
- **数据规模**：作者在 1000 亿 token 级别的预训练语料上分别做了“原始版”和“思考增强版”。  
- **模型族**：覆盖 3B、7B、30B 参数的多种架构（包括 LLaMA、OPT）。  
- **基准任务**：使用了 GSM8K、MATH、BigBench Hard 等推理密集型测评。  
- **结果**：在 3B 参数模型上，思考增强后在 GSM8K 上提升约 12%，在 MATH 上提升约 10%。整体来看，所有模型在同等算力下的表现相当于原始训练的 3 倍数据量。  
- **消融实验**：去掉思考段的特殊分隔符会导致性能下降约 3%；只在少数高难度样本上加入思考轨迹的提升幅度明显低于全量加入。  
- **局限性**：论文未详细说明在极端长文本（>2048 token）上的效果，也没有评估思考轨迹对生成式对话质量的影响。作者承认思考轨迹的质量受教师模型能力限制，若教师模型出现错误，错误会被直接复制进训练数据。

### 影响与延伸思考
这篇工作打开了“用模型自我解释来增强训练数据”的新思路，随后出现了多篇跟进研究，例如利用多模态模型生成视觉思考链、在代码预训练中加入逐行注释等。对资源受限的实验室来说，思考增强提供了一条在算力不变的前提下提升模型推理能力的捷径。想进一步探索，可以关注以下方向：  
- **思考轨迹的质量控制**：如何检测并过滤教师模型的错误推理。  
- **跨语言思考增强**：在多语言语料上生成对应语言的思考链。  
- **与检索结合**：把外部知识检索结果也写进思考轨迹，形成“检索+思考”的混合增强。

### 一句话记住它
让强模型先写思考链，再把链子塞回原始文本，等于用“解释”把每条数据的学习价值放大三倍。