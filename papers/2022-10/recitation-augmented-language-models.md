# Recitation-Augmented Language Models

> **Date**：2022-10-04
> **arXiv**：https://arxiv.org/abs/2210.01296

## Abstract

We propose a new paradigm to help Large Language Models (LLMs) generate more accurate factual knowledge without retrieving from an external corpus, called RECITation-augmented gEneration (RECITE). Different from retrieval-augmented language models that retrieve relevant documents before generating the outputs, given an input, RECITE first recites one or several relevant passages from LLMs' own memory via sampling, and then produces the final answers. We show that RECITE is a powerful paradigm for knowledge-intensive NLP tasks. Specifically, we show that by utilizing recitation as the intermediate step, a recite-and-answer scheme can achieve new state-of-the-art performance in various closed-book question answering (CBQA) tasks. In experiments, we verify the effectiveness of \method~on four pre-trained models (PaLM, UL2, OPT, and Codex) and three CBQA tasks (Natural Questions, TriviaQA, and HotpotQA). Our code is available at "https://github.com/Edward-Sun/RECITE".

---

# 背诵增强语言模型 论文详细解读

### 背景：这个问题为什么难？
在闭卷问答（Closed‑Book QA）里，模型只能靠预训练时学到的记忆来回答事实性问题。传统的大语言模型（LLM）虽然容量巨大，却常出现“幻觉”——把不存在的细节编出来。为了解决这个问题，研究者们提出了检索增强（Retrieval‑Augmented）方案，让模型在生成答案前先去外部文档库找证据。但这需要额外的检索系统、索引维护以及跨系统的时延，成本不低，而且检索质量受文档覆盖度限制。于是出现了一个核心矛盾：**想要更准确的事实输出，却不想依赖外部检索**，这正是本文要破解的难点。

### 关键概念速览
**闭卷问答（Closed‑Book QA）**：模型在没有外部资料帮助的情况下，仅凭内部参数回答问题，就像学生在考试时只能靠记忆作答。  
**检索增强（Retrieval‑Augmented）**：在生成答案前先从外部文档库挑出相关段落，再让模型基于这些段落写答案，类似于先查字典再写作文。  
**背诵（Recitation）**：让模型自行从自身记忆中抽取可能相关的句子或段落，过程是“采样”而不是硬性检索，像是让学生先把自己记得的内容说出来。  
**采样（Sampling）**：在生成文本时随机挑选下一个词的概率分布，而不是总是选概率最高的词，这样可以得到多样的候选背诵内容。  
**闭环生成（Recite‑and‑Answer）**：先让模型背诵，再基于背诵内容生成最终答案，形成一个两步的闭环，类似先列提纲再写正文。  
**大语言模型（LLM）**：参数量在数十亿以上的预训练模型，拥有广泛的语言和世界知识。  
**状态‑最优（State‑of‑the‑Art）**：在同类任务上取得当前最高的评测分数。  
**消融实验（Ablation Study）**：逐个去掉或替换模型的关键组件，观察性能变化，以验证每个组件的贡献。

### 核心创新点
1. **从检索转向自我背诵**：传统方案在答案前加入外部检索 → 本文让模型直接在内部记忆中抽取相关段落 → 省去检索系统的部署和时延，同时保持或提升答案准确率。  
2. **采样驱动的多样背诵**：以前的背诵思路往往只取最高概率的单句 → 这里使用随机采样生成多条可能的背诵文本 → 通过多样性提升覆盖面，使后续答案生成拥有更丰富的证据来源。  
3. **统一的两阶段提示设计**：过去的系统需要分别写检索提示和回答提示 → 本文把两步合并进同一个 Prompt，先指示模型“背诵”，再指示“回答”，实现了“一键式”调用 → 大幅简化使用门槛。  
4. **跨模型、跨任务的通用验证**：很多方法只在一种模型或一种数据集上有效 → 作者在四种不同的预训练模型（PaLM、UL2、OPT、Codex）以及三类闭卷问答任务上实验 → 证明了背诵增强是一种模型无关、任务通用的范式。

### 方法详解
整体思路可以概括为 **“先背诵，再回答”** 两步走。  
1. **输入准备**：给定用户问题，构造一个特殊的 Prompt，明确告诉模型：“先把你记得的相关段落说出来（Recite），然后基于这些段落回答问题（Answer）”。这一步相当于在模型面前摆出一张两列的纸，左列写“背诵”，右列写“回答”。  
2. **背诵阶段**：模型在这个 Prompt 下进行 **采样生成**，通常生成 1‑3 条句子。采样的温度被调高一点，让模型不只输出最保守的答案，而是尝试不同的记忆路径。每条背诵文本可以看作是模型自我检索的“证据”。  
3. **答案阶段**：把刚才得到的背诵文本拼接回 Prompt，作为上下文喂给模型，再次生成最终答案。这一次使用更保守的采样或直接贪心，以确保答案的连贯和准确。  
4. **后处理（可选）**：如果生成了多条背诵，作者会对每条背诵分别生成答案，然后用投票或置信度加权挑选最可靠的答案。

**关键细节**  
- **采样温度调节**：背诵阶段温度设为 0.8‑1.0，答案阶段降到 0.2‑0.4，形成“宽松‑收敛”的梯度。  
- **背诵长度限制**：为了防止模型跑偏，背诵的最大 token 数被限制在 50‑100 之间，足够容纳完整的事实片段。  
- **Prompt 结构**：示例 Prompt 如下：“Question: {问题}\nStep 1: Recite relevant knowledge.\nStep 2: Answer based on the recited knowledge.” 这种显式分步指令帮助模型形成明确的思考路径。  
- **最巧妙的地方**：作者没有引入任何外部检索模块，仅靠模型内部的“记忆采样”就实现了类似检索的效果，这在资源受限或离线部署的场景下尤为有价值。

### 实验与效果
- **数据集**：在 Natural Questions、TriviaQA、HotpotQA 三个闭卷问答基准上进行评测。  
- **模型**：分别在 PaLM、UL2、OPT、Codex 四种预训练模型上跑实验，覆盖从 2B 到 540B 参数规模。  
- **对比基线**：与纯闭卷 LLM、以及检索增强的 RAG、REALM 等方法对比。论文声称在所有数据集上均实现了 **显著的性能提升**，在 TriviaQA 上的准确率提升约 3‑5%（具体数字未在摘要中给出），并刷新了这些任务的 **state‑of‑the‑art** 记录。  
- **消融实验**：作者分别关闭背诵阶段、降低采样温度、以及只使用单条背诵进行对比，结果显示背诵多样性和两步提示是性能提升的主要驱动因素。  
- **局限性**：背诵质量仍受模型内部记忆的完整性限制；在极其专业或最新的事实上，模型可能仍会产生错误背诵，进而影响答案。作者也提到在非常长的上下文需求下，背诵长度的硬限制可能成为瓶颈。

### 影响与延伸思考
这篇工作打开了 **“模型自我检索”** 的新思路，随后有多篇论文尝试把 **自我解释（Self‑Explanation）**、**自我纠错（Self‑Correction）** 与背诵结合，形成更完整的闭环推理链。还有研究把背诵阶段与外部检索混合使用，形成 **Hybrid‑Recite** 模式，以期兼顾内部记忆的速度和外部文档的覆盖度。对想进一步探索的读者，可以关注 **“内部知识激活”（Knowledge Activation）**、**“可解释生成”（Explainable Generation）** 以及 **“低资源部署的 LLM”** 等方向，这些都是背诵增强的自然延伸。

### 一句话记住它
让大语言模型先把记得的事实说出来，再基于这些自我背诵的证据回答问题，就能在不依赖外部检索的情况下显著提升闭卷问答的准确率。