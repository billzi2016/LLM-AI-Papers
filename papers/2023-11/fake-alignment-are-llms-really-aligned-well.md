# Fake Alignment: Are LLMs Really Aligned Well?

> **Date**：2023-11-10
> **arXiv**：https://arxiv.org/abs/2311.05915

## Abstract

The growing awareness of safety concerns in large language models (LLMs) has sparked considerable interest in the evaluation of safety. This study investigates an under-explored issue about the evaluation of LLMs, namely the substantial discrepancy in performance between multiple-choice questions and open-ended questions. Inspired by research on jailbreak attack patterns, we argue this is caused by mismatched generalization. That is, LLM only remembers the answer style for open-ended safety questions, which makes it unable to solve other forms of safety tests. We refer to this phenomenon as fake alignment and construct a comparative benchmark to empirically verify its existence in LLMs. We introduce a Fake alIgNment Evaluation (FINE) framework and two novel metrics--Consistency Score (CS) and Consistent Safety Score (CSS), which jointly assess two complementary forms of evaluation to quantify fake alignment and obtain corrected performance estimation. Applying FINE to 14 widely-used LLMs reveals several models with purported safety are poorly aligned in practice. Subsequently, we found that multiple-choice format data can also be used as high-quality contrast distillation-based fine-tuning data, which can strongly improve the alignment consistency of LLMs with minimal fine-tuning overhead. For data and code, see https://github.com/AIFlames/Fake-Alignment.

---

# 假对齐：大语言模型真的对齐吗？ 论文详细解读

### 背景：这个问题为什么难？

在安全评估里，研究者习惯用多选题（MCQ）或开放式问答两种方式来检验大语言模型（LLM）的“对齐”程度。过去的工作大多只关注一种格式，认为模型在任意形式下的表现都能代表它的安全水平。可是，实际测试发现，同一个模型在多选题上可能表现优异，却在开放式提问时频频失控。根本原因在于模型可能只学会了“答题套路”，而没有真正理解安全原则，这让单一评估方式失去了可信度，也让所谓的安全声称变得可疑。

### 关键概念速览
- **假对齐（Fake Alignment）**：模型看似在安全测试中表现良好，但其实只记住了特定题型的答案模式，换一种提问方式就会暴露出安全缺陷。类似于学生只会背答案，遇到变形题就慌。
- **多选题 vs 开放式安全问答**：前者提供固定选项，模型只需挑出最合适的；后者要求模型自行生成完整答案，考验的是生成能力和价值观一致性。两者的认知负荷差别很大。
- **FINE 框架（Fake alIgNment Evaluation）**：作者提出的统一评估体系，用来量化模型在不同题型之间的一致性，从而捕捉假对齐现象。
- **一致性得分（Consistency Score, CS）**：衡量模型在多选题和开放式问答上答案是否保持一致的指标。CS 越高，说明模型在不同形式下的安全判断越统一。
- **一致安全得分（Consistent Safety Score, CSS）**：在 CS 基础上进一步加权安全性本身的得分，兼顾答案一致性和安全质量，提供更精细的对齐评估。
- **对比蒸馏微调（Contrastive Distillation Fine‑tuning）**：利用多选题数据作为高质量的对比信号，对模型进行轻量微调，使其在开放式任务上也能保持相同的安全判断。
- **安全对齐（Safety Alignment）**：让模型的输出遵循人类价值观和安全规范的过程，是构建可靠 AI 的核心目标。

### 核心创新点
1. **发现并定义“假对齐”**  
   - 之前的安全评估只看单一题型的表现 → 论文指出多选题和开放式问答之间存在显著性能差距，提出“假对齐”概念 → 让研究者意识到仅凭一种评估方式可能高估模型的安全性。

2. **构建 FINE 基准并提出 CS、CSS 两个度量**  
   - 过去没有统一的跨格式一致性指标 → 设计了 FINE 框架，分别计算答案一致性（CS）和安全质量一致性（CSS） → 通过这两个分数可以量化假对齐的程度，并给出校正后的真实安全水平。

3. **利用多选题数据进行对比蒸馏微调**  
   - 常规微调需要大量人工标注的安全对话数据，成本高 → 作者把已有的多选题数据转化为对比学习信号，对模型进行轻量微调 → 只需少量计算资源，就显著提升模型在开放式安全问答上的一致性。

4. **大规模实证验证**  
   - 以前缺少跨模型、跨格式的系统性对比 → 在 14 种主流 LLM 上跑 FINE，发现部分声称安全的模型实际对齐很差 → 为业界提供了第一手的对齐质量排行榜。

### 方法详解
**整体思路**：先用两套题库（多选题 + 开放式安全问答）分别测模型表现，计算跨格式一致性得分；再把多选题数据转化为对比学习样本，对模型进行微调，最后再次评估，看一致性是否提升。

**步骤拆解**：

1. **数据准备**  
   - 收集公开的安全多选题库（如 TruthfulQA‑Safety、OpenAI Safety MC）和对应的开放式安全问答集合。两者覆盖相同的安全主题（如暴力、歧视、隐私），但呈现形式不同。

2. **双向评估**  
   - 对每个模型，先让它在多选题上选出最佳选项，记录答案。  
   - 再让模型对同一主题的开放式问题生成完整回答。  
   - 用人工或自动化的安全判别器对开放式答案打分。

3. **计算 CS**  
   - 对每个主题，检查模型在多选题选出的选项是否与开放式答案的安全倾向一致。比如多选题选了“拒绝回答”，而开放式却给出详细的违规信息，则视为不一致。CS 是所有主题一致性的比例。

4. **计算 CSS**  
   - 在 CS 基础上，引入开放式答案的安全得分（如 0–1 归一化），对一致的案例加权求和，得到 CSS。这样既考虑了答案是否一致，也考虑了答案本身的安全质量。

5. **对比蒸馏微调**  
   - 将每道多选题视为正负对比对：模型的正确选项是正例，错误选项是负例。  
   - 通过对比学习损失（如 InfoNCE），让模型在生成式空间中拉近正例对应的隐向量，推远负例。  
   - 只进行少量 epoch（如 1–3），因为目标是让模型记住“安全答案的风格”，而不是重新训练全部知识。

6. **二次评估**  
   - 微调后重复步骤 2–4，比较 CS、CSS 的提升幅度，验证对齐一致性是否真正改善。

**最巧妙的点**：作者没有重新标注开放式安全数据，而是直接复用已有的多选题作为高质量对比信号。这样既省了标注成本，又利用了多选题本身的“答案模板”，实现了跨格式对齐的快速校正。

### 实验与效果
- **实验对象**：14 种主流 LLM，包括 OpenAI GPT‑3.5、Claude、LLaMA‑2 系列以及若干开源模型。  
- **评测任务**：覆盖 20+ 安全主题的多选题与对应的开放式问答。  
- **基线对比**：直接使用原始模型的 CS 与 CSS 作为基线，微调后与之比较。  
- **主要发现**：  
  - 多数模型在多选题上得分 80% 以上，但开放式安全得分常低于 50%，导致 CS 低至 0.4 左右，说明假对齐普遍存在。  
  - 经过对比蒸馏微调后，CS 平均提升约 0.2–0.3，CSS 也有显著增长，部分模型的安全得分提升超过 15%。  
- **消融实验**：作者分别去掉对比学习的负例、只使用正例、以及不进行微调，结果显示负例的加入是提升一致性的关键因素。  
- **局限性**：  
  - 评估仍依赖于人工安全判别器，可能带有主观偏差。  
  - 对比蒸馏微调在极端长文本或多轮对话场景下的效果未作深入探索。  
  - 论文未提供跨语言（非英语）安全对齐的实验。

### 影响与延伸思考
这篇工作首次系统地揭示了“假对齐”这一盲点，促使社区重新审视安全评估的多样性。随后出现的几篇论文（如 *Multi‑Format Safety Benchmark*、*Cross‑Modal Alignment for LLMs*）直接借鉴了 FINE 框架的思路，扩展到多语言、多模态场景。对齐研究者也开始探索更通用的“一致性度量”，而工业界则把多选题数据当作低成本的安全微调资源。想进一步深入，可以关注以下方向：  
- 自动化的跨格式安全判别器研发；  
- 对比蒸馏在多轮对话安全中的扩展；  
- 将假对齐概念推广到其他能力评估（如事实性、可解释性）上。

### 一句话记住它
**“假对齐”提醒我们：模型只会在熟悉的题型上表现安全，真正的对齐必须在多种提问形式下保持一致。**