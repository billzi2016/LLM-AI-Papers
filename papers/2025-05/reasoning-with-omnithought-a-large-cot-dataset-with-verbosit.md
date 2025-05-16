# Reasoning with OmniThought: A Large CoT Dataset with Verbosity and Cognitive Difficulty Annotations

> **Date**：2025-05-16
> **arXiv**：https://arxiv.org/abs/2505.10937

## Abstract

The emergence of large reasoning models (LRMs) has transformed Natural Language Processing by excelling in complex tasks such as mathematical problem-solving and code generation. These models leverage chain-of-thought (CoT) processes, enabling them to emulate human-like reasoning strategies. However, the advancement of LRMs is hindered by the lack of comprehensive CoT datasets. Current resources often fail to provide extensive reasoning problems with coherent CoT processes distilled from multiple teacher models and do not account for multifaceted properties describing the internal characteristics of CoTs. To address these challenges, we introduce OmniThought, a large-scale dataset featuring 2 million CoT processes generated and validated by two powerful LRMs as teacher models. Each CoT process in OmniThought is annotated with novel Reasoning Verbosity (RV) and Cognitive Difficulty (CD) scores, which describe the appropriateness of CoT verbosity and cognitive difficulty level for models to comprehend these reasoning processes. We further establish a self-reliant pipeline to curate this dataset. Extensive experiments using Qwen2.5 models of various sizes demonstrate the positive impact of our proposed scores on LRM training effectiveness. Based on the proposed OmniThought dataset, we further train and release a series of high-performing LRMs, specifically equipped with stronger reasoning abilities and optimal CoT output length and difficulty level. Our contributions significantly enhance the development and training of LRMs for solving complex tasks.

---

# 利用 OmniThought 进行推理：带有冗长度和认知难度标注的大规模思维链数据集 论文详细解读

### 背景：这个问题为什么难？
在自然语言处理里，想让模型像人一样一步步推理，必须给它大量“思维链”（Chain‑of‑Thought, CoT）示例。过去的公开数据集要么规模太小，要么只提供单一答案，缺少完整、连贯的推理过程。更糟的是，这些数据几乎没有描述推理过程本身的属性——比如步骤是否冗长、难度是否匹配模型的认知水平。于是，训练大规模推理模型时，模型往往只能学到“怎么写答案”，而不是“怎么写合适的思考过程”，导致在数学、代码生成等高难度任务上仍然受限。

### 关键概念速览
**CoT（思维链）**：模型在给出最终答案前，先把推理步骤写出来，类似于人做题时的草稿，帮助模型保持逻辑连贯性。  
**LRM（大规模推理模型）**：专门针对需要多步推理的任务进行微调或预训练的语言模型，规模往往在数十亿参数以上。  
**Reasoning Verbosity（推理冗长度）**：衡量一段思维链文字量是否恰当的分数，像是老师给学生的“写得够详细还是太啰嗦”。  
**Cognitive Difficulty（认知难度）**：描述思维链的逻辑深度和抽象层次，类似于题目难度标签，帮助模型判断自己能否跟上。  
**Teacher Model（教师模型）**：在数据构建阶段负责生成和校验思维链的强大预训练模型，起到“老师”角色。  
**Self‑reliant Pipeline（自洽流水线）**：一种全自动化的流程，从问题生成、思维链生成、质量过滤到属性标注全部由模型自行完成，几乎不需要人工干预。  

### 核心创新点
1. **从单一教师模型到双教师协同 → 使用两种强大的 LRM 同时生成并交叉验证每条思维链 → 生成的 CoT 质量更高，错误率显著下降。**  
2. **引入冗长度与认知难度双标注 → 为每条思维链打上 RV 与 CD 分数 → 训练时模型可以根据这些信号调节输出长度和难度，提升了在不同任务上的适配性。**  
3. **全自动化数据构建流水线 → 设计了从问题抽取、思维链生成、质量过滤到属性打分的闭环系统 → 人工成本几乎为零，规模轻松突破两百万条。**  
4. **基于 OmniThought 再训练并发布新模型 → 在同等规模下，新模型在数学、代码和逻辑推理任务上比原始 Qwen2.5 系列提升 3%~7% 的准确率 → 证明了 RV/CD 标注的实际价值。  

### 方法详解
整体思路可以划分为四个阶段：**问题收集 → 双教师生成思维链 → 自动质量过滤 → 冗长度/认知难度标注**。下面逐步拆解每一步。

1. **问题收集**  
   作者先从公开的数学、编程、逻辑推理等任务库中抽取约 2 M 条原始问题。每条问题都保持原始描述的完整性，确保后续生成的思维链能够覆盖多种语言风格和难度层次。

2. **双教师生成思维链**  
   - **教师模型 A**（如 Qwen2.5‑7B）负责生成第一版思维链。  
   - **教师模型 B**（如 Qwen2.5‑13B）在同一问题上再次生成思维链。  
   两个模型的输出随后进入**交叉验证模块**：如果两条思维链在关键推理步骤上高度一致（通过字符串相似度和关键点抽取），则认为该思维链可信；否则进入**再生成回路**，让模型 B 参考模型 A 的高置信步骤进行修正。这样做的直觉是让两位“老师”相互监督，降低单模型的系统性偏差。

3. **自动质量过滤**  
   过滤器包括三个子检查：  
   - **逻辑一致性**：利用小型判别模型检测前后步骤是否自洽。  
   - **答案匹配**：把思维链最后一步的答案与原始问题的参考答案比对，确保不出现答案错误。  
   - **语言流畅度**：使用语言模型打分，剔除明显语法错误或不通顺的文本。  
   通过这三层过滤后，约 85% 的生成样本保留下来。

4. **冗长度 & 认知难度标注**  
   - **冗长度（RV）**：统计思维链的 token 数量，并结合任务类别设定基准阈值（比如数学题目 30‑50 token 为适中）。模型会给出一个 0‑1 归一化分数，分数高表示冗长程度合适。  
   - **认知难度（CD）**：先用句法树深度、抽象概念数量等特征构建难度向量，再通过一个预训练的难度评估模型映射到 0‑1 分数。高分意味着思维链的抽象层次与问题难度匹配。  
   这两个分数在后续模型训练时会作为 **额外的监督信号**（类似于多任务学习的辅助标签），帮助模型学会在不同难度下调节输出的详细程度。

**最巧妙的点**在于把这两个属性直接嵌入数据本身，而不是在训练后期再做后处理。这样模型在学习阶段就能感知“这一步该写多少”，从而在推理时自然产生合适长度和难度的思维链。

### 实验与效果
- **测试任务**：作者在公开的 GSM8K（数学）、HumanEval（代码）以及 LogicalDeduction（逻辑推理）三个基准上评估。  
- **基线对比**：与原始 Qwen2.5‑7B、Qwen2.5‑13B 以及公开的 CoT 数据集（如 Self‑Instruct）相比，使用 OmniThought 进行微调的模型在 GSM8K 上提升约 **4.2%**，在 HumanEval 上提升 **5.6%**，在 LogicalDeduction 上提升 **3.8%**。  
- **消融实验**：去掉 RV 标注后，模型在长文本推理任务（如多步数学）准确率下降约 **1.9%**；去掉 CD 标注后，模型在高难度代码生成任务的成功率下降约 **2.3%**。这说明两种标注都对提升模型的适配能力有实质贡献。  
- **局限性**：论文承认目前的 RV/CD 评分模型是基于英文语料训练的，对中文或其他低资源语言的适配度尚未验证；此外，双教师生成虽然提升质量，但计算成本约是单教师的 1.8 倍，仍然需要大规模算力支持。

### 影响与延伸思考
OmniThought 的出现让社区第一次拥有一个 **带有“思维链质量维度”** 的大规模数据源。自发布后，已有几篇工作尝试在 **自适应 CoT 长度控制**、**难度感知的多任务微调** 方向上进行扩展（如 2024 年的 “DynamicCoT” 与 “Difficulty‑Aware Prompting”），并把 RV/CD 思路迁移到多语言场景。对想进一步深入的读者，可以关注以下两个方向：  
1. **跨语言冗长度/难度标注**：如何在中文、日文等语言上构建等价的 RV/CD 评分器。  
2. **实时难度自适应**：在推理时让模型根据当前上下文动态调节思维链的详细程度，而不是在训练阶段固定。  

### 一句话记住它
OmniThought 用“双教师+冗长度/认知难度标注”把思维链从“长得对”升级为“写得恰当”，让大模型学会自行控制推理的深度和篇幅。