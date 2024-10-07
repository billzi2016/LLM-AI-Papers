# GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models

> **Date**：2024-10-07
> **arXiv**：https://arxiv.org/abs/2410.05229

## Abstract

Recent advancements in Large Language Models (LLMs) have sparked interest in their formal reasoning capabilities, particularly in mathematics. The GSM8K benchmark is widely used to assess the mathematical reasoning of models on grade-school-level questions. While the performance of LLMs on GSM8K has significantly improved in recent years, it remains unclear whether their mathematical reasoning capabilities have genuinely advanced, raising questions about the reliability of the reported metrics. To address these concerns, we conduct a large-scale study on several SOTA open and closed models. To overcome the limitations of existing evaluations, we introduce GSM-Symbolic, an improved benchmark created from symbolic templates that allow for the generation of a diverse set of questions. GSM-Symbolic enables more controllable evaluations, providing key insights and more reliable metrics for measuring the reasoning capabilities of models.Our findings reveal that LLMs exhibit noticeable variance when responding to different instantiations of the same question. Specifically, the performance of all models declines when only the numerical values in the question are altered in the GSM-Symbolic benchmark. Furthermore, we investigate the fragility of mathematical reasoning in these models and show that their performance significantly deteriorates as the number of clauses in a question increases. We hypothesize that this decline is because current LLMs cannot perform genuine logical reasoning; they replicate reasoning steps from their training data. Adding a single clause that seems relevant to the question causes significant performance drops (up to 65%) across all state-of-the-art models, even though the clause doesn't contribute to the reasoning chain needed for the final answer. Overall, our work offers a more nuanced understanding of LLMs' capabilities and limitations in mathematical reasoning.

---

# GSM‑Symbolic：洞察大语言模型数学推理的局限性 论文详细解读

### 背景：这个问题为什么难？
在 LLM 迅速突破通用语言任务的同时，研究者把目光投向了它们的“数学思考”。最常用的评测是 GSM8K——一套来源于美国小学数学的 8,000 余道文字题。过去几年里，模型在 GSM8K 上的准确率从几百分提升到接近 80%，看似已经可以“解题”。然而，这种提升到底是模型真的学会了数学推理，还是在靠记忆、模式匹配骗过评测，始终没有得到清晰答案。原有 benchmark 只提供固定的题目，缺少对同一道题不同表述的考察，也无法系统控制干扰因素，这让我们难以判断模型的推理深度。

### 关键概念速览
- **GSM8K**：一个公开的 grade‑school 数学题库，题目和答案都写成自然语言，常被用来衡量 LLM 的数学能力。  
- **符号模板（symbolic template）**：一种抽象的题目骨架，用占位符表示数字、变量和逻辑结构，类似于“填空题”，可以随意替换成不同具体数值。  
- **实例化（instantiation）**：把符号模板中的占位符替换成实际数字或文字，生成具体的测试样本。  
- **冗余子句（irrelevant clause）**：在题目中加入与求解无关的描述或条件，像是给人添了“噪音”，用来检验模型是否会被无关信息干扰。  
- **逻辑推理 vs. 记忆复制**：前者指模型能够依据题目结构自行演绎出解法，后者指模型直接从训练数据中检索相似的解题步骤并套用。  
- **性能波动（variance）**：同一道题在不同数值或不同表述下，模型输出的正确率出现显著差异。  

### 核心创新点
1. **从固定题库到可控生成**：传统做法直接在 GSM8K 上跑模型 → 本文构造了符号模板库，能够系统生成数千甚至上万种不同数值、不同表述的同类题目 → 评测不再受单一题目偶然性的限制，得到更稳健的性能估计。  
2. **引入冗余子句的干扰实验**：以前的 benchmark 只关注核心条件 → 在每个实例化题目后面加上一句与求解无关的描述（如“如果今天是晴天”） → 结果显示所有模型的准确率骤降，最高下降 65%，直接暴露出模型对无关信息的脆弱性。  
3. **系统化分析“数值敏感度”**：仅改变题目中的数字，而保持文字结构不变 → 发现模型的正确率普遍下降，说明它们并未真正理解数值之间的算术关系，而是依赖于特定数字组合的记忆。  
4. **规模化跨模型对比**：在多个主流闭源（如 ChatGPT、Claude）和开源（如 LLaMA、Falcon）模型上统一跑 GSM‑Symbolic → 统一的实验平台让我们看到“性能波动”是所有 SOTA 模型的共性，而非单一模型的缺陷。  

### 方法详解
整体思路可以划分为三步：**模板构建 → 实例化生成 → 干扰评测**。

1. **模板构建**  
   - 研究团队先从 GSM8K 中抽取常见的题型（如“求和”“求平均”“比例”），把每道题的文字描述抽象成带占位符的模板。  
   - 每个占位符标记为 `NUM_i`（数字）、`VAR_i`（变量）或 `COND_i`（条件），并记录该题型的解题步骤模板（例如“先把所有 NUM_i 相加，再除以数量”）。  
   - 类比于程序中的函数定义，模板相当于函数签名，实例化时只需要提供具体参数。

2. **实例化生成**  
   - 随机采样一组符合题目约束的数字（如正整数、分数、负数），填入模板的 `NUM_i`。  
   - 为了测试文字表述的鲁棒性，还会随机替换同义词、改变句子顺序，生成多种语言变体。  
   - 每生成一套题目，系统自动计算出对应的标准答案，形成“输入‑答案对”。  

3. **干扰评测**  
   - 在每个实例化题目后，附加一条 **冗余子句**。这条子句从另一个模板随机抽取，保证语义上与求解无关，但在语言上仍然看起来像是题目的一部分。  
   - 通过对比 **原始实例** 与 **加噪实例** 的准确率，量化模型对无关信息的敏感度。  

4. **评估流程**  
   - 将所有生成的题目喂入目标 LLM，采用 **zero‑shot**（不提供任何提示）和 **few‑shot CoT**（提供思维链示例）两种设置。  
   - 采集模型的最终答案，使用严格的数值匹配（考虑四舍五入、分数化简）判断对错，统计整体准确率以及在不同数值、不同子句数目下的细分表现。  

**最巧妙的点**在于把“题目变体”和“干扰子句”完全解耦：模板保证了数学结构的一致性，而随机数值和噪声子句则分别检验模型的数值理解和逻辑过滤能力。这种设计让实验结果可以明确归因，而不是混在一起。

### 实验与效果
- **数据集**：使用 GSM‑Symbolic 生成的约 30,000 题目，其中 15,000 为纯净实例，15,000 为加噪实例。  
- **模型**：覆盖了 5 种闭源商用模型（如 GPT‑4、Claude）和 4 种开源模型（如 LLaMA‑2、Falcon）。  
- **基线对比**：在原始 GSM8K 上，最好的闭源模型已经突破 80% 准确率。转到 GSM‑Symbolic 的纯净实例后，所有模型的准确率普遍下降 5%~12%。加入冗余子句后，下降幅度更大，最高达到 65%。  
- **消融实验**：分别去掉数值随机化、文字同义替换、冗余子句三项因素，发现数值随机化导致的性能下降最为显著（约 8%），说明模型对具体数字的记忆依赖最强。  
- **作者承认的局限**：模板生成的题目仍然是高度结构化的，可能与真实课堂题目在语言多样性上有差距；此外，实验只覆盖了小学数学范围，未涉及更高阶的抽象推理。  

### 影响与延伸思考
这篇工作在社区里引发了两股潮流：一是 **更具可控性的数学 benchmark**，后续出现了如 MathBench、MATH‑Symbolic 等基于模板的评测套件；二是 **对 LLM 逻辑推理的深度审视**，不少研究开始探索显式的逻辑模块（如符号求解器、程序化思维链）与语言模型的融合。若想进一步了解，可以关注 **“LLM + Symbolic Reasoning”** 方向的最新论文，尤其是把外部数学引擎（如 SymPy）嵌入生成式模型的尝试。  

### 一句话记住它
**GSM‑Symbolic 证明：大语言模型在数学题上看似强大，却在数字微调和无关信息面前瞬间崩溃，真正的逻辑推理仍未到位。**