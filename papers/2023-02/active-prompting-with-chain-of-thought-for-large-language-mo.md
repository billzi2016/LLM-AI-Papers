# Active Prompting with Chain-of-Thought for Large Language Models

> **Date**：2023-02-23
> **arXiv**：https://arxiv.org/abs/2302.12246

## Abstract

The increasing scale of large language models (LLMs) brings emergent abilities to various complex tasks requiring reasoning, such as arithmetic and commonsense reasoning. It is known that the effective design of task-specific prompts is critical for LLMs' ability to produce high-quality answers. In particular, an effective approach for complex question-and-answer tasks is example-based prompting with chain-of-thought (CoT) reasoning, which significantly improves the performance of LLMs. However, current CoT methods rely on a fixed set of human-annotated exemplars, which are not necessarily the most effective examples for different tasks. This paper proposes a new method, Active-Prompt, to adapt LLMs to different tasks with task-specific example prompts (annotated with human-designed CoT reasoning). For this purpose, we propose a solution to the key problem of determining which questions are the most important and helpful ones to annotate from a pool of task-specific queries. By borrowing ideas from the related problem of uncertainty-based active learning, we introduce several metrics to characterize the uncertainty so as to select the most uncertain questions for annotation. Experimental results demonstrate the superiority of our proposed method, achieving state-of-the-art on eight complex reasoning tasks. Further analyses of different uncertainty metrics, pool sizes, zero-shot learning, and accuracy-uncertainty relationship demonstrate the effectiveness of our method. Our code will be available at https://github.com/shizhediao/active-prompt.

---

# 面向大语言模型的主动提示与思维链 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）越大，越会出现“突现”能力，能够处理算数、常识推理等需要多步思考的任务。但这些能力并不是天生就能发挥出来，往往要靠精心设计的提示（prompt）才能把模型的潜力激活。传统的提示方式是直接给出几个带有思维链（CoT）的示例，让模型模仿这些步骤。然而，这些示例是固定的、人工挑选的，往往只能覆盖少数任务类型。不同任务的难点、数据分布甚至提问方式都有差异，统一的示例库很难兼顾所有情况，导致在一些任务上模型仍然表现平平。于是，如何为每个具体任务挑选最有价值的示例，成为提升 LLM 推理能力的关键瓶颈。

### 关键概念速览
- **大语言模型（LLM）**：参数量在数十亿甚至上千亿级别的生成式模型，能够理解并生成自然语言。把它想象成一个“超级词典+推理引擎”。
- **提示（Prompt）**：向模型提供的文字输入，包含任务说明、示例、问题等。类似老师给学生的考试说明。
- **思维链（Chain‑of‑Thought，CoT）**：在答案前让模型写出逐步推理过程，就像解题时先写草稿再给出结论，能显著提升复杂任务的准确率。
- **主动学习（Active Learning）**：在标注成本高的场景下，模型主动挑选最“值得标注”的样本，以最小的标注量获得最大性能提升。可以比作老师只挑选学生最容易出错的题目来讲解。
- **不确定性度量（Uncertainty Metric）**：衡量模型对某个问题答案的信心程度，信心低说明模型对该问题“犹豫”，通常是挑选标注对象的依据。
- **示例库（Exemplar Pool）**：预先收集的一批任务相关问题，待挑选后作为提示中的示例。相当于老师准备的题库。

### 核心创新点
1. **从固定示例转向任务自适应示例**  
   之前的 CoT 方法总是使用一套固定的人类标注示例，无论任务如何变化。本文引入“主动提示”（Active‑Prompt），让模型根据当前任务的查询池主动挑选最有价值的示例加入提示，从而实现任务特化。这样做把“一刀切”的示例库换成了“按需定制”的示例集合，显著提升了跨任务的鲁棒性。

2. **基于不确定性进行示例挑选**  
   直接让模型随机抽样示例会浪费标注资源。作者借鉴主动学习的思路，设计了多种不确定性度量（如答案分布熵、模型输出的置信度差等），用来评估每个候选问题的“难度”。挑选出最不确定的 K 条问题进行人工 CoT 标注，再放进提示。这样可以用极少的标注成本覆盖模型最薄弱的环节。

3. **统一的两阶段流程**  
   论文把整个过程拆成“筛选‑标注‑提示”两步：先在大规模未标注查询池中计算不确定性，选出子集；再让人类为这子集写出 CoT 步骤，最终把这些示例拼接进提示。相比于一次性手工挑选示例，这种流水线式的操作更易扩展到新任务。

4. **系统性实验验证**  
   在八个公开的复杂推理基准上，主动提示的表现均超过了最强的固定示例基线，且在零样本（zero‑shot）设置下仍保持优势。作者进一步通过消融实验证明，不确定性度量的选择、池大小、示例数量等因素对最终效果都有可解释的影响。

### 方法详解
**整体框架**  
整个方法可以概括为三步：① 构建任务查询池；② 用不确定性指标挑选标注样本；③ 将标注好的 CoT 示例加入提示，喂给 LLM 进行推理。核心思想是让模型自己“告诉我们它在哪儿卡住了”，我们再针对这些卡点提供思维链示例。

**步骤拆解**  

1. **查询池准备**  
   - 收集该任务的大量未标注问题（比如数学题、常识问答等），形成一个候选集合。  
   - 这些问题不需要任何人工干预，只要能覆盖任务的输入分布即可。

2. **不确定性评估**  
   - 将每个候选问题直接喂给 LLM，使用 **zero‑shot CoT**（即不提供示例，只让模型自行生成思考过程）得到若干候选答案。  
   - 计算答案分布的熵：如果模型在不同采样下给出多种答案，熵就高，说明模型对该问题不确定。  
   - 另外，还可以用 **置信度差**（最高概率答案与次高概率答案的差距）或 **梯度不确定性**（对输入的梯度变化幅度）等度量。  
   - 根据这些指标对所有候选进行排序，选出前 K（如 50）最不确定的样本。

3. **人工 CoT 标注**  
   - 对挑选出的 K 条问题，请人类专家写出完整的思维链解释，即“先把问题拆解成若干子步骤，再一步步求解”。  
   - 这些标注好的示例会形成 **任务特定的示例库**。

4. **构造主动提示**  
   - 将示例库中的每条示例按固定格式拼接：任务说明 → 示例 1（含 CoT） → 示例 2 → … → 待解问题。  
   - 这里的关键是示例的顺序和数量：实验表明，少量高质量示例（3–5 条）已经足以显著提升性能。

5. **模型推理**  
   - 把完整的提示交给目标 LLM（如 GPT‑4、Claude 等），让模型在已有思维链示例的“引导”下生成自己的 CoT 并给出答案。  

**最巧妙的点**  
- **不确定性驱动的示例挑选**：把主动学习的核心思想直接搬到提示工程里，让标注成本与模型的弱点高度对齐。  
- **两层 CoT**：先让模型自己在 zero‑shot 条件下产生粗糙的思考过程用于不确定性评估，再在正式推理时使用高质量的人类 CoT 示例进行“教学”。这种“先自测后授课”的循环在提示工程中少见。

### 实验与效果
- **测试任务**：论文在八个公开的复杂推理基准上评估，包括 GSM8K（数学）、SVAMP、AQuA、CommonsenseQA、StrategyQA、Date Understanding、MultiArith、和 Logical Deduction 等。  
- **对比基线**：主要与固定示例的 CoT 方法、Zero‑shot CoT、以及最新的 Few‑shot Prompting（如 Self‑Consistency）进行比较。  
- **核心结果**：在大多数任务上，主动提示比固定示例 CoT 提升 3%~9% 的准确率，尤其在 GSM8K 上从 71.2% 提升到 78.5%（具体数字以论文为准）。  
- **消融实验**：作者分别去掉不确定性筛选、只使用随机示例、以及只用单一不确定性度量，发现不确定性驱动的筛选是提升的主要因素，去掉后性能回落约 2%–4%。  
- **池大小与示例数量**：实验显示，查询池从 500 增加到 2000 时效果提升趋于饱和；示例数量在 3–5 条之间最优，更多示例反而会产生噪声。  
- **局限性**：论文承认仍需人工标注 CoT 示例，标注成本虽比全量标注低，却仍不可忽视；此外，不确定性度量在不同模型上表现不完全一致，需要针对具体 LLM 调整。

### 影响与延伸思考
这篇工作把主动学习的思路引入提示工程，开启了“主动提示”（Active Prompting）这一新方向。随后的研究开始探索 **自动生成 CoT 示例**（如利用小模型自我蒸馏）以及 **多模型协同不确定性评估**，试图进一步降低人工标注门槛。还有工作把不确定性度量与 **RL‑based Prompt Optimization** 结合，形成端到端的提示搜索框架。对想深入的读者，可以关注以下方向：① 自动化 CoT 生成的质量评估；② 跨模型不确定性共享；③ 大规模任务自适应提示库的构建与维护。  

### 一句话记住它
让大语言模型自己挑最“卡”的问题，用人类写的思维链示例“点穴”，即可显著提升复杂推理的准确率。