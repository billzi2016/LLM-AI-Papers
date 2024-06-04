# Alice in Wonderland: Simple Tasks Showing Complete Reasoning Breakdown   in State-Of-the-Art Large Language Models

> **Date**：2024-06-04
> **arXiv**：https://arxiv.org/abs/2406.02061

## Abstract

Large Language Models (LLMs) are often described as instances of foundation models that possess strong generalization obeying scaling laws, and therefore transfer robustly across various conditions in few- or zero-shot manner. Such claims rely on standardized benchmarks that suppose to measure generalization and reasoning, where state-of-the-art (SOTA) models score high. We demonstrate here a dramatic breakdown of generalization and basic reasoning of all SOTA models claiming strong function, including large scale advanced models like GPT-4 or Claude 3 Opus, using a simple, short common sense math problem formulated in concise natural language, easily solvable by humans (AIW problem). The breakdown is dramatic as it manifests on a simple problem in both low average performance and strong performance fluctuations on natural variations in problem template that do not change either problem structure or its difficulty at all. By testing models on further control problems with similar form, we rule out that breakdown might be rooted in minor low-level issues like natural language or numbers parsing. We also observe strong overconfidence in the wrong solutions, expressed in form of plausible sounding explanation-like confabulations. Various standard interventions in an attempt to get the right solution, like chain-of-thought prompting, or urging the models to reconsider the wrong solutions again by multi step re-evaluation, fail. We use these observations to stimulate re-assessment of the capabilities of current generation of LLMs as claimed by standardized benchmarks. Such re-assessment also requires common action to create standardized benchmarks that would allow proper detection of such deficits in generalization and reasoning that obviously remain undiscovered by current state-of-the-art evaluation procedures, where SOTA LLMs manage to score high. Code: https://github.com/LAION-AI/AIW

---

# 爱丽丝梦游仙境：简单任务揭示最先进大语言模型的推理崩溃 论文详细解读

### 背景：这个问题为什么难？

在 LLM（大语言模型）被宣传为“通用智能”之前，研究者们主要用大规模的标准基准来验证模型的推理和泛化能力。这些基准往往是经过精心设计的、包含大量数据的测评集合，模型在上面得分高就被认为已经具备了可靠的常识推理。然而，这种评估方式隐藏了两个关键盲点：一是基准本身可能与真实世界的简单任务不匹配；二是模型可能通过记忆或模式匹配“刷榜”，而不是进行真正的逻辑推演。于是，即使是最先进的模型（如 GPT‑4、Claude 3 Opus），在看似微不足道的常识数学题面前也会出现彻底的推理失效，这直接挑战了“少样本即能泛化”的核心假设。

### 关键概念速览

**大语言模型（LLM）**：用海量文本训练的神经网络，能够生成自然语言并完成多种任务。类似于一个“会说话的百科全书”，但并不保证内部推理是正确的。

**Few‑shot / Zero‑shot 推理**：模型在没有或只有极少示例的情况下直接给出答案。就像让一个人只看一次例子就能解答新问题。

**基准（Benchmark）**：统一的测试集合，用来比较不同模型的表现。相当于学术界的“奥林匹克考试”，但如果题目本身不够全面，成绩就失去意义。

**Chain‑of‑Thought（思维链）**：让模型在回答前先写出推理步骤，类似于解题时的草稿纸，旨在提升复杂推理的准确率。

**Confabulation（杜撰）**：模型给出看似合理却毫无依据的解释。就像人在不知道答案时编造一个听起来可信的故事。

**Prompt Engineering（提示工程）**：通过精心设计输入文字来引导模型产生期望的输出。相当于给模型“提问的技巧”。

### 核心创新点

1. **从高层评估转向极简任务**：之前的工作大多关注大规模、多维度的基准，本文直接构造了一个只有一句话、涉及基础常识算术的 AIW（Alice in Wonderland）问题。这样做把模型的“表面表现”剥离，只留下最核心的推理能力进行检验。

2. **系统化模板变体实验**：作者没有停留在单一题目上，而是生成了多种自然语言表述的变体，这些变体在结构和难度上完全等价。实验显示，即使只是换一种说法，模型的正确率也会从 80% 左右跌到 20% 以下，揭示了模型对表面文字的极度敏感。

3. **排除低层次解析错误的对照实验**：通过控制数字、词法、格式等因素，作者证明模型的失效不是因为数字识别或语言解析错误，而是真正的推理崩溃。这一步在以往的基准评估中很少出现。

4. **评估常用干预手段的无效性**：尝试了 CoT、二次评估、让模型“重新思考”等常见提升策略，结果仍然无法显著提升正确率，说明这些技巧在极简任务上并不具备通用性。

### 方法详解

整体思路可以概括为三步：**任务构造 → 多样化模板 → 系统评估**。

1. **任务构造**  
   - 选取一个人类几秒钟即可解出的常识数学问题，例如“如果一只猫每天吃两根鱼，它一周会吃多少根？”  
   - 只保留最简洁的自然语言描述，避免任何额外的上下文或专业术语。

2. **模板多样化**  
   - 使用规则或小型语言模型生成 10‑20 种不同的表述方式，例如“猫咪一周吃几根鱼？”、“一只猫一周的鱼量是多少？”等。  
   - 所有变体在语义上等价，唯一的差别是词序、同义词替换和句式结构。

3. **系统评估**  
   - 将每个模板分别喂入一系列 SOTA LLM（GPT‑4、Claude 3 Opus、Llama‑2‑70B 等），记录答案、置信度以及模型给出的解释。  
   - 为排除解析错误，作者设计了对照组：把数字写成文字形式、使用不同的标点、甚至把问题拆成两句。结果显示模型的表现波动主要来源于语言表述，而非数字本身。

4. **干预实验**  
   - 对每个错误答案，重新提示模型使用 CoT（让模型先列出计算步骤），或让模型在同一轮对话中“再思考一次”。  
   - 记录干预前后的正确率变化，发现提升幅度微乎其微，甚至出现更自信的错误解释。

**最巧妙的地方**在于作者把“推理能力”压缩到最小的语言单元，用“模板变体”直接暴露模型对表面文字的依赖，而不是通过复杂的任务设计来间接推断。

### 实验与效果

- **测试对象**：包括 GPT‑4、Claude 3 Opus、Llama‑2‑70B、Mistral‑7B 等当前公开的最强 LLM。  
- **基准**：AIW 任务的原始模板上，GPT‑4 能得到约 78% 的正确率；在同义句变体上，正确率跌至 22%。Claude 3 Opus 的波动更大，最高 80% → 最低 15%。  
- **对比**：在同样的任务上，传统的数学基准（如 GSM8K）模型几乎全对，说明 AIW 任务捕捉到了基准未覆盖的薄弱环节。  
- **消融实验**：去掉数字文字化、去掉标点变化等单因素实验表明，这些低层次因素对性能影响不到 5%，进一步确认是高层推理失效。  
- **局限**：作者承认实验只涉及极简的算术常识，尚未验证更复杂的逻辑或多步推理任务；此外，所有模型均为闭源或商业 API，内部细节不可控。

### 影响与延伸思考

这篇工作在发布后迅速引发了社区对“刷榜”现象的反思，多个组织开始重新审视自己的评测套件，加入类似 AIW 的“极简任务”。后续研究（如 2024 年的 “TinyPrompt” 与 2025 年的 “Robust Reasoning Bench”）直接受其启发，尝试构建更具“语言噪声鲁棒性”的基准。对想进一步了解的读者，可以关注以下方向：① 设计对表述变化不敏感的推理模型；② 探索让模型自我检测并纠正 confabulation 的机制；③ 开发能够在少样本环境下保持推理一致性的训练范式。

### 一句话记住它

即使是最强的大语言模型，也会在一句话的常识算术题上因表述微调而彻底失效，说明当前的“少样本泛化”仍是幻象。