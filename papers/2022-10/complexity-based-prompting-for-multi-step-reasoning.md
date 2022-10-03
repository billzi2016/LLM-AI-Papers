# Complexity-Based Prompting for Multi-Step Reasoning

> **Date**：2022-10-03
> **arXiv**：https://arxiv.org/abs/2210.00720

## Abstract

We study the task of prompting large-scale language models to perform multi-step reasoning. Existing work shows that when prompted with a chain of thoughts (CoT), sequences of short sentences describing intermediate reasoning steps towards a final answer, large language models can generate new reasoning chains and predict answers for new inputs. A central question is which reasoning examples make the most effective prompts. In this work, we propose complexity-based prompting, a simple and effective example selection scheme for multi-step reasoning. We show that prompts with higher reasoning complexity, i.e., chains with more reasoning steps, achieve substantially better performance on multi-step reasoning tasks over strong baselines. We further extend our complexity-based criteria from prompting (selecting inputs) to decoding (selecting outputs), where we sample multiple reasoning chains from the model, then choose the majority of generated answers from complex reasoning chains (over simple chains). When used to prompt GPT-3 and Codex, our approach substantially improves multi-step reasoning accuracy and achieves new state-of-the-art (SOTA) performance on three math benchmarks (GSM8K, MultiArith, and MathQA) and two BigBenchHard tasks (Date Understanding and Penguins), with an average +5.3 and up to +18 accuracy improvements. Compared with existing example selection schemes like manual tuning or retrieval-based selection, selection based on reasoning complexity is intuitive, easy to implement, and annotation-efficient. Further results demonstrate the robustness of performance gains from complex prompts under format perturbation and distribution shift.

---

# 基于复杂度的多步推理提示 论文详细解读

### 背景：这个问题为什么难？
在让大语言模型（LLM）解答需要多步推理的题目时，模型往往直接给出答案，却忽略了中间的思考过程，导致错误率居高不下。早期的“思维链”（Chain‑of‑Thought，CoT）技巧通过让模型先写出推理步骤，显著提升了数学和逻辑题的表现，但仍存在一个关键盲点：到底该挑选哪些示例来做提示？过去的做法要么靠人工经验手动挑选，要么使用检索系统找相似问题，却没有系统地衡量示例本身的推理难度。缺少对示例“复杂度”的认识，使得提示的质量参差不齐，限制了多步推理的上限。

### 关键概念速览
**CoT（思维链）**：让模型在给出最终答案前先把每一步推理写出来，类似于人解题时的草稿本，帮助模型保持逻辑连贯。  
**Prompt（提示）**：向模型提供的输入文本，包括任务描述、示例和问题本身，起到“引导”模型思考的作用。  
**示例选择（Example Selection）**：在 few‑shot 场景下，从大量可用的示例中挑出几条放进提示里，决定模型的参考框架。  
**推理复杂度（Reasoning Complexity）**：指一个思维链包含的推理步骤数，步骤越多，链条越长，代表更高的逻辑深度。  
**解码策略（Decoding Strategy）**：模型生成答案的方式，例如一次性采样、束搜索或后处理投票。  
**多数投票（Majority Voting）**：对多次生成的答案取出现次数最多的那个，以降低随机噪声的影响。  
**分布漂移（Distribution Shift）**：训练或检索得到的示例与真实测试题目在风格或难度上的差异，常导致性能下降。  

### 核心创新点
1. **从“多少步”出发挑选示例**  
   之前的工作多关注示例与目标问题的表面相似度，忽略了示例内部的推理深度。作者提出直接统计每条 CoT 中的句子或步骤数，优先把步骤多的示例放进提示。这样做让模型在提示阶段就感受到“需要更细致的思考”，从而在后续生成时倾向于展开更完整的推理链。实验显示，使用高复杂度示例的提示在所有测试集上平均提升约 5% 以上。

2. **把复杂度延伸到解码阶段**  
   传统的解码只关注生成的答案本身，或在多个候选答案上做简单投票。本文在采样出若干完整的思维链后，先筛选出步骤数较多的链，再在这些链对应的答案上执行多数投票。相当于让“更长的思考”拥有更大的话语权，进一步压制了短浅、偶然正确的答案。

3. **统一的复杂度度量与实现**  
   作者没有引入额外的模型或人工标注，只用一个轻量的计数器统计每条 CoT 的步骤数。整个流程可以在几行代码里完成，兼容任何支持 few‑shot 的 LLM（如 GPT‑3、Codex），实现成本极低，却带来显著的性能提升。

4. **对抗格式扰动和分布漂移的鲁棒性**  
   为验证方法的通用性，作者在提示格式被随机打乱、或在与训练数据分布不同的任务上进行实验，发现基于复杂度的选择仍保持优势，说明该策略并非依赖特定数据形态，而是捕捉到了更根本的推理需求。

### 方法详解
整体思路可以划分为两大阶段：**示例挑选**和**答案生成与筛选**。下面按步骤展开。

1. **准备候选示例库**  
   收集一批已经标注好的 CoT 示例，每条示例包括问题、完整的思维链和最终答案。这里不要求示例与目标问题相似，只要覆盖任务类型即可。

2. **计算推理复杂度**  
   对每条示例的思维链进行句子切分（或使用特殊标记 `\n`），统计得到的步骤数。步骤数即为该示例的复杂度得分。实现上，只需遍历字符串并计数，几乎不消耗算力。

3. **构造提示**  
   按复杂度从高到低排序，取前 K 条（K 通常为 4~8）作为 few‑shot 示例，拼接到提示的开头。提示的其余部分是任务说明和待解问题。因为示例本身已经是“长思维链”，模型在阅读提示时自然会被“长链”诱导。

4. **多链采样**  
   使用温度采样或束搜索，让模型针对同一问题生成 N 条完整的思维链（N 常取 10~20）。每条链都包含从第一步到答案的完整过程。

5. **复杂度过滤**  
   对这 N 条生成的链再次计数步骤数，保留步骤数在前 M%（如前 50%）的链。这样做的直觉是：如果模型自己产生了一个短链，说明它可能在偷懒或未充分展开推理。

6. **多数投票决定答案**  
   在保留下来的链对应的答案上执行多数投票，得到最终输出。如果出现平票，可回退到全部链投票或直接选取最高置信度的答案。

**最巧妙的点**在于把“复杂度”这一本来只用于示例挑选的度量，贯穿到生成后处理阶段，实现了从输入到输出的全链路一致性。整个流程不需要额外的训练，只是对提示和解码做了两次轻量的过滤。

### 实验与效果
- **测试任务**：三大数学基准（GSM8K、MultiArith、MathQA）以及 BigBenchHard 中的 Date Understanding 与 Penguins 两项。前者要求逐步算数推理，后者涉及日期计算和逻辑归纳。
- **基线对比**：与普通 few‑shot CoT、手工挑选示例的 CoT、以及检索式示例选择（基于向量相似度）进行比较。  
- **主要结果**：在 GPT‑3 与 Codex 上，使用复杂度提示的模型在 GSM8K 上提升约 7.2%（从 71% 到 78%），在 MultiArith 提升 6.5%，MathQA 提升 5.8%。在 BigBenchHard 两项任务上，最高提升达 18%（Penguins）。整体平均提升约 5.3%。  
- **消融实验**：去掉复杂度过滤，仅保留高复杂度示例，提升仍在 3% 左右；去掉解码阶段的复杂度投票，提升回落约 2%。说明两阶段都贡献显著。  
- **鲁棒性测试**：在提示格式被随机打乱（如示例顺序、换行符位置变化）后，性能下降不到 1%；在使用与训练分布不同的数学题目（如新出题库）时，仍保持 4% 左右的优势。  
- **局限性**：作者指出，复杂度度量仅基于步骤数，未考虑步骤质量；在极端长链（>30 步）时，采样成本会显著上升，且多数投票可能被少数错误链主导。原文未提供对非常大模型（如 GPT‑4）或非英文任务的实验。

### 影响与延伸思考
这篇工作把“思考深度”直接量化并用于提示设计，开启了“复杂度感知提示”（complexity‑aware prompting）的新方向。随后的研究开始探索更细粒度的复杂度指标，如逻辑层级、公式数量或信息增益，并尝试将复杂度预测模型嵌入到自动提示生成系统中。还有工作把复杂度与任务难度标签结合，做出自适应的 few‑shot 规模调节。想进一步了解，可以关注以下方向：  
- **复杂度预测模型**：训练小模型估计给定问题需要多少推理步骤，从而动态决定提示长度。  
- **跨语言复杂度提示**：验证中文、日文等非英文任务中，步骤数是否同样是有效的选择信号。  
- **与检索结合**：将向量相似度检索与复杂度过滤并行，兼顾语义相似和思考深度。  
- **大模型内部解释**：利用注意力或激活模式直接衡量模型内部的推理深度，形成更精准的复杂度度量。

### 一句话记住它
让模型看到“更长的思考链”，它就会自己写出更长的思考链，从而显著提升多步推理的准确率。