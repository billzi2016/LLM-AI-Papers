# Synthetic Prompting: Generating Chain-of-Thought Demonstrations for   Large Language Models

> **Date**：2023-02-01
> **arXiv**：https://arxiv.org/abs/2302.00618

## Abstract

Large language models can perform various reasoning tasks by using chain-of-thought prompting, which guides them to find answers through step-by-step demonstrations. However, the quality of the prompts depends on the demonstrations given to the models, and creating many of them by hand is costly. We introduce Synthetic prompting, a method that leverages a few handcrafted examples to prompt the model to generate more examples by itself, and selects effective demonstrations to elicit better reasoning. Our method alternates between a backward and forward process to generate new examples. The backward process generates a question that match a sampled reasoning chain, so that the question is solvable and clear. The forward process produces a more detailed reasoning chain for the question, improving the quality of the example. We evaluate our method on numerical, symbolic, and algorithmic reasoning tasks, and show that it outperforms existing prompting techniques.

---

# 合成提示：为大语言模型生成思维链示例 论文详细解读

### 背景：这个问题为什么难？
在大语言模型（LLM）上使用思维链（Chain‑of‑Thought，CoT）提示可以让模型一步步推理，从而在数学、逻辑等复杂任务上取得更高准确率。但 CoT 的效果高度依赖于示例质量：如果示例本身推理不完整或表述晦涩，模型往往会复制错误的思路。手工收集、编写大量高质量 CoT 示例成本极高，尤其是面对多种任务时更是难上加难。于是出现了“少量示例、批量生成”的需求：如何让模型自己产生更多可靠的 CoT 示例，而不牺牲推理质量？

### 关键概念速览
**思维链（Chain‑of‑Thought，CoT）**：让模型在给出最终答案前先写出逐步推理过程，类似人做数学题时的草稿，能够提升模型的逻辑连贯性。  
**提示（Prompt）**：向模型提供的输入文本，包含任务说明、示例等，用来引导模型的输出。  
**合成示例（Synthetic Example）**：不是人工编写，而是模型自行生成的示例，包括问题、推理链和答案。  
**前向过程（Forward Process）**：从已有问题出发，让模型扩展出更详细、更完整的推理链。  
**逆向过程（Backward Process）**：从一段已知的推理链出发，逆向生成一个对应的问题，使得该链成为该问题的自然解答。  
**示例筛选（Demo Selection）**：在大量合成示例中挑选出最能激发模型正确推理的若干条，作为最终的提示示例。  
**Few‑shot Prompting**：在提示中只放入少量示例（几条），让模型在此基础上完成新任务。

### 核心创新点
1. **逆向‑前向双向生成 → 通过逆向过程先生成“可解”问题，再用前向过程细化推理链 → 合成的示例兼具可解性和推理深度，克服了单向生成常出现的“答案不匹配”或“推理不完整”问题。  
2. **示例自适应筛选 → 在大量合成示例中使用模型自身的评估（如答案一致性、推理连贯性）挑选出最具诱导力的几条 → 只保留高质量示例，显著提升了最终 Few‑shot CoT 的效果。  
3. **少量手工种子驱动 → 只需要几条人工编写的高质量 CoT 示例作为种子，剩余示例全部由模型生成 → 大幅降低了人工标注成本，同时保持了示例的任务覆盖面。  
4. **任务通用框架 → 方法在数值推理、符号推理和算法推理三个截然不同的任务上均可直接套用 → 证明了合成提示的跨任务可迁移性。

### 方法详解
整体思路可以概括为“三步走”：**种子准备 → 双向合成 → 示例筛选 → 形成最终提示**。下面逐层拆解。

1. **种子准备**  
   研究者先手工挑选 3‑5 条高质量的 CoT 示例，这些示例覆盖目标任务的核心推理模式。每条示例包括：问题、逐步推理链、最终答案。种子数量少到可以在几分钟内完成，却足以让模型学习到任务的基本解题套路。

2. **逆向过程（Backward Generation）**  
   - **输入**：从种子示例中抽取一段完整的推理链（不包括问题本身）。  
   - **目标**：让模型生成一个“对应的问题”，使得这段推理链成为该问题的自然解答。  
   - **实现**：在提示中写明“下面是一段推理过程，请给出一个能被这段过程解答的问题”。模型会依据语言模式逆向构造问题，确保问题的表述清晰、答案唯一。  
   - **直觉类比**：就像老师先写出解题步骤，再让学生想出对应的题目，确保步骤真的能解这道题。

3. **前向过程（Forward Generation）**  
   - **输入**：逆向过程得到的问题。  
   - **目标**：让模型在该问题上生成更细致、更完整的思维链。  
   - **实现**：提示模型“请一步步解答下面的问题，并写出每一步的推理”。模型会在原有推理链的基础上补充遗漏的中间计算、解释或检查步骤，使示例更具教学价值。  
   - **关键点**：前向过程使用的是与种子相同的 CoT 提示模板，保证生成的推理风格与种子保持一致。

4. **示例筛选（Demo Selection）**  
   - **生成池**：逆向‑前向循环可以产生上百条合成示例。  
   - **筛选标准**：  
     a. **答案一致性**：模型再次对问题进行推理，若得到的答案与前向过程的答案相同，则视为自洽。  
     b. **推理连贯性**：使用语言模型评分（如 perplexity）或专门的逻辑一致性检测器，剔除逻辑跳跃或表述模糊的示例。  
     c. **多样性**：确保最终挑选的示例在问题类型、数值范围、推理路径上有一定差异，防止提示过于单一。  
   - **最终提示**：从筛选后保留的示例中抽取 k 条（常见取 4‑8 条），与任务描述一起拼接成 Few‑shot CoT 提示，送入目标模型进行推理。

**最巧妙的地方**在于逆向过程的“问题生成”。传统的 CoT 扩展只会让模型继续写更长的推理，却不保证对应的问题本身是合理的。通过先固定推理链，再让模型逆向构造问题，作者确保每条合成示例都是“可解且清晰”的，从根本上提升了示例质量。

### 实验与效果
- **测试任务**：数值推理（如算术题、单位换算）、符号推理（如代数化简、逻辑公式求值）以及算法推理（如排序、图遍历的步骤描述）。  
- **对比基线**：原始 Few‑shot CoT（仅使用人工示例）、Self‑Consistency（多次采样后投票）、Chain‑of‑Thought Prompting with Retrieval（从外部数据库检索示例）等。  
- **论文声称**：在上述三类任务上，Synthetic Prompting 均超过了最强基线，提升幅度在 5%~15% 之间，尤其在需要长链推理的算法任务上表现最为突出。  
- **消融实验**：去掉逆向过程、只保留前向生成或不做示例筛选时，性能显著下降，验证了每个模块的必要性。  
- **局限性**：方法依赖于模型本身的生成能力；在极端复杂或高度专业化的任务上，逆向生成的“问题”可能不够自然；此外，筛选过程仍需要额外的计算资源。

### 影响与延伸思考
Synthetic Prompting 为“让模型自我扩展提示库”提供了可行路径，随后出现的工作多聚焦于：① 更高效的逆向生成（如使用结构化模板），② 跨语言或跨模态的合成示例生成，③ 将合成示例与检索式提示结合形成混合系统。对想进一步探索的读者，可以关注 **自监督提示生成**、**多模态 CoT** 以及 **大模型自校准** 等方向，这些都是在 Synthetic Prompting 思路上自然延伸的研究热点。

### 一句话记住它
让模型先写好推理链，再逆向生成对应问题，循环产生高质量 CoT 示例，从而用极少的人工示例实现大幅提升的推理能力。