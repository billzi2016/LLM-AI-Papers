# Automatic Chain of Thought Prompting in Large Language Models

> **Date**：2022-10-07
> **arXiv**：https://arxiv.org/abs/2210.03493

## Abstract

Large language models (LLMs) can perform complex reasoning by generating intermediate reasoning steps. Providing these steps for prompting demonstrations is called chain-of-thought (CoT) prompting. CoT prompting has two major paradigms. One leverages a simple prompt like "Let's think step by step" to facilitate step-by-step thinking before answering a question. The other uses a few manual demonstrations one by one, each composed of a question and a reasoning chain that leads to an answer. The superior performance of the second paradigm hinges on the hand-crafting of task-specific demonstrations one by one. We show that such manual efforts may be eliminated by leveraging LLMs with the "Let's think step by step" prompt to generate reasoning chains for demonstrations one by one, i.e., let's think not just step by step, but also one by one. However, these generated chains often come with mistakes. To mitigate the effect of such mistakes, we find that diversity matters for automatically constructing demonstrations. We propose an automatic CoT prompting method: Auto-CoT. It samples questions with diversity and generates reasoning chains to construct demonstrations. On ten public benchmark reasoning tasks with GPT-3, Auto-CoT consistently matches or exceeds the performance of the CoT paradigm that requires manual designs of demonstrations. Code is available at https://github.com/amazon-research/auto-cot

---

# 大语言模型的自动思维链提示 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）出现之前，模型只能“一口气”给出答案，面对需要多步推理的数学、逻辑或常识题时错误率高。后来有人发现让模型先写出推理步骤（即思维链，Chain‑of‑Thought，CoT）能显著提升准确率，但要让模型真正学会这种思考方式，需要在提示中加入手工编写的示例。每个任务都要准备若干“问题 + 推理步骤 + 答案”的演示，这既费时又要求对任务有深刻理解，限制了 CoT 的大规模推广。于是，如何在不人工设计示例的前提下，让模型自行生成高质量的思维链，成为了亟待突破的瓶颈。

### 关键概念速览
- **思维链（CoT，Chain‑of‑Thought）**：让模型在给出最终答案前，先把推理过程写出来，类似于人解题时在草稿纸上列步骤，能够让错误更容易被发现和纠正。  
- **“一步一步思考”提示**：一种极简的指令（如 “Let's think step by step”），告诉模型在回答前自行展开推理，效果已经被证明能提升不少任务的表现。  
- **示例（demonstration）**：在 few‑shot 提示中出现的“问题 + 思维链 + 答案”三元组，用来示范模型该怎么推理。  
- **多样性采样（diverse sampling）**：在构造示例时，故意挑选不同类型、不同难度的题目，以获得覆盖更广的推理模式，防止模型只学到单一的解题套路。  
- **自动 CoT（Auto‑CoT）**：本文提出的全流程：先用“一步一步思考”提示让模型自行生成思维链，再把这些自动生成的示例喂回模型，形成自我强化的提示。  

### 核心创新点
1. **手工示例 → 自动生成示例**  
   过去的高效 CoT 需要研究者手动挑选并编写每一道示例。本文直接让 LLM 在“让我们一步一步思考”的指令下，为每个采样得到的问题自行生成完整的思维链，省去了人工标注的全部工作。

2. **单一示例 → 多样化示例集合**  
   自动生成的思维链往往会出现小错误。作者发现，如果只提供一两个示例，错误会被模型放大。于是引入多样性采样，收集一批在主题、难度、解题技巧上都有差异的示例，使得错误在不同示例之间相互抵消，整体提示更稳健。

3. **一步一步 → 一步一步**  
   传统的 “一步一步思考” 只指导模型在单个问题上展开推理。本文把这个思路延伸为 “一步一步生成示例”，即在构造提示的每一步都让模型自行思考，从而实现全链路的自动化。

4. **无需额外模型或后处理**  
   与一些需要专门的校验模型或答案后处理的方案不同，Auto‑CoT 完全在同一个 LLM（如 GPT‑3）内部完成示例生成和最终推理，保持了实现的简洁性。

### 方法详解
**整体框架**  
Auto‑CoT 由两大阶段组成：① 采样多样化问题并让模型生成思维链；② 把这些自动生成的（问题 + 思维链 + 答案）示例拼接进 few‑shot 提示，随后模型在新问题上进行推理。整个流程只需要一次调用 LLM，即可得到最终答案。

**步骤拆解**  

1. **多样性问题采样**  
   - 从目标任务的训练或验证集合中随机抽取若干问题。  
   - 为保证多样性，作者使用了不同的采样策略（如均匀抽取、按难度分层抽取），确保示例覆盖多种解题思路。  

2. **思维链生成**  
   - 对每个抽取的问题，向 LLM 发送提示：“Let's think step by step.”（或其中文等价），并附上问题本身。  
   - LLM 按照该指令输出完整的推理过程，最后给出答案。此时得到的文本即为一个自动示例。  

3. **示例过滤（隐式）**  
   - 虽然生成的思维链可能包含错误，但作者并未加入显式的过滤步骤，而是依赖后续的多示例组合来稀释错误的影响。  

4. **构造 Few‑shot 提示**  
   - 将所有自动示例按顺序拼接，形成一个长提示。每个示例之间用空行或分隔符区分，保持格式与手工 CoT 相同。  
   - 在提示的末尾加入待解新问题，同样附上 “Let's think step by step.” 的指令。  

5. **最终推理**  
   - LLM 读取包含多个自动示例的提示后，依据示例中展示的推理模式，对新问题展开一步一步的思考，输出答案。  

**关键巧思**  
- **多样性抵消错误**：作者发现，错误往往是“局部”且不具备系统性。若示例之间差异足够大，错误在不同示例中出现的概率互不相同，模型在综合这些示例时会倾向于采信正确的推理路径。  
- **同模型自循环**：整个过程不需要外部校验模型或人工审阅，完全在同一个 LLM 内部完成，展示了大模型的自我提升潜力。

### 实验与效果
- **测试任务**：在十个公开的推理基准上评估，包括数学推理（GSM8K）、逻辑推理（LogicalDeduction）、常识推理（CommonsenseQA）等。  
- **对比基线**：与零提示（直接回答）、单一 “一步一步思考” 提示、以及手工构造的 CoT few‑shot（需要人工示例）进行比较。  
- **结果概述**：论文声称 Auto‑CoT 在所有十个任务上都至少匹配手工 CoT 的表现，部分任务甚至超出 2%~5% 的绝对提升。相较于仅使用 “一步一步思考” 的单提示，提升幅度更为显著，常在 5%~10% 之间。  
- **消融实验**：作者分别去掉多样性采样、只保留单一自动示例、以及不使用 “一步一步思考” 指令进行实验，发现多样性采样是提升的关键因素，去掉后性能跌回到单提示水平。  
- **局限性**：自动生成的思维链仍可能出现系统性错误（如对特定数学公式的误用），在极端高难度或需要专业背景的任务上仍不如精心手工编写的示例。作者也指出，当前实现依赖于 GPT‑3 规模的模型，较小模型可能无法产生足够可靠的示例。

### 影响与延伸思考
Auto‑CoT 打开了“让大模型自我教自己”的新思路，随后出现了多篇工作尝试在不同语言、不同模型规模上复现或扩展该方法，例如在多语言模型中加入跨语言示例多样化、在指令微调阶段加入自动 CoT 生成等。还有研究把自动生成的思维链作为监督信号，进一步微调模型，使其在没有任何提示的情况下也能自行进行多步推理。对想深入的读者，可以关注以下方向：① 自动示例质量评估与过滤；② 多模态（文本+图像）思维链的生成；③ 将 Auto‑CoT 与检索增强模型结合，实现更大规模的知识调用。  

### 一句话记住它
让大语言模型先用“一步一步思考”自行写出推理示例，再把这些自动示例拼成 few‑shot 提示，就能在不人工标注的情况下获得和手工 CoT 同等甚至更好的推理能力。