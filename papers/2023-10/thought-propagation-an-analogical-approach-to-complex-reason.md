# Thought Propagation: An Analogical Approach to Complex Reasoning with   Large Language Models

> **Date**：2023-10-06
> **arXiv**：https://arxiv.org/abs/2310.03965

## Abstract

Large Language Models (LLMs) have achieved remarkable success in reasoning tasks with the development of prompting methods. However, existing prompting approaches cannot reuse insights of solving similar problems and suffer from accumulated errors in multi-step reasoning, since they prompt LLMs to reason \textit{from scratch}. To address these issues, we propose \textbf{\textit{Thought Propagation} (TP)}, which explores the analogous problems and leverages their solutions to enhance the complex reasoning ability of LLMs. These analogous problems are related to the input one, with reusable solutions and problem-solving strategies. Thus, it is promising to propagate insights of solving previous analogous problems to inspire new problem-solving. To achieve this, TP first prompts LLMs to propose and solve a set of analogous problems that are related to the input one. Then, TP reuses the results of analogous problems to directly yield a new solution or derive a knowledge-intensive plan for execution to amend the initial solution obtained from scratch. TP is compatible with existing prompting approaches, allowing plug-and-play generalization and enhancement in a wide range of tasks without much labor in task-specific prompt engineering. Experiments across three challenging tasks demonstrate TP enjoys a substantial improvement over the baselines by an average of 12\% absolute increase in finding the optimal solutions in Shortest-path Reasoning, 13\% improvement of human preference in Creative Writing, and 15\% enhancement in the task completion rate of LLM-Agent Planning.

---

# 思维传播：一种基于类比的复杂推理方法 论文详细解读

### 背景：这个问题为什么难？

大型语言模型（LLM）在零样本或少样本提示下已经能完成不少推理任务，但它们仍然有两大痛点：第一，模型每次都要“从头”思考，无法把过去解决过的类似问题的经验直接搬进来；第二，在需要多步推理的场景里，错误会像雪球一样越滚越大，最终答案往往偏离真相。现有的 Chain‑of‑Thought（思维链）或 Self‑Consistency（自洽）等技巧只能让模型把思考过程写出来，却没有机制让模型复用已有的解法，也没有办法主动纠正已经产生的错误。因此，如何让 LLM 像人类一样“举一反三”，把相似问题的洞见迁移到新问题上，成为提升复杂推理能力的关键瓶颈。

### 关键概念速览
- **思维链（CoT）**：让模型在给出最终答案前先把推理步骤写出来，类似于人在解数学题时先列草稿，帮助模型保持逻辑连贯性。  
- **类比问题**：与目标问题在结构或求解思路上相似的辅助问题，就像老师在课堂上先让学生练习“相似题”，帮助他们抓住关键技巧。  
- **解答复用**：把已经求解出的类比问题的答案或推理路径直接拿来使用，而不是重新让模型从零开始思考。  
- **知识密集型计划**：在类比答案不足以直接给出目标答案时，模型会生成一套执行步骤（计划），类似于先列出“行动清单”，再一步步完成。  
- **Plug‑and‑Play 提示**：一种可以无缝叠加在已有提示技术之上的模块，使用时只需要少量额外指令，不需要重新设计任务专属的提示模板。  
- **错误传播抑制**：通过类比答案的校验或计划修正，阻止早期错误在后续推理中被放大。  

### 核心创新点
1. **从“单一推理”到“类比驱动”**  
   - 之前的提示方法每次都让模型独立完成目标任务，等于每次都重新写一遍解题稿。  
   - 本文让模型先主动生成一批与目标问题结构相近的类比问题，并求解它们。  
   - 这样模型可以在解答目标问题时直接引用已有的解法或思路，显著提升了答案的正确率和一致性。

2. **双轨答案生成：直接复用 vs. 计划修正**  
   - 传统方法只有“一条路”：从头推理得到答案。  
   - TP 先检查类比答案是否能直接映射到目标问题，若能则直接复用；若不能，则把类比答案当作“知识库”，让模型生成一套执行计划来修正初始答案。  
   - 这种“先用后补”的策略在多步推理中有效抑制错误累积，提升了任务完成率。

3. **与现有提示技术的兼容层**  
   - 过去的改进往往需要重新设计提示模板或额外的微调。  
   - TP 只在原有提示前后加上一段“生成类比问题并求解”的指令，几乎不改变原有工作流。  
   - 这种即插即用的特性让它可以在各种任务上快速验证，实验中即表现出跨任务的提升。

4. **类比问题的自动筛选与关联**  
   - 类比问题如果随意生成，可能与目标任务毫不相关，反而增加噪声。  
   - 作者让模型在生成类比问题时使用“关联度评估”，只保留与目标问题在求解策略上高度相似的子集。  
   - 这一步骤在实验中被证明是提升整体效果的关键因素。

### 方法详解
**整体框架**  
TP 把一次完整的推理过程拆成三大阶段：  
1）**类比问题生成** → 让模型基于目标问题的描述，创造若干结构相似的辅助问题。  
2）**类比解答** → 对每个辅助问题使用标准的 CoT 或其他提示技术求解，得到一组解答和推理链。  
3）**答案传播** → 根据类比解答的可复用程度，决定是直接映射到目标答案，还是生成一套计划来修正原始的“从头”答案。

**关键模块拆解**  

- **类比生成器**  
  - 输入：目标问题的自然语言描述。  
  - 操作：模型被提示“请列出 3–5 个与下面问题在求解思路上相似的变体”，并要求每个变体保留核心约束。  
  - 类比度评估：模型再对每个生成的变体打分，保留分数最高的前 N 条（N 通常为 3），确保后续解答的相关性。

- **类比求解器**  
  - 对每条类比问题，使用已有的 CoT 提示让模型写出完整的推理链并给出答案。  
  - 这里的输出既是“答案”，也是“思考过程”，后者将在传播阶段提供结构化的知识。

- **传播引擎**  
  - **直接复用路径**：检查类比答案是否可以通过简单的映射（如变量替换、约束平移）得到目标答案。若匹配成功，直接输出。  
  - **计划修正路径**：若直接映射失败，传播引擎把类比推理链当作“知识库”，让模型生成一段“基于以下类比经验的执行计划”，该计划会在目标问题的上下文中重新执行，以校正最初的从头答案。  
  - **错误抑制机制**：在计划生成后，模型会再次检查计划执行的中间结果是否与类比答案保持一致，若出现冲突则回滚并尝试另一条类比路径。

**最巧妙的设计**  
- 把类比问题的求解过程本身当作一种“检索”，而不是外部检索系统。所有信息都在同一个 LLM 内部流转，避免了跨模型的接口开销。  
- 采用“双轨”传播（直接复用 vs. 计划修正）让模型在面对不同难度的任务时可以自适应选择最省力的路径，类似于人类在解题时先尝试套用已有公式，若不行再重新推导。

### 实验与效果
- **测试任务**  
  1. **Shortest‑Path Reasoning**（最短路径推理）：给出图结构和起止点，要求模型推导最短路径。  
  2. **Creative Writing**（创意写作）：让模型根据提示生成短篇故事，并通过人工评审判断质量。  
  3. **LLM‑Agent Planning**（LLM 代理规划）：模型需要为一个虚拟助理生成可执行的多步骤计划。

- **对比基线**  
  - 标准零样本提示、Few‑Shot 示例、Chain‑of‑Thought、Self‑Consistency 等常用方法。  

- **主要结果**（论文声称）  
  - 在最短路径任务上，TP 将找到最优解的比例提升了 **12%**（绝对值）。  
  - 创意写作的人工偏好评分提升了 **13%**，说明生成的故事更符合人类审美。  
  - LLM‑Agent 规划的任务完成率提升了 **15%**，表明计划更可靠、执行成功率更高。

- **消融实验**  
  - 去掉类比生成阶段，直接使用传统 CoT，性能回落到基线水平，说明类比问题是提升的核心。  
  - 只保留直接复用路径而不使用计划修正，创意写作的提升幅度下降约 6%，说明计划修正在处理高噪声任务时仍然重要。  

- **局限性**  
  - 类比问题的质量高度依赖 LLM 本身的生成能力，若模型本身在特定领域知识薄弱，类比质量会受限。  
  - 生成和求解类比问题会带来额外的计算开销，尤其在需要大量类比时成本上升。  
  - 论文未提供对大规模真实业务场景的评估，实际部署时的响应时延仍是未知数。

### 影响与延伸思考
TP 把“类比思考”正式引入 LLM 提示工程，随后出现的工作多聚焦于 **检索增强生成（RAG）+类比**、**跨任务知识迁移**以及 **自监督类比学习**。例如 2024 年的 “Analogical Prompting” 进一步将外部知识库的相似案例与内部类比生成结合，取得了更高的鲁棒性。  
如果想继续深挖，可以关注以下方向：  
- **自动类比质量评估**：设计更精细的相似度度量，减少噪声类比。  
- **多模态类比**：把图像、代码等非语言信息也纳入类比生成，拓宽适用场景。  
- **低成本类比生成**：利用小模型或缓存机制降低计算开销。  

### 一句话记住它
让大模型先“找相似题再搬答案”，把类比问题的解法直接搬进新问题的推理里，从而显著削减错误累积并提升复杂任务的成功率。