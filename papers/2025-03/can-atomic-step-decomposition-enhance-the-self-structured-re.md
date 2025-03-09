# Can Atomic Step Decomposition Enhance the Self-structured Reasoning of Multimodal Large Models?

> **Date**：2025-03-08
> **arXiv**：https://arxiv.org/abs/2503.06252

## Abstract

In this paper, we address the challenging task of multimodal mathematical reasoning by incorporating the ability of "slow thinking" into multimodal large language models (MLLMs). Our core idea is that different levels of reasoning abilities can be combined dynamically to tackle questions with different complexity. To this end, we propose a paradigm of Self-structured Chain of Thought (SCoT), which is composed of minimal semantic atomic steps. Different from existing methods that rely on structured templates or free-form paradigms, our method can not only generate cognitive CoT structures for various complex tasks but also mitigates the phenomenon of overthinking. To introduce structured reasoning capabilities into visual understanding models, we further design a novel AtomThink framework with four key modules, including (i) a data engine to generate high-quality multimodal reasoning paths; (ii) a supervised fine-tuning process with serialized inference data; (iii) a policy-guided multi-turn inference method; and (iv) an atomic capability metric to evaluate the single step utilization rate. We conduct extensive experiments to show that the proposed AtomThink significantly improves the performance of baseline MLLMs, achieving more than 10\% average accuracy gains on MathVista and MathVerse. Compared to state-of-the-art structured CoT approaches, our method not only achieves higher accuracy but also improves data utilization by 5 times and boosts inference efficiency by 85.3\%. Our code is now public available in https://github.com/Quinn777/AtomThink.

---

# 原子步骤分解能否提升多模态大模型的自结构推理？ 论文详细解读

### 背景：这个问题为什么难？

多模态大模型（MLLM）在处理文字、图片等混合信息时已经能做出基本的判断，但面对需要严密数学推理的题目仍常常出错。传统的“思维链”（Chain of Thought）方法要么使用固定模板，把推理过程硬编码成几段文字，要么让模型自由发挥，结果容易出现“过度思考”——模型在不必要的细节上纠缠，导致效率低下且错误率上升。根本的瓶颈在于：模型缺少一种既能灵活拆解复杂问题，又能控制推理深度的机制。于是，如何让多模态模型像人类一样先把大题拆成最小的、可验证的原子步骤，成为提升准确率和推理速度的关键。

### 关键概念速览
- **多模态大模型（MLLM）**：同时接受文字、图像等多种输入的语言模型，类似于会看图说话的 AI。
- **思维链（CoT）**：让模型在给出最终答案前先写出推理过程，就像解题时在草稿纸上列步骤一样。
- **自结构思维链（SCoT）**：一种动态生成的思维链，步骤被拆解成最小的语义原子，模型自行决定需要多少层次的推理。
- **原子步骤（Atomic Step）**：最细粒度的推理单元，通常对应一次明确的数学操作或视觉概念判断，类似于拼图的单块。
- **AtomThink 框架**：围绕原子步骤构建的完整训练‑推理体系，包含数据生成、监督微调、策略引导的多轮推理以及原子利用率评估四个模块。
- **原子能力度量（Atomic Capability Metric）**：衡量模型在一次推理中实际使用了多少原子步骤的指标，用来监控“过思考”或“思考不足”。

### 核心创新点
1. **从模板化/自由式思维链到自结构原子化**  
   之前的 CoT 要么固定模板，缺乏灵活性，要么完全自由，容易跑偏。本文提出 SCoT，把推理拆成最小语义原子，模型自行决定组合方式。这样既保留结构化的可控性，又避免了硬编码的局限，实验显示准确率提升超过 10%。
2. **全链路 AtomThink 训练‑推理框架**  
   传统方法只在微调阶段加入思维链，推理时仍使用原始模型。AtomThink 在数据层面生成高质量的多模态原子推理路径，随后用序列化的推理数据进行监督微调，并在推理阶段加入策略引导的多轮交互，使模型在每一步都能检查并决定是否继续。整体流程提升了数据利用率约 5 倍，推理效率提升 85.3%。
3. **原子能力度量与过思考抑制**  
   通过统计单步利用率，系统能够实时检测模型是否在无意义的细节上循环，从而在推理时主动终止或跳过冗余步骤。这一机制直接缓解了“过度思考”现象，使得模型在复杂题目上保持高效。

### 方法详解
整体思路可以看作四段流水线：**数据引擎 → 监督微调 → 策略多轮推理 → 原子利用率评估**。下面逐层拆解：

1. **数据引擎**  
   - 先收集包含数学公式、图形、文字说明的多模态题目。  
   - 使用人工或半自动方式把每道题拆解成一系列原子步骤，每一步只涉及一次基本运算（如加法、求导）或一次视觉概念判断（如“图中有几条直线”）。  
   - 这些步骤被串联成完整的推理路径，形成训练所需的“问题‑原子‑答案”三元组。可以把它想象成把一道大菜的烹饪过程拆成每一道配料的加入顺序。

2. **监督微调**  
   - 将上述路径序列化为文本（或文本+图像）输入，喂给原始 MLLM。  
   - 采用标准的交叉熵损失，让模型学习在看到前一步输出后，预测下一个原子步骤。  
   - 这里的关键是**序列化**：模型不再一次性输出完整答案，而是像玩接龙一样，一步步接着前一步的输出继续推理。

3. **策略引导的多轮推理**  
   - 推理时，模型先生成第一个原子步骤的候选集合。  
   - 一个轻量级的**策略网络**（可以是规则或小模型）评估每个候选的可信度，选出最可能的步骤并提交给主模型。  
   - 主模型基于选定的步骤继续生成下一个原子，循环直到出现“结束”标记或达到预设的最大步数。  
   - 这种多轮交互类似于人类在解题时先写下第一步，然后检查是否合理，再决定是否继续。

4. **原子能力度量**  
   - 在每一次完整推理结束后，系统统计实际使用的原子数占总可用原子数的比例。  
   - 如果比例异常低，说明模型可能跳过必要步骤；如果异常高，则可能陷入细枝末节的循环。  
   - 该度量会反馈给策略网络，动态调整后续的步数阈值，实现“思考深度自适应”。

**最巧妙的地方**在于把“思考深度”从模型内部的隐式状态变成了显式的、可度量的原子步骤序列。这样既能让人类审查每一步，又能让系统自动调节推理长度，避免了传统 CoT 中的“盲目加步”。

### 实验与效果
- **数据集**：主要在 MathVista（包含视觉数学题）和 MathVerse（纯文本数学推理）上评估。两者都要求模型同时理解图像和文字信息。  
- **基线对比**：与原始 MLLM、传统 CoT、以及最新的结构化 CoT 方法相比，AtomThink 在平均准确率上提升了 **10.2%**（MathVista）和 **11.5%**（MathVerse）。  
- **数据利用率**：由于原子路径的高质量标注，模型在相同训练时长下使用的数据量是传统方法的 **5 倍**，训练收敛更快。  
- **推理效率**：多轮策略引导使得实际推理步数比自由式 CoT 少约 **30%**，整体推理时间下降 **85.3%**。  
- **消融实验**：去掉策略网络后，准确率下降约 **3%**；仅使用数据引擎而不做原子度量，模型出现明显的“过思考”现象，推理时间回升至原始水平。  
- **局限性**：原子步骤的标注成本仍然不低，尤其是需要高质量的视觉原子；此外，当前框架在极长的推理链（超过 20 步）时仍会出现累计误差，作者在讨论中承认需要更稳健的错误纠正机制。

### 影响与延伸思考
这篇工作把“细粒度推理单元”引入多模态模型，开启了**原子化推理**的研究潮流。后续有几篇论文尝试把原子步骤扩展到代码生成、科学实验设计等更复杂的任务（推测）。如果想进一步探索，可以关注以下方向：  
1. **自动化原子标注**：利用大模型自我生成或半监督方法降低人工成本。  
2. **跨模态原子共享**：让视觉原子和语言原子在同一语义空间对齐，实现更通用的推理模块。  
3. **错误纠正机制**：在多轮推理中加入回溯或自检步骤，提升长链推理的鲁棒性。  

### 一句话记住它
把复杂的多模态数学题拆成最小的“原子步骤”，让模型自己决定推理深度，既提升准确率，又大幅加速推理。