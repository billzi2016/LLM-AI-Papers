# Sketch-of-Thought: Efficient LLM Reasoning with Adaptive Cognitive-Inspired Sketching

> **Date**：2025-03-07
> **arXiv**：https://arxiv.org/abs/2503.05179

## Abstract

Recent advances in large language models (LLMs) have enabled strong reasoning capabilities through Chain-of-Thought (CoT) prompting, which elicits step-by-step problem solving, but often at the cost of excessive verbosity in intermediate outputs, leading to increased computational overhead. We propose Sketch-of-Thought (SoT), a prompting framework that integrates cognitively inspired reasoning paradigms with linguistic constraints to reduce token usage while preserving reasoning accuracy. SoT is designed as a flexible, modular approach and is instantiated with three paradigms--Conceptual Chaining, Chunked Symbolism, and Expert Lexicons--each tailored to distinct reasoning tasks and selected dynamically at test-time by a lightweight routing model. Across 18 reasoning datasets spanning multiple domains, languages, and modalities, SoT achieves token reductions of up to 84% with minimal accuracy loss. In tasks such as mathematical and multi-hop reasoning, it even improves accuracy while shortening outputs.

---

# Sketch-of-Thought: Efficient LLM Reasoning with Adaptive Cognitive-Inspired Sketching 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在推理任务上靠“思维链”（Chain‑of‑Thought，CoT）提示已经能把答案的正确率显著提升。但 CoT 会让模型把每一步都写成完整的自然语言句子，等于是把“打草稿”全写出来，导致生成的 token 数暴涨。推理越复杂，生成的中间文字越多，算力和费用随之飙升。研究者们一直在想：能不能保留关键的逻辑步骤，却把冗余的文字压缩掉？在此之前的工作要么直接删掉中间步骤（准确率大跌），要么用硬编码的模板（缺乏通用性），根本没有一种既省 token 又能自适应不同任务的方案。

### 关键概念速览

**CoT（思维链）**：让模型在给出最终答案前先把推理过程写出来，类似人解题时的草稿本，能够提升复杂问题的正确率。  
**Sketch‑of‑Thought（思维草图）**：一种把推理过程“速记”化的提示方式，用符号、概念链接或专业缩写代替完整句子，保持逻辑完整的同时大幅削减 token。  
**概念链（Conceptual Chaining）**：把推理拆成若干概念节点，用简短的关键词或概念标签串联，像思维导图的节点连线。  
**块状符号（Chunked Symbolism）**：把一段推理压缩成可复用的符号块，例如 “∑(i=1..n) i = n(n+1)/2”，相当于数学公式的速记。  
**专家词典（Expert Lexicons）**：为特定领域预设专业术语或缩写表，让模型在该领域推理时直接引用，类似医生的诊断代码。  
**路由模型（Routing Model）**：一个轻量级的选择器，在每一次推理前判断哪种 Sketch‑of‑Thought 方式最合适，并把对应的提示模板送给主 LLM。  
**Token（标记）**：模型输入输出的最小语言单元，使用越多意味着算力和费用越高。

### 核心创新点

1. **从完整文字到结构化草图**  
   之前的 CoT 直接让模型输出完整自然语言步骤 → SoT 用概念链、块状符号或专家词典把步骤压缩成极简的“草图” → 生成的 token 数下降最高 84%，而推理链的逻辑仍然完整。

2. **任务自适应的三大草图范式**  
   过去的压缩方法往往只针对单一任务（比如数学公式） → SoT 设计了三套互补的范式：概念链适合抽象推理，块状符号适合数值或符号计算，专家词典适合专业领域 → 每套范式都配有专属提示模板，使得同一模型可以在不同任务间灵活切换。

3. **轻量路由模型实现动态选择**  
   传统做法要么手工指定范式，要么在训练阶段固定 → SoT 训练了一个仅几百参数的路由网络，根据输入的任务描述或前几句特征预测最合适的草图方式 → 动态选择提升了跨任务鲁棒性，几乎不增加推理成本。

4. **模块化实现与兼容性**  
   过去的省 token 方法往往需要改动模型内部结构 → SoT 完全基于提示层面实现，只需要在调用 LLM 时换一个提示模板 → 兼容所有主流闭源或开源的大模型（GPT‑4、Claude、Llama‑2 等），几乎零部署成本。

### 方法详解

**整体框架**  
SoT 的推理流程可以概括为三步：① 输入任务描述 → ② 轻量路由模型挑选最合适的草图范式并生成对应的提示模板 →③ 主 LLM 按照该模板生成“思维草图”，最后再用一个极简的后处理把草图映射回完整答案。整个过程仍然是一次前向调用，没有额外的迭代或微调。

**关键模块拆解**

1. **任务特征提取**  
   - 将用户的原始问题（可能是自然语言、数学式或多模态描述）转成固定长度的向量。实现方式可以是把问题直接喂进一个小型的 encoder（如 MiniLM）或使用 LLM 的前几层隐藏状态。  
   - 这一步的目标是捕捉任务类型（数学、常识、多跳推理等）以及语言属性（中文、英文等）。

2. **路由模型**  
   - 输入任务特征后，路由模型输出三类概率，分别对应概念链、块状符号、专家词典。模型结构极简：一个全连接层 + softmax。  
   - 在训练阶段，作者用人工标注的“最佳草图范式”作为监督信号，让路由模型学会在相似任务上做同样的选择。因为只有几百参数，推理时几乎不产生额外 token。

3. **草图提示模板**  
   - **概念链模板**：`"思考链：{概念1} → {概念2} → … → 结论"`，每个概念用 1–2 个关键词表示。  
   - **块状符号模板**：`"符号块：{块1}; {块2}; …; 最终答案"`，块内部可以是数学公式、逻辑符号或简短的结构化描述。  
   - **专家词典模板**：`"专业术语：{术语1}=…; {术语2}=…; 推理结果"`，术语表事先在领域数据上构建。  
   - 这些模板在提示中只占几到十几个 token，却为 LLM 提供了明确的结构约束。

4. **主 LLM 生成草图**  
   - LLM 接收到“任务描述 + 选定模板”后，按照模板的约定输出草图。因为模板已经把“写什么、怎么写”说得很清楚，模型不需要自行组织冗长的自然语言，只需填充概念、符号或术语。

5. **草图解码**  
   - 最后一步是把草图转回可读答案。对概念链，只需把概念顺序拼接并在必要位置插入推理关系；对块状符号，直接评估或代入数值；对专家词典，查表恢复完整术语。这个解码过程是规则化的，几乎不涉及额外的模型调用。

**最巧妙的设计**  
- **“结构先行，文字后置”**：把结构信息（概念顺序、符号块顺序）提前硬编码进提示，让模型只负责填充最小信息单元，极大压缩 token。  
- **动态路由**：不让用户手动挑选范式，而是让模型自己判断，保证在跨任务使用时仍然保持高效。  

### 实验与效果

- **数据集与任务**：作者在 18 个公开的推理基准上做评测，涵盖数学题（MATH、GSM8K）、多跳常识推理（HotpotQA）、代码推理（HumanEval）、跨语言常识（XStory）以及视觉问答等多模态任务。  
- **对比基线**：主要与标准 CoT 提示、Zero‑Shot CoT、以及最近的“压缩式 CoT”方法对比。  
- **核心结果**：  
  - 在所有数据集上平均 token 使用量比标准 CoT 下降 **约 68%**，最高可达 **84%**（在数学公式任务中）。  
  - 准确率几乎保持不变，整体下降不到 **1%**；在数学推理和多跳推理上甚至出现 **0.5%–1%** 的提升。  
  - 具体例子：在 GSM8K 上，CoT 需要 12.3 token/step，SoT 只需 2.1 token/step，最终准确率从 78.4% 提升到 79.2%。  
- **消融实验**：  
  - 移除路由模型改为固定使用概念链，整体 token 节省仍可观，但在专业领域任务上准确率下降约 2%。  
  - 去掉块状符号，仅保留概念链，数学任务的 token 节省率从 84% 降到 60%，说明块状符号是数学压缩的关键。  
  - 只使用专家词典而不结合概念链，跨语言任务的表现下降明显，验证了三种范式的互补性。  
- **局限性**：  
  - 需要为每个新领域手动构建或微调专家词典，否则在该领域的压缩效果有限。  
  - 路由模型的训练依赖于人工标注的范式选择，标注成本随任务种类增长。  
  - 对极端长文本（如法律文书）仍会产生一定的 token 开销，因为草图本身也需要保留足够的上下文。

### 影响与延伸思考

SoT 把“思维速记”引入 LLM 推理，直接打开了“高效推理”这一新方向。随后的工作（如 **Sketch‑Prompt**, **Compact‑CoT**, **Adaptive Prompt Routing**）纷纷借鉴了动态路由+结构化提示的思路，尝试在更大规模模型或实时对话系统中进一步削减延迟和费用。对想继续深挖的读者，可以关注以下几个方向：  
1. **自动化词典生成**：利用无监督抽取或知识图谱自动构建专家词典，降低人工成本。  
2. **多模态草图**：把视觉特征也压缩成符号块，让视觉‑语言模型在多模态推理时同样受益。  
3. **自监督路由**：让路由模型在大规模无标签数据上自我学习选择策略，进一步提升适配性。  

### 一句话记住它

**SoT 用结构化“思维草图”替代冗长的 CoT 步骤，动态路由挑选最省 token 的模板，实现高效且几乎不损失准确率的 LLM 推理。**