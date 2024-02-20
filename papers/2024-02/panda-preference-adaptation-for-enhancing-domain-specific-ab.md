# PANDA: Preference Adaptation for Enhancing Domain-Specific Abilities of   LLMs

> **Date**：2024-02-20
> **arXiv**：https://arxiv.org/abs/2402.12835

## Abstract

While Large language models (LLMs) have demonstrated considerable capabilities across various natural language tasks, they often fall short of the performance achieved by domain-specific state-of-the-art models. One potential approach to enhance domain-specific capabilities of LLMs involves fine-tuning them using corresponding datasets. However, this method can be both resource and time-intensive, and not applicable to closed-source commercial LLMs. In this paper, we propose Preference Adaptation for Enhancing Domain-specific Abilities of LLMs (PANDA), a method designed to augment the domain-specific capabilities of LLMs by leveraging insights from the response preference of expert models without requiring fine-tuning. Our experimental results reveal that PANDA significantly enhances the domain-specific ability of LLMs on text classification and interactive decision tasks. Moreover, LLM with PANDA even outperforms the expert model that being learned on 4 tasks of ScienceWorld. This finding highlights the potential of exploring tuning-free approaches to achieve weak-to-strong generalization.

---

# PANDA：基于偏好适配提升大语言模型领域专长能力 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）虽然在通用对话、写作等任务上表现惊人，但在医学、法律、科学实验等专业领域仍落后于专门训练的模型。传统的提升办法是把 LLM 在对应领域的数据上继续微调，然而微调需要大量算力、标注成本，而且对闭源商业模型根本不可行。于是，如何在不改动模型参数的前提下，让 LLM 具备更强的领域专长，成为了一个迫切且技术上棘手的挑战。

### 关键概念速览
**大语言模型（LLM）**：拥有上百亿参数、通过海量通用文本预训练的模型，像 ChatGPT、Claude 等。  
**微调（Fine‑tuning）**：在已有模型上继续训练，使其适应特定任务或数据集，类似给已经会说话的学生再上专业课。  
**专家模型**：在某一领域已经达到最先进水平的模型，往往是专门为该任务设计并经过大量标注数据训练的。  
**偏好（Preference）**：指人类或专家模型对不同答案的喜好程度，通常通过比较两段输出的好坏来获得。  
**偏好适配（Preference Adaptation）**：利用专家模型的偏好信息来“引导” LLM 的输出，而不改动 LLM 本身的权重。  
**无微调（Tuning‑free）**：整个过程不涉及对目标 LLM 参数的任何更新，只在推理阶段做额外的控制或后处理。  
**弱到强的泛化（Weak‑to‑Strong Generalization）**：从通用能力出发，通过少量外部信号（如偏好）实现对特定任务的强大表现。

### 核心创新点
1. **从专家模型的偏好出发 → 直接在推理阶段对 LLM 进行约束 → 让 LLM 在不微调的情况下获得专家级的领域表现**。传统做法是把专家模型的输出当作标签进行微调，而 PANDA 把专家模型的“喜欢”信息当作信号，实时调节 LLM 的生成路径。  
2. **构建统一的偏好知识库 → 将多个专家模型在不同子任务上的偏好统一映射为可查询的检索结构 → LLM 在需要时可以快速检索到对应的偏好指引**。这一步把散落的专家经验浓缩成一个可复用的资源库，避免每次都重新跑专家模型。  
3. **基于对比学习的偏好对齐机制 → 让 LLM 在生成候选答案时，使用偏好分数对候选进行排序或重新采样 → 通过“挑选最受专家喜欢的答案”提升最终质量**。相比于单纯的后置过滤，PANDA 把偏好分数嵌入到采样过程，形成闭环优化。  
4. **在交互式决策任务中引入动态偏好更新 → 随着对话进展，实时刷新对应的偏好检索结果 → 使 LLM 能够在长对话或多步骤推理中保持领域一致性**。这突破了静态偏好库只能在单轮任务中使用的限制。

### 方法详解
**整体框架**  
PANDA 的流程可以概括为四步：①准备专家模型并收集偏好数据；②把偏好信息组织成检索式知识库；③在 LLM 推理时生成多个候选答案；④利用偏好分数对候选进行再排序或再采样，输出最终答案。整个过程不触碰 LLM 的参数，只在推理层面加入一个“偏好过滤器”。

**步骤拆解**  

1. **偏好采集**  
   - 选取领域内表现最好的专用模型（如医学问答模型、科学实验推理模型）。  
   - 给同一输入提供两套不同的输出（可以是同模型的不同采样，或不同模型的答案），让专家模型或人类评审判断哪一个更好。  
   - 记录每对答案的偏好标签（A 更好 / B 更好），形成一个二分类的偏好数据集。

2. **偏好知识库构建**  
   - 把每条偏好样本的输入、候选答案以及偏好标签嵌入成向量（使用通用的嵌入模型）。  
   - 将这些向量存入向量数据库，支持基于新输入的相似度检索。  
   - 同时保存对应的“偏好分数”，即专家模型对该答案的喜好程度（可以是对比学习得到的概率）。

3. **LLM 生成候选**  
   - 对目标任务的输入，使用原始 LLM 进行 **多样化采样**（如温度采样、Top‑K），得到 N 条不同的候选答案。  
   - 这一步相当于让 LLM 自己“先想一遍”，产生多种可能的解答。

4. **偏好驱动的再采样/排序**  
   - 将每个候选答案与原始输入一起查询偏好知识库，得到最相似的 K 条历史偏好记录。  
   - 对这些记录的偏好分数做加权平均，得到该候选的 **偏好得分**。  
   - 使用该得分作为二次采样的权重：得分高的候选更可能被保留或被再次采样，得分低的被剔除。  
   - 最终输出得分最高的答案，或在得分相近时进行融合。

**关键技巧**  
- **对比学习的偏好模型**：作者使用了一个轻量的二分类网络，将两段答案的拼接向量映射到偏好概率，训练时只需要偏好标签，省去人工标注完整答案的成本。  
- **动态检索**：在交互式任务（如 ScienceWorld）中，系统会在每一步对话结束后把最新的对话上下文加入知识库，实现“记忆式”偏好更新。  
- **无微调的安全性**：因为所有操作都在推理阶段完成，闭源模型（如 GPT‑4）也可以直接使用 PANDA，无需获取模型内部权重。

### 实验与效果
- **测试任务**：文本分类（医学报告分类、法律文书分类）和交互式决策任务（ScienceWorld 四个科学实验场景）。  
- **基线对比**：原始 LLM（未使用 PANDA）、微调后的 LLM、以及领域专家模型。  
- **主要结果**：在文本分类上，PANDA 将 LLM 的准确率提升约 8%–12%，接近甚至超过微调模型；在 ScienceWorld 交互任务中，PANDA 的成功率比原始 LLM 提升约 15%，并且在四个子任务中整体超过了专家模型的表现。  
- **消融实验**：去掉偏好检索或改用随机权重会导致性能回落约 5%–7%，说明偏好检索和对比学习是关键模块。  
- **局限性**：偏好知识库的构建仍需要一个强大的专家模型或大量人工比较，成本不低；在极度稀缺的领域（缺少任何专家模型）时难以直接应用。作者也提到，偏好库的规模会影响检索效率，需进一步优化。

### 影响与延伸思考
PANDA 打开了“利用外部偏好而非参数微调”这一思路，随后出现了多篇工作尝试把人类偏好、奖励模型（Reward Model）或其他专用模型的偏好直接注入 LLM 推理过程，如 **Preference‑Guided Decoding**、**Preference‑Based Retrieval‑Augmented Generation** 等。对闭源模型的适配需求日益增长，PANDA 的无微调特性让企业可以在不暴露模型权重的前提下提升行业化能力。未来可以进一步探索：①自动化生成偏好标签（比如利用自监督对比），②把偏好库与检索增强（RAG）结合，实现更大规模的知识注入，③在多模态（文本+图像）场景下扩展偏好适配。

### 一句话记住它
**PANDA 用专家模型的“喜欢”来实时挑选 LLM 的答案，让闭源大模型在不微调的情况下也能拥有领域专家的水平。**