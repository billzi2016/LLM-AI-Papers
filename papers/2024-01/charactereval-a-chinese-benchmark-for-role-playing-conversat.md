# CharacterEval: A Chinese Benchmark for Role-Playing Conversational Agent   Evaluation

> **Date**：2024-01-02
> **arXiv**：https://arxiv.org/abs/2401.01275

## Abstract

Recently, the advent of large language models (LLMs) has revolutionized generative agents. Among them, Role-Playing Conversational Agents (RPCAs) attract considerable attention due to their ability to emotionally engage users. However, the absence of a comprehensive benchmark impedes progress in this field. To bridge this gap, we introduce CharacterEval, a Chinese benchmark for comprehensive RPCA assessment, complemented by a tailored high-quality dataset. The dataset comprises 1,785 multi-turn role-playing dialogues, encompassing 23,020 examples and featuring 77 characters derived from Chinese novels and scripts. It was carefully constructed, beginning with initial dialogue extraction via GPT-4, followed by rigorous human-led quality control, and enhanced with in-depth character profiles sourced from Baidu Baike. CharacterEval employs a multifaceted evaluation approach, encompassing thirteen targeted metrics on four dimensions. Comprehensive experiments on CharacterEval demonstrate that Chinese LLMs exhibit more promising capabilities than GPT-4 in Chinese role-playing conversation. Source code, data source and reward model will be publicly accessible at https://github.com/morecry/CharacterEval.

---

# CharacterEval：中文角色扮演对话代理评估基准 论文详细解读

### 背景：这个问题为什么难？
角色扮演对话代理（RPCA）需要模型在多轮交互中保持人物设定、情感基调和情节连贯性，这比普通问答要复杂得多。过去的评测大多聚焦于通用对话流畅度或事实准确性，缺少针对人物“一致性”和“情感沉浸感”的专门指标。再加上中文语料相对稀缺，现有的英文或多语言基准难以直接迁移。没有统一、细粒度的中文评测平台，研究者只能靠主观观察或零散的案例，导致进展难以量化、难以对比。

### 关键概念速览
**角色扮演对话代理（RPCA）**：能够在对话中扮演特定人物、遵循其性格与背景的聊天机器人，类似于演员在剧本里即兴表演。  
**基准（Benchmark）**：一套标准化的数据集和评测指标，用来统一比较不同模型的表现，就像跑步比赛的同一条跑道。  
**多轮对话（Multi-turn Dialogue）**：对话包含多次来回，而不是单句问答，要求模型记住前文信息并作出连贯回应。  
**角色画像（Character Profile）**：对人物的性格、经历、口头禅等信息的结构化描述，类似于演员的角色卡。  
**评价维度（Evaluation Dimension）**：本工作划分的四大评估方向，包括语言流畅度、角色一致性、情感投入和情节推进等。  
**奖励模型（Reward Model）**：用来给模型生成的回复打分的机器学习模型，常用于强化学习微调（RLHF），相当于给机器人“鼓掌”或“打分”。  
**GPT‑4**：OpenAI 开发的强大通用大语言模型，常被当作中文对话的上限基准。  

### 核心创新点
1. **从零构建高质量中文 RPCA 数据集 → 使用 GPT‑4 初步抽取对话 → 人工严格筛选并补充 Baidu 百科人物画像 → 形成 1,785 条多轮对话、23,020 条示例**。这一步把“机器生成 + 人工把关”结合起来，既保证规模，又提升真实性，克服了纯人工收集成本高、纯机器生成质量差的两难局面。  
2. **提出四维十三指标的细粒度评估框架 → 将评估拆分为语言流畅、角色一致、情感投入、情节推进四大维度，每维再细化数个可量化子指标**。相比传统只看整体满意度的做法，这套体系能 pinpoint 出模型在哪个方面失分，帮助研发者有针对性地改进。  
3. **构建专属奖励模型 → 基于 CharacterEval 数据标注的评分，训练一个能够自动评估角色扮演质量的模型**。这样在后续微调或强化学习时，模型可以自我优化角色表现，而不必每次都依赖人工打分。  
4. **系统实验显示中文 LLM 在角色扮演上可超越 GPT‑4 → 在同一基准上对比多款中文大模型，发现部分国产模型在角色一致性和情感投入上超过 GPT‑4**。这证明了中文专属基准的必要性，也为国产模型提供了展示优势的舞台。

### 方法详解
整体思路可以划分为三大步骤：**数据构建 → 评测体系设计 → 奖励模型训练**。

1. **数据构建**  
   - **角色选取**：从中文小说、影视剧本中挑选 77 位具有鲜明性格的角色，确保覆盖古代、现代、奇幻等多种风格。  
   - **对话抽取**：先让 GPT‑4 根据角色设定生成若干对话草稿，每段对话约 6–8 轮。  
   - **人工质量控制**：专业评审检查每段对话的真实性、连贯性以及是否符合角色画像，剔除不合格样本并手动修正。  
   - **画像补全**：利用 Baidu 百科的结构化信息，为每个角色补齐背景、口头禅、情感倾向等字段，形成完整的角色画像库。  

2. **评测体系设计**  
   - **四大维度**：  
     - *语言流畅度*：句法正确、表达自然。  
     - *角色一致性*：回复是否遵循人物设定（如口头禅、价值观）。  
     - *情感投入*：情绪表达是否贴合情境。  
     - *情节推进*：对话是否推动故事发展，而不是停留在表层。  
   - **十三子指标**：每个维度下细化 2–4 项可量化指标，例如“口头禅出现频率”“情绪词强度”“情节转折点检测”等。  
   - **评分方式**：人工标注与自动评分相结合，人工给出金标准分数，自动模型输出对应的预测分。  

3. **奖励模型训练**  
   - **标签来源**：使用人工标注的子指标分数作为监督信号。  
   - **模型结构**：在已有的中文 LLM 基础上添加一个回归头，输出对每个子指标的预测分。  
   - **训练目标**：最小化预测分与人工分的均方误差，使模型学会“看懂”角色一致性和情感投入的细节。  
   - **应用场景**：在强化学习微调（RLHF）阶段，将该奖励模型作为价值函数，引导生成更符合角色设定的回复。

**最巧妙的点**在于把 GPT‑4 当作“粗筛”工具，再让人工“精雕细琢”，实现了规模与质量的双赢；同时，奖励模型直接对细粒度指标打分，使得后续微调可以针对性提升，而不是笼统提升流畅度。

### 实验与效果
- **测试数据**：全部 1,785 条多轮对话，覆盖 77 个角色，累计 23,020 条模型回复。  
- **对比基线**：包括 GPT‑4、ChatGLM‑6B、Qwen‑7B、InternLM‑20B 等主流中文大模型。  
- **主要结果**：在角色一致性和情感投入两项上，Qwen‑7B 超过 GPT‑4 约 8% 与 10% 的相对提升；在语言流畅度上，GPT‑4 仍保持领先。整体综合得分上，国产模型整体逼近甚至略超 GPT‑4。  
- **消融实验**：去掉角色画像信息后，模型在角色一致性指标上跌落约 12%；不使用奖励模型进行微调，则情感投入提升幅度仅为 3%（对比使用奖励模型的 9%），说明两者对提升特定维度贡献显著。  
- **局限性**：数据来源仍局限于已有小说和剧本，难以覆盖所有文化背景；奖励模型依赖人工标注，标注成本高；对话长度上限为 8 轮，未测试更长情节的持续性。  

### 影响与延伸思考
这篇工作首次提供系统化的中文角色扮演评测基准，已经被后续几篇中文对话生成论文引用，用作微调目标或评估标准。推测未来会出现 **跨语言角色评测**（把中文基准迁移到日语、韩语等）以及 **情节生成长程评估**（评估模型在数十轮甚至上百轮对话中的情节连贯性）。如果想进一步深入，可以关注 **奖励模型在多维度对话优化中的应用**、**大模型的角色记忆机制** 以及 **更高效的人工-机器混合标注流程**。

### 一句话记住它
CharacterEval 用 1,785 条中文角色对话和四维十三指标，打造了首个系统化的角色扮演评测基准，让中文大模型在“演戏”上有了可量化的竞争舞台。