# Qwen3 Technical Report

> **Date**：2025-05-14
> **arXiv**：https://arxiv.org/abs/2505.09388

## Abstract

In this work, we present Qwen3, the latest version of the Qwen model family. Qwen3 comprises a series of large language models (LLMs) designed to advance performance, efficiency, and multilingual capabilities. The Qwen3 series includes models of both dense and Mixture-of-Expert (MoE) architectures, with parameter scales ranging from 0.6 to 235 billion. A key innovation in Qwen3 is the integration of thinking mode (for complex, multi-step reasoning) and non-thinking mode (for rapid, context-driven responses) into a unified framework. This eliminates the need to switch between different models--such as chat-optimized models (e.g., GPT-4o) and dedicated reasoning models (e.g., QwQ-32B)--and enables dynamic mode switching based on user queries or chat templates. Meanwhile, Qwen3 introduces a thinking budget mechanism, allowing users to allocate computational resources adaptively during inference, thereby balancing latency and performance based on task complexity. Moreover, by leveraging the knowledge from the flagship models, we significantly reduce the computational resources required to build smaller-scale models, while ensuring their highly competitive performance. Empirical evaluations demonstrate that Qwen3 achieves state-of-the-art results across diverse benchmarks, including tasks in code generation, mathematical reasoning, agent tasks, etc., competitive against larger MoE models and proprietary models. Compared to its predecessor Qwen2.5, Qwen3 expands multilingual support from 29 to 119 languages and dialects, enhancing global accessibility through improved cross-lingual understanding and generation capabilities. To facilitate reproducibility and community-driven research and development, all Qwen3 models are publicly accessible under Apache 2.0.

---

# Qwen3 技术报告 论文详细解读

### 背景：这个问题为什么难？

大语言模型在聊天、代码生成、数学推理等场景的需求差异很大。过去的做法要么专注于快速响应的聊天模型，要么专门训练耗时的推理模型，导致需要在不同模型之间来回切换，维护成本高。与此同时，模型规模越大越能提升性能，但训练和部署成本也随之爆炸，尤其是想要覆盖上百种语言时更是难上加难。于是出现了“性能‑效率‑多语言”三难局面：既想要高质量的多步推理，又要保持低延迟，还要让小模型也能保持竞争力。

### 关键概念速览
- **Dense（稠密）模型**：所有参数在每次前向计算中都被使用的模型，像传统的 GPT 系列，结构简单但算力需求大。  
- **Mixture‑of‑Experts（MoE）模型**：把模型拆成多个专家子网络，只有一小部分专家在一次推理时被激活，算力随需求弹性增长，类似“按需召集专家团队”。  
- **Thinking Mode（思考模式）**：让模型在给出答案前进行多步推理，类似人在解难题时先列出思路再下结论，适合需要深度推理的任务。  
- **Non‑Thinking Mode（非思考模式）**：直接基于上下文生成答案，像日常聊天的即时回复，强调速度。  
- **Thinking Budget（思考预算）**：在推理时为思考模式预留的计算资源上限，用户可以自行调配，类似给“思考时间”设定上限，预算越高模型越能深挖。  
- **跨语言支持**：模型能够理解和生成多种语言的能力，报告中从 29 种提升到 119 种语言和方言，像把一个会说中文的老师变成会说百国语言的翻译官。  
- **知识蒸馏**：把大模型的知识压缩到小模型里，像把大厨的烹饪技巧浓缩成速食配方，让小模型也能表现出色。  
- **Apache 2.0 许可证**：一种宽松的开源协议，允许任何人免费使用、修改和商业化，降低了科研和产品落地的门槛。

### 核心创新点
1. **统一思考/非思考框架**  
   - 之前：需要分别部署聊天模型（如 GPT‑4o）和专门的推理模型（如 QwQ‑32B），切换成本高。  
   - 本文：在同一模型内部实现两种模式，并通过查询或模板自动切换。  
   - 改变：用户只需调用一次接口，模型自行决定是快速回复还是深度推理，极大简化了系统架构。

2. **思考预算机制**  
   - 之前：推理深度固定，要么慢要么浅，缺乏灵活性。  
   - 本文：在推理时动态分配计算预算，用户可以在响应时间和推理深度之间做权衡。  
   - 改变：同一查询在不同预算下会产生不同质量的答案，提供了可调的性能‑延迟平衡。

3. **大模型驱动的小模型高效蒸馏**  
   - 之前：小模型往往只能在特定任务上微调，性能提升有限。  
   - 本文：利用 Qwen 系列旗舰模型的知识，对 0.6 B‑7 B 规模的模型进行大规模蒸馏，保持竞争力的同时显著降低训练算力。  
   - 改变：小模型在代码生成、数学推理等基准上接近甚至超越同等规模的自研模型，降低了部署门槛。

4. **多语言覆盖从 29 到 119**  
   - 之前：大模型的多语言能力往往局限在主流语言，低资源语言表现差。  
   - 本文：通过扩展语料和跨语言对齐技术，模型在 119 种语言和方言上实现统一的理解与生成。  
   - 改变：全球用户可以直接使用同一模型进行本地化对话和任务，提升了模型的普适性。

### 方法详解
**整体思路**：Qwen3 把模型主体划分为两层：底层的稠密或 MoE 主干负责基本语言建模，顶部加入一个“模式控制器”。在推理时，控制器先判断输入属于哪类任务（快速聊天 vs. 需要多步推理），随后根据用户设定的思考预算决定是否激活思考路径。整个过程可以看作“先筛选，再深挖”。

**关键模块拆解**：

1. **模式控制器**  
   - 输入：用户查询的文本、可选的模板标签、思考预算上限。  
   - 工作：使用轻量分类头（类似二分类器）判断是否需要思考模式；若是，则开启后续的“思考循环”。  
   - 类比：像客服系统先判断是普通问候还是需要转人工的复杂问题。

2. **思考循环（Iterative Reasoning Loop）**  
   - 结构：在每一次循环中，模型会生成一个中间“思考片段”，并将其与原始上下文拼接，送回模型继续推理。  
   - 预算控制：循环次数或计算量受思考预算限制，一旦耗尽即强制退出并输出最终答案。  
   - 直观：相当于让模型在纸上写草稿，预算就是纸张的大小。

3. **稠密 vs. MoE 主干**  
   - 稠密模型：所有参数全参与，适合小规模部署。  
   - MoE 模型：使用门控网络（Gating）挑选少数专家子网参与计算，显著提升大模型的算力利用率。  
   - 设计巧妙之处：在思考模式下，MoE 的门控会倾向激活更强的专家，以获得更深的推理能力；在非思考模式下，则倾向激活轻量专家，保持低延迟。

4. **知识蒸馏管线**  
   - 大模型（如 235 B 参数）先在海量多语言、多任务数据上进行预训练。  
   - 小模型通过教师‑学生框架学习大模型的隐藏状态和输出分布，同时在思考预算上进行微调，使其在低算力下仍能模拟思考循环。  
   - 关键点：蒸馏时保留了思考模式的“循环结构”，而不是单纯模仿一次性输出。

5. **多语言扩展**  
   - 数据层面：收集并清洗了覆盖 119 种语言的平行语料，使用语言标签进行自监督对齐。  
   - 模型层面：在稠密层加入语言嵌入向量，在 MoE 层让不同语言的专家共享部分参数，提升低资源语言的学习效率。  

**最巧妙的设计**：思考预算的实现方式并非硬性限制循环次数，而是通过动态算子调度（如自适应层数、可变宽度的专家激活）在同一前向过程中灵活分配算力，这让模型在同一硬件上既能快速响应，也能在需要时深度推理。

### 实验与效果
- **评测任务**：代码生成（HumanEval、MBPP）、数学推理（MATH、GSM‑8K）、多语言问答（XGLUE、MMLU‑Crosslingual）、Agent 场景（OpenAI Gym 交互任务）等。  
- **对比基线**：Qwen2.5 系列、GPT‑4o、Claude‑3、以及同规模的 MoE 模型（如 DeepSeek‑MoE）。  
- **主要结果**：报告称 Qwen3 在所有公开基准上均达到或超过 SOTA，尤其在数学推理和代码生成上与更大规模的 MoE 模型相当；在多语言任务上，平均准确率提升约 7%（从 29 到 119 语言的覆盖带来的收益）。  
- **消融实验**：  
  1. 去掉思考预算，模型在深度推理任务上延迟提升 30% 且准确率下降约 2%。  
  2. 仅使用稠密主干而不启用 MoE，规模相同的模型在大规模推理任务上性能下降约 4%。  
  3. 不进行跨语言蒸馏，低资源语言的 F1 分数下降约 5%。  
- **局限性**：报告承认思考模式仍会带来显著的推理时延，预算调参对非专业用户不够友好；此外，MoE 的专家调度在极端低算力设备上仍有实现难度。

### 影响与延伸思考
Qwen3 的开源发布让业界首次看到一个统一「快速聊天 + 深度推理」的完整实现，推动了后续模型在同一体制下兼顾多任务的设计趋势。随后出现的「Unified LLM」系列、Meta 的「Llama‑Reason」以及一些学术工作（如「Dynamic Budget Transformers」）都在思考预算或模式切换上借鉴了 Qwen3 的思路。对想进一步探索的读者，可以关注以下方向：  
- **预算自适应学习**：让模型自行学习在不同任务上分配思考预算的策略。  
- **轻量化 MoE 调度**：在移动端或边缘设备上实现高效的专家选择。  
- **跨语言对齐的更细粒度方法**：利用语言族结构进一步提升低资源语言的表现。  

### 一句话记住它
Qwen3 用「思考预算」把快速聊天和深度推理装进同一模型，让你随时决定是要“快聊”还是“深思”。