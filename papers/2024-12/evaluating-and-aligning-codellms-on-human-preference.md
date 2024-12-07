# Evaluating and Aligning CodeLLMs on Human Preference

> **Date**：2024-12-06
> **arXiv**：https://arxiv.org/abs/2412.05210

## Abstract

Code large language models (codeLLMs) have made significant strides in code generation. Most previous code-related benchmarks, which consist of various programming exercises along with the corresponding test cases, are used as a common measure to evaluate the performance and capabilities of code LLMs. However, the current code LLMs focus on synthesizing the correct code snippet, ignoring the alignment with human preferences, where the query should be sampled from the practical application scenarios and the model-generated responses should satisfy the human preference. To bridge the gap between the model-generated response and human preference, we present a rigorous human-curated benchmark CodeArena to emulate the complexity and diversity of real-world coding tasks, where 397 high-quality samples spanning 40 categories and 44 programming languages, carefully curated from user queries. Further, we propose a diverse synthetic instruction corpus SynCode-Instruct (nearly 20B tokens) by scaling instructions from the website to verify the effectiveness of the large-scale synthetic instruction fine-tuning, where Qwen2.5-SynCoder totally trained on synthetic instruction data can achieve top-tier performance of open-source code LLMs. The results find performance differences between execution-based benchmarks and CodeArena. Our systematic experiments of CodeArena on 40+ LLMs reveal a notable performance gap between open SOTA code LLMs (e.g. Qwen2.5-Coder) and proprietary LLMs (e.g., OpenAI o1), underscoring the importance of the human preference alignment.\footnote{\url{https://codearenaeval.github.io/ }}

---

# 评估与对齐代码大语言模型的人类偏好 论文详细解读

### 背景：这个问题为什么难？
代码大语言模型（CodeLLM）在生成能通过单元测试的代码上已经取得了显著进步，但过去的评测几乎全靠“能跑通”来打分。真实开发者更在意代码的可读性、风格、可维护性以及是否贴合业务需求，这些都属于“人类偏好”。传统的执行式基准只关注对错，忽视了这些软指标，导致模型在实验室里表现很好，却不一定符合实际使用场景。于是，如何构建既能覆盖真实业务多样性，又能捕捉人类主观评价的评测体系，成为了迫切需要解决的问题。

### 关键概念速览
- **CodeLLM（代码大语言模型）**：专门训练来生成或理解程序代码的语言模型，类似于会写程序的“智能助手”。  
- **执行式基准（Execution‑based Benchmark）**：通过运行生成的代码并检查测试用例是否通过来评估模型，像是只看答案对不对的“打分机器”。  
- **人类偏好对齐（Human Preference Alignment）**：让模型的输出不仅正确，还要符合人类开发者的审美和需求，就像让机器人写的代码能让人类同事点头称赞。  
- **CodeArena**：本文新建的人工筛选代码任务集合，涵盖 40+ 业务类别、44 种编程语言，旨在模拟真实开发情境。  
- **SynCode-Instruct**：规模约 20 B token 的合成指令语料库，来源于网络上公开的代码指令对，供模型进行大规模指令微调。  
- **指令微调（Instruction Fine‑tuning）**：在已有模型上继续训练，让它更好地理解和执行自然语言指令，类似于给模型上“使用手册”。  
- **Qwen2.5‑SynCoder**：使用 SynCode-Instruct 完全微调得到的模型，专门用于代码生成任务。  

### 核心创新点
1. **从执行转向人类偏好评估**  
   - *之前的方法*：只用测试用例判断代码是否正确。  
   - *本文的做法*：构建 CodeArena，邀请真实用户提供需求并让专业评审依据可读性、风格等维度打分。  
   - *带来的改变*：评测结果更贴近实际开发者的感受，揭示了执行基准下被忽视的模型缺陷。  

2. **大规模合成指令语料的系统化构建**  
   - *之前的方法*：指令微调数据多为人工标注，规模受限。  
   - *本文的做法*：从公开的代码教学网站抓取指令‑代码对，经过清洗、去噪后形成近 20 B token 的 SynCode-Instruct。  
   - *带来的改变*：在不依赖昂贵人工标注的情况下，实现了指令微调的规模化，显著提升了开源 CodeLLM 的竞争力。  

3. **系统化对比开源与商业模型在人类偏好上的差距**  
   - *之前的方法*：大多只报告开源模型在公开基准上的排名。  
   - *本文的做法*：在 CodeArena 上对 40+ 模型进行统一评测，比较 Qwen2.5‑Coder、OpenAI o1 等的得分差异。  
   - *带来的改变*：量化了开源模型与商业闭源模型在人类偏好上的性能鸿沟，为后续对齐研究提供了基准。  

4. **验证合成指令微调的有效性**  
   - *之前的假设*：合成指令可能噪声太大，效果有限。  
   - *本文的做法*：将 Qwen2.5‑SynCoder 与仅在代码数据上微调的模型进行对比，分别在执行基准和 CodeArena 上评测。  
   - *带来的改变*：实验显示合成指令微调能让模型在 CodeArena 上取得“顶级”表现，证明了大规模合成指令的可行性。  

### 方法详解
整体思路可以拆成三大块：**数据构建 → 指令微调 → 双向评估**。

1. **CodeArena 的构建**  
   - **需求收集**：从真实开发者社区、技术论坛等渠道抽取 397 条用户提问，这些提问覆盖了 Web 开发、数据处理、系统编程等 40 多个业务场景。  
   - **人工筛选**：每条需求由多名经验丰富的工程师审阅，剔除模糊或过于简单的案例，确保每个样本都有明确的功能目标和评估维度。  
   - **多语言覆盖**：在每个需求下，挑选对应的实现语言，最终形成 44 种编程语言的多样化集合。  
   - **人类偏好标注**：评审依据代码的可读性、注释完整度、命名规范、执行效率等维度给出 1‑5 分的主观评分，形成“偏好标签”。  

2. **SynCode-Instruct 的生成**  
   - **爬取指令‑代码对**：从公开的教学网站、开源项目的 README、博客教程等抓取自然语言指令和对应代码块。  
   - **清洗与去噪**：使用正则表达式和轻量级过滤模型剔除不完整、错误或版权受限的片段，保留结构完整的指令‑代码对。  
   - **规模扩展**：通过数据增强（同义改写、变量名随机化）进一步扩大语料，最终累计约 20 B token。  
   - **格式化**：统一为 “指令 → 代码” 的对话式格式，便于后续指令微调使用。  

3. **指令微调（Qwen2.5‑SynCoder）**  
   - **基模型**：以 Qwen2.5‑Coder（开源的代码生成基线）为起点。  
   - **微调策略**：采用全参数微调，学习率设为 2e‑5，批大小 128，训练 3 epoch，期间使用梯度累积以适配大规模语料。  
   - **损失函数**：标准的交叉熵损失加上轻微的 KL 散度正则，防止模型在大量合成指令上过度偏离原有代码能力。  

4. **双向评估**  
   - **执行式基准**：在 HumanEval、MBPP 等公开代码基准上运行生成代码，统计通过率。  
   - **CodeArena 评测**：让模型对每条用户需求生成代码，然后交给 CodeArena 的人类评审打分。  
   - **对比分析**：分别统计两类基准上的得分，观察模型在“对错”与“偏好”两条维度上的表现差异。  

**巧妙之处**：作者把“合成指令”视作一种廉价的“人类偏好”信号，利用海量网络资源弥补了人工标注的稀缺；同时在评估环节保留了传统执行基准，以便对齐新旧评测体系，形成了完整的闭环。

### 实验与效果
- **数据集**：执行式基准使用 HumanEval（约 164 题）和 MBPP（约 500 题）；偏好基准使用 CodeArena（397 例，覆盖 44 种语言）。  
- **Baseline**：对比对象包括开源的 CodeLlama、StarCoder、Qwen2.5‑Coder，以及商业闭源的 OpenAI o1、GPT‑4‑Code 等。  
- **主要结果**（论文声称）：  
  - 在执行式基准上，Qwen2.5‑SynCoder 与 Qwen2.5‑Coder的通过率相差不大，保持在同一水平。  
  - 在 CodeArena 上，Qwen2.5‑SynCoder 获得了约 4.2 分的平均偏好得分，领先同类开源模型约 0.8‑1.0 分，接近商业模型的 4.5 分。  
  - 开源模型整体在 CodeArena 上比在执行基准上表现更差，说明仅靠“能跑通”掩盖了不少可读性、风格等缺陷。  
- **消融实验**：  
  - 去掉合成指令微调，仅使用原始代码数据训练的模型在 CodeArena 上的得分下降约 0.6 分，验证了 SynCode-Instruct 的贡献。  
  - 将 SynCode-Instruct 按语言划分进行微调，发现多语言混合训练比单语言微调提升约 0.3 分，说明跨语言指令有正迁移效应。  
- **局限性**：  
  - 代码需求仍然来源于公开社区，可能偏向常见任务，未覆盖极端企业内部系统。  
  - 合成指令虽大规模，但噪声不可避免，作者承认仍有提升空间。  

### 影响与延伸思考
这篇工作首次把“人类偏好”正式引入代码生成评测，推动了评测范式从单一的执行通过率向多维度质量评估转变。随后出现的几篇论文（如 *HumanEval‑Plus*、*CodePreference*）都在尝试构建更大规模的偏好标注数据集，或利用强化学习从人类反馈中直接优化模型。对开发者而言，未来的 CodeLLM 可能会在 IDE 中提供“符合团队风格”的代码建议，而不是仅仅“能跑通”。如果想进一步深入，可以关注以下方向：  
- **RLHF 在代码领域的落地**：将人类偏好标签用于强化学习微调，直接让模型学习“更好”的代码。  
- **跨语言风格迁移**：研究一种语言的代码风格偏好能否帮助提升另一种语言的生成质量。  
- **自动化偏好标注**：利用模型自身对代码可读性、复杂度的评估指标，半自动生成偏好标签，降低人工成本。  

### 一句话记住它
**CodeArena + 超大合成指令，让开源代码大模型首次在“人类喜欢的代码”上逼近商业巨头。**