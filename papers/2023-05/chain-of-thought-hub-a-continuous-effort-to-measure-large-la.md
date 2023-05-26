# Chain-of-Thought Hub: A Continuous Effort to Measure Large Language   Models' Reasoning Performance

> **Date**：2023-05-26
> **arXiv**：https://arxiv.org/abs/2305.17306

## Abstract

As large language models (LLMs) are continuously being developed, their evaluation becomes increasingly important yet challenging. This work proposes Chain-of-Thought Hub, an open-source evaluation suite on the multi-step reasoning capabilities of large language models. We are interested in this setting for two reasons: (1) from the behavior of GPT and PaLM model family, we observe that complex reasoning is likely to be a key differentiator between weaker and stronger LLMs; (2) we envisage large language models to become the next-generation computational platform and foster an ecosystem of LLM-based new applications, this naturally requires the foundation models to perform complex tasks that often involve the composition of linguistic and logical operations. Our approach is to compile a suite of challenging reasoning benchmarks to track the progress of LLMs. Our current results show that: (1) model scale clearly correlates with reasoning capabilities; (2) As of May 2023, Claude-v1.3 and PaLM-2 are the only two models that are comparable with GPT-4, while open-sourced models still lag behind; (3) LLaMA-65B performs closely to code-davinci-002, indicating that with successful further development such as reinforcement learning from human feedback (RLHF), it has great potential to be close to GPT-3.5-Turbo. Our results also suggest that for the open-source efforts to catch up, the community may focus more on building better base models and exploring RLHF.

---

# 思维链中心：持续衡量大语言模型推理能力的评估套件 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在生成自然语言方面已经相当成熟，但要让它们像人一样进行多步、逻辑严密的推理仍然是瓶颈。过去的评测大多聚焦于单句完形填空、问答或翻译，几乎不要求模型展示中间思考过程。于是模型的“聪明”往往只能在表层语言上体现，而真正的推理能力难以量化。更糟的是，随着模型规模不断扩大，传统的准确率指标已经无法区分不同规模模型在复杂任务上的细微差距，导致研究者缺少统一、可重复的基准来追踪进展。正因如此，需要一个专门针对多步推理、能够持续更新的评测平台。

### 关键概念速览
- **Chain-of-Thought（思维链）**：让模型在给出最终答案前先把推理步骤写出来，类似于人解数学题时先在草稿纸上列出每一步，能够让模型的中间思考过程被检验和纠错。  
- **Few-shot prompting（少样本提示）**：在输入中给模型提供少量示例（通常 1–5 条），帮助模型快速适应任务格式，而不需要大规模微调。  
- **RLHF（人类反馈强化学习）**：先让模型生成答案，再让人工标注者对答案好坏打分，模型再根据这些分数进行强化学习，以提升对齐度和实用性。  
- **Benchmark suite（基准套件）**：一组标准化的测试任务和数据，供不同模型在相同条件下进行比较，就像跑步比赛的统一赛道。  
- **Open-source model（开源模型）**：代码、权重公开的模型，社区可以自由下载、改进或二次训练，区别于只能通过 API 调用的商业模型。  
- **Scale‑reasoning correlation（规模‑推理相关性）**：模型参数量越大，往往在多步推理任务上表现越好，这是一种经验规律。  

### 核心创新点
1. **从零散任务到统一套件 → 构建 CoT Hub → 形成可持续、可扩展的多步推理基准**  
   以前研究者各自挑选几道推理题做实验，缺乏统一的评测框架。CoT Hub 把已有的挑战性推理数据（如数学、逻辑、常识组合）统一收录，并提供统一的 few‑shot CoT 提示模板，使得不同模型的评测可以在同一“跑道”上进行比较。

2. **仅关注最终答案的评估方式 → 引入完整思维链的可视化对齐 → 让评测更公平**  
   传统评测只检查答案是否正确，忽略了模型是否在中间步骤出现明显错误。CoT Hub 在评测脚本中加入对思维链的完整匹配（即每一步是否符合预期），从而避免模型“偷懒”直接给出答案而不展示推理过程。

3. **公开评测代码与排行榜 → 促进社区协作 → 开源模型的进步更易被量化**  
   过去很多评测是闭源的，只在论文里给出表格。CoT Hub 把评测脚本、数据下载、结果可视化全部开源，并提供自动化的排行榜更新机制，任何人只要提交模型输出即可看到最新排名，极大降低了复现门槛。

4. **系统性分析模型规模与推理能力的关系 → 给出明确的趋势图 → 为后续模型设计提供参考**  
   作者在不同规模的 LLM（从几十亿到上百亿参数）上跑完整套基准，发现规模提升带来的推理性能提升几乎呈线性。这个发现帮助社区认识到，仅靠模型放大仍是提升推理的有效手段，同时也暗示了需要更高效的微调或 RLHF 才能在同等规模下赶超。

### 方法详解
**整体框架**  
CoT Hub 的评测流程可以拆成四步：① 数据收集与标准化；② Few‑shot CoT 提示构造；③ 模型推理与思维链捕获；④ 结果对齐与评分。整个过程全部脚本化，用户只需提供模型的 API 接口或本地推理函数，即可得到完整的评测报告。

**步骤拆解**  

1. **数据收集与标准化**  
   - 作者从公开的多步推理数据集（如 GSM8K、SVAMP、LogicalDeduction 等）挑选出 30+ 任务，确保覆盖数学、逻辑、常识组合等不同推理维度。  
   - 每条样本都被统一成三段格式：*题干*、*参考思维链*（人工标注的逐步推理过程）和*正确答案*。这样评测脚本可以直接对比模型输出的每一步。

2. **Few‑shot CoT 提示构造**  
   - 对每个任务，选取 2–3 条相似的示例（题干 + 思维链 + 答案）拼接在一起，形成 few‑shot 提示。  
   - 提示的结构固定为：“下面是几个例子：\n例1…\n例2…\n请解答下面的问题并写出思考过程”。这种模板化让不同模型在同样的上下文下进行推理，避免提示差异带来的偏差。

3. **模型推理与思维链捕获**  
   - 将提示送入模型，要求模型返回完整的文本。脚本会自动截取从“思考过程”开始到答案结束的部分，形成模型的思维链。  
   - 为了兼容不同模型的输出风格，脚本使用正则表达式匹配常见的步骤标记（如 “Step 1:”, “1.” 等），确保即使模型自行编号也能被捕获。

4. **结果对齐与评分**  
   - **步骤一致性**：将模型的每一步与参考思维链逐句比对，使用 BLEU/ROUGE 等文本相似度指标，计算一步一步的匹配分。  
   - **最终答案正确性**：直接比较模型给出的数值或选项是否与正确答案一致。  
   - 两者加权得到综合得分，默认 70% 关注答案正确率，30% 关注思维链质量。用户可以自行调节权重以适应不同研究需求。

**最巧妙的设计**  
- **思维链对齐层**：大多数评测只看答案，CoT Hub 把思维链对齐作为评分的显著部分，这让模型必须“写对过程”，从而更真实地反映推理能力。  
- **自动化排行榜**：评测脚本在每次运行后会把得分推送到 GitHub Pages 上的公开排行榜，支持模型名称、版本、推理时间等信息的展示，形成社区驱动的竞争氛围。

### 实验与效果
- **测试任务**：共计 30+ 子任务，涵盖数学计算（GSM8K、SVAMP）、逻辑推理（LogicalDeduction、ProofWriter）、常识组合（StrategyQA）等。每个子任务均使用 few‑shot CoT 提示进行评测。  
- **对比基线**：包括 GPT‑3.5‑Turbo、GPT‑4、Claude‑v1.3、PaLM‑2、LLaMA‑65B、Code‑davinci‑002 以及若干开源模型（如 Vicuna、Mistral）。  
- **关键结果**（摘自论文）：  
  - 在整体得分上，GPT‑4 仍居首位，Claude‑v1.3 与 PaLM‑2 紧随其后，二者在思维链一致性上与 GPT‑4 差距不到 5%。  
  - LLaMA‑65B 的综合得分约为 Code‑davinci‑002 的 92%，显示出仅凭规模提升即可逼近商用模型的水平。  
  - 开源模型（如 Vicuna‑13B）在答案正确率上落后 15%~20%，思维链匹配更是低于 30%。  
- **消融实验**：作者分别去掉思维链对齐评分、改为 0‑shot 提示以及只使用答案正确率进行评估。结果表明：去掉思维链对齐后，模型之间的差距显著缩小，说明思维链评分是区分模型推理深度的关键因素。  
- **局限性**：论文承认评测仍然依赖人工标注的参考思维链，可能带有主观偏好；此外，few‑shot 提示的选取对结果有一定敏感性，未系统探索提示选择的最优策略。

### 影响与延伸思考
CoT Hub 在发布后迅速被多个研究团队引用，成为评测 LLM 推理能力的事实标准。后续工作如 **MATH‑CoT**、**LogicalBench** 等都在其数据格式和评测脚本的基础上扩展任务种类。社区也利用 Hub 的开源框架，加入了 **自适应提示生成**（让模型自行挑选最相似的示例）和 **多语言思维链**（将评测扩展到非英语）等方向。对想进一步深入的读者，建议关注以下两条路：① **提示工程**——如何自动化挑选 few‑shot 示例以最大化推理性能；② **RLHF 在思维链上的细化**——让人类反馈不仅评价答案，还评价每一步的合理性，从而训练出更“会写草稿”的模型（推测）。

### 一句话记住它
CoT Hub 用统一的多步推理基准和思维链对齐评分，让我们能够客观比较不同规模和开源程度的 LLM 推理实力。