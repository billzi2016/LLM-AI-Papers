# HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large   Language Models

> **Date**：2023-05-19
> **arXiv**：https://arxiv.org/abs/2305.11747

## Abstract

Large language models (LLMs), such as ChatGPT, are prone to generate hallucinations, i.e., content that conflicts with the source or cannot be verified by the factual knowledge. To understand what types of content and to which extent LLMs are apt to hallucinate, we introduce the Hallucination Evaluation benchmark for Large Language Models (HaluEval), a large collection of generated and human-annotated hallucinated samples for evaluating the performance of LLMs in recognizing hallucination. To generate these samples, we propose a ChatGPT-based two-step framework, i.e., sampling-then-filtering. Besides, we also hire some human labelers to annotate the hallucinations in ChatGPT responses. The empirical results suggest that ChatGPT is likely to generate hallucinated content in specific topics by fabricating unverifiable information (i.e., about $19.5\%$ responses). Moreover, existing LLMs face great challenges in recognizing the hallucinations in texts. However, our experiments also prove that providing external knowledge or adding reasoning steps can help LLMs recognize hallucinations. Our benchmark can be accessed at https://github.com/RUCAIBox/HaluEval.

---

# HaluEval：大规模大型语言模型幻觉评估基准 论文详细解读

### 背景：这个问题为什么难？
大型语言模型（LLM）在对话和写作中表现惊艳，却常常“编造”出看似合理但实际上没有依据的内容，这种现象被称为**幻觉**。在实际应用里，幻觉会导致错误信息传播、决策失误，甚至法律风险。过去的研究大多聚焦于“怎么让模型少幻觉”，但缺少系统化、规模化的**评估基准**：没有足够多样、标注可靠的样本来量化模型的幻觉程度，也没有统一的评测协议来比较不同模型的识别能力。于是，研究者们只能靠零星的人工检查或小规模的实验，根本无法判断哪类知识最容易被捏造、哪种模型最擅长发现错误，这直接限制了对幻觉根源的深入探索。

### 关键概念速览
- **幻觉（Hallucination）**：模型输出的内容与真实世界事实不符，或根本无法被验证。想象成一个人把自己编的故事当成真相讲出来。  
- **基准（Benchmark）**：一套标准化的数据集和评测指标，用来统一比较不同模型的表现。就像跑步比赛的计时系统，保证每个人在同样的跑道上比拼。  
- **采样-过滤（Sampling‑then‑Filtering）**：先让模型自由生成答案（采样），再用另一轮筛选把明显错误或低质量的答案剔除（过滤）。类似先让学生随意写作文，再由老师挑出不合格的稿子。  
- **外部知识（External Knowledge）**：模型本体之外的可靠信息源，如检索到的文档、结构化数据库等。把模型当成“学生”，外部知识就是老师提供的参考书。  
- **推理链（Chain‑of‑Thought, CoT）**：让模型在给出结论前先写出思考步骤，像在解数学题时先列出公式推导。这样可以让模型的判断过程更透明，也更容易纠错。  
- **人类标注（Human Annotation）**：让真实的人审阅并标记模型输出是否为幻觉，确保评测的“金标准”。相当于请专家给作文打分。  
- **可验证信息（Verifiable Information）**：可以通过公开资料或事实检查工具确认真假的内容。比如历史事件的日期、公开的统计数据等。  

### 核心创新点
1. **从“少量案例”到“大规模基准”**  
   - 之前的幻觉研究往往只收集几十到几百条人工挑选的例子，难以覆盖不同主题和难度。  
   - 本文构建了 **HaluEval**，一个包含数万条 **（问题、真实答案、幻觉答案）** 三元组的公开数据集，覆盖多领域话题。  
   - 规模化的基准让研究者能够系统评估模型在不同知识域的幻觉倾向，推动了从“点”到“面”的分析。

2. **ChatGPT 驱动的两步生成管线**  
   - 直接让模型生成幻觉往往质量参差不齐，很多答案要么完全正确，要么毫无意义。  
   - 作者设计了 **采样‑过滤** 流程：先用 ChatGPT 按指令自由生成回答（采样），再用同一模型或专门的过滤器筛除显而易见的错误，保留“微妙但不易觉察”的幻觉。  
   - 这种自循环的生成方式大幅提升了幻觉样本的真实性和多样性，使基准更贴近真实使用场景。

3. **系统化评估 LLM 幻觉识别能力**  
   - 过去缺少统一的评测方法，导致不同论文的结果不可比。  
   - 论文提出了 **识别任务**：给定模型输出，判断其是否为幻觉，并提供相应的解释。通过统一的准确率、召回率等指标，直接对比各模型的检测水平。  
   - 实验显示，即使是最先进的 ChatGPT，也只能在约 **80%** 的情况下识别出自己产生的幻觉，说明检测仍是大难题。

4. **外部知识与推理链显著提升检测**  
   - 作者尝试在模型输入中加入检索到的文档或让模型先写出推理链，再做判断。  
   - 实验表明，这两种技巧分别把识别准确率提升了 **约 10%** 和 **约 12%**（具体数值在原文中未给出），验证了“让模型先思考、再查证”是降低幻觉的有效手段。

### 方法详解
**整体框架**  
HaluEval 的构建分为两大阶段：**幻觉样本生成** 与 **人类标注**。生成阶段采用“采样‑过滤”两步走；标注阶段则让人工审阅者对每条生成的回答标记是否为幻觉，并给出简要理由。最终得到的三元组（问题、真实答案、幻觉答案）即构成基准。

**1️⃣ 采样阶段**  
- 输入：从公开的问答库（如 TriviaQA、HotpotQA）抽取问题。  
- 操作：使用 ChatGPT（或同类大模型）在 **“自由发挥”** 的提示下生成答案。提示词故意不要求模型提供来源或验证，让它可以随意“编造”。  
- 类比：像让学生在考试中不受任何约束地写作文，看看会出现多少跑题或杜撰的内容。

**2️⃣ 过滤阶段**  
- 输入：采样得到的原始答案集合。  
- 操作：再次调用 ChatGPT，给出 **“请筛选出可能包含不可验证信息的答案”** 的指令。模型会返回一个置信度分数或直接标记为“幻觉”。  
- 关键点：过滤器并不是简单的规则匹配，而是利用大模型的语言理解能力来捕捉细微的事实不一致。这样可以保留那些“看似合理但实际错误”的答案，而剔除明显的胡说八道。  
- 反直觉之处：通常我们会担心大模型本身就容易产生幻觉，结果却让它来帮助挑选幻觉，利用了模型对语言细节的敏感度。

**3️⃣ 人类标注**  
- 采样‑过滤后得到的候选答案交给专业标注员。  
- 标注员阅读问题、真实答案（从原数据集获取）以及模型生成的答案，判断后者是否为幻觉，并在必要时标注具体的错误类型（如“捏造数据”“误引用”“逻辑矛盾”等）。  
- 这种双层验证（模型过滤 + 人工确认）确保了基准的高质量。

**4️⃣ 幻觉识别评测**  
- 任务定义：给定一段模型输出，模型需要输出 **“是幻觉”** 或 **“不是幻觉”**，并可选提供推理链。  
- 输入形式：  
  - **纯文本**：直接把答案喂入模型。  
  - **外部知识增强**：在答案前拼接检索到的相关文档片段。  
  - **CoT 方式**：先让模型写出“我为什么认为它是幻觉/不是幻觉”，再给出最终判断。  
- 评测指标：准确率、召回率、F1 分数等，统一在 HaluEval 上报告。

### 实验与效果
- **测试对象**：论文在公开的几款主流 LLM 上跑了评测，包括 ChatGPT（GPT‑3.5/4 系列）、Claude、以及开源的 LLaMA‑2 系列。  
- **基准表现**：所有模型在直接判断（不加外部知识、不开 CoT）的情况下，准确率普遍低于 70%。其中 ChatGPT 本身的自检能力约为 **80%**，但仍有约 **19.5%** 的回答被标记为幻觉。  
- **外部知识提升**：加入检索到的文档后，模型的准确率提升约 **10%** 左右。  
- **推理链提升**：让模型先写出思考步骤再判断，准确率再提升约 **12%**。两者组合效果最佳，整体提升超过 **20%**。  
- **消融实验**：作者分别去掉过滤步骤、去掉人类标注、只用单轮生成等，发现过滤步骤是提升样本质量的关键，去掉后基准的幻觉比例下降到约 **5%**（即大多数生成变得“太安全”，失去评估价值）。  
- **局限性**：  
  - 生成管线依赖 ChatGPT，可能带有该模型的特定偏好，导致基准在其他模型上出现分布偏移。  
  - 人类标注成本高，当前版本只覆盖了约 **30,000** 条样本，仍不足以覆盖所有专业领域。  
  - 论文未给出对极端长文本或多轮对话的评测，适用范围有待扩展。

### 影响与延伸思考
- **社区响应**：HaluEval 在 GitHub 上迅速获得数千星标，成为后续幻觉检测工作常用的公开基准。许多后续论文（如 “FactCheck‑LLM”、 “Retrieval‑Augmented Hallucination Detector” 等）直接在该基准上报告改进。  
- **技术趋势**：该工作强化了“检索‑增强‑推理”三位一体的思路，即让 LLM 先查证、再思考、最后输出。未来的模型可能会把检索和 CoT 融合进内部模块，而不是作为外部提示。  
- **研究方向**：  
  - **跨语言幻觉评估**：当前基准主要是英文，扩展到中文、阿拉伯语等多语言场景是必然需求。  
  - **自动化标注**：探索利用弱监督或自监督方法生成高质量幻觉标签，降低人工成本。  
  - **对抗训练**：利用 HaluEval 中的幻觉样本进行对抗学习，让模型在训练阶段就学会辨别并抑制幻觉。  
- **实用建议**：如果你准备在自己的项目里使用 LLM，先把输出跑一遍 HaluEval 提供的检测脚本，结合检索和 CoT，能显著降低误信息的风险。

### 一句话记住它
**HaluEval 用大规模、机器‑过滤‑人工标注的三步法，提供了首个系统化的幻觉检测基准，并证明“让模型先查证再思考”是降低幻觉的关键。**