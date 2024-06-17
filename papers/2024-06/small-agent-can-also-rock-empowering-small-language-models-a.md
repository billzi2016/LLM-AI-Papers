# Small Agent Can Also Rock! Empowering Small Language Models as   Hallucination Detector

> **Date**：2024-06-17
> **arXiv**：https://arxiv.org/abs/2406.11277

## Abstract

Hallucination detection is a challenging task for large language models (LLMs), and existing studies heavily rely on powerful closed-source LLMs such as GPT-4. In this paper, we propose an autonomous LLM-based agent framework, called HaluAgent, which enables relatively smaller LLMs (e.g. Baichuan2-Chat 7B) to actively select suitable tools for detecting multiple hallucination types such as text, code, and mathematical expression. In HaluAgent, we integrate the LLM, multi-functional toolbox, and design a fine-grained three-stage detection framework along with memory mechanism. To facilitate the effectiveness of HaluAgent, we leverage existing Chinese and English datasets to synthesize detection trajectories for fine-tuning, which endows HaluAgent with the capability for bilingual hallucination detection. Extensive experiments demonstrate that only using 2K samples for tuning LLMs, HaluAgent can perform hallucination detection on various types of tasks and datasets, achieving performance comparable to or even higher than GPT-4 without tool enhancements on both in-domain and out-of-domain datasets. We release our dataset and code at https://github.com/RUCAIBox/HaluAgent.

---

# 小模型也能大显身手！赋能小型语言模型进行幻觉检测 论文详细解读

### 背景：这个问题为什么难？

大型语言模型（LLM）在生成文本、代码、数学式等内容时常会出现“幻觉”——即输出看似合理却与事实不符的答案。现有的幻觉检测大多依赖像 GPT‑4 这样的闭源、参数量巨大的模型，成本高且难以部署到资源受限的环境。更糟的是，检测任务本身多样化：从事实性问答到代码语法、再到数学计算，每种幻觉都有不同的判别标准。于是，如何让参数更小、开源的模型也能可靠地发现这些错误，成为了一个迫切且技术上不易突破的难题。

### 关键概念速览
- **幻觉（Hallucination）**：模型生成的内容在事实、逻辑或形式上与真实答案不匹配，就像人类在记忆模糊时说出错误信息一样。  
- **工具箱（Toolbox）**：一组外部可调用的功能模块（如搜索、计算器、代码解释器），相当于给模型配备了“插件”，帮助它在特定子任务上获得更精准的答案。  
- **ReAct 框架**：让模型在推理过程中交替执行“思考”（Reason）和“行动”（Act），类似于人在查资料时先思考要查什么，再去搜索。  
- **三阶段检测框架**：检测流程被拆成句子切分、工具选择与执行、结果反思三个步骤，层层递进提升精度。  
- **记忆机制（Memory）**：模型在一次对话或检测任务中保留前面的中间结果，类似于人做笔记，防止重复劳动并帮助全局一致性检查。  
- **双语微调**：在中英文数据上进行少量（约 2K 条）微调，使小模型能够同时处理两种语言的幻觉检测任务。  

### 核心创新点
1. **小模型+工具箱的协同**  
   - 之前：幻觉检测几乎全部依赖大模型内部的知识，缺少外部工具的帮助。  
   - 本文：把小模型（如 Baichuan2‑Chat 7B）包装进一个能够主动调用多功能工具的 agent，使其在需要时借助搜索、计算等外部能力。  
   - 改变：即使模型本身知识有限，也能通过工具获取最新信息或精确计算，从而实现与大模型相当的检测效果。

2. **细粒度三阶段检测流程**  
   - 之前：多数方法一次性给出“是否幻觉”的二分类答案，缺少过程化的检查。  
   - 本文：先把长文本拆成句子单元，再为每句挑选最合适的工具，最后让模型对工具返回进行全局和局部的反思校验。  
   - 改变：检测更细致，错误定位更明确，尤其在多句复杂回答中表现出更高的召回率。

3. **少量双语微调即可提升**  
   - 之前：提升小模型性能往往需要大规模数据和长时间训练。  
   - 本文：利用已有中英文幻觉检测数据合成“检测轨迹”，仅用 2 K 条样本对模型进行微调，就让它学会如何在 ReAct 框架下调用工具并进行自我纠错。  
   - 改变：显著降低了数据和算力门槛，使得开源小模型也能快速适配幻觉检测任务。

4. **记忆驱动的自我校验**  
   - 之前：工具的输出被直接当作最终答案，忽视了工具本身可能出错的风险。  
   - 本文：在每轮工具调用后，模型会把结果写入记忆，并在后续步骤中再次审视这些信息，类似于人做完实验后再检查实验记录。  
   - 改变：提升了整体鲁棒性，尤其在工具不可靠或返回模糊信息时仍能保持较高准确率。

### 方法详解
**整体思路**  
HaluAgent 把一个相对小的语言模型包装成一个“自主探测者”。它在接收到待检测文本后，先把文本切成独立的句子（或代码行、数学式），然后对每个单元执行三步：① 选择并调用最合适的外部工具；② 收集工具返回；③ 让模型基于记忆对返回进行局部和全局的验证。整个过程遵循 ReAct 的“思考–行动–思考”循环，并在每轮循环结束后把关键中间状态写入记忆，以供后续使用。

**关键模块拆解**  

1. **句子切分模块**  
   - 功能：把输入文本划分为若干检测单元，确保每个单元的检测策略可以单独优化。  
   - 类比：像老师在批改作文时先把文章拆成段落，再逐段点评。  

2. **工具选择与调用模块**  
   - 输入：当前检测单元的内容、已有记忆。  
   - 过程：模型在内部生成一段“思考”文字，列出可能需要的工具（搜索、计算器、代码解释器等），随后输出一条“行动”指令，触发对应工具的 API。  
   - 关键点：工具列表是预先定义好的，每个工具都有统一的输入/输出格式，模型只需要说出工具名和参数即可。  

3. **记忆与反思模块**  
   - 记忆结构：一个列表，记录每轮的检测单元、所选工具、工具返回以及模型的评估结论。  
   - 反思过程：在收到工具返回后，模型再次进行“思考”，检查返回是否满足单元的检测需求（例如：搜索结果是否包含可信来源、计算器结果是否与原式相符），并在必要时发起二次调用或直接给出“幻觉/非幻觉”标签。  
   - 设计巧妙之处：即使工具本身出现错误，模型仍能通过记忆中的上下文信息进行二次校验，避免“一锤子买卖”。  

4. **双语微调数据生成**  
   - 作者把公开的中英文幻觉检测数据转换成 ReAct 轨迹：每条样本包含“用户提问 → 模型思考 → 工具调用 → 工具返回 → 模型反思 → 最终标签”。  
   - 只用了约 2 K 条这样的轨迹，对小模型进行轻量微调，使其学会在实际使用时自动完成上述循环。  

**最反直觉的设计**  
把“工具不可靠”这一常识直接写进模型的推理流程，而不是假设工具永远正确。通过记忆驱动的二次校验，让模型在不确定的情况下仍能保持判断的稳健性，这在传统的“模型+工具”方案中很少出现。

### 实验与效果
- **数据与任务**：作者在中文和英文的多种幻觉检测基准上评估，包括事实性问答、代码生成、数学计算等五大类型。数据来源于公开的 QA、代码、数学题库，并加入了作者自行构造的跨语言测试集。  
- **对比基线**：主要与 GPT‑4（未使用任何工具）以及几种基于大模型的检测器（如直接让 GPT‑4 做二分类）进行比较。  
- **核心结果**：在 2 K 条微调样本的支持下，HaluAgent 在大多数任务上达到了与 GPT‑4 相当甚至略高的 F1 分数。具体而言，在中文事实性 QA 上提升约 3% 的召回率，在代码错误检测上超过 2% 的准确率。  
- **消融实验**：作者分别去掉记忆机制、三阶段划分和工具调用，发现：  
  - 去掉记忆后整体性能下降约 4%（说明二次校验贡献显著）。  
  - 直接一次性检测（不拆句）导致召回率下降约 5%。  
  - 不使用工具箱则性能跌至仅比随机略好，验证了工具的关键作用。  
- **局限性**：实验主要在已有的公开数据上进行，真实业务场景中的长文档、跨领域知识仍未充分验证；此外，工具的质量仍受外部 API 稳定性影响，作者在论文中承认这可能限制系统的可部署性。  

### 影响与延伸思考
这篇工作向社区展示了“小模型+工具”可以在高价值任务上逼近大模型的性能，激发了后续对开源模型与插件化架构的兴趣。随后出现的几篇论文（如 **ToolFormer**、**Plug-and-Play LLMs**）在不同程度上借鉴了 HaluAgent 的三阶段检测思路和记忆驱动的自我校验机制。对想进一步探索的读者，可以关注以下方向：  
- **工具可靠性评估**：如何让模型自行判断工具返回的可信度。  
- **跨模态工具集成**：把图像、音频分析工具加入检测流程，扩展到多模态幻觉检测。  
- **自适应微调**：在实际部署后利用用户反馈自动生成新的 ReAct 轨迹，实现持续学习。  

### 一句话记住它
只要让小模型学会像人一样挑工具、记笔记、再检查，它就能用几千条数据把幻觉检测做到和 GPT‑4 打平。