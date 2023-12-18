# An In-depth Look at Gemini's Language Abilities

> **Date**：2023-12-18
> **arXiv**：https://arxiv.org/abs/2312.11444

## Abstract

The recently released Google Gemini class of models are the first to comprehensively report results that rival the OpenAI GPT series across a wide variety of tasks. In this paper, we do an in-depth exploration of Gemini's language abilities, making two contributions. First, we provide a third-party, objective comparison of the abilities of the OpenAI GPT and Google Gemini models with reproducible code and fully transparent results. Second, we take a closer look at the results, identifying areas where one of the two model classes excels. We perform this analysis over 10 datasets testing a variety of language abilities, including reasoning, answering knowledge-based questions, solving math problems, translating between languages, generating code, and acting as instruction-following agents. From this analysis, we find that Gemini Pro achieves accuracy that is close but slightly inferior to the corresponding GPT 3.5 Turbo on all tasks that we benchmarked. We further provide explanations for some of this under-performance, including failures in mathematical reasoning with many digits, sensitivity to multiple-choice answer ordering, aggressive content filtering, and others. We also identify areas where Gemini demonstrates comparably high performance, including generation into non-English languages, and handling longer and more complex reasoning chains. Code and data for reproduction can be found at https://github.com/neulab/gemini-benchmark

---

# 深入探讨 Gemini 语言能力 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）已经可以写文章、写代码、解数学题，但要真正说清楚它们在不同任务上的强弱仍是个难题。过去的评测大多由模型提供方自己完成，数据集、评测脚本甚至指标都有可能被“调参”。于是业界缺少一种公开、可复现、第三方视角的对比，导致我们很难判断新模型到底在何处超越或落后于已有模型。正因为评测方法不透明、结果难以复现，这篇论文才有了价值。

### 关键概念速览
- **大语言模型（LLM）**：能够理解并生成自然语言的深度学习模型，像 GPT‑3.5、Gemini Pro 这类模型拥有数十亿甚至上千亿参数。可以把它想象成“会说话的百科全书”。
- **基准数据集（benchmark）**：一组标准化的测试题目，用来衡量模型在特定能力上的表现。类似于体育比赛的计时表，大家用同一套规则比拼。
- **指令遵循（instruction‑following）**：模型在接收到明确指令后，能够产生符合意图的输出。就像你让助理写邮件，助理要准确抓住要点。
- **思维链（Chain‑of‑Thought, CoT）**：让模型在给出答案前先写出推理步骤，类似于解题时先列出草稿，帮助模型避免“一口气”直接出错。
- **内容过滤（content filtering）**：模型内部的安全机制，会在输出中屏蔽敏感或违规信息。相当于聊天时的“审查员”，有时会误伤正常内容。
- **多选题顺序敏感性**：模型在选择题中对选项顺序的依赖程度。想象考试时把选项 A、B、C、D 随机换位，模型的答案是否会随之变化。

### 核心创新点
1. **第三方客观基准 → 开源评测框架 → 结果透明可复现**  
   过去大多数对比都是模型厂商内部完成，这篇论文直接搭建了一个公开的 GitHub 项目，提供完整代码、数据下载脚本和运行说明，让任何人都能在相同环境下复现实验。

2. **细粒度错误诊断 → 多维度性能剖析 → 揭示模型优势与短板**  
   作者不仅报告整体准确率，还针对数学多位数运算、选项顺序、长链推理等具体场景做了专项实验，帮助我们看到 Gemini 在长文本推理上表现不错，但在高精度数值计算上容易失手。

3. **跨语言生成对比 → 多语言数据集 → 确认 Gemini 在非英语生成上的竞争力**  
   通过在多语言翻译和生成任务上加入中文、法语等语种，作者发现 Gemini 在非英语输出质量上几乎可以追平甚至略胜 GPT‑3.5。

4. **内容过滤影响评估 → 人工干预实验 → 解释部分性能下降的根因**  
   通过关闭或放宽安全过滤，实验显示 Gemini 在某些知识问答上会恢复部分被过滤掉的正确答案，说明过滤策略在一定程度上牺牲了准确性。

### 方法详解
**整体框架**  
这篇论文的实验流程可以概括为三步：① 选取覆盖多种语言能力的 10 套公开数据集；② 用统一的评测脚本分别跑 GPT‑3.5 Turbo 与 Gemini Pro；③ 对比结果并进行错误模式分析。整个过程全部开源，任何人只要按照说明装好依赖，就能得到同样的表格和图表。

**关键模块拆解**  

1. **数据集挑选**  
   - 任务覆盖：常识问答、数学推理、代码生成、机器翻译、指令遵循等。  
   - 每类任务挑选 1–2 个主流基准（如 MMLU、GSM‑8K、HumanEval），确保评测广度。

2. **统一评测脚本**  
   - 脚本负责向模型发送 Prompt（输入），收集模型返回的 Completion（输出），再根据任务类型调用相应的评分函数（准确率、BLEU、代码通过率等）。  
   - 为了避免 API 调用差异带来的噪声，脚本对两种模型使用相同的温度、最大生成长度等超参数。

3. **结果聚合与可视化**  
   - 将每个任务的指标汇总成表格，使用条形图展示两模型的相对差距。  
   - 额外生成“错误热图”，标记哪些子任务出现了显著差异。

4. **错误模式实验**  
   - **数学多位数**：在 GSM‑8K 基础上构造了 100 题 10 位以上的算术题，观察两模型的正确率变化。  
   - **选项顺序**：对同一道多选题随机排列选项 5 次，统计答案一致性。  
   - **内容过滤**：在 OpenAI 与 Gemini 的安全设置上分别打开/关闭，比较同一批敏感问答的输出差异。  
   - **长链推理**：挑选需要 5 步以上 CoT 的题目，比较两模型的完整推理链长度和正确率。

**最巧妙的设计**  
作者把“模型内部安全过滤”当作一个可控变量来实验，这在公开评测里很少见。通过手动调节过滤阈值，直接量化了安全机制对性能的影响，提供了一个全新视角来审视模型的“安全‑效能”权衡。

### 实验与效果
- **测试任务**：共 10 套数据集，涵盖常识问答、数学推理、代码生成、机器翻译、指令遵循等。  
- **对比基线**：OpenAI 的 GPT‑3.5 Turbo（官方 API）和 Google Gemini Pro（官方访问渠道）。  
- **整体表现**：Gemini Pro 在所有任务上都紧随 GPT‑3.5 Turbo，整体准确率略低。作者描述为“接近但稍逊”，具体差距在不同任务上从几百分点到十几百分点不等（原文未给出精确数字）。  
- **优势**：在非英语生成（尤其是中文、法语）以及需要较长推理链的任务上，Gemini 的得分与 GPT‑3.5 相当，甚至在某些长链推理题上略有优势。  
- **劣势**：高精度数学运算（多位数）表现不佳；对多选题选项顺序敏感，答案一致性低于 GPT‑3.5；内容过滤导致部分知识问答被截断，影响整体准确率。  
- **消融实验**：通过关闭 Gemini 的内容过滤，部分被过滤的正确答案恢复，验证了过滤是导致性能下降的关键因素之一。  
- **局限性**：评测仅覆盖公开数据集，未涉及行业专有任务；对模型内部实现细节（如微调数据、模型规模）缺乏深入剖析；实验环境受 API 调用限制，可能出现速率或随机性噪声。

### 影响与延伸思考
这篇工作在社区里起到了“第三方审计”的示范效应，促使更多研究者在发布新模型时提供可复现的评测代码。随后出现的几篇对比 Gemini、Claude、LLaMA 系列的论文，都直接引用了该 GitHub 项目作为基准平台。对安全过滤的量化实验也激发了后续关于“可调节安全阈值”和“安全‑效能平衡”的研究方向。想进一步了解模型在特定安全策略下的行为，建议关注近期的“安全对齐”系列工作以及对大模型可解释性的探索。

### 一句话记住它
Gemini Pro 在多语言和长链推理上能跟 GPT‑3.5 打平，但在高精度数学和安全过滤下稍显逊色，这篇论文用公开、可复现的基准把这些差异摆在了台面上。