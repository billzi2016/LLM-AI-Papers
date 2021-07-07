# Evaluating Large Language Models Trained on Code

> **Date**：2021-07-07
> **arXiv**：https://arxiv.org/abs/2107.03374

## Abstract

We introduce Codex, a GPT language model fine-tuned on publicly available code from GitHub, and study its Python code-writing capabilities. A distinct production version of Codex powers GitHub Copilot. On HumanEval, a new evaluation set we release to measure functional correctness for synthesizing programs from docstrings, our model solves 28.8% of the problems, while GPT-3 solves 0% and GPT-J solves 11.4%. Furthermore, we find that repeated sampling from the model is a surprisingly effective strategy for producing working solutions to difficult prompts. Using this method, we solve 70.2% of our problems with 100 samples per problem. Careful investigation of our model reveals its limitations, including difficulty with docstrings describing long chains of operations and with binding operations to variables. Finally, we discuss the potential broader impacts of deploying powerful code generation technologies, covering safety, security, and economics.

---

# 评估在代码上训练的大语言模型 论文详细解读

### 背景：这个问题为什么难？
在自然语言处理里，GPT 系列已经展示了强大的文本生成能力，但把同样的技术搬到代码上并不直接。代码不像普通句子，它必须严格符合语法、类型和运行时语义，任何小错误都会导致程序崩溃。早期的代码生成系统大多基于检索或模板，缺乏对复杂逻辑的理解，面对需要多步推理的题目几乎无能为力。于是研究者迫切需要一种既能捕捉大规模语言模式，又能在细粒度上遵守编程规则的模型，这正是本文要解决的核心难点。

### 关键概念速览
**大语言模型（LLM）**：在海量文本上训练的神经网络，能够预测下一个词或字符。把它想象成一个“会写作文的机器人”，只要给它足够的例子，它就能续写下去。  
**微调（Fine‑tuning）**：在已有的通用模型基础上，用特定领域的数据再训练几轮，让模型更专注于该领域的细节。类似于先学会通用英语，再专门学医学术语。  
**代码生成（Code Generation）**：模型直接输出可执行代码，而不是仅仅解释或翻译。相当于让机器人从零写出一个完整的程序。  
**文档字符串（Docstring）**：函数或模块顶部的文字说明，通常用自然语言描述功能、输入输出等。模型需要把这些文字“翻译”成实际代码。  
**功能正确性（Functional Correctness）**：生成的代码在给定测试用例下能够得到正确结果，而不是仅仅语法上合法。就像考试不仅要写出答案，还要能通过所有检查点。  
**采样（Sampling）**：在模型生成时，根据概率分布随机挑选下一个 token，而不是总是选概率最高的。可以把它看作让机器人在每一步都有一点“创意”。  
**HumanEval**：本文新推出的评测集，包含 164 条 Python 小题，每题都有 docstring 与隐藏的单元测试，用来衡量模型的功能正确率。  
**GitHub Copilot**：基于 Codex 的商业化插件，能在编辑器里实时给出代码建议。它是本文技术的直接落地产品。

### 核心创新点
**公开代码微调 → Codex**：之前的 GPT‑3 只在通用文本上训练，直接用于写代码时几乎没有成功。作者把 GitHub 上公开的数十亿行代码当作微调语料，让模型在代码的统计规律上再学习一次。结果是模型在代码生成任务上出现了质的飞跃。  
**HumanEval 基准 → 可测量的功能正确率**：过去缺少统一的、能自动评判代码对错的基准。本文构建了 HumanEval，配合隐藏单元测试，提供了一个“一键跑分”的评估方式，使得不同模型的比较变得透明。  
**大量采样策略 → 70% 通过率**：直觉上一次采样已经够了，但作者发现对同一个提示进行多次随机采样，然后挑选第一个通过所有测试的解，成功率大幅提升。用 100 次采样，正确率从 28.8% 跳到 70.2%。这是一种极其简单却意外有效的技巧。  
**系统性局限性分析 → 识别长链和变量绑定难点**：作者细致检查错误案例，发现模型在处理需要多步操作的长 docstring 以及把中间结果绑定到变量时表现不佳，为后续改进指明了方向。

### 方法详解
整体思路可以分为三步：**数据准备 → 模型微调 → 推理采样**。

1. **数据准备**  
   - 从 GitHub 上抓取公开仓库，过滤掉许可证不兼容或质量极低的文件，最终得到约 159 GB 的 Python、JavaScript、Go 等多语言代码。  
   - 对每段代码做标准化处理：去除注释、统一缩进、分割成 token 序列。这里的 token 与 GPT‑3 使用的 BPE（字节对编码）保持一致，保证模型能够直接接受代码。  

2. **模型微调**  
   - 基础模型是 GPT‑3 的 12‑层、175 B 参数版本（称为 “davinci”）。  
   - 训练目标仍是 **下一个 token 预测**：给定前面的代码片段，模型要预测下一个 token 的概率分布。  
   - 采用 **AdamW** 优化器，学习率在 1e‑5 左右，训练约 2 周。因为代码的语法结构比自然语言更严格，微调过程中模型会逐渐学会匹配括号、缩进和常见 API 调用。  

3. **推理采样**  
   - 输入为函数的 docstring（例如 “返回两个整数的最大公约数”），模型在 **temperature=0.8**、**top‑k=50** 的设置下开始生成代码。temperature 控制随机程度，top‑k 限制候选 token 的范围。  
   - 关键技巧是 **重复采样**：对同一 prompt 进行 N 次（本文实验 N=1、10、100）独立采样，每次得到完整的函数实现。随后把每个实现送入隐藏的单元测试，只要有一次通过，就算该 prompt 成功。  
   - 这种做法的背后逻辑是：模型的概率分布本身已经包含了多个潜在解，单次采样只能捕获其中一个，而多次采样相当于在解空间里做“盲目搜索”，大幅提升命中正确解的概率。  

最让人意外的地方是 **不需要任何后处理或搜索算法**，仅靠随机采样和测试过滤，就实现了接近 70% 的成功率，这在当时的代码生成领域是前所未有的。

### 实验与效果
- **评测数据**：HumanEval 包含 164 条 Python 小题，每题都有 3–5 条隐藏测试。  
- **基线对比**：  
  - GPT‑3（未微调）在单次采样下的通过率为 0%。  
  - GPT‑J（开源 6 B 参数模型）单次采样通过率为 11.4%。  
  - Codex（本文模型）单次采样通过率为 **28.8%**，采样 10 次提升到约 **55%**，采样 100 次则达到 **70.2%**。  
- **消融实验**：论文中对比了只使用通用文本微调、只使用代码微调以及两者混合的效果，发现仅代码微调的提升最显著，说明代码语料的质量是关键。  
- **局限性**：模型在处理 **长链操作**（docstring 描述多步计算）和 **变量绑定**（需要把中间结果存入变量再使用）时错误率明显上升。作者把这些归因于模型缺乏显式的“工作记忆”，只能靠隐式的上下文窗口来记忆信息。  

### 影响与延伸思考
这篇论文直接催生了 **GitHub Copilot**，让数百万开发者在日常编码时体验到 AI 辅助。随后出现的 **AlphaCode**、**CodeGen**、**StarCoder** 等模型，都在微调策略、采样技巧或评测基准上向 Codex 看齐。  
从安全角度看，论文也提醒业界代码生成模型可能产生 **漏洞代码** 或 **版权争议**，推动了对模型输出审计和许可证合规性的研究。  
如果想进一步深入，可以关注以下方向：  
- **检索增强生成**：把代码库检索结果作为模型的显式记忆，帮助解决长链推理。  
- **结构化约束解码**：在采样时加入语法或类型约束，降低语法错误率。  
- **更细粒度的评测**：除了功能正确性，还要评估代码的可读性、效率和安全性。  

### 一句话记住它
Codex 通过在公开代码上微调的 GPT，并配合大量随机采样，首次让机器在自动生成可运行 Python 程序上实现了突破性成功。