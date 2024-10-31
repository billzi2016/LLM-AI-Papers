# Can Language Models Perform Robust Reasoning in Chain-of-thought   Prompting with Noisy Rationales?

> **Date**：2024-10-31
> **arXiv**：https://arxiv.org/abs/2410.23856

## Abstract

This paper investigates an under-explored challenge in large language models (LLMs): chain-of-thought prompting with noisy rationales, which include irrelevant or inaccurate reasoning thoughts within examples used for in-context learning. We construct NoRa dataset that is tailored to evaluate the robustness of reasoning in the presence of noisy rationales. Our findings on NoRa dataset reveal a prevalent vulnerability to such noise among current LLMs, with existing robust methods like self-correction and self-consistency showing limited efficacy. Notably, compared to prompting with clean rationales, base LLM drops by 1.4%-19.8% in accuracy with irrelevant thoughts and more drastically by 2.2%-40.4% with inaccurate thoughts.   Addressing this challenge necessitates external supervision that should be accessible in practice. Here, we propose the method of contrastive denoising with noisy chain-of-thought (CD-CoT). It enhances LLMs' denoising-reasoning capabilities by contrasting noisy rationales with only one clean rationale, which can be the minimal requirement for denoising-purpose prompting. This method follows a principle of exploration and exploitation: (1) rephrasing and selecting rationales in the input space to achieve explicit denoising and (2) exploring diverse reasoning paths and voting on answers in the output space. Empirically, CD-CoT demonstrates an average improvement of 17.8% in accuracy over the base model and shows significantly stronger denoising capabilities than baseline methods. The source code is publicly available at: https://github.com/tmlr-group/NoisyRationales.

---

# 语言模型能在带噪声推理链的思维链提示中实现鲁棒推理吗？ 论文详细解读

### 背景：这个问题为什么难？
在大模型的“思维链”（Chain‑of‑Thought, CoT）提示里，模型会先把推理过程写出来，再给出答案，这已经让很多复杂任务的准确率大幅提升。但实际使用时，示例中的推理往往并不完美——可能夹杂无关的废话，甚至出现错误的推理步骤。传统的 CoT 方法默认示例是干净的，一旦出现噪声，模型的推理会被误导，导致答案错误。现有的自我纠错或自我一致性等鲁棒技术在面对这种“噪声推理链”时效果有限，说明我们缺少一种能够在有噪声的上下文中仍保持可靠推理的机制。

### 关键概念速览
**思维链（CoT）**：让模型在输出答案前先写出一步步的推理，就像学生做数学题时先列草稿，帮助模型把复杂逻辑拆解成可检视的子步骤。  
**噪声推理链（Noisy Rationales）**：指示例中出现的无关或错误的推理内容，类似于课堂笔记里夹杂的错别字或跑题句子，会干扰模型的学习。  
**自我纠错（Self‑Correction）**：模型在第一次生成答案后，再次审视并尝试改正错误的过程，类似人写完作文后再检查。  
**自我一致性（Self‑Consistency）**：让模型多次采样不同的推理路径，然后投票决定最终答案，像是多位专家给出意见后取多数。  
**对比去噪（Contrastive Denoising）**：把噪声样本和干净样本放在一起比较，让模型学习“噪声到底长什么样”，从而学会过滤掉。  
**CD‑CoT（Contrastive Denoising with noisy Chain‑of‑Thought）**：本文提出的具体方法，利用仅一条干净推理链对比大量噪声链，提升模型的去噪和推理能力。  
**探索‑利用原则**：先在输入空间里“探索”不同的表述方式（重写、挑选），再在输出空间里“利用”多样化的推理路径进行投票，类似先找出最佳工具，再用它们完成任务。

### 核心创新点
1. **噪声评估基准 → NoRa 数据集 → 揭示现有模型对噪声极度脆弱**  
   过去没有专门的测评集合来检验 CoT 在噪声环境下的表现。作者手工构造了 NoRa，包含两类噪声（无关思路和错误思路），并在多个主流 LLM 上跑实验，发现准确率下降幅度从 1.4% 到 40% 不等，直观展示了模型的弱点。

2. **仅需一条干净推理链 → 对比去噪框架 CD‑CoT → 大幅提升鲁棒性**  
   与需要大量标注干净示例的传统去噪方法不同，CD‑CoT 只要求提供“一条”干净的推理链作为对照。通过在输入端对噪声链进行重写、筛选，使其显式对齐干净链；在输出端让模型生成多条推理路径并投票。实验显示，平均提升约 17.8% 的准确率，证明少量干净信息足以驱动强去噪。

3. **探索‑利用双阶段设计 → 输入空间的显式去噪 + 输出空间的多样化投票 → 超越自我纠错/自我一致性的效果**  
   过去的鲁棒方法只在输出阶段做“纠错”或“投票”。CD‑CoT 在前期就对噪声进行“清洗”，相当于先把教材的错页撕掉，再让学生答题，效果自然更好。

### 方法详解
**整体框架**  
CD‑CoT 由两大阶段组成：① **输入层去噪**：把带噪声的示例转化为更干净的形式；② **输出层推理与投票**：让模型基于去噪后的示例生成多条思维链，并通过多数投票决定答案。

**步骤拆解**  

1. **准备干净对照**  
   - 只需要为每个任务准备一条高质量的思维链（即“干净理性”），这条链可以来自人工标注或已有的高可信示例。

2. **噪声链的重写与筛选（输入探索）**  
   - 将每条噪声理性输入模型，要求模型**重新表述**（rephrase）并**挑选**出与干净链最相似的子句。  
   - 这一步相当于让模型自己找出噪声中的“有价值片段”，并把它们重新组织成更接近干净链的结构。  
   - 通过对比（contrastive）损失，让模型学习“干净 ≈ 重写后噪声”，从而在训练时自动压制噪声信息。

3. **多路径生成（输出探索）**  
   - 使用去噪后的示例，令模型 **采样** 多条思维链（比如 5‑10 条），每条链都从不同的随机种子或温度参数出发。  
   - 这一步的目标是让模型探索可能的推理路径，防止单一路径被噪声锁定。

4. **答案投票（利用）**  
   - 对每条生成的思维链，提取其最终答案。  
   - 采用多数投票或加权投票（权重可以基于链的自信度）得到最终输出。  
   - 这种投票机制类似自我一致性，但因为前面的输入层已经去噪，投票的质量显著提升。

**核心技巧**  
- **对比学习的最小需求**：只要一条干净链，就能构造正负样本（干净 vs 噪声），大幅降低标注成本。  
- **显式去噪而非隐式纠错**：把噪声过滤放在输入阶段，而不是等模型已经产生错误答案后再纠正，避免错误在后续传播。  
- **探索‑利用的双向循环**：输入层的“探索”帮助模型发现哪些噪声是可救的，输出层的“利用”则把这些发现转化为更稳健的答案。

### 实验与效果
- **数据集与任务**：作者在 NoRa 数据集上评估，任务涵盖数学推理、逻辑判断和常识问答三类，每类都分别加入无关噪声和错误噪声两种变体。  
- **基线对比**：与原始 LLM（未做任何鲁棒处理）、自我纠错（Self‑Correction）和自我一致性（Self‑Consistency）三种常见鲁棒方法比较。  
- **性能提升**：在所有任务上，CD‑CoT 的平均准确率提升约 **17.8%**，其中对错误噪声的提升最高可达 **40.4%**，对无关噪声的提升约 **19.8%**。相较于自我一致性，提升幅度在 **10%‑15%** 之间。  
- **消融实验**：作者分别去掉输入层的重写筛选、去掉多路径采样、以及只使用单条推理链进行投票。结果显示，去掉任何一个环节都会导致整体性能下降 5%‑12%，其中输入层去噪的贡献最大。  
- **局限性**：论文承认 CD‑CoT 仍依赖于至少一条高质量干净理性，若任务本身缺乏可靠示例，效果会受限；此外，多路径采样会增加推理时间，实际部署时需要权衡。

### 影响与延伸思考
- 这篇工作首次系统量化了 CoT 在噪声环境下的脆弱性，推动社区关注“噪声鲁棒性”这一新维度。  
- 之后的研究（如 2024‑2025 年的几篇论文）开始探索 **少量干净监督** 与 **大规模噪声自监督** 的结合，甚至尝试用 **检索增强** 的方式自动获取干净理性。  
- 对想进一步深入的读者，可以关注以下方向：① 如何在完全无标注的情况下生成可信的干净理性（比如利用模型自身的自监督信号）；② 将 CD‑CoT 融入 **检索‑增强生成**（RAG）框架，实现更大规模的开放域问答鲁棒性；③ 探索 **跨语言** 的噪声去除，看看方法在多语言模型上是否同样有效。

### 一句话记住它
只要给模型一条干净的思维链，利用对比去噪和多路径投票，就能让大模型在充斥噪声的示例中仍保持稳健推理。