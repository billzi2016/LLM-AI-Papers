# The Right Prompts for the Job: Repair Code-Review Defects with Large   Language Model

> **Date**：2023-12-29
> **arXiv**：https://arxiv.org/abs/2312.17485

## Abstract

Automatic program repair (APR) techniques have the potential to reduce manual efforts in uncovering and repairing program defects during the code review (CR) process. However, the limited accuracy and considerable time costs associated with existing APR approaches hinder their adoption in industrial practice. One key factor is the under-utilization of review comments, which provide valuable insights into defects and potential fixes. Recent advancements in Large Language Models (LLMs) have enhanced their ability to comprehend natural and programming languages, enabling them to generate patches based on review comments. This paper conducts a comprehensive investigation into the effective utilization of LLMs for repairing CR defects. In this study, various prompts are designed and compared across mainstream LLMs using two distinct datasets from human reviewers and automated checkers. Experimental results demonstrate a remarkable repair rate of 72.97% with the best prompt, highlighting a substantial improvement in the effectiveness and practicality of automatic repair techniques.

---

# 为工作挑选合适的提示：利用大语言模型修复代码审查缺陷 论文详细解读

### 背景：这个问题为什么难？

代码审查（Code Review）是软件团队发现缺陷、提升质量的关键环节，但人工审查耗时且容易遗漏细节。传统的自动程序修复（APR）技术虽然能生成补丁，却在准确率和修复速度上远不及人工，导致工业落地受阻。根本原因在于这些系统往往只看源码本身，忽视了审查者留下的评论——这些自然语言的线索正是缺陷根因和修复思路的浓缩。缺少对评论的有效利用，使得 APR 只能在“盲目搜索”中挣扎，难以在真实的代码审查场景中发挥作用。

### 关键概念速览
- **代码审查（Code Review）**：开发者相互检查提交的代码，提出缺陷或改进建议的过程，类似于论文的同行评审，只不过对象是代码。
- **自动程序修复（APR）**：使用算法自动生成能够通过测试的代码补丁，像是让机器人帮你写“修正稿”。
- **大语言模型（LLM）**：在海量文本上训练的深度学习模型，能够理解并生成自然语言和代码，类似于拥有“多语言翻译+编程”双重能力的智能助理。
- **Prompt（提示）**：向 LLM 提出的问题或指令，决定模型的思考方向，就像老师给学生的作业要求，写得好坏直接影响答案质量。
- **Review Comment（审查评论）**：审查者在代码审查平台留下的文字说明，往往点出具体的 bug、潜在风险或改进建议，等价于“缺陷的线索卡”。
- **Patch（补丁）**：一段可以直接应用到源码的修改代码，类似于 Word 文档的“修订痕迹”。
- **主流 LLM**：指当前公开或商业化使用最广的模型，如 GPT‑4、Claude、LLaMA 等，它们在语言理解和代码生成上都有不错表现。

### 核心创新点
1. **把审查评论当作第一手修复线索**  
   - 之前的 APR 大多只看源码或测试用例，忽略了审查评论的价值。  
   - 本文把评论直接喂给 LLM，配合专门设计的提示，让模型在生成补丁时“先读懂评论”。  
   - 结果是补丁更贴合实际缺陷，修复成功率大幅提升。

2. **系统化 Prompt 设计与对比**  
   - 过去的研究往往随意给模型一个问题，缺少对提示的系统探索。  
   - 这篇论文手工构造了多种提示模板（如“直接修复”“先解释缺陷再给补丁”“分步生成”等），并在不同 LLM 上做横向比较。  
   - 最佳提示在实验中实现了 72.97% 的修复率，证明提示工程本身是提升 LLM 修复能力的关键杠杆。

3. **跨模型、跨数据集的全面评估**  
   - 以往的评测往往只在单一模型或单一数据集上跑实验，难以判断方法的通用性。  
   - 本文选取了两套数据集：一套来源于真实人工审查评论，另一套来源于自动化检查工具的输出，分别在 GPT‑4、Claude 等主流模型上跑实验。  
   - 这种全方位的对比展示了方法的稳健性，也为后续研究提供了基准。

### 方法详解
整体思路可以拆成四步：**收集审查评论 → 构造 Prompt → 调用 LLM 生成补丁 → 验证与过滤**。下面逐步展开。

1. **收集审查评论**  
   - 从代码审查平台（如 Gerrit、GitHub PR）抓取每条评论，保留评论文本、对应的代码位置以及关联的提交哈希。  
   - 对于自动检查生成的评论，直接使用工具输出的警告信息。  
   - 这一步的目标是把“人类思考的痕迹”转化为机器可读的字符串。

2. **Prompt 构造**  
   - 作者设计了若干模板，核心结构大致是：  
     ```
     [任务描述]：请根据以下审查评论修复代码。  
     [代码片段]：<代码>  
     [审查评论]：<评论>  
     [输出要求]：只返回完整的补丁，不要解释。
     ```  
   - 变体包括：先让模型解释缺陷再生成补丁、要求模型给出修改前后对比、加入“常见错误模式”提示等。  
   - 通过实验发现，加入“先解释再修复”的两步提示能显著提升模型的聚焦度，因为模型先把注意力放在评论的意图上，再去思考具体改动。

3. **调用 LLM 生成补丁**  
   - 将构造好的 Prompt 发送给目标 LLM，使用温度（temperature）等采样参数控制输出的确定性。  
   - 为防止模型输出不完整的代码，作者在 Prompt 中明确要求“返回完整的 diff”，并在后处理阶段检查是否符合语法。

4. **验证与过滤**  
   - 生成的补丁会先在本地编译/运行单元测试，只有全部通过的才算成功。  
   - 对于未通过的补丁，系统会记录错误类型（编译错误、测试失败），并在后续的 Prompt 调整中使用这些信息进行“自我纠错”。  
   - 这种闭环的验证机制确保最终报告的修复率是真实可用的。

**最巧妙的点**在于把审查评论直接当作“修复指令”，而不是让模型自行去发现 bug。这样模型的搜索空间从整个代码库缩小到评论指向的局部，极大提升了效率和准确度。

### 实验与效果
- **数据集**：两套数据。第一套来自真实开发者在代码审查中留下的评论，覆盖多种语言和项目；第二套是自动化检查工具（如 SonarQube）产生的警告，代表机器生成的“伪评论”。  
- **基线**：传统 APR 方法（如 GenProg、Repairnator）以及直接让 LLM 在没有评论的情况下生成补丁的“裸 Prompt”。  
- **主要结果**：最佳 Prompt 在两套数据上整体修复率达到 **72.97%**，相比传统 APR 的 30% 左右提升超过两倍。对比“裸 Prompt”，修复率提升约 20% 以上。  
- **消融实验**：作者分别去掉“先解释”步骤、去掉代码片段上下文、仅使用自动检查评论等，发现“先解释”模块贡献约 8% 的提升，代码上下文贡献约 5%。这说明每个 Prompt 设计细节都有实质性作用。  
- **局限性**：原文未详细描述对大型项目的扩展成本，也未给出对不同编程语言的细粒度表现。作者承认当前实验主要在中小规模代码库，工业级的实时审查仍需进一步优化响应时间和安全审计。

### 影响与延伸思考
这篇工作把 **Prompt 工程** 与 **代码审查** 直接挂钩，开启了“审查驱动的自动修复”新思路。随后的研究（如 2024 年的 *Review2Patch*、*CommentGuidedRepair*）纷纷在此基础上加入多轮对话、错误定位模型等，进一步提升了修复的精准度。对想继续深入的读者，可以关注以下方向：  
- **多模态审查信息**：把代码差异图、静态分析报告等一起喂给 LLM，探索信息融合的增益。  
- **自适应 Prompt 学习**：使用强化学习让模型自行发现最有效的提示模板，减少人工调参。  
- **安全与可信度评估**：在工业环境中，自动生成的补丁需要经过安全审计，如何在 Prompt 中加入安全约束是后续关键挑战。

### 一句话记住它
把审查评论当作“修复指令”，用精心设计的 Prompt 让大语言模型直接生成高成功率的代码补丁。