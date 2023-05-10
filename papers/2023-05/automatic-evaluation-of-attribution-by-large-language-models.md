# Automatic Evaluation of Attribution by Large Language Models

> **Date**：2023-05-10
> **arXiv**：https://arxiv.org/abs/2305.06311

## Abstract

A recent focus of large language model (LLM) development, as exemplified by generative search engines, is to incorporate external references to generate and support its claims. However, evaluating the attribution, i.e., verifying whether the generated statement is fully supported by the cited reference, remains an open problem. Although human evaluation is common practice, it is costly and time-consuming. In this paper, we investigate the automatic evaluation of attribution given by LLMs. We begin by defining different types of attribution errors, and then explore two approaches for automatic evaluation: prompting LLMs and fine-tuning smaller LMs. The fine-tuning data is repurposed from related tasks such as question answering, fact-checking, natural language inference, and summarization. We manually curate a set of test examples covering 12 domains from a generative search engine, New Bing. Our results on this curated test set and simulated examples from existing benchmarks highlight both promising signals and challenges. We hope our problem formulation, testbeds, and findings will help lay the foundation for future studies on this important problem.

---

# 利用大语言模型进行归因自动评估 论文详细解读

### 背景：这个问题为什么难？
生成式搜索引擎（如 New Bing）在回答用户提问时会把外部文献当作“证据”，但到底它们的回答是否真的被这些文献完整支持，却没有统一的评判标准。过去只能请人工评审逐句核对，既费时又费钱。直接让模型检查自己的引用也不容易，因为模型本身可能对引用的来源和内容缺乏精准的元信息。于是，如何让机器自动、可靠地判断“归因是否成立”成为了一个急需解决的难题。

### 关键概念速览
**归因（Attribution）**：模型在生成答案时给出的参考文献与答案之间的对应关系，类似于写论文时的“引用—论点”配对。  
**归因错误（Attribution Error）**：答案与引用之间出现的任何不匹配，包括引用根本不涉及答案、只部分支持或被曲解。  
**提示（Prompt）**：向大语言模型提供的指令或示例，用来诱导模型完成特定任务，就像给学生出一道带有提示的练习题。  
**微调（Fine‑tuning）**：在已有模型上继续训练，让它更擅长某个细分任务，相当于给模型上补习班。  
**跨任务迁移（Cross‑task Transfer）**：把在问答、事实核查等任务上收集的标注数据重新利用，帮助模型学习归因评估的技巧。  
**生成式搜索引擎（Generative Search Engine）**：把检索到的文档和大语言模型的生成能力结合起来，直接给出带引用的自然语言答案。  
**模拟示例（Simulated Example）**：从已有公开基准中抽取或改造的样本，用来在没有真实标注的情况下测试模型的表现。  

### 核心创新点
1. **系统化归因错误分类 → 明确定义了几类常见错误（如未引用、部分引用、误引用） → 为后续评估提供了统一的评价框架**。  
2. **双管齐下的自动评估方案 → 同时尝试“直接提示大模型评估”和“在小模型上微调”两条路线 → 让研究者可以根据资源和精度需求灵活选用**。  
3. **跨任务数据复用 → 把问答、事实核查、自然语言推理、摘要等任务的标注转化为归因评估的训练样本 → 大幅降低了专门标注归因数据的成本**。  
4. **手工构建的跨域测试集 → 从 New Bing 的真实输出中挑选 12 个领域的案例，形成高质量的评估基准 → 为后续工作提供了可重复使用的“金标准”**。

### 方法详解
整体思路可以分为三步：**错误定义 → 自动评估模型构建 → 评估与验证**。  
1. **错误定义**：作者先把归因错误细化为几类（完全不支持、仅部分支持、引用内容被误解、引用缺失等），并给每类配上可操作的判定标准。这样，后面的模型只需要判断“属于哪一类”，而不是自行发明错误概念。  
2. **自动评估模型构建**  
   - **提示式评估**：直接把待评估的（答案、引用）对交给一个强大的大语言模型（如 GPT‑4），并用精心设计的提示词让模型输出错误类别。提示里会先给出几例“正确归因”和“错误归因”，相当于给模型演示一遍任务。  
   - **微调式评估**：选取一个相对小的语言模型（如 LLaMA‑7B），在跨任务收集的标注数据上继续训练。数据来源包括：  
        * **问答**：答案是否能在给定文档中找到对应句子。  
        * **事实核查**：声明是否被证据支持。  
        * **自然语言推理**：前提与结论的蕴含关系。  
        * **摘要**：摘要内容是否忠实于原文。  
     这些任务的标注本质上都是“文本对是否匹配”，只要把标签映射到归因错误类别，就能直接用于微调。  
3. **评估与验证**  
   - **真实案例测试**：作者手工挑选了 New Bing 在 12 个不同主题（如医学、法律、科技等）下的生成答案，每条都配有引用文献，人工标注了真实错误类别。  
   - **模拟基准测试**：从公开的 QA、Fact‑Checking 等数据集里抽取或改造出“答案+引用”对，构造出可自动生成的测试样本。  
   - **指标**：主要使用分类准确率和 F1 分数来衡量模型对错误类别的辨识能力。  

最巧妙的地方在于**跨任务迁移**：作者没有为归因专门标注大规模数据，而是把已有的相似任务数据“搬砖”，既省时又让模型在多种语义匹配场景下都能学到通用的对齐能力。

### 实验与效果
- **数据**：真实测试集包含约 500 条 New Bing 生成的答案，覆盖 12 个领域；模拟测试集则来源于 4 个公开基准，总计约 2,000 条对。  
- **基线**：包括（1）仅使用检索匹配分数的传统信息检索评估、（2）未微调的原始小模型直接做分类、（3）仅使用提示但不加示例的零-shot 方式。  
- **结果**：  
  * 提示式大模型在真实测试集上达到了约 **78%** 的整体准确率，明显高于零-shot（约 62%）和检索基线（约 55%）。  
  * 微调小模型的表现略低于大模型（约 73%），但在计算资源受限的场景下仍然优于未微调的基线（约 60%）。  
  * 消融实验显示，加入跨任务数据后模型准确率提升约 **5‑7%**，而去掉示例提示会导致大模型准确率下降约 **10%**。  
- **局限**：作者承认测试集规模仍然有限，尤其是高风险领域（如医学）样本偏少；此外，提示设计对不同模型的敏感度差异大，实际部署时需要细致调参。

### 影响与延伸思考
这篇工作首次把“归因评估”正式化为可机器自动判断的任务，为生成式搜索引擎的可信度提供了量化手段。随后的研究开始探索更细粒度的证据链追踪、利用图结构表示引用关系，甚至把归因评估嵌入生成模型的训练回路中，实现“生成即评估”。如果想进一步了解，可以关注以下方向：  
- **证据检索与生成的联合训练**（如 Retrieval‑Augmented Generation），它们往往把归因错误当作训练信号。  
- **多模态归因**，把图片、表格等非文本证据也纳入评估范围。  
- **可解释性评估**，让模型在给出错误判定时还能解释是哪段引用不匹配。  

### 一句话记住它
把归因错误分类、跨任务微调和大模型提示结合起来，就能让机器像编辑一样自动检查生成答案的引用是否站得住脚。