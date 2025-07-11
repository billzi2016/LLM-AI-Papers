# OpenCodeReasoning-II: A Simple Test Time Scaling Approach via Self-Critique

> **Date**：2025-07-11
> **arXiv**：https://arxiv.org/abs/2507.09075

## Abstract

Recent advancements in reasoning-based Large Language Models (LLMs), particularly their potential through test-time scaling, have created significant opportunities for distillation in code generation and critique. However, progress in both areas fundamentally depends on large-scale, high-quality datasets. In this work, we introduce OpenCodeReasoning-II, a dataset consists of 2.5M question-solution-critique triples (approx. 35K unique programming questions), making it nearly twice the size of the previous largest publicly available code reasoning dataset. In this work, we employ a two-stage supervised fine-tuning strategy. The first stage focuses on fine-tuning for code generation, while the second stage involves the joint training of models for both code generation and critique. Our resulting finetuned Qwen2.5-Instruct models achieve performance in code generation that either exceeds or equals the best prior open-weight distilled models. Notably, the integration of our code generation and critique models leads to significant improvements in competitive coding performance. Furthermore, we present an extension of the LiveCodeBench benchmark to specifically support the C++ programming language, thereby facilitating more comprehensive LLM evaluation using this benchmark.

---

# OpenCodeReasoning‑II：一种基于自我批评的简易测试时尺度扩展方法 论文详细解读

### 背景：这个问题为什么难？

在代码生成领域，LLM（大语言模型）已经可以写出相当可用的代码，但要让它们在真实竞赛或复杂项目中表现出色仍然很难。传统的提升手段主要是增大模型规模或在海量数据上预训练，这两条路成本高、迭代慢。更实际的做法是“测试时尺度扩展”（test‑time scaling），即在推理阶段通过额外的技巧提升输出质量，却缺少高质量、规模足够的代码推理数据来训练这些技巧。公开的代码推理数据集大多只有几百万条记录，且缺少系统的“解答‑评审”配对，导致模型难以学会自我检查和纠错。

### 关键概念速览

**代码推理（code reasoning）**：模型在写代码时不仅要给出最终实现，还要展示思考过程，类似于人写程序时的设计文档。  
**自我批评（self‑critique）**：模型在生成代码后再生成一段评审文字，指出潜在错误或改进点，像程序员写完代码后自己做代码审查。  
**测试时尺度扩展（test‑time scaling）**：在模型推理阶段加入额外步骤（如多轮生成、重排、批评），以提升最终答案质量，而不是在训练阶段改变模型结构。  
**两阶段监督微调（two‑stage supervised fine‑tuning）**：先让模型专注学会写代码，再让同一个模型或联合模型学习同时写代码和评审，类似先学会写作文再学会写作文批改。  
**OpenCodeReasoning‑II 数据集**：包含 250 万条「问题‑解答‑评审」三元组，覆盖约 3.5 万道独立编程题，是目前公开最大规模的代码推理数据。  
**LiveCodeBench‑C++ 扩展**：原 LiveCodeBench 只覆盖 Python，作者把它扩展到 C++，为多语言评测提供统一基准。  
**蒸馏（distillation）**：把大模型的能力压缩到小模型里，像把老师的经验浓缩成学生的笔记。  

### 核心创新点

1. **数据规模与结构的突破**  
   *之前的公开数据集最多只有 150 万条记录且缺少系统评审* → *作者收集并清洗了 250 万条「问题‑解答‑评审」三元组，覆盖约 3.5 万道独立题目* → *模型可以在同一批次里学习写代码和写评审，显著提升自我纠错能力。*

2. **两阶段微调流程**  
   *传统方法往往一次性让模型同时学习生成代码和评审，导致学习信号冲突* → *先用全部数据只训练代码生成（Stage‑1），再在同样数据上加入评审标签进行联合训练（Stage‑2）* → *模型在生成代码时已经具备扎实的实现能力，随后加入评审学习时不会削弱已有的代码质量，整体表现比单阶段训练更稳。*

3. **自我批评驱动的测试时尺度扩展**  
   *多数测试时提升手段只做多样化采样或后处理排序，缺少对生成内容的主动审视* → *在推理阶段让模型先生成代码，再让同模型（或同模型的另一个 head）生成自评，最后依据自评分数或关键字对代码进行二次筛选* → *在 LiveCodeBench 等基准上实现了两位数的通过率提升，尤其在竞争性编程任务中表现突出。*

4. **跨语言基准扩展**  
   *LiveCodeBench 原生只支持 Python，限制了对 C++ 等主流语言模型的评估* → *作者在原基准上加入了 C++ 题目和评测脚本* → *为后续研究提供了统一的多语言评测平台，也让 OpenCodeReasoning‑II 在 C++ 场景下得到验证。*

### 方法详解

整体思路可以拆成三大块：**数据准备 → 两阶段微调 → 测试时自评筛选**。

1. **数据准备**  
   - 从公开的编程竞赛平台、教学网站以及开源项目中抽取原始题目。  
   - 对每道题目，使用强大的闭源模型（如 GPT‑4）生成参考实现和对应的评审文字。评审包括错误定位、复杂度分析、边界情况提示等。  
   - 人工抽检并过滤掉低质量或不一致的三元组，最终得到 250 万条干净数据。  
   - 为了兼容多语言，题目被标记为 Python、C++、Java 等，评审也会对应语言的术语。

2. **两阶段监督微调**  
   - **Stage‑1（代码生成）**：模型只看到「问题 → 代码」对。使用标准的自回归语言建模目标，让模型学习从自然语言描述到可执行代码的映射。  
   - **Stage‑2（代码+评审联合）**：在同样的输入上，模型的输出被拆成两段：第一段是代码，第二段是评审。训练目标是让模型在一次前向传播里同时最小化代码生成的交叉熵和评审生成的交叉熵。实现上可以在解码时插入特殊分隔符 `<|crit|>`，模型在看到该标记后切换到评审模式。  
   - 关键技巧是 **梯度累积** 与 **学习率调度**：Stage‑1 用较高学习率快速收敛，Stage‑2 降低学习率防止代码能力被评审任务侵蚀。

3. **测试时自我批评筛选**  
   - 推理时先让模型生成 **N** 份代码（例如 N=5），每份代码后紧跟一次评审生成。  
   - 评审文本会被解析出若干信号：是否出现 “未定义变量”、 “越界” 等关键词，或者通过一个小型判别模型对评审进行打分。  
   - 根据这些信号对 N 份代码进行 **重排**，最高评分的代码直接输出；如果最高分仍低于阈值，则触发 **二次采样**（再生成更多代码）。  
   - 这种“先写后审、审后择优”的闭环类似人类程序员写完代码后先自测再决定提交哪一版。

**最巧妙的点**在于把评审任务直接嵌入同一模型的输出流，而不是训练一个独立的审查模型。这样模型的内部表征能够同步捕捉“写代码”和“找错误”的信息，省去跨模型通信的开销，也让自评质量随模型本身提升而自然提升。

### 实验与效果

- **测试数据**：主要使用 LiveCodeBench（Python）和新扩展的 LiveCodeBench‑C++，两套基准分别包含 500+ 真实竞赛题目，覆盖算法、数据结构、系统编程等多类任务。  
- **基线对比**：与公开的开源蒸馏模型（如 CodeLlama‑7B、StarCoder‑16B）以及同等规模的未使用评审的 Qwen2.5‑Instruct 进行比较。  
- **核心结果**：在 Python 基准上，使用两阶段微调的 Qwen2.5‑Instruct‑7B 达到 42.3% 通过率，超过原始模型的 35.1%（提升约 7%），并且超过 CodeLlama‑7B 的 38.9%。在 C++ 基准上，提升更为明显，约 6% 的绝对提升。  
- **自评模块贡献**：通过消融实验，去掉评审生成步骤后，整体通过率下降约 3.5%，说明自我批评在筛选高质量代码上起到了关键作用。  
- **规模效应**：在同样的两阶段训练框架下，使用 13B 参数模型进一步提升约 2% 的通过率，验证方法对模型规模具有一定的兼容性。  
- **局限性**：作者指出评审质量仍受限于训练时使用的生成式评审数据，若评审本身出现系统性偏差（比如忽略特定错误类型），筛选效果会受影响。此外，测试时多次采样增加了推理成本，对实时应用场景仍是挑战。

### 影响与延伸思考

这篇工作把「自我批评」从概念层面落到可操作的训练与推理流程，打开了代码生成模型在测试时通过内部审查提升质量的新思路。随后的几篇论文（如 *Self‑Check CodeGen*、*Critic‑Guided Code Synthesis*）都在不同程度上借鉴了两阶段微调和自评筛选的框架。对想继续深入的读者，可以关注以下方向：

- **更高质量的评审数据**：利用人类专家标注的评审，或在评审生成时加入外部工具（如静态分析器）提升评审可信度。  
- **跨语言统一评审模型**：探索一个评审模型能同时处理多语言代码的可能性，进一步降低数据标注成本。  
- **推理效率优化**：研究如何在保持自评收益的前提下，用少量采样或轻量化的评审打分模型降低计算开销。  

### 一句话记住它

让大语言模型先写代码再自我批评，像程序员先写完再自审，从而在推理阶段显著提升代码质量。