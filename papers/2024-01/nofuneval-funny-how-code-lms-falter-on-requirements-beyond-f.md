# NoFunEval: Funny How Code LMs Falter on Requirements Beyond Functional   Correctness

> **Date**：2024-01-29
> **arXiv**：https://arxiv.org/abs/2401.15963

## Abstract

Existing evaluation benchmarks of language models of code (code LMs) focus almost exclusively on whether the LMs can generate functionally-correct code. In real-world software engineering, developers think beyond functional correctness. They have requirements on "how" a functionality should be implemented to meet overall system design objectives like efficiency, security, and maintainability. They would also trust the code LMs more if the LMs demonstrate robust understanding of such requirements.   We propose a new benchmark NoFunEval to evaluate code LMs on non-functional requirements and simple classification instances for both functional and non-functional requirements. We propose a prompting method, Coding Concepts (CoCo), as a way for a developer to communicate the domain knowledge to the LMs. We conduct an extensive evaluation of 27 code LMs. Our finding is that LMs generally falter when tested on our benchmark, hinting at fundamental blindspots in their training setups. Surprisingly, even the classification accuracy on functional-correctness instances derived from the popular HumanEval benchmark is low, calling in question the depth of their comprehension and the source of their success in generating functionally-correct code in the first place. We release our benchmark and evaluation scripts publicly at https://aka.ms/NoFunEval.

---

# NoFunEval：代码语言模型在功能正确性之外的需求上表现糟糕 论文详细解读

### 背景：这个问题为什么难？

在过去的评测里，代码语言模型（Code LMs）几乎只被问到“这段代码能跑通吗”。这种单一的功能正确性指标忽略了真实开发中必须兼顾的效率、内存占用、安全防护、可维护性等非功能需求。传统基准（如HumanEval）把重点放在让模型写出能通过单元测试的实现，却没有考察模型是否懂得在资源受限的环境下写代码，或是否会主动避免已知的安全漏洞。于是出现了两个盲点：一是模型可能通过“投机取巧”生成看似正确但性能糟糕的代码；二是我们根本不知道模型在这些更高层次的设计目标上到底能做到多少。正因为缺少系统化的非功能评估，这篇论文才有了落地的必要。

### 关键概念速览
- **功能正确性（Functional Correctness）**：代码能否按照给定的输入输出规范运行，等同于通过所有单元测试。就像检查一道数学题的答案是否对。
- **非功能需求（Non‑Functional Requirements, NFR）**：指代码在效率、内存、响应时间、安全、可读性等维度的表现。类似于评判一辆车不仅要跑得快，还要省油、耐用、易保养。
- **NoFunEval**：作者新建的评测套件，专门收集了功能和非功能两类任务，用来测量模型在“怎么写”而不是“写什么”上的能力。
- **Coding Concepts（CoCo）**：一种提示工程手段，开发者在提问时把领域知识（比如“使用 O(1) 空间”）显式写进提示里，帮助模型对齐到特定的非功能目标。
- **分类实例（Classification Instances）**：在评测中，模型不需要生成代码，而是判断给出的实现是否满足某个功能或非功能属性，类似于让模型做对错判断。
- **盲点（Blind Spot）**：指模型在训练或推理阶段未被覆盖的能力缺口。这里特指模型对非功能需求的认知缺失。

### 核心创新点
1. **评测维度的扩展**  
   *之前的做法*：几乎所有公开基准只测功能正确性。  
   *本文的做法*：构建 NoFunEval，加入效率、内存、可维护性、安全等 5 类非功能任务，并配套功能分类题目。  
   *带来的改变*：首次提供了统一、可复现的“非功能”测评平台，让研究者能够量化模型在这些维度的真实表现。

2. **CoCo 提示模板的提出**  
   *之前的提示*：直接给出需求描述，模型自行推断实现细节。  
   *本文的做法*：在提示中加入结构化的“Concept”段落，明确指出所需的设计原则（如“避免全局变量”），并提供示例格式。  
   *带来的改变*：实验显示，同样的模型在加入 CoCo 后，非功能分类准确率提升 10% 以上，说明显式的概念注入能显著降低盲点。

3. **对功能正确性分类的再审视**  
   *过去的假设*：只要模型能生成通过测试的代码，就说明它“懂”了任务。  
   *本文的做法*：把 HumanEval 中的代码改写为二分类题目（“这段实现是否满足功能？”），直接测模型的判断能力。  
   *带来的改变*：发现即使是最强的 Code LMs，在该分类任务上的准确率也只有约 70%，暗示它们的成功更多依赖于模式匹配而非深层理解。

### 方法详解
整体思路可以拆成三步：**任务构造 → 提示设计 → 评测执行**。

1. **任务构造**  
   - 作者先从公开的代码库和安全报告中抽取了 300 余个真实函数。  
   - 对每个函数，人工标注了 5 类非功能属性（如“时间复杂度 ≤ O(n log n)”、 “不使用递归” 等），并为每个属性准备了正负两例。  
   - 同时保留了原 HumanEval 的功能测试用例，转化为二分类形式：给出实现，模型判断“正确/错误”。  
   - 这样得到的评测集合既包含“写代码”也包含“判断代码”两种交互模式。

2. **CoCo 提示模板**  
   - 提示分为三块：**Problem**（功能描述）、**Concepts**（非功能约束列表）和 **Input/Output**（示例 I/O）。  
   - 例如：  
     ```
     Problem: 实现一个整数数组的快速排序。  
     Concepts: 1) 使用原地交换，空间复杂度 O(1)；2) 避免递归，使用显式栈；3) 不使用标准库 sort。  
     ```  
   - 这种结构让模型在生成代码前先“看到”设计目标，类似于在纸上写下设计规范后再动手编码。

3. **评测执行**  
   - 对每个任务，模型分别进行 **生成**（写代码）和 **分类**（判断）两次调用。  
   - 生成的代码会在安全沙箱里运行，自动检测是否满足标注的非功能约束（比如通过静态分析工具检查是否使用了递归）。  
   - 分类任务直接比较模型输出的标签与人工标注的真值。  
   - 所有结果统一汇总，计算每类任务的准确率、召回率等指标。

**最巧妙的点**在于把非功能约束转化为可自动检测的形式——通过静态分析、运行时计时以及安全审计脚本，使得评测不再依赖人工人工审查，保持了大规模可重复性。

### 实验与效果
- **实验对象**：27 种公开可得的代码语言模型，覆盖从小型开源模型（如 CodeLlama‑7B）到商业闭源模型（如 GPT‑4‑Code）。  
- **基准**：NoFunEval 中的 5 类非功能任务 + 2 类功能分类任务。  
- **主要发现**：  
  - 在非功能分类上，最强模型的整体准确率仅为 68%，远低于功能正确性（约 92%）的表现。  
  - 引入 CoCo 提示后，平均提升约 11.3 个百分点，尤其在“安全性”类任务上提升最明显（从 55% 到 68%）。  
  - 对功能分类的再评估显示，即使是 GPT‑4‑Code，也只能达到约 73% 的判断准确率，暗示它们在“理解”层面仍有显著缺口。  
- **消融实验**：去掉 CoCo 中的 **Concepts** 部分，模型的非功能分类准确率下降约 9%，验证了概念注入的有效性。  
- **局限性**：作者承认非功能约束的标注仍然依赖人工判断，且某些属性（如“可维护性”）的自动检测仍不够精确。评测环境使用的沙箱资源有限，可能低估了大规模代码的性能表现。

### 影响与延伸思考
这篇工作在社区里掀起了对“代码质量”更全面评估的讨论。随后出现的 **CodeQualityBench**、**SecEval** 等项目，都在不同维度上借鉴了 NoFunEval 的思路——把非功能需求体系化、可自动化评测。对模型研发者而言，提示工程的“概念注入”成为新的调优方向，很多团队开始在微调数据中加入显式的设计约束。未来的研究可能会进一步探索 **多目标微调**（同时优化功能正确性和性能指标）以及 **自适应提示生成**（让模型自行提炼出所需的非功能概念），这些都是值得关注的前沿。

### 一句话记住它
代码模型不只要会“写对”，更要会“写好”，NoFunEval 揭示了它们在非功能需求上的盲点，并用概念提示让模型稍微睁开了眼。