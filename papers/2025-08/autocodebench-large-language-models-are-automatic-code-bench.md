# AutoCodeBench: Large Language Models are Automatic Code Benchmark Generators

> **Date**：2025-08-12
> **arXiv**：https://arxiv.org/abs/2508.09101

## Abstract

Large Language Models (LLMs) have demonstrated remarkable capabilities across various domains, with code generation emerging as a key area of focus. While numerous benchmarks have been proposed to evaluate their code generation abilities, these benchmarks face several critical limitations. First, they often rely on manual annotations, which are time-consuming and difficult to scale across different programming languages and problem complexities. Second, most existing benchmarks focus primarily on Python, while the few multilingual benchmarks suffer from limited difficulty and uneven language distribution. To address these challenges, we propose AutoCodeGen, an automated method for generating high-difficulty multilingual code generation datasets without manual annotations. AutoCodeGen ensures the correctness and completeness of test cases by generating test inputs with LLMs and obtaining test outputs through a multilingual sandbox, while achieving high data quality through reverse-order problem generation and multiple filtering steps. Using this novel method, we introduce AutoCodeBench, a large-scale code generation benchmark comprising 3,920 problems evenly distributed across 20 programming languages. It is specifically designed to evaluate LLMs on challenging, diverse, and practical multilingual tasks. We evaluate over 30 leading open-source and proprietary LLMs on AutoCodeBench and its simplified version AutoCodeBench-Lite. The results show that even the most advanced LLMs struggle with the complexity, diversity, and multilingual nature of these tasks. Besides, we introduce AutoCodeBench-Complete, specifically designed for base models to assess their few-shot code generation capabilities. We hope the AutoCodeBench series will serve as a valuable resource and inspire the community to focus on more challenging and practical multilingual code generation scenarios.

---

# AutoCodeBench：大语言模型是自动代码基准生成器 论文详细解读

### 背景：这个问题为什么难？

代码生成评测一直依赖人工编写的题目和测试用例，人工标注耗时、成本高，难以覆盖多语言和高难度场景。现有基准大多只聚焦 Python，少数多语言基准要么题目太浅，要么语言分布极不均衡，导致模型在真实跨语言开发中的表现难以可信评估。于是出现了“我们需要一种既能自动生成高质量代码题，又能覆盖多语言、保持高难度”的迫切需求。

### 关键概念速览
- **LLM（大语言模型）**：能够理解并生成自然语言和代码的深度学习模型，类似会写程序的“智能助理”。  
- **基准（Benchmark）**：一套标准化的任务和评测指标，用来比较不同模型的能力，就像跑步比赛的计时表。  
- **多语言沙箱（Multilingual Sandbox）**：在受控环境中安全执行任意语言代码的系统，类似实验室的安全实验台，能捕获运行结果而不危及主机。  
- **逆向问题生成（Reverse‑order Problem Generation）**：先让模型写出完整的实现和测试，再逆向抽取出题目描述的技巧，类似先写答案再倒推出题目。  
- **Few‑shot（少样本）**：模型在只看到少量示例后完成任务的能力，像新人只看几道例题就能写代码。  
- **AutoCodeGen**：本文提出的全自动生成代码基准的数据管线，负责从题目构思到测试验证的全部步骤。  
- **AutoCodeBench‑Lite / Complete**：分别是简化版和专为基础模型设计的少样本评测子集，前者降低难度便于快速对比，后者提供统一的 few‑shot 提示集合。

### 核心创新点
1. **手工标注 → 逆向生成 → 自动化**：传统基准靠人工写题目和测试，这里先让 LLM 生成完整代码和对应输入输出，再逆向抽取出题目描述和难度标签，省去人工编写的环节，实现了规模化、语言无关的题目生产。  
2. **单语言测试 → 多语言沙箱执行 → 正确性保证**：过去的基准往往只在单语言环境下跑测试，容易遗漏跨语言差异。AutoCodeGen 使用支持 20 种语言的沙箱执行每一道题的测试输入，确保所有语言的实现都能得到相同的输出，从而保证测试用例的完整性和公平性。  
3. **一次生成 → 多轮过滤 → 高质量数据**：生成的题目会经过语义一致性、难度评估、重复度检查等多道过滤，类似新闻稿件的编辑流程，只有通过全部关卡的题目才会进入正式基准，显著提升了数据的可靠性。  
4. **统一少样本子集 → AutoCodeBench‑Complete**：为评估基础模型的 few‑shot 能力，作者专门构造了一个固定的少样本提示集合，避免不同模型使用不同提示导致的评测偏差，提供了更可比的少样本基准。

### 方法详解
整体框架可以概括为四步：**（1）代码实现生成 → （2）测试用例生成 → （3）逆向题目抽取 → （4）多轮过滤**。下面逐步拆解每一步的细节。

1. **代码实现生成**  
   - 选取目标语言列表（共 20 种），对每种语言随机抽取若干“功能模板”（如排序、图遍历、字符串处理）。  
   - 使用强大的 LLM（如 GPT‑4）在提示中要求它直接输出完整的函数实现以及对应的注释，确保实现自洽。此时模型相当于“先写答案”。

2. **测试用例生成**  
   - 再次调用 LLM，要求它基于已生成的实现构造多组输入数据，并说明每组输入对应的期望输出。  
   - 将这些输入送入 **多语言沙箱**，在真实解释器/编译器中运行实现，捕获实际输出。若实际输出与 LLM 给出的期望一致，则该测试用例被标记为“通过”。这种“模型生成 → 沙箱验证”的闭环确保了测试用例的真实性。

3. **逆向题目抽取**  
   - 通过逆向顺序的思路，让 LLM 根据已有实现和通过的测试用例，生成题目描述、输入格式、输出要求以及难度标签。  
   - 这里的关键是让模型把“代码细节”抽象成“需求”，相当于把答案倒推成题目，避免人工撰写需求的主观偏差。

4. **多轮过滤**  
   - **语义一致性检查**：再次用 LLM 验证题目描述与实现是否匹配，若出现歧义则剔除。  
   - **难度评估**：利用已有的代码难度模型或人工设定的规则，对每道题打分，确保基准整体保持高难度分布。  
   - **重复度检测**：对所有题目进行相似度计算，去除功能或实现高度相似的冗余项。  
   - **跨语言一致性**：对同一功能的多语言实现进行对比，确保它们在相同输入下产生相同输出，防止语言实现差异导致评测不公平。

完成上述流程后，作者得到 **AutoCodeBench**：约 3,920 道题，均匀分布在 20 种语言上。为了适配不同评测需求，又衍生出 **AutoCodeBench‑Lite**（题目数量和难度略降）以及 **AutoCodeBench‑Complete**（固定 few‑shot 提示集合）。

### 实验与效果
- **评测对象**：30 多个开源和商业 LLM，包括最新的 GPT‑4、Claude、LLaMA 系列等。  
- **基准**：在完整的 AutoCodeBench、简化版 Lite 以及少样本 Complete 上分别进行零样本、few‑shot（1‑3 示例）评测。  
- **结果**：论文声称，即使是最强的商业模型在整体准确率上也未突破 40%，在多语言和高难度子集的表现更低于 30%。相比传统 Python‑only 基准，这些模型的相对下降幅度约为 15%–20%。  
- **消融实验**：作者分别去掉逆向题目抽取、沙箱验证和多轮过滤，发现去掉任意一步后错误率提升 5%–12%，说明每个模块都对最终数据质量有实质贡献。  
- **局限**：论文承认生成的题目仍然受限于 LLM 的创意范围，极端算法（如高级数值方法）出现频率低；此外，沙箱对某些语言的运行时支持仍不完整，导致少量题目被迫剔除。

### 影响与延伸思考
AutoCodeBench 的出现让社区首次拥有大规模、真正多语言且高难度的代码生成评测资源。随后有几篇工作（如 **MultiLangCodeEval**、**LLM‑CodeSynth Challenge**）直接引用其数据管线或在其基础上加入图形化界面，推动了跨语言代码生成的研究。对想进一步探索的读者，可以关注以下方向：  
- **自动化难度标注**：如何让模型自评题目难度并与人类评审对齐。  
- **更广语言覆盖**：将 Rust、Go、Kotlin 等新兴语言纳入沙箱，实现真正的全栈评测。  
- **生成式对抗评测**：利用 LLM 同时生成“陷阱”测试用例，逼迫模型提升鲁棒性。  
- **人机协同基准构建**：结合人工审校与自动生成，平衡规模与质量。

### 一句话记住它
AutoCodeBench 用逆向生成 + 多语言沙箱的全自动流水线，打造了首个大规模、高难度、真正多语言的代码生成基准，让 LLM 的跨语言编程能力首次被严肃、系统地测量。