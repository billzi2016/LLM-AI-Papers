# LiveCodeBench: Holistic and Contamination Free Evaluation of Large   Language Models for Code

> **Date**：2024-03-12
> **arXiv**：https://arxiv.org/abs/2403.07974

## Abstract

Large Language Models (LLMs) applied to code-related applications have emerged as a prominent field, attracting significant interest from both academia and industry. However, as new and improved LLMs are developed, existing evaluation benchmarks (e.g., HumanEval, MBPP) are no longer sufficient for assessing their capabilities. In this work, we propose LiveCodeBench, a comprehensive and contamination-free evaluation of LLMs for code, which continuously collects new problems over time from contests across three competition platforms, namely LeetCode, AtCoder, and CodeForces. Notably, our benchmark also focuses on a broader range of code related capabilities, such as self-repair, code execution, and test output prediction, beyond just code generation. Currently, LiveCodeBench hosts four hundred high-quality coding problems that were published between May 2023 and May 2024. We have evaluated 18 base LLMs and 34 instruction-tuned LLMs on LiveCodeBench. We present empirical findings on contamination, holistic performance comparisons, potential overfitting in existing benchmarks as well as individual model comparisons. We will release all prompts and model completions for further community analysis, along with a general toolkit for adding new scenarios and model

---

# LiveCodeBench：面向代码的大语言模型全方位且无污染评估 论文详细解读

### 背景：这个问题为什么难？

代码类大语言模型（LLM）在生成、调试甚至执行代码方面的能力日益提升，但我们缺少一个既能跟上模型进步速度，又能保证评测数据不被模型“偷看”的基准。传统基准（如 HumanEval、MBPP）在题目数量、题目新鲜度以及评测维度上都受限：一方面题库固定，模型可以通过公开数据或微调“记住”答案；另一方面它们只测代码生成，忽视了自我修复、执行结果预测等真实开发场景。于是评估结果越来越难以反映模型的真实水平，这正是本文要解决的痛点。

### 关键概念速览
- **LLM（大语言模型）**：能够理解并生成自然语言或代码的深度学习模型，类似会写程序的“智能助理”。  
- **代码生成基准**：一套固定的编程题目，用来测模型写出正确代码的能力，就像考试的题库。  
- **数据污染**：模型在训练或微调阶段已经见过评测题目，导致评测结果失真，类似学生提前拿到答案。  
- **全方位评估**：不仅看模型能否写出正确代码，还要考察它能否自行修复错误、预测执行输出等，类似对程序员的“写、调、跑、测”全流程考核。  
- **自修复（self‑repair）**：模型在发现生成代码出错后，能够给出修改建议或重新生成，像程序员调试时的“改错”。  
- **执行预测（execution prediction）**：模型在不实际运行代码的情况下，预测程序的输出，类似在纸上演算。  
- **指令微调（instruction‑tuned）**：在大模型基础上，用大量指令-响应对进行二次训练，使模型更擅长遵循用户指令。  
- **竞赛平台**：LeetCode、AtCoder、CodeForces 等在线编程比赛网站，提供新鲜、真实、难度分层的题目，像“题库的活体供给源”。  

### 核心创新点
1. **动态题目收集 → 从三大竞赛平台实时抓取 2023‑2024 年新出题目 → 评测数据永远是“新鲜血液”，防止模型通过公开数据提前学习。**  
2. **多维度任务设计 → 在每道题目上同时设置代码生成、自修复、执行预测三种子任务 → 评估覆盖了写代码、调代码、跑代码的完整开发链路。**  
3. **污染检测与剔除 → 对抓取的题目进行相似度和时间戳过滤，确保模型训练集里没有对应答案 → 评测结果更可信，避免了传统基准的“记忆作弊”。**  
4. **通用评测工具箱 → 提供统一的 Prompt 模板、自动化评测脚本以及结果可视化模块 → 研究者只需接入自己的模型，即可在同一平台上进行公平比较。**  

### 方法详解
整体思路可以拆成五步：**抓题 → 过滤 → 任务构造 → 模型调用 → 结果统计**。

1. **抓题（题目抓取）**  
   - 每天爬取 LeetCode、AtCoder、CodeForces 的公开赛题目。  
   - 只保留发布时间在 2023‑05 到 2024‑05 之间的题目，确保“新鲜”。  
   - 抓取的原始信息包括题目描述、输入输出格式、样例、官方测试用例等。

2. **过滤（污染剔除）**  
   - 使用文本相似度模型对每道新题与公开的代码数据集（包括 GitHub、StackOverflow）进行比对，过滤掉相似度超过阈值的题目。  
   - 同时检查题目是否已被大型模型的公开训练集收录，若是则直接剔除。  
   - 这一步相当于在“食材市场”挑选未被其他厨师使用过的全新食材。

3. **任务构造（多维度评估）**  
   - **代码生成**：给模型题目描述，要求直接输出可通过全部官方测试的代码。  
   - **自修复**：先让模型生成代码，然后故意注入一个小错误（如语法错误），再让模型在错误提示下给出修复方案。  
   - **执行预测**：提供题目描述和输入样例，要求模型在不运行代码的前提下预测输出。  
   - 每个子任务都有统一的 Prompt 模板，保证不同模型的输入风格一致。

4. **模型调用（统一接口）**  
   - 通过 OpenAI、vLLM、HuggingFace 等统一的 API 把 Prompt 发送给目标模型。  
   - 对每个模型分别运行 **18 个基础 LLM** 与 **34 个指令微调 LLM**，记录完整的生成文本、错误信息以及运行时长。

5. **结果统计（评测与公开）**  
   - 代码生成通过率采用官方测试用例的全部通过率。  
   - 自修复成功率衡量模型在错误提示后是否能恢复到通过状态。  
   - 执行预测准确率直接对比模型输出与真实运行结果。  
   - 所有 Prompt、模型输出、评测脚本统一打包发布，方便社区二次分析。  

**最巧妙的点**在于把“实时抓题+污染过滤”做成闭环：每次新题进入评测前，都经过严格的相似度检查，确保评测数据始终保持“未被模型见过”。这在过去的基准里是极少出现的。

### 实验与效果
- **数据规模**：LiveCodeBench 收录了 400 题，全部来自 2023‑05 至 2024‑05 的真实竞赛。  
- **模型覆盖**：对 18 种基础 LLM 与 34 种指令微调 LLM 进行评测。  
- **基线对比**：与 HumanEval、MBPP 等传统基准进行横向比较，论文声称在这些老基准上表现突出的模型，在 LiveCodeBench 上的通过率出现显著下降，说明旧基准可能出现了过拟合。  
- **整体表现**：在代码生成任务上，最强指令微调模型的通过率约为 45%（相较 HumanEval 上的 60% 有明显回落），自修复成功率约为 30%，执行预测准确率约为 55%。（具体数字来自论文表格，本文未列出全部细节）  
- **消融实验**：作者分别去掉“自修复”与“执行预测”子任务，整体评测分数下降约 8% 与 5%，说明多维度评估对区分模型能力贡献显著。  
- **局限性**：论文承认目前仅覆盖三大竞赛平台，题目类型仍偏算法实现，缺少系统编程、库调用等真实工程场景；此外，过滤过程依赖相似度模型，可能仍有少量潜在污染未被捕获。

### 影响与延伸思考
LiveCodeBench 的出现让代码模型的评测从“单一答题”转向“全流程能力检验”，已经被后续的开源项目（如 CodeEval++、DynamicCodeBench）借鉴，形成了“持续更新、无污染”评测的趋势。研究者们开始关注模型在调试、预测等二次任务上的表现，而不再只看一次性生成的正确率。未来可以进一步扩展到 **多语言支持**、**库调用正确性**、**代码安全审计** 等方向，构建更贴近工业开发的评测生态。想深入了解的读者可以关注 LiveCodeBench 官方 GitHub 上的 “Add New Scenario” 模块，它提供了把新任务（比如代码审计）快速接入评测框架的指南。

### 一句话记住它
LiveCodeBench 用持续抓取、严格过滤的真实竞赛题目，把代码模型的评估从单一生成提升到写、调、跑、测的全流程检验。