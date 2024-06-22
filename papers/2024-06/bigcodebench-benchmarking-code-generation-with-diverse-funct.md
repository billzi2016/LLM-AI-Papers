# BigCodeBench: Benchmarking Code Generation with Diverse Function Calls   and Complex Instructions

> **Date**：2024-06-22
> **arXiv**：https://arxiv.org/abs/2406.15877

## Abstract

Task automation has been greatly empowered by the recent advances in Large Language Models (LLMs) via Python code, where the tasks ranging from software engineering development to general-purpose reasoning. While current benchmarks have shown that LLMs can solve tasks using programs like human developers, the majority of their evaluations are limited to short and self-contained algorithmic tasks or standalone function calls. Solving challenging and practical tasks requires the capability of utilizing diverse function calls as tools to efficiently implement functionalities like data analysis and web development. In addition, using multiple tools to solve a task needs compositional reasoning by accurately understanding complex instructions. Fulfilling both of these characteristics can pose a great challenge for LLMs.To assess how well LLMs can solve challenging and practical tasks via programs, we introduce BigCodeBench, a benchmark that challenges LLMs to invoke multiple function calls as tools from 139 libraries and 7 domains for 1,140 fine-grained tasks. To evaluate LLMs rigorously, each task encompasses 5.6 test cases with an average branch coverage of 99%. In addition, we propose a natural-language-oriented variant of BigCodeBench, BigCodeBench-Instruct, that automatically transforms the original docstrings into short instructions only with essential information. Our extensive evaluation of 60 LLMs shows that LLMs are not yet capable of following complex instructions to use function calls precisely, with scores up to 60%, significantly lower than the human performance of 97%. The results underscore the need for further advancements in this area.

---

# BigCodeBench：多样函数调用与复杂指令的代码生成基准 论文详细解读

### 背景：这个问题为什么难？
在 LLM 生成代码的早期评测里，任务大多是短小的算法题或单一函数调用，模型只需要写出一个自洽的实现。真实的开发场景却常常需要调动 dozens of third‑party libraries，组合多个工具完成数据清洗、网络请求、可视化等复杂流程。现有 benchmark 缺少这种“工具箱”式的需求，也没有对长指令的细粒度理解进行考察，导致我们不知道模型在真正的工程任务中到底能走多远。

### 关键概念速览
**函数调用（Function Call）**：代码中向已有库的某个功能发起请求，就像在厨房里叫厨师去切菜、烤箱去烤面包。  
**库（Library）**：一组预先实现好的函数集合，提供特定领域的能力，例如 pandas 用来做表格运算。  
**分支覆盖率（Branch Coverage）**：测试用例触及代码中不同分支的比例，覆盖率高说明评测更严格，类似于检查一辆车的每个部件是否都被测试过。  
**指令复杂度（Instruction Complexity）**：自然语言任务描述的长度与层次，越复杂越像真实需求文档，需要模型进行多步推理。  
**BigCodeBench‑Instruct**：把原始函数的 docstring 自动压缩成简短指令的版本，专注于让模型只看到必要信息。  
**人类基准（Human Baseline）**：让真实开发者完成同样任务得到的正确率，用来衡量模型的上限。  
**多域（Multi‑Domain）**：涉及数据分析、网页抓取、机器学习等七大应用领域，类似于让模型在不同的工作岗位上轮岗。  
**细粒度任务（Fine‑grained Task）**：每个任务只要求实现一个具体功能，而不是一个大项目，类似于把一整本书拆成若干章节来考。

### 核心创新点
1. **从单函数到多库调用 → 构建了包含 139 个库、7 大领域的 1,140 条细粒度任务 → 评测不再局限于“写一个排序函数”，而是考察模型能否像工程师一样挑选并组合工具。**  
2. **从宽松指令到高复杂度指令 → 设计了自然语言指令版 BigCodeBench‑Instruct，自动把完整 docstring 精炼为关键要点 → 模型必须在信息稀缺的情况下完成正确的函数调用，逼近真实需求文档的阅读难度。**  
3. **从少量测试到高覆盖率评估 → 为每个任务准备约 5.6 条测试用例，整体分支覆盖率达 99% → 代码错误几乎不可能逃过检测，确保评测结果可信。**  
4. **从单模型对比到大规模横向评测 → 对 60 种公开或内部 LLM 进行统一测评 → 揭示了当前主流模型在复杂指令和多工具使用上的普遍短板。

### 方法详解
整体思路可以拆成三步：任务构造、指令转化、评测执行。

1. **任务构造**  
   - 先挑选 7 个代表性领域（如数据分析、网络爬虫、机器学习等），每个领域再抽取若干常用库，累计 139 个。  
   - 在每个库中挑选若干函数，围绕它们设计真实业务场景的需求。例如，要求使用 pandas 读取 CSV、过滤行、计算统计指标。  
   - 每个需求对应一个细粒度任务，确保只需要调用 1‑3 个函数，保持任务可解但仍需组合工具。

2. **指令转化（BigCodeBench‑Instruct）**  
   - 原始任务提供完整的 Python docstring，里面包含函数签名、参数解释、返回值说明等全部信息。  
   - 使用自动化脚本抽取关键动词、输入输出约束，去掉冗余背景，生成一句到两句的自然语言指令。  
   - 这样模型在推理时只能依赖最核心的信息，模拟真实开发者阅读需求文档的情形。

3. **评测执行**  
   - 对每个任务生成约 5.6 条测试样例，覆盖所有可能的分支路径，确保 99% 的代码分支被执行。  
   - 让模型在给定指令后输出完整的 Python 代码块。系统自动将模型输出与参考实现进行对比，运行所有测试用例，统计通过率。  
   - 同时邀请人类开发者完成同样任务，记录其通过率作为上限基准（97%）。

**最巧妙的设计**在于把“库调用选择”与“指令理解”这两个本来分散的难点合并进同一评测框架。模型必须先解读指令的意图，再在 139 个库中快速定位合适的函数，这相当于让模型在巨大的工具箱里找钥匙，而不是只在桌面上找螺丝刀。

### 实验与效果
- **数据规模**：1,140 条细粒度任务，覆盖 7 大领域，涉及 139 个第三方库。每任务约 5.6 条测试用例，整体分支覆盖率 99%。  
- **模型阵容**：共 60 种公开或内部大语言模型，包括最新的指令微调模型和代码专用模型。  
- **整体表现**：最高得分约 60%，远低于人类的 97%。这说明即使是最强的 LLM，也只能在约三分之二的情况下正确完成复杂指令下的多库调用。  
- **对比基线**：与传统代码生成 benchmark（如 HumanEval、MBPP）相比，BigCodeBench 的通过率下降约 30%‑40%，凸显了多工具、复杂指令的额外难度。  
- **消融实验**：作者分别去掉指令精炼、测试覆盖率提升、库数量限制等因素，发现指令精炼对模型表现影响最大——去掉 Instruct 版本后，模型通过率提升约 10%，说明模型在信息充足时仍能较好完成任务。  
- **局限性**：评测仍然基于 Python 生态，未覆盖其他语言；任务虽细粒度，但仍是单一函数实现，未涉及跨文件项目结构；作者承认自动生成的指令可能存在信息丢失的风险。

### 影响与延伸思考
BigCodeBench 把“工具调用”拉进了代码生成评测的主流视野，随后出现的工作如 **ToolBench**、**CodeToolEval** 等，都在进一步扩展库的种类或加入真实的 IDE 环境。对想深入的读者，可以关注以下方向：  
- **工具选择策略**：让模型学习在海量库中快速检索最匹配的函数，类似信息检索与代码搜索的交叉。  
- **多步计划生成**：把复杂指令拆解成子任务序列，再逐步调用工具，类似人类的“分解任务”。  
- **跨语言工具链**：把 Java、JavaScript、Rust 等生态的库也纳入评测，检验模型的通用工具使用能力。  
- **人机协同**：研究如何让模型在不确定时主动请求人类澄清，形成交互式的代码生成流程。

### 一句话记住它
BigCodeBench 用上千个真实库和高覆盖率测试，首次让 LLM 必须在复杂指令下精准调用多工具，暴露了当前代码生成模型仍远未达到工程师水平。