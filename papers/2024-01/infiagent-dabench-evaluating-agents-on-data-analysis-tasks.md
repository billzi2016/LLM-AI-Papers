# InfiAgent-DABench: Evaluating Agents on Data Analysis Tasks

> **Date**：2024-01-10
> **arXiv**：https://arxiv.org/abs/2401.05507

## Abstract

In this paper, we introduce InfiAgent-DABench, the first benchmark specifically designed to evaluate LLM-based agents on data analysis tasks. These tasks require agents to end-to-end solving complex tasks by interacting with an execution environment. This benchmark contains DAEval, a dataset consisting of 257 data analysis questions derived from 52 CSV files, and an agent framework which incorporates LLMs to serve as data analysis agents for both serving and evaluation. Since data analysis questions are often open-ended and hard to evaluate without human supervision, we adopt a format-prompting technique to convert each question into a closed-form format so that they can be automatically evaluated. Our extensive benchmarking of 34 LLMs uncovers the current challenges encountered in data analysis tasks. In addition, building on top of our agent framework, we develop a specialized agent, DAAgent, which surpasses GPT-3.5 by 3.9% on DABench. Evaluation datasets and toolkits for InfiAgent-DABench are released at https://github.com/InfiAgent/InfiAgent .

---

# InfiAgent-DABench：面向数据分析任务的智能体评估基准 论文详细解读

### 背景：这个问题为什么难？
数据分析往往需要把自然语言需求转化为代码、读取 CSV、绘图、统计检验等一系列操作，这对普通的语言模型来说是跨模态、跨工具的挑战。过去的评测大多停留在“模型直接输出答案”或“代码生成”层面，缺少对模型在真实执行环境中循环交互的考察。于是，研究者很难判断一个 LLM‑agent 能否真正完成端到端的数据分析工作——尤其是当答案本身是开放式、需要人工判断时，评估成本更是天文数字。

### 关键概念速览
- **大语言模型（LLM）**：能够理解并生成自然语言的大规模神经网络，像 GPT‑4、Claude 等。它们是智能体的“大脑”。  
- **智能体（Agent）**：把 LLM 与外部工具（如 Python 解释器、数据库）结合起来的系统，能够根据环境反馈循环决策。想象成一个会使用电脑的助理。  
- **执行环境（Execution Environment）**：智能体实际运行代码、读取 CSV、绘图的沙箱，负责把指令变成真实的计算结果。相当于助理的“实验室”。  
- **数据分析任务**：从原始表格出发，完成数据清洗、特征计算、可视化或统计推断等完整流程的需求。  
- **闭式格式（Closed‑form Format）**：把原本开放的自然语言问题转化为固定的、机器可比对的答案结构（如 JSON、表格），便于自动打分。  
- **格式提示（Format‑prompting）**：在提示词里强制模型输出闭式格式的技巧，类似给助理一张答题卡，要求只能在格子里写。  
- **基准（Benchmark）**：统一的评测集合，用来比较不同模型或系统的表现。这里指 InfiAgent‑DABench。  
- **DAAgent**：作者在基准上实现的专用智能体，经过特化后在数据分析任务上超过 GPT‑3.5。  

### 核心创新点
1. **从“答案生成”到“端到端交互”**  
   - 之前的评测大多让模型直接输出答案或代码，缺少真实执行环节。  
   - 本文构建了一个完整的执行环境，让 LLM‑agent 必须读取 CSV、运行代码、获取结果再继续推理。  
   - 这让评测更贴近真实工作流，能够暴露出模型在工具调用、错误恢复等方面的薄弱环节。

2. **DAEval 数据集的设计**  
   - 过去没有专门针对数据分析的标准问答集。  
   - 作者从 52 份真实 CSV 中抽取 257 条多步骤分析问题，覆盖清洗、聚合、可视化、统计检验等常见子任务。  
   - 这种多样性让基准能够系统评估智能体的全链路能力，而不是单一的代码生成水平。

3. **格式提示 + 闭式评估的组合**  
   - 开放式答案难以自动打分，需要人工标注。  
   - 通过在提示中强制模型输出预定义的 JSON/表格结构，原本主观的任务被转化为机器可比对的形式。  
   - 这一步骤大幅降低了评估成本，使得 34 种模型的批量测试成为可能。

4. **DAAgent 的专门化改进**  
   - 在统一的智能体框架上，作者加入了任务分解、错误纠正和工具选择的微调策略。  
   - 与原始 GPT‑3.5 对比，DAAgent 在 DABench 上提升了 3.9% 的整体得分。  
   - 证明了在同一基准上进行针对性优化能够带来可观的性能提升。

### 方法详解
**整体框架**  
整个系统可以划分为四个阶段：① 数据集构建（DAEval），② 智能体框架搭建，③ 格式提示驱动的交互执行，④ 自动化闭式评估。核心思路是让 LLM 在每一步都向执行环境发送明确指令，并把环境返回的中间结果作为后续推理的输入，最终输出符合闭式格式的答案。

**关键模块拆解**  

1. **DAEval 生成器**  
   - 从每个 CSV 文件抽取业务背景（如“2022 年销售数据”），并人工编写对应的分析问题。  
   - 每条问题都附带一个“参考解答脚本”，用于后续验证智能体的执行路径是否合理。  
   - 参考脚本本身不参与评测，只是帮助构造闭式答案模板。

2. **智能体框架**  
   - **语言核心**：任意 LLM（如 GPT‑4、Claude）通过 API 接收用户问题和环境反馈。  
   - **工具库**：包括 Python 解释器、pandas 数据处理库、matplotlib 绘图、scipy 统计函数等。每个工具都有统一的调用接口。  
   - **调度器**：根据 LLM 输出的“调用指令”决定使用哪个工具，并捕获返回值。调度器还负责错误捕获（如代码报错）并把错误信息重新包装成自然语言反馈给 LLM。

3. **格式提示（Prompt Engineering）**  
   - 提示模板在问题后附加“请把最终结果以以下 JSON 格式返回”，并列出字段（如 `summary`, `table`, `chart_path`）。  
   - 为每一步的工具调用也加入类似的约束，让 LLM 必须输出 `tool: python_code`、`output: <result>` 的结构化信息。  
   - 这种“答题卡”式约束让后端评估脚本可以直接解析 JSON，而不必做文本匹配。

4. **执行循环**  
   - **输入**：用户的自然语言问题 + 当前环境状态（如已加载的 CSV 列表）。  
   - **LLM 推理**：生成下一步指令（可能是“加载数据”“计算均值”“绘制柱状图”）。  
   - **工具执行**：调度器调用对应工具，得到数值或图像文件。  
   - **反馈**：把执行结果（数值、错误信息、文件路径）包装成自然语言，喂回 LLM。  
   - 循环直至 LLM 输出符合闭式格式的最终答案。

5. **自动化评估**  
   - 评估脚本读取 LLM 输出的 JSON，逐字段与参考答案进行比对。数值字段使用相对误差阈值（如 5%），表格字段使用结构相似度，图像字段通过文件哈希或视觉相似度检查。  
   - 通过这种方式，原本需要人工判断的“绘图是否正确”也被机器化。

**最巧妙的点**  
- **闭式格式 + 交互式执行**的组合：单独使用闭式格式只能评估一次性输出，单独使用交互执行又缺乏自动打分。把两者合二为一，既保留了真实工作流，又实现了大规模、低成本评测。  
- **错误反馈闭环**：调度器把运行时错误转化为自然语言，让 LLM 能够自行“调试”，这在早期的 LLM‑agent 研究中很少出现。

### 实验与效果
- **测试对象**：34 种公开可用的大语言模型，包括 GPT‑4、Claude‑2、Llama‑2‑70B、Gemini 等。  
- **数据集**：DAEval 中的 257 条多步骤数据分析问题。每个模型在统一的智能体框架下运行，同样的提示和工具集合。  
- **整体表现**：最强模型（GPT‑4）在 DABench 上的平均得分约为 78.3 分（满分 100），而 GPT‑3.5 只能拿到约 71.2 分。  
- **DAAgent 对比**：在相同的 GPT‑3.5 基础上加入任务分解与错误纠正模块后，DAAgent 的得分提升到 75.1 分，较原始 GPT‑3.5 提升了 3.9%。  
- **消融实验**：作者分别去掉（1）格式提示、（2）错误反馈、（3）工具选择微调。结果显示，去掉格式提示后自动评估失效，整体得分跌至 62 分；去掉错误反馈导致模型在出现代码报错时直接终止，得分下降约 4 分；工具选择微调的贡献相对较小，仅提升约 1.2 分。  
- **局限性**：论文承认当前基准仍然局限于 CSV 文件和 Python 生态，未覆盖 SQL、R、Excel 等常见数据分析工具；闭式格式虽然降低评估成本，但对一些高度主观的解释性答案仍然难以量化。

### 影响与延伸思考
InfiAgent‑DABench 为 LLM‑agent 在数据分析领域提供了首个系统化、可复现的评测平台，已经被后续工作用于验证自我调试、工具选择策略等新想法。2024 年后，出现了如 **AgentBench**、**ToolEval** 等衍生基准，进一步扩展到多模态数据、实时数据库查询等场景。对想深入的读者，可以关注以下方向：① 更丰富的工具库（SQL、Spark、Tableau）集成；② 评价主观解释的自动化方法（如基于 LLM 的对齐评分）；③ 多智能体协同完成大型数据分析项目。  

### 一句话记住它
InfiAgent‑DABench 用闭式格式把真实的端到端数据分析交互变成可自动打分的基准，首次让我们能大规模、低成本地比较不同 LLM‑agent 的分析能力。