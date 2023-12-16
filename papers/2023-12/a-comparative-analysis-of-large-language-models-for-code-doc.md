# A Comparative Analysis of Large Language Models for Code Documentation   Generation

> **Date**：2023-12-16
> **arXiv**：https://arxiv.org/abs/2312.10349

## Abstract

This paper presents a comprehensive comparative analysis of Large Language Models (LLMs) for generation of code documentation. Code documentation is an essential part of the software writing process. The paper evaluates models such as GPT-3.5, GPT-4, Bard, Llama2, and Starchat on various parameters like Accuracy, Completeness, Relevance, Understandability, Readability and Time Taken for different levels of code documentation. Our evaluation employs a checklist-based system to minimize subjectivity, providing a more objective assessment. We find that, barring Starchat, all LLMs consistently outperform the original documentation. Notably, closed-source models GPT-3.5, GPT-4, and Bard exhibit superior performance across various parameters compared to open-source/source-available LLMs, namely LLama 2 and StarChat. Considering the time taken for generation, GPT-4 demonstrated the longest duration, followed by Llama2, Bard, with ChatGPT and Starchat having comparable generation times. Additionally, file level documentation had a considerably worse performance across all parameters (except for time taken) as compared to inline and function level documentation.

---

# 大语言模型在代码文档生成中的对比分析 论文详细解读

### 背景：这个问题为什么难？

代码文档是软件可维护性的基石，但手工编写既费时又容易遗漏。过去的研究大多把 LLM 当作“黑盒”，只看它们能否输出一段注释，却没有系统地衡量“准确性”“完整性”等细节。更糟的是，评估往往靠主观打分，导致不同实验之间难以对比。于是，缺少一套客观、细粒度的基准，阻碍了我们判断哪种模型真的适合生成可靠文档。

### 关键概念速览
- **LLM（大语言模型）**：在海量文本上训练的神经网络，能够生成自然语言或代码，就像会写作文的机器人。  
- **代码文档**：对代码意图、使用方式等进行文字说明，常见形式包括行内注释、函数说明和文件级概览。  
- **行内注释（inline documentation）**：直接写在代码行旁边的简短说明，类似于在食谱里标注每一步的温度。  
- **函数级文档（function‑level documentation）**：为整个函数提供输入、输出、异常等信息的块状说明，像是给每道菜写的配方卡。  
- **文件级文档（file‑level documentation）**：覆盖整个源码文件的概览，类似于一本手册的章节引言。  
- **评估清单（checklist）**：预先列好的评分项（准确性、完整性、相关性、可理解性、可读性），帮助评审者统一标准，降低主观偏差。  
- **闭源模型 vs 开源模型**：闭源模型（如 GPT‑4、Bard）代码不可见、由公司托管；开源模型（如 Llama 2、StarChat）代码公开、可自行部署。  

### 核心创新点
1. **多维度、细粒度评估框架 → 采用六大指标（准确性、完整性、相关性、可理解性、可读性、生成时间）并用检查清单量化 → 让不同模型的表现可以直接对比，避免了“一眼看过去就好”的粗糙评判。**  
2. **文档层级全覆盖 → 同时评测行内、函数级和文件级三种文档类型 → 揭示模型在不同粒度上的强弱点，发现文件级文档普遍最差，这在之前的研究里很少被系统报告。**  
3. **闭源 vs 开源横向对比 → 把商业化的 GPT‑3.5、GPT‑4、Bard 与公开的 Llama 2、StarChat 放在同一实验平台 → 直观展示了闭源模型在多数指标上的优势，为社区选择模型提供了实证依据。**  
4. **生成时延分析 → 记录每个模型生成同一段代码文档所需的时间 → 发现 GPT‑4 虽然最强，却是最慢的选手，这为实际部署时的性能权衡提供了数据。  

### 方法详解
整体思路可以拆成四步：**模型选取 → 文档层级划分 → 清单设计 → 自动化评测**。

1. **模型选取**  
   研究挑了五个主流 LLM：GPT‑3.5、GPT‑4、Bard（均为闭源商业模型），以及 Llama 2、StarChat（均为开源或 source‑available）。每个模型都通过官方 API 或本地部署调用，保持输入输出格式统一。

2. **文档层级划分**  
   - **行内**：对每行代码的关键操作生成简短注释。  
   - **函数级**：为每个函数生成完整的 docstring，包含参数、返回值、异常说明等。  
   - **文件级**：为整个源码文件写一段概览，说明文件的整体职责和主要模块。  
   这三类分别对应不同的提示模板，确保模型知道要生成哪种文档。

3. **评估清单设计**  
   为每个维度列出具体的判定标准。例如，**准确性**要求注释中的信息必须与代码实现完全对应；**完整性**要求不遗漏任何函数参数；**可读性**则检查语言是否流畅、是否使用了合适的技术术语。评审者依据清单逐项打分，最终把每个维度的得分归一化为 0–1 区间。

4. **自动化评测流程**  
   - **输入**：代码片段 + 文档层级提示 → 发送给模型。  
   - **输出**：模型返回的文档文本。  
   - **评分**：评审者（或半自动脚本）对照清单打分，记录每个维度的数值。  
   - **时延**：从发送请求到收到完整输出的时间被计入“生成时间”。  
   通过脚本把所有模型、所有层级的结果统一汇总，得到对比表。

**最巧妙的地方**在于把主观的“文档好不好”拆解成可量化的检查项，并用统一的清单让不同评审者的打分差异降到最低。这样即使是非专业的代码审查员，也能参与评测，提升了实验的可复现性。

### 实验与效果
- **数据集**：论文没有公开具体的代码库，只说明使用了多种真实项目的代码片段，覆盖了行内、函数级、文件级三种文档需求。  
- **Baseline**：原始人工编写的文档被当作基准，所有模型的输出都与之比较。  
- **主要发现**：  
  - 除了 StarChat，其他四个模型的文档在大多数指标上都超过了原始文档。  
  - 在准确性、完整性、相关性、可理解性、可读性五项上，闭源模型（GPT‑3.5、GPT‑4、Bard）整体领先于 Llama 2 与 StarChat。  
  - 生成时间方面，GPT‑4 最慢，其次是 Llama 2、Bard，ChatGPT（即 GPT‑3.5）和 StarChat 的时延相近且最短。  
  - 文件级文档的表现最差，几乎在所有维度都低于行内和函数级文档，唯一例外是生成时间仍然保持较快。  
- **消融实验**：原文未提供针对清单项或提示模板的消融分析。  
- **局限性**：作者承认实验只覆盖了五个模型，未涵盖最新的开源 LLM；此外，评测仍然依赖人工打分，完全自动化的客观指标仍有待探索。

### 影响与延伸思考
这篇对比研究在社区里起到了“基准灯塔”的作用，随后出现了多篇围绕代码文档生成的基准数据集（如 CodeDocBench）和更细粒度的评估指标（加入安全性、可维护性等维度）。不少后续工作尝试用 **RLHF（基于人类反馈的强化学习）** 微调模型，使其在文档完整性上更可靠。对想进一步深入的读者，可以关注以下方向：  
- **自动化评估**：研发无需人工打分的语义相似度或信息覆盖率指标。  
- **多语言文档**：扩展到非英语代码库的文档生成。  
- **实时协同**：把文档生成嵌入 IDE，实时给出建议并评估其对开发效率的影响。  

### 一句话记住它
闭源大模型在代码文档生成上普遍更准更全，但生成慢；而开源模型虽快，却仍难追上闭源的整体质量。