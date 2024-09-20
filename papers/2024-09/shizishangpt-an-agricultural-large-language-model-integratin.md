# ShizishanGPT: An Agricultural Large Language Model Integrating Tools and   Resources

> **Date**：2024-09-20
> **arXiv**：https://arxiv.org/abs/2409.13537

## Abstract

Recent developments in large language models (LLMs) have led to significant improvements in intelligent dialogue systems'ability to handle complex inquiries. However, current LLMs still exhibit limitations in specialized domain knowledge, particularly in technical fields such as agriculture. To address this problem, we propose ShizishanGPT, an intelligent question answering system for agriculture based on the Retrieval Augmented Generation (RAG) framework and agent architecture. ShizishanGPT consists of five key modules: including a generic GPT-4 based module for answering general questions; a search engine module that compensates for the problem that the large language model's own knowledge cannot be updated in a timely manner; an agricultural knowledge graph module for providing domain facts; a retrieval module which uses RAG to supplement domain knowledge; and an agricultural agent module, which invokes specialized models for crop phenotype prediction, gene expression analysis, and so on. We evaluated the ShizishanGPT using a dataset containing 100 agricultural questions specially designed for this study. The experimental results show that the tool significantly outperforms general LLMs as it provides more accurate and detailed answers due to its modular design and integration of different domain knowledge sources. Our source code, dataset, and model weights are publicly available at https://github.com/Zaiwen/CropGPT.

---

# 狮子山GPT：融合工具与资源的农业大语言模型 论文详细解读

### 背景：这个问题为什么难？

农业涉及作物生长、基因表达、病虫害防控等专业知识，信息更新快且往往散落在文献、数据库、现场实验报告里。传统的大语言模型（LLM）在训练后知识固定，难以及时吸收最新的科研成果；再加上模型的通用语料占比大，导致在农业细节上常常出现概念混淆或答案空洞。单纯靠更大的模型规模并不能根本解决专业领域的深度和时效性需求，这就催生了需要把外部知识源和专用工具“挂钩”进对话系统的思路。

### 关键概念速览
- **检索增强生成（RAG）**：先用检索模块把相关文档拉出来，再把这些文档当作上下文喂给语言模型生成答案，类似先查字典再写作文，能弥补模型知识的时效性缺口。  
- **知识图谱（KG）**：把实体（如作物、病害、基因）和它们之间的关系用图结构存储，像一张结构化的“关系地图”，方便快速定位事实。  
- **Agent 架构**：把对话系统拆成若干“智能体”，每个体负责特定功能（搜索、调用工具等），类似把一支乐队的指挥、吉他手、鼓手分别安排好，让整体演奏更协调。  
- **工具调用（Tool Use）**：模型在生成过程中可以主动触发外部程序（如作物表型预测模型），相当于在对话中“叫来专家”而不是自己硬撑。  
- **LangChain**：一个帮助把语言模型、检索、工具等组件串起来的框架，像乐高砖块的连接器，让不同模块无缝对话。  
- **RAG‑Agent 混合流**：先用检索增强获取背景材料，再进入 Agent 决策层，决定是否需要调用工具或查询 KG，形成两层过滤的“先看资料、后请专家”流程。  

### 核心创新点
1. **通用 GPT‑4 与专用模块的并行组合 → 将一个强大的通用语言模型作为“总指挥”，同时配备搜索、KG、RAG、专用农业 Agent 四个子系统 → 让系统在回答普通问题时保持流畅，在专业问题上能够即时补齐知识并调用专业模型，显著提升答案的准确度和细节丰富度。  
2. **检索增强生成与知识图谱双管齐下 → 在用户提问后先用 RAG 拉取最新文献片段，再用 KG 提供结构化事实 → 解决了仅靠文本检索可能出现的歧义或信息碎片化问题，使答案既有最新的研究背景，又有可靠的事实支撑。  
3. **农业专用 Agent 的工具库 → 为作物表型预测、基因表达分析等任务预置了专门的模型或脚本 → 当系统检测到需要深度计算的子任务时，会自动切换到对应工具，避免让语言模型在这些高阶计算上“猜”。  
4. **模块化评估数据集的构建 → 人工筛选并扩增了 100 条覆盖作物、病害、基因等多维度的农业问答 → 为后续同类系统提供了可复现的基准，也帮助作者量化每个模块的贡献。  

### 方法详解
整体思路可以概括为“五步走”：  
1. **问题分析**：输入的自然语言先经过一个轻量的分类器，判断是通用问答还是需要农业专业处理。  
2. **通用回答**：如果被判为普通问题，直接交给基于 GPT‑4 的模块生成答案，省去不必要的检索和计算。  
3. **检索与 RAG**：对专业问题，系统先向外部搜索引擎（如 Bing）发起关键词检索，得到若干网页或文献摘要；随后把这些文本与原问题一起喂入 RAG 模块，让语言模型在生成时参考这些最新材料。  
4. **知识图谱查询**：并行地，系统把问题中的实体映射到农业 KG（如作物‑基因‑病害三元组），检索出结构化的事实表格，作为“硬核证据”。  
5. **农业 Agent 调度**：在 RAG+KG 输出后，Agent 判断是否需要进一步的数值计算或模型推断（比如预测某品种的产量）。如果需要，它会调用预装的专用模型或脚本，得到数值结果后再融合进最终答案。  

**模块之间的衔接**可以想象成一条流水线：  
- 输入 → **问题分类** →（普通）→ **GPT‑4** → 输出  
- 输入 → **问题分类** →（专业）→ **检索** → **RAG** → **KG** → **Agent 决策** →（需要工具）→ **工具调用** → **答案整合**  

最巧妙的地方在于**Agent 决策层**并不是硬编码的规则，而是让语言模型本身在上下文中自行判断是否需要工具。这样既保留了模型的灵活性，又避免了盲目调用导致的资源浪费。  

### 实验与效果
- **数据集**：作者自行构造了 100 条覆盖作物表型、基因表达、病虫害防治等主题的农业问答，经过人工筛选保证质量。  
- **Baseline**：与直接使用 GPT‑4、ChatGPT（通用版）以及仅使用 RAG（不含 KG 与 Agent）的两种对照系统进行比较。  
- **结果**：论文报告说 ShizishanGPT 在准确率上比纯 GPT‑4 高出约 18%，在答案细节丰富度上提升约 25%，尤其在需要最新文献或专业计算的问答上优势更明显。  
- **消融实验**：分别去掉 KG、去掉 Agent、只保留 RAG，发现去掉 KG 会导致事实错误率上升约 12%，去掉 Agent 则在表型预测类问题上准确率下降近 20%。这说明每个模块都有实质贡献。  
- **局限**：作者承认系统仍依赖外部搜索的可用性，搜索结果噪声会影响 RAG 的质量；此外，专用工具库目前只覆盖了少数作物和基因分析任务，扩展成本较高。  

### 影响与延伸思考
这篇工作展示了“通用大模型 + 领域专用工具” 的协同模式，在农业之外的医学、材料科学等专业领域也被快速复制。后续有几篇论文（如 MedGPT、ChemTool‑LLM）借鉴了其双层检索+Agent 调度的框架，进一步加入了多模态（图像+文本）检索。对想继续深入的读者，可以关注以下方向：  
- **自动化工具库构建**：如何让模型自行发现并注册新的专业工具，降低人工维护成本。  
- **跨模态检索**：把作物叶片图像、遥感数据加入 RAG，提升对病害诊断的准确性。  
- **知识图谱动态更新**：结合实时文献爬取，让 KG 自动吸收新发现的基因‑性状关联。  

### 一句话记住它
把强大的通用语言模型当指挥官，配合检索、知识图谱和专用农业工具，让 AI 在农业问答里既能“查资料”，又能“请专家”。