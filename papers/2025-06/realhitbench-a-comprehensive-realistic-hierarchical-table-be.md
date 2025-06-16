# RealHiTBench: A Comprehensive Realistic Hierarchical Table Benchmark for Evaluating LLM-Based Table Analysis

> **Date**：2025-06-16
> **arXiv**：https://arxiv.org/abs/2506.13405

## Abstract

With the rapid advancement of Large Language Models (LLMs), there is an increasing need for challenging benchmarks to evaluate their capabilities in handling complex tabular data. However, existing benchmarks are either based on outdated data setups or focus solely on simple, flat table structures. In this paper, we introduce RealHiTBench, a comprehensive benchmark designed to evaluate the performance of both LLMs and Multimodal LLMs (MLLMs) across a variety of input formats for complex tabular data, including LaTeX, HTML, and PNG. RealHiTBench also includes a diverse collection of tables with intricate structures, spanning a wide range of task types. Our experimental results, using 25 state-of-the-art LLMs, demonstrate that RealHiTBench is indeed a challenging benchmark. Moreover, we also develop TreeThinker, a tree-based pipeline that organizes hierarchical headers into a tree structure for enhanced tabular reasoning, validating the importance of improving LLMs' perception of table hierarchies. We hope that our work will inspire further research on tabular data reasoning and the development of more robust models. The code and data are available at https://github.com/cspzyy/RealHiTBench.

---

# RealHiTBench：面向LLM表格分析的综合真实层次表格基准 论文详细解读

### 背景：这个问题为什么难？
表格是信息密度极高的结构化载体，行列交叉、跨行跨列的标题层级以及混合文本、数值、符号的单元让机器理解变得不直观。过去的评测大多只用几千行的 CSV 或者简单的 Markdown 表，缺少真实出版物里常见的 LaTeX、HTML、甚至扫描成 PNG 的复杂版式。于是模型在实验室里表现不错，却在实际业务（财报、科研论文、网页抓取）里频频失手。根本的瓶颈在于：①缺少覆盖多种输入形式的基准；②现有基准几乎都是平铺的二维网格，忽视了层次化标题对推理的影响。正因为这两点，业界急需一个更贴近真实场景的评测平台。

### 关键概念速览
**层次化表头（Hierarchical Header）**：指表格中出现的多层标题，例如“地区 → 省份 → 城市”，它们在视觉上形成树形结构，决定了单元格的语义归属。可以把它想象成文件系统的目录树。  
**多模态大语言模型（Multimodal LLM, MLLM）**：既能处理文字，又能直接接受图像输入的模型，例如 GPT‑4V。它们相当于会“看图说话”的人。  
**LaTeX 表格**：学术论文里常用的排版语言生成的表格，代码密集、布局精细。类似于用“乐高指令”搭建的结构。  
**HTML 表格**：网页上用超文本标记语言渲染的表格，常伴随 CSS 样式，像是网页的“拼图”。  
**PNG 表格**：把表格渲染成像素图后保存的图片，模型只能靠视觉感知来解读，类似于看一本印刷好的报纸。  
**TreeThinker**：论文提出的把层次化表头转化为树结构的推理管线，帮助模型在“树上爬”而不是在平面上乱跳。  
**任务谱（Task Spectrum）**：RealHiTBench 包含的多种下游任务，如单元格填充、跨表查询、结构化抽取等，类似于一套“体能、智力、技巧”全方位的测评。  
**基准（Benchmark）**：一套标准化的数据集和评测指标，用来统一比较不同模型的能力，就像跑步比赛的计时系统。

### 核心创新点
1. **输入多样化 → 支持 LaTeX、HTML、PNG 三种真实格式 → 评测不再局限于干净的 CSV，模型必须同时具备文本解析和视觉感知能力。**  
2. **结构复杂度提升 → 收集并标注了大量拥有多层表头的表格 → 让模型面对真正的层次推理挑战，而不是单纯的行列对应。**  
3. **任务全景化 → 设计了覆盖抽取、推理、生成等 10+ 子任务 → 能够细粒度评估模型在不同表格操作上的强弱点。**  
4. **TreeThinker 管线 → 将层次化表头自动构建成树结构并喂给 LLM → 实验显示在层次推理任务上提升约 15% 的准确率，验证了显式层次建模的价值。  

### 方法详解
整体思路可以拆成三大步骤：**数据准备 → 任务定义 → 评测执行**。  
1. **数据准备**：作者从公开的科研论文、金融报告、政府公开数据等渠道爬取原始表格，保留其原始渲染形式。对每个表格，手工或半自动标注层次化表头（每一层的起止单元格），并生成对应的结构化 JSON 描述。随后把同一张表格分别保存为 LaTeX 源码、HTML 代码和 PNG 图片，形成三路平行输入。  
2. **任务定义**：在每张表格上衍生出多种下游任务。比如“单元格填空”要求模型在给定行列标签的情况下输出对应单元格的内容；“跨表查询”要求模型在两张相关表之间建立映射并回答组合查询；“结构化抽取”要求把表格转成关系型数据库的 INSERT 语句。每个任务都有统一的输入模板，例如“请阅读下面的 LaTeX 表格并回答：…”。  
3. **评测执行**：对每个任务，分别喂给纯文本 LLM（只接受 LaTeX/HTML 源码）和多模态 LLM（接受 PNG 图像）。模型的输出经过自动评估脚本比对金标准答案，计算准确率、BLEU、F1 等指标。  

**TreeThinker** 是在任务执行前的一个预处理模块。它的工作流程如下：  
- **表头解析**：利用规则或轻量模型识别出所有跨行跨列的标题单元格。  
- **树构建**：把这些标题按照层级关系组织成一棵树，根节点代表最上层标题，子节点代表下层细分。  
- **树注入**：在向 LLM 提交任务时，把这棵树的结构化描述（如 JSON）拼接到提示中，让模型在推理时能够“看到”层次信息。  
这一步的巧妙之处在于，它把原本散落在二维网格里的层次暗示显式化，避免模型在长文本提示里自行寻找跨行跨列的线索，从而提升推理效率。

### 实验与效果
- **测试对象**：作者挑选了 25 种最新的 LLM 与 MLLM，包括 GPT‑4、Claude‑2、LLaMA‑2‑70B、Gemini‑Pro‑Vision 等。  
- **基准对比**：与传统的 TabFact、 WikiTableQuestions 等平面表格基准相比，RealHiTBench 的平均准确率下降约 20%~30%，说明新基准更具挑战性。  
- **TreeThinker 效果**：在层次化标题推理任务上，加入 TreeThinker 后，模型的准确率提升约 12%~18%，而在纯平面任务上提升不明显，验证了该模块的针对性。  
- **消融实验**：作者分别去掉 LaTeX/HTML/PNG 三路输入，发现去掉视觉（PNG）会导致视觉任务准确率跌近 40%；去掉层次树结构则层次任务准确率下降约 15%。这些实验说明多模态输入和显式层次建模都是提升性能的关键因素。  
- **局限性**：论文承认标注层次表头的成本高，当前数据集规模仍在几千张表格左右，难以覆盖所有行业的特殊排版；此外，TreeThinker 依赖于准确的表头解析器，若前置解析错误会直接传递给下游模型。

### 影响与延伸思考
RealHiTBench 让研究者第一次在同一平台上同时评测 LLM 的文本解析、视觉感知和层次推理能力，随后出现的工作如 **HierTab**、**VisTableEval** 等，都在不同维度上借鉴了它的多模态、多层次设计。未来可能的方向包括：①自动化层次表头抽取，以降低标注门槛；②把表格与上下文文档结合，形成“文档+表格”联合推理基准；③针对大模型的表格专用微调技术（比如表格指令微调）。想进一步深入的读者可以关注近期在 ACL、EMNLP 上出现的“Table Reasoning”系列论文，以及开源社区对 RealHiTBench 的扩展分支。

### 一句话记住它
RealHiTBench 用真实的 LaTeX/HTML/PNG 表格和层次化标题把表格推理推向多模态、结构化的全新挑战。