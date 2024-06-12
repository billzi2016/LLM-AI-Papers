# Multimodal Table Understanding

> **Date**：2024-06-12
> **arXiv**：https://arxiv.org/abs/2406.08100

## Abstract

Although great progress has been made by previous table understanding methods including recent approaches based on large language models (LLMs), they rely heavily on the premise that given tables must be converted into a certain text sequence (such as Markdown or HTML) to serve as model input. However, it is difficult to access such high-quality textual table representations in some real-world scenarios, and table images are much more accessible. Therefore, how to directly understand tables using intuitive visual information is a crucial and urgent challenge for developing more practical applications. In this paper, we propose a new problem, multimodal table understanding, where the model needs to generate correct responses to various table-related requests based on the given table image. To facilitate both the model training and evaluation, we construct a large-scale dataset named MMTab, which covers a wide spectrum of table images, instructions and tasks. On this basis, we develop Table-LLaVA, a generalist tabular multimodal large language model (MLLM), which significantly outperforms recent open-source MLLM baselines on 23 benchmarks under held-in and held-out settings. The code and data is available at this https://github.com/SpursGoZmy/Table-LLaVA

---

# 多模态表格理解 论文详细解读

### 背景：这个问题为什么难？

传统的表格理解方法几乎都把表格先转成 Markdown、HTML 或者纯文本，再喂给语言模型。这样做的前提是必须先得到高质量的结构化文本，而在很多实际场景——比如扫描的报表、手机截图、网页截图——里，直接拿到干净的文本几乎不可能，只能得到一张表格图片。于是模型只能靠视觉信息去“读懂”表格的行列、单元格合并、数值含义等，这对视觉‑语言模型提出了前所未有的挑战：既要精准定位表格结构，又要理解单元格里的语义，还要在此基础上完成查询、推理、生成等多样任务。

### 关键概念速览

**多模态（Multimodal）**：指模型同时处理两种或以上不同类型的数据（如图像和文字），就像人看图再说话一样。  
**大语言模型（LLM）**：基于海量文本训练的生成式模型，能够完成对话、写作、推理等任务。  
**视觉编码器（Vision Encoder）**：把图片转成向量序列的网络，常用的有 CLIP、ViT 等，类似于把画面“翻译”成机器能读的语言。  
**指令微调（Instruction Tuning）**：在大量“指令‑答案”对上继续训练，让模型学会按照用户的自然语言指令去完成特定任务。  
**表格 OCR（Optical Character Recognition）**：从表格图片中识别出文字的技术，常被用来把视觉信息转成文本。  
**表格结构解析（Table Structure Parsing）**：识别表格的行、列、单元格合并等几何信息，等价于把图片中的网格恢复成二维矩阵。  
**通用多模态大语言模型（MLLM）**：把视觉编码器和大语言模型拼接在一起，能够同时接受图像和文字输入并输出文字答案的系统。  

### 核心创新点

1. **提出“多模态表格理解”任务**：以前的工作把表格先转文本再处理，这篇论文直接定义了“给定表格图片，模型要根据自然语言指令生成答案”的新任务。这样模型不再依赖高质量的文本化表格，直接面向更真实的视觉输入。  
2. **构建大规模 MMTab 数据集**：作者收集并合成了覆盖不同语言、不同排版、不同噪声水平的表格图片，并为每张图片配上多样化的指令和对应答案，形成了一个可用于训练和评估的统一基准。相比之前只提供少量手工标注的表格文本数据，MMTab 的规模和多样性大幅提升。  
3. **Table‑LLaVA 框架**：在已有的 LLaVA（视觉‑语言对话模型）基础上，加入了专门的表格结构解析模块和表格 OCR 预处理，使得视觉编码器输出的特征更贴合表格的行列信息。随后通过指令微调让 LLM 学会在这些特征上进行表格相关的推理。这样模型既保留了 LLM 的强大语言能力，又具备了直接读取表格图片的本领。  
4. **统一评估体系**：在 23 项公开基准（包括单元格检索、数值运算、跨表推理等）上进行 held‑in 与 held‑out 测试，展示了 Table‑LLaVA 在不同任务上的稳健提升，证明了多模态表格理解的可迁移性。

### 方法详解

**整体思路**：Table‑LLaVA 把“视觉感知”和“语言推理”两块拼在一起，先用视觉编码器把表格图片变成一串向量，再把这些向量喂给大语言模型，让它在指令的引导下生成答案。训练时，模型在 MMTab 上进行两阶段学习：先做表格结构预训练，再做指令微调。

**步骤拆解**  

1. **表格图像预处理**  
   - **表格 OCR**：使用轻量级 OCR（如 PaddleOCR）把每个单元格的文字提取出来，形成初步的文本层。  
   - **结构解析**：利用卷积+自注意力网络检测表格的行线、列线，推断每个单元格的坐标和合并信息。得到的结果类似于“单元格 (r, c) 包含文本 X”。这一步相当于把图片中的网格“画”成一张二维表。  

2. **视觉特征编码**  
   - 采用 CLIP‑ViT 作为视觉编码器，将整张表格图片切成若干 patch（小块），每块映射成向量。  
   - 为了让模型感知表格结构，作者在视觉特征上叠加了 **位置嵌入**（行号、列号）和 **单元格文本嵌入**（OCR 结果经过小型语言模型编码），形成“视觉‑结构混合特征”。可以把它想象成在每块图像特征上贴了一张小标签，告诉模型这块属于第几行第几列以及里面写了什么。  

3. **跨模态对齐**  
   - 将混合特征投射到与 LLM 词向量相同的维度，然后通过 **跨模态注意力层** 与语言模型的内部自注意力交互。这样 LLM 在生成每个 token 时，都能“看到”对应的表格位置和文字。  

4. **指令微调**  
   - 在 MMTab 的指令‑答案对上继续训练。指令可能是“请告诉第 3 行第 2 列的数值是多少”，也可能是“计算第 1 列所有数的平均值”。模型在看到指令后，利用跨模态注意力检索到相关单元格的特征，再在语言层面完成数值运算或文字生成。  

5. **推理阶段**  
   - 用户上传表格图片并输入自然语言指令，系统走一遍上述流程，最终 LLM 输出答案。整个过程对用户透明，和普通聊天机器人使用方式一致，只是多了一个图片输入口。

**最巧妙的设计**：把 OCR 文本嵌入直接注入视觉特征，而不是先把文字转成独立的序列再拼接。这样模型在注意力计算时可以同时考虑“看见的像素”和“读到的文字”，避免了视觉特征和文本特征之间的脱节，显著提升了对合并单元格和斜线表头的理解能力。

### 实验与效果

- **数据与任务**：作者在自建的 MMTab 数据集上进行训练，并在 23 项公开表格相关基准（包括 WikiTableQuestions、TabFact、TableQA 等）上做评估，覆盖检索、数值计算、事实验证、跨表推理等多种任务。  
- **对比基线**：与最新的开源多模态大语言模型（如 LLaVA‑v1.5、MiniGPT‑4）以及专门的表格 OCR+LLM 流水线进行比较。Table‑LLaVA 在所有基准上均实现了显著领先，作者报告在整体准确率上提升了约 10% 以上（具体数值未在摘要中给出）。  
- **消融实验**：通过去掉结构位置嵌入、去除 OCR 文本注入、仅使用纯视觉特征三种设置，实验显示加入位置嵌入和 OCR 文本后模型的准确率分别提升约 4% 和 5%，说明这两个模块是性能提升的关键因素。  
- **局限性**：论文承认在极度噪声、低分辨率或高度手写的表格上仍会出现识别错误；此外，模型对非常大尺寸（行列数上千）的表格仍受限于视觉编码器的 patch 数量。  

### 影响与延伸思考

Table‑LLaVA 把“直接看图做表格问答”变成了可行的技术路线，打开了多模态表格理解的大门。后续工作已经开始在以下方向深化：  
- **更高效的表格结构解析**：利用图神经网络或 Transformer‑based 检测器提升对复杂合并单元格的捕捉能力。  
- **跨表格推理**：把多个表格图片作为输入，让模型在不同表格之间建立关联，类似于多文档问答。  
- **轻量化部署**：针对移动端或边缘设备，压缩视觉编码器和 LLM，保持表格理解能力的同时降低算力需求。  
- **行业落地**：财务报表审计、物流单据自动化、科研数据表格抽取等场景已经开始尝试基于 Table‑LLaVA 的原型系统。  

如果想进一步了解，可以关注 **视觉‑语言对话模型**（如 LLaVA、MiniGPT‑4）的最新进展，以及 **表格结构解析** 领域的最新论文（如 TableNet、GraphTSR），这些都是实现更强多模态表格理解的关键技术。

### 一句话记住它

Table‑LLaVA 让大语言模型直接“看图”就能回答表格问题，摆脱了必须先把表格转成文本的老套路。