# AssertCoder: LLM-Based Assertion Generation via Multimodal Specification Extraction

> **Date**：2025-07-14
> **arXiv**：https://arxiv.org/abs/2507.10338

## Abstract

Assertion-Based Verification (ABV) is critical for ensuring functional correctness in modern hardware systems. However, manually writing high-quality SVAs remains labor-intensive and error-prone. To bridge this gap, we propose AssertCoder, a novel unified framework that automatically generates high-quality SVAs directly from multimodal hardware design specifications. AssertCoder employs a modality-sensitive preprocessing to parse heterogeneous specification formats (text, tables, diagrams, and formulas), followed by a set of dedicated semantic analyzers that extract structured representations aligned with signal-level semantics. These representations are utilized to drive assertion synthesis via multi-step chain-of-thought (CoT) prompting. The framework incorporates a mutation-based evaluation approach to assess assertion quality via model checking and further refine the generated assertions. Experimental evaluation across three real-world Register-Transfer Level (RTL) designs demonstrates AssertCoder's superior performance, achieving an average increase of 8.4% in functional correctness and 5.8% in mutation detection compared to existing state-of-the-art approaches.

---

# AssertCoder：基于大语言模型的断言生成通过多模态规格提取 论文详细解读

### 背景：这个问题为什么难？

在硬件设计的验证阶段，工程师需要手写 SystemVerilog Assertions（SVAs）来捕捉信号之间的时序关系。手工编写不仅耗时，还极易遗漏细节，导致功能错误在后期才被发现。传统的自动化工具大多只能处理结构化的文本或单一的 RTL 代码，对图表、公式、表格等多种规格形式的理解非常薄弱。于是，如何让机器像人一样从各种文档中抽取语义并生成高质量断言，成为了制约验证效率的核心瓶颈。

### 关键概念速览
- **SVAs（SystemVerilog Assertions）**：在硬件描述语言里写的断言，用来在仿真或形式验证时检查信号是否满足预期的时序约束。相当于软件里的单元测试，只不过对象是硬件信号。
- **多模态规格**：指设计文档中出现的文本、表格、时序图、数学公式等不同表现形式。就像一本教材里既有文字解释又有图示，信息分散在多个“频道”里。
- **模态感知预处理**：针对不同规格类型使用专门的解析器（OCR、表格抽取、公式识别等），把它们统一转成机器可读的结构化数据。类似于把不同语言的字幕都翻译成同一种语言再进行分析。
- **链式思考（Chain‑of‑Thought, CoT）提示**：在大语言模型（LLM）生成答案前，先让模型输出一步步的推理过程。好比老师让学生先写解题步骤，再给出最终答案，以提升复杂任务的可靠性。
- **突变评估（Mutation‑Based Evaluation）**：故意在 RTL 代码里植入小错误（突变），检查生成的断言能否捕获这些错误。相当于给模型的检测能力做一次“压力测试”。
- **信号层语义对齐**：把抽取出的规格信息映射到具体的硬件信号上，确保生成的断言在信号层面是有意义的。可以想象成把自然语言的“温度上升”对应到具体的 sensor_temp 信号。

### 核心创新点
1. **之前的方法 → 本文的做法 → 带来的改变**  
   传统工具只能读取纯文本或直接分析 RTL，面对图表、公式时往往失效。AssertCoder 引入了模态感知预处理，分别使用 OCR、表格抽取和公式解析，把所有规格统一成结构化的信号语义图。这样一来，系统能够从完整的设计说明书中捕获细粒度的时序约束，显著提升了断言的覆盖率。

2. **之前的方法 → 本文的做法 → 带来的改变**  
   早期的 LLM 驱动生成通常采用一次性提示，容易产生语义漂移。本文采用多步 CoT 提示，让模型先列出涉及的信号、再推导时序关系、最后输出完整的 SVA。这个分层思考过程把复杂的规格转化为可验证的子任务，生成的断言在语法和语义上都更可靠。

3. **之前的方法 → 本文的做法 → 带来的改变**  
   自动化验证缺少客观的质量评估手段，往往只能靠人工审查。AssertCoder 加入了突变评估环节：在 RTL 中随机注入错误，使用模型检查生成的断言是否能捕获这些错误。实验表明，这种闭环反馈机制帮助模型自我纠错，使最终的断言在实际验证中表现更好。

### 方法详解
整体框架可以划分为四个阶段：**多模态预处理 → 语义抽取 → CoT 断言合成 → 突变回环优化**。

1. **多模态预处理**  
   - 文本：直接用自然语言处理模型抽取关键词。  
   - 表格：利用表格解析器把行列映射成“信号‑属性”对。  
   - 时序图：先用 OCR 识别图形，再用图结构分析算法恢复信号之间的时序箭头。  
   - 公式：采用 MathPix 类的公式识别，将 LaTeX 转成抽象语法树（AST），再映射到对应信号。  
   这些子模块的输出都是统一的 **信号语义节点**，每个节点记录信号名、位宽、时序约束来源等信息。

2. **语义抽取**  
   将所有节点送入专门的 **信号层语义对齐器**，它会构建一个有向图，节点是信号，边是“先后关系”或“数值约束”。对齐器利用规则库（如“上升沿触发”对应 posedge）以及 LLM 的零样本推理，确保每条边都有明确的硬件语义解释。

3. **CoT 断言合成**  
   - **步骤 1**：模型列出涉及的信号集合。  
   - **步骤 2**：对每对信号，模型依据对齐图输出时序关系的自然语言描述。  
   - **步骤 3**：模型把自然语言转化为 SystemVerilog Assertion 语法。  
   这里的提示模板是层层递进的，确保模型在每一步都有明确的上下文，避免一次性生成时出现遗漏或语法错误。

4. **突变回环优化**  
   - 在原始 RTL 中随机插入位翻转、时序偏移等突变。  
   - 用模型检查生成的断言是否能在仿真或形式验证中捕获这些突变。  
   - 对未捕获的突变，系统会把对应的信号对重新送回 CoT 合成模块，要求模型补全或修正断言。  
   经过若干轮迭代后，断言集合的检测率显著提升。

**最巧妙的点**在于把“多模态信息抽取”与“链式思考生成”解耦：前者负责把杂乱的规格转成结构化图，后者只需要在图上做推理。这样既利用了 LLM 的语言理解优势，又规避了它在图像/表格直接处理上的短板。

### 实验与效果
- **测试对象**：三套真实的 Register‑Transfer Level（RTL）设计，分别来自通信、存储和处理器子系统。每套设计都配有完整的规格文档（含文本、表格、时序图和公式）。
- **对比基线**：传统的基于规则的断言生成工具、纯文本‑LLM 生成方案以及最新的开源硬件验证框架。  
- **核心指标**：功能正确率提升 8.4%，突变检测率提升 5.8%。这意味着在相同的验证时间内，AssertCoder 能捕获更多潜在的功能错误。  
- **消融实验**：论文对每个模块做了独立去除的实验。去掉多模态预处理后，整体性能下降约 6%；去掉 CoT 提示链，错误率上升约 4%；不使用突变回环，检测率下降约 3%。这些数字说明每个创新点都对最终效果有实质贡献。  
- **局限性**：原文未详细描述在极端规格（如高度自定义的 DSL）下的适配成本；此外，LLM 本身仍可能产生幻觉，需要人工审查作为安全网。

### 影响与延伸思考
AssertCoder 把多模态文档解析与 LLM 推理结合，打开了硬件验证自动化的新思路。后续工作已经开始探索 **跨语言模型协同**（比如把专用图像模型和代码模型分别负责不同模态，再通过统一的语义图合并），以及 **自适应提示学习**，让模型在每轮突变回环中自动优化提示模板。对想进一步深入的读者，可以关注 **硬件规格的知识图谱构建** 与 **大模型在形式验证中的安全性** 两大方向，这两块正逐渐成为学术和工业的热点。

### 一句话记住它
AssertCoder 用多模态解析 + 链式思考，让大语言模型直接从完整的硬件规格文档生成高质量断言，显著提升验证覆盖率。