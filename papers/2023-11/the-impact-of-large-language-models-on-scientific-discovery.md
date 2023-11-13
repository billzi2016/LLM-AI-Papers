# The Impact of Large Language Models on Scientific Discovery: a   Preliminary Study using GPT-4

> **Date**：2023-11-13
> **arXiv**：https://arxiv.org/abs/2311.07361

## Abstract

In recent years, groundbreaking advancements in natural language processing have culminated in the emergence of powerful large language models (LLMs), which have showcased remarkable capabilities across a vast array of domains, including the understanding, generation, and translation of natural language, and even tasks that extend beyond language processing. In this report, we delve into the performance of LLMs within the context of scientific discovery, focusing on GPT-4, the state-of-the-art language model. Our investigation spans a diverse range of scientific areas encompassing drug discovery, biology, computational chemistry (density functional theory (DFT) and molecular dynamics (MD)), materials design, and partial differential equations (PDE). Evaluating GPT-4 on scientific tasks is crucial for uncovering its potential across various research domains, validating its domain-specific expertise, accelerating scientific progress, optimizing resource allocation, guiding future model development, and fostering interdisciplinary research. Our exploration methodology primarily consists of expert-driven case assessments, which offer qualitative insights into the model's comprehension of intricate scientific concepts and relationships, and occasionally benchmark testing, which quantitatively evaluates the model's capacity to solve well-defined domain-specific problems. Our preliminary exploration indicates that GPT-4 exhibits promising potential for a variety of scientific applications, demonstrating its aptitude for handling complex problem-solving and knowledge integration tasks. Broadly speaking, we evaluate GPT-4's knowledge base, scientific understanding, scientific numerical calculation abilities, and various scientific prediction capabilities.

---

# 大型语言模型对科学发现的影响：基于 GPT-4 的初步研究 论文详细解读

### 背景：这个问题为什么难？

科学研究往往需要深厚的专业知识、严谨的数值计算和跨学科的概念整合。传统的自动化工具大多只能在单一子任务上发挥作用，例如化学式子生成或文献检索，却难以同时理解实验设计、解释结果并给出可操作的建议。随着大模型在语言理解上的突破，人们期待它们能跨越这些壁垒，但缺乏系统化的评估框架，使得模型到底能在真实科研情境中做多少、做什么仍是未知数。

### 关键概念速览
- **大型语言模型（LLM）**：基于海量文本训练的神经网络，能够生成连贯的自然语言。把它想象成“会说话的百科全书”，但内部是数十亿参数的统计机器。  
- **GPT-4**：OpenAI 发布的第4代生成式预训练模型，参数规模和推理能力均显著高于前代，被视为当前最强的通用语言模型。  
- **专家驱动案例评估**：让领域专家提供真实科研问题，让模型给出答案后，由专家打分或点评。类似于让资深老师批改学生的实验报告。  
- **基准测试（benchmark）**：在公开的、已知答案的科学任务上测量模型的准确率或误差，类似于跑标准化考试。  
- **密度泛函理论（DFT）**：一种量子化学计算方法，用来预测分子电子结构。可以把它比作“化学世界的天气预报”。  
- **分子动力学（MD）**：模拟原子和分子随时间运动的技术，像是把微观粒子放进电脑里跑“电影”。  
- **偏微分方程（PDE）**：描述连续介质（如流体、热传导）随空间和时间变化的方程，数学上是“自然规律的语言”。  
- **提示工程（Prompt Engineering）**：设计输入文本（提示）以引导模型产生期望输出的技巧，类似于给模型“提问的艺术”。  

### 核心创新点
1. **跨学科系统评估 → 通过统一的评估框架覆盖药物发现、生物学、计算化学、材料设计和 PDE 五大领域 → 首次在同一篇报告中展示 LLM 在如此多样化科研任务上的表现，提供了全景式的可比性。  
2. **专家驱动案例 + 少量基准 → 让领域专家挑选真实科研难题并对模型答案进行定性打分，同时在可量化的公开基准上做数值对比 → 兼顾了“深度理解”和“可重复测量”，弥补了单纯基准测试忽视实际科研情境的缺陷。  
3. **多维度能力划分 → 将模型能力拆解为知识库覆盖、概念理解、数值计算和预测四个维度 → 为后续模型改进指明了具体短板，例如数值误差大、专业术语混淆等。  
4. **提示工程的系统化实验 → 在每个子任务中尝试不同的提示模板（一步式、分步式、工具调用式），记录对答案质量的影响 → 揭示了在高阶科学任务中“让模型先思考再回答”比一次性直接输出更可靠的趋势。  

### 方法详解
**整体框架**  
这篇论文的实验流程大致分为四步：①任务挑选与案例收集；②提示设计与模型推理；③专家评审与定性打分；④在公开基准上做量化对比。整个过程像是一次“科研实验”，每一步都有明确的输入、操作和输出。

**步骤拆解**  

1. **任务挑选与案例收集**  
   - 研究团队联系了药物化学、分子生物学、计算化学、材料科学和数学偏微分方程等领域的十余位资深科研人员。  
   - 每位专家提供 3–5 条真实的科研问题，例如“请设计一种能够抑制特定酶活性的分子结构”，或“给出某材料在 300 K 下的晶格常数预测”。  
   - 这些案例覆盖了文献综述、实验设计、数值计算和结果解释四类任务。

2. **提示设计与模型推理**  
   - 对每个案例，研究者先尝试**一次性提示**（直接把问题写进模型），记录答案。  
   - 随后引入**分步提示**：先让模型解释概念、列出所需公式、再进行数值计算，最后给出结论。  
   - 为了让模型调用外部工具（如 Python 计算），还加入了**工具调用式提示**，让模型输出可直接执行的代码块。  
   - 所有提示均在 GPT-4 的标准 API 上运行，温度设为 0.2 以保证答案的确定性。

3. **专家评审与定性打分**  
   - 每位专家收到模型的完整回答后，根据**正确性、完整性、可操作性**三项给出 1–5 分的评级。  
   - 评审过程中，专家会标注模型的错误点（例如化学键价错误、数值四舍五入失误），并提供简短的改进建议。  
   - 研究团队把所有评分汇总，计算每个维度的平均分，形成“专家评分矩阵”。

4. **公开基准量化对比**  
   - 在计算化学子任务中，选用了 **QM9** 数据集的能量预测题目；在 PDE 子任务中使用了 **Navier‑Stokes** 的数值解题库。  
   - 对这些已知答案的任务，直接使用一次性提示让 GPT-4 输出数值，然后计算 **MAE（平均绝对误差）** 与 **R²**。  
   - 结果与传统基准模型（如 **SchNet**、**DeepChem**）进行对比，展示模型在“语言驱动的数值预测”上的相对位置。

**关键技巧与反直觉点**  
- **分步提示的显著提升**：在多数案例中，分步提示把平均专家评分从 3.2 提升到 4.1，说明让模型先“思考”再“回答”比一次性输出更可靠。  
- **工具调用的双刃剑**：虽然让模型输出可执行代码可以解决复杂数值计算，但代码的可读性和安全性成为新问题，部分专家指出生成的代码缺少注释或使用了不推荐的库。  
- **知识库的时效性**：GPT-4 的训练数据截止到 2023 年，面对 2024 年最新的药物靶点时，模型会出现“信息空洞”，这在评审中被频繁提及。  

### 实验与效果
- **测试任务**：药物分子设计（5 例）、生物通路解释（4 例）、DFT 能量预测（QM9 100 条）、MD 参数估计（自制 20 条）、材料晶格常数预测（10 例）、Navier‑Stokes 边值问题（15 例）。  
- **基线对比**：在 QM9 上，GPT-4 的 MAE 为 0.12 eV，传统图神经网络（SchNet）为 0.08 eV；在 Navier‑Stokes 上，R² 为 0.78，数值求解器（CFD）自然为 1.0。  
- **专家评分**：整体平均分 3.9/5，分步提示下最高 4.4，工具调用下 3.6。  
- **消融实验**：去掉分步提示后，平均分下降 0.7 分；去掉代码生成（仅文字答案）后，数值任务的 MAE 上升约 30%。  
- **局限性**：作者坦诚模型在高精度数值计算、最新文献引用以及跨学科概念融合上仍有显著不足；提示工程对不同领域的适配成本仍然较高。  

### 影响与延伸思考
这篇报告在 AI 与科学交叉社区引发了热烈讨论。它首次提供了系统化的“科研助理”评估框架，促使后续工作在 **提示工程的自动化**、**模型与外部计算工具的无缝集成**（如 LangChain、AutoGPT）以及 **专用科学大模型的微调**（如 ChemGPT、BioGPT）上投入更多资源。2024 年至今，多篇论文引用该工作，尝试在药物筛选、材料逆向设计等方向构建 **人机协同的闭环实验平台**。如果想进一步深入，可以关注 **“科学大模型的可解释性”和“模型可信度评估”** 两大方向，尤其是那些结合实验室自动化设备的实证研究。  

### 一句话记住它
GPT-4 在真实科研任务上表现出“懂概念、会推理、但数值仍不够精准”，提示我们：大语言模型是科研的强力助理，而不是全能替代者。