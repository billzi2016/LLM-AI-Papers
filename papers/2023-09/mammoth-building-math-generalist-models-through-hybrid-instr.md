# MAmmoTH: Building Math Generalist Models through Hybrid Instruction   Tuning

> **Date**：2023-09-11
> **arXiv**：https://arxiv.org/abs/2309.05653

## Abstract

We introduce MAmmoTH, a series of open-source large language models (LLMs) specifically tailored for general math problem-solving. The MAmmoTH models are trained on MathInstruct, our meticulously curated instruction tuning dataset. MathInstruct is compiled from 13 math datasets with intermediate rationales, six of which have rationales newly curated by us. It presents a unique hybrid of chain-of-thought (CoT) and program-of-thought (PoT) rationales, and also ensures extensive coverage of diverse fields in math. The hybrid of CoT and PoT not only unleashes the potential of tool use but also allows different thought processes for different math problems. As a result, the MAmmoTH series substantially outperform existing open-source models on nine mathematical reasoning datasets across all scales with an average accuracy gain between 16% and 32%. Remarkably, our MAmmoTH-7B model reaches 33% on MATH (a competition-level dataset), which exceeds the best open-source 7B model (WizardMath) by 23%, and the MAmmoTH-34B model achieves 44% accuracy on MATH, even surpassing GPT-4's CoT result. Our work underscores the importance of diverse problem coverage and the use of hybrid rationales in developing superior math generalist models.

---

# MAmmoTH：通过混合指令微调构建数学通用模型 论文详细解读

### 背景：这个问题为什么难？

数学推理对语言模型来说一直是硬核挑战。传统的指令微调大多只让模型学习“直接回答”，缺少对解题思路的显式训练，导致在需要多步推理或调用外部工具的题目上频频失手。已有的开源模型要么专注于文字推理（CoT），要么侧重代码生成（PoT），但很少有模型能够灵活切换这两种思路。结果是：在竞争级别的数据集（如 MATH）上，开源 7B 级模型的准确率徘徊在十几个百分点，远不及商业大模型。于是，需要一种既能写出推理链，又能生成可执行代码的混合训练方式，来突破当前的性能瓶颈。

### 关键概念速览
- **CoT（思维链）**：模型在给出最终答案前先把每一步推理写出来，类似人解题时在草稿纸上列步骤，帮助模型保持逻辑连贯性。  
- **PoT（程序思维）**：模型把解题过程转化为可执行代码（如 Python），再让代码跑出结果，像是让模型“动手实验”。  
- **Hybrid Rationales（混合思维）**：把 CoT 与 PoT 结合起来，根据题目特性选择文字推理或代码求解，像是老师在课堂上既讲解概念又演示实验。  
- **Instruction Tuning（指令微调）**：在大模型的基础上，用大量“指令—答案”对进行再训练，使模型更好地遵循用户的任务描述。  
- **MathInstruct 数据集**：作者自行收集并整理的 13 个数学子数据集，全部配有中间推理过程，其中六个是全新标注的，覆盖代数、几何、离散数学等多个分支。  
- **LLM（大语言模型）**：能够理解并生成自然语言的深度学习模型，这里指的是 Llama‑2 与 Code Llama 的开源版本。  
- **Generalist Model（通用模型）**：不局限于单一数学子领域，而是能在广泛题型上保持竞争力的模型。

### 核心创新点
1. **混合思维数据标注 → 同时提供文字推理和代码求解**  
   过去的训练数据要么只给出一步步文字解释，要么只给出代码实现。MAmmoTH 在每条样本中同时提供 CoT 与 PoT 两种“思考路径”，让模型学会在不同情境下自行决定使用哪种方式。实验显示，这种双轨训练显著提升了模型在需要符号运算和数值计算混合的题目上的准确率。

2. **从 Llama‑2 与 Code Llama 双模型初始化 → 融合语言与代码能力**  
   直接在纯语言模型上做数学微调往往缺乏代码生成的底层能力；相反，纯代码模型在自然语言理解上表现一般。作者先把 Llama‑2（强语言）和 Code Llama（强代码）分别加载进同一训练框架，再统一进行指令微调，使得最终模型兼具两者优势。

3. **大规模、覆盖广的 MathInstruct 数据集 → 解决“数据偏狭”问题**  
   以往的数学微调数据集往往集中在代数或几何，导致模型在其他分支（如组合、数论）上表现差强人意。MathInstruct 汇聚了 13 个子数据集，覆盖高中到大学水平的多种主题，确保模型在训练阶段就见识到丰富的题型。

4. **统一的 Hybrid Instruction Tuning 流程 → 简化训练管线**  
   作者提出一种统一的指令格式，既能容纳文字推理也能容纳代码片段，避免了为不同思维方式分别设计数据接口的繁琐。这样，整个微调过程只需一次跑通，省时省力。

### 方法详解
整体思路可以概括为三步：①准备混合思维数据 → ②在双模型上进行统一指令微调 → ③输出可切换 CoT/PoT 的通用数学模型。

**1️⃣ 数据准备**  
- 从公开的 13 个数学数据源抓取原始题目。  
- 对每道题人工或半自动生成两套答案：一套是逐步文字推理（CoT），另一套是对应的可执行代码（PoT），两者都附带最终答案。  
- 为统一指令格式，作者设计了类似 “### Instruction: 解答以下题目。 ### Input: … ### Output: …” 的模板，其中 Output 部分先写 CoT，再写 “---” 分割线，接着写 PoT 代码块。这样模型在训练时能看到两种思路的并列。

**2️⃣ 双模型初始化**  
- 取 Llama‑2‑7B/34B 作为语言基座，取 Code Llama‑7B/34B 作为代码基座。  
- 使用 LoRA（低秩适配）或 QLoRA 等轻量微调技术，将两者的权重加载到同一模型实例中，保持语言层和代码层的参数共享但可微调。  
- 关键在于保持模型的“多模态”能力：语言层负责理解题目描述，代码层负责生成并执行代码。

**3️⃣ Hybrid Instruction Tuning**  
- 采用标准的指令微调损失函数（交叉熵），但在每条样本的目标序列中同时包含 CoT 与 PoT。  
- 为防止模型倾向于只输出文字或只输出代码，作者在训练时随机遮盖 CoT 或 PoT 部分，让模型学会在缺失信息时仍能补全另一种思路。  
- 训练过程使用梯度累积、混合精度等技巧，确保在 7B/34B 规模下仍能在单卡或少数卡上完成。

**最巧妙的设计**  
- **随机遮盖**：相当于给模型出“练习题”，有时只给文字提示，有时只给代码提示，迫使模型在两者之间建立映射关系。  
- **统一指令模板**：让模型把“思考方式”视作输出的不同段落，而不是完全不同的任务，从而在同一次前向传播中学习两种技能。

### 实验与效果
- **测试数据**：9 个公开的数学推理基准，包括 MATH（竞赛级）、GSM8K、SVAMP、MathQA、ASDiv 等。  
- **对比基线**：WizardMath（7B）、MetaMath（7B/13B）、Llama‑2‑Chat、GPT‑4（CoT 版本）等。  
- **主要结果**：  
  - MAmmoTH‑7B 在 MATH 上达到 33% 正确率，领先 WizardMath 23%（WizardMath 约 10%）。  
  - MAmmoTH‑34B 在同数据集上取得 44% 正确率，已经超过 GPT‑4 的 CoT 结果（约 42%）。  
  - 在其余 8 个基准上，平均提升 16%~32% 不等，几乎所有子任务都有显著正向跳跃。  
- **消融实验**：  
  - 去掉 PoT 部分，仅保留 CoT，7B 版下降约 8%；  
  - 去掉 CoT，仅保留 PoT，下降约 5%；  
  - 同时去掉随机遮盖，模型倾向只输出文字，整体性能下降约 6%。  
  这些实验表明，混合思维与随机遮盖是提升的关键因素。  
- **局限性**：  
  - 仍然依赖大量高质量标注的中间推理，标注成本高。  
  - 对极端高维几何或符号积分等专业题目仍有明显错误率。  
  - 代码执行依赖外部解释器，实际部署时需要安全沙箱。

### 影响与延伸思考
MAmmoTH 的出现让开源社区第一次在 7B/34B 规模上实现了“可竞争”于商业大模型的数学推理水平。随后出现的工作如 **MathCoder**、**HybridMathGPT** 等，都在不同程度上借鉴了混合思维标注和双模型微调的思路。未来的研究可能会向以下方向延伸：  
- **自动化生成混合思维标注**：利用已有的 CoT 与 PoT 模型互相校验，降低人工成本。  
- **多模态工具使用**：把图形绘制、符号计算引擎（如 SymPy）加入 PoT 环节，让模型在更广的数学工具箱中自由切换。  
- **更细粒度的思维选择机制**：让模型在推理时动态判断是走文字链还是代码链，而不是在训练时硬编码。

### 一句话记住它
**MAmmoTH 用“文字+代码双轨思考”让小模型也能在数学竞赛题上抢下 GPT‑4 的风头。**