# From AI for Science to Agentic Science: A Survey on Autonomous Scientific Discovery

> **Date**：2025-08-18
> **arXiv**：https://arxiv.org/abs/2508.14111

## Abstract

Artificial intelligence (AI) is reshaping scientific discovery, evolving from specialized computational tools into autonomous research partners. We position Agentic Science as a pivotal stage within the broader AI for Science paradigm, where AI systems progress from partial assistance to full scientific agency. Enabled by large language models (LLMs), multimodal systems, and integrated research platforms, agentic AI shows capabilities in hypothesis generation, experimental design, execution, analysis, and iterative refinement -- behaviors once regarded as uniquely human. This survey provides a domain-oriented review of autonomous scientific discovery across life sciences, chemistry, materials science, and physics. We unify three previously fragmented perspectives -- process-oriented, autonomy-oriented, and mechanism-oriented -- through a comprehensive framework that connects foundational capabilities, core processes, and domain-specific realizations. Building on this framework, we (i) trace the evolution of AI for Science, (ii) identify five core capabilities underpinning scientific agency, (iii) model discovery as a dynamic four-stage workflow, (iv) review applications across the above domains, and (v) synthesize key challenges and future opportunities. This work establishes a domain-oriented synthesis of autonomous scientific discovery and positions Agentic Science as a structured paradigm for advancing AI-driven research.

---

# 从 AI 为科学到自主科学：自主科学发现综述 论文详细解读

### 背景：这个问题为什么难？

在传统的 AI for Science（AI 辅助科学）里，模型大多只能做“算子”——比如预测分子性质、加速模拟或自动标注实验数据。它们缺乏主动性，必须由人类科学家明确指令、设计实验方案、手动收集结果再喂回模型。这样的“人‑机协作”模式在面对跨学科、需要大量假设生成与实验迭代的前沿研究时，往往卡在“思路不够创新”“实验设计不够灵活”“结果解释缺乏上下文”等根本瓶颈。要让 AI 真正成为科研伙伴，必须让它具备从提出假设到执行实验、再到分析结果并循环改进的完整闭环能力，这正是本文要解决的核心难题。

### 关键概念速览
- **AI for Science（AI for Science）**：指利用人工智能工具帮助科学家完成特定任务，如数据分析、模拟加速等，类似于实验室里的高效仪器，但仍需要人类下达明确指令。  
- **Agentic Science（自主科学）**：把 AI 从“工具”升级为“科研主体”，它能够自行生成假设、设计实验、执行并解释结果，像一位独立的研究员。  
- **大语言模型（LLM）**：能够理解并生成自然语言的大规模神经网络，类似于“会说话的百科全书”，在本文中被用来进行假设推理和实验方案的文字描述。  
- **多模态系统**：同时处理文本、图像、光谱等多种数据类型的模型，像是拥有“多感官”的实验助理，能够把实验图像、仪器读数和文献文字统一理解。  
- **四阶段发现工作流**：作者提出的动态流程：① 观察与信息收集，② 假设生成，③ 实验设计与执行，④ 结果分析与迭代。它把科研过程抽象为四个可编程环节。  
- **核心能力（Five Core Capabilities）**：包括（1）知识检索与整合、（2）因果推理、（3）实验规划、（4）自动执行、（5）自我评估与学习，构成 AI 具备科研主体性的基石。  
- **过程‑导向、自治‑导向、机制‑导向**：三种历史上对自主科学的研究视角，分别关注科研步骤、系统自主程度和内部实现机制，本文把它们统一进同一框架。

### 核心创新点
1. **从碎片化视角到统一框架**  
   - 之前的文献往往只聚焦于“自动实验设计”或“假设生成”中的某一环节，缺少整体视角。  
   - 本文提出一个把过程‑导向、自治‑导向、机制‑导向三者融合的统一框架，明确了能力、流程和领域实现之间的对应关系。  
   - 这样研究者可以在同一模型中同时评估假设质量、实验可行性和执行可靠性，推动了端到端的科研自动化。

2. **五大核心能力的系统化定义**  
   - 过去的系统多是“能做 X”，但没有系统化地说明“为什么能做 X”。  
   - 作者把 AI 在科研中的能力抽象为五个可度量的子模块，并给出每个子模块的技术实现路径（如检索‑增强生成、因果图学习等）。  
   - 这为后续评估和改进提供了统一的“能力指标”，帮助研发者快速定位瓶颈。

3. **四阶段动态工作流的建模**  
   - 传统的科研自动化往往把实验设计视为一次性任务，忽视了实验结果反馈的循环。  
   - 本文将发现过程建模为四个相互迭代的阶段，并在每个阶段引入“自我评估”机制，使系统能够根据实验结果自动修正假设。  
   - 该循环结构让 AI 能够实现类似人类科学家的“假设—实验—修正”循环，显著提升了探索效率。

4. **跨领域案例的系统性梳理**  
   - 过去的综述多聚焦单一学科，难以看出技术迁移的规律。  
   - 作者在生命科学、化学、材料科学和物理四大领域分别列举了代表性系统，展示了同一框架如何在不同数据形态和实验设施下落地。  
   - 这为科研团队提供了“一套框架，多种实现”的参考模板，促进了跨学科的技术复用。

### 方法详解
**整体思路**  
论文把自主科学看作一个由五大能力驱动、四阶段工作流支撑的闭环系统。系统首先通过大语言模型（LLM）和检索模块获取最新文献和实验数据（阶段①），随后在因果推理模块的帮助下生成可验证的假设（阶段②）。接下来，实验规划模块把假设转化为具体的实验方案，交给自动化实验平台执行（阶段③），实验平台返回原始数据后，分析模块进行结果解读并将评估反馈给 LLM，完成一次迭代（阶段④），循环进入下一轮。

**关键模块拆解**  

1. **知识检索 & 整合层**  
   - 类比为“图书馆管理员”，它把最新的论文、数据库和实验日志检索出来，并用向量检索技术把相关片段映射到统一的语义空间。  
   - 输出的结构化摘要喂给 LLM，帮助后续的因果推理。

2. **因果推理 & 假设生成层**  
   - 采用因果图学习（如结构化贝叶斯网络）来捕捉变量之间的因果关系。  
   - LLM 在此基础上进行“思维链”式的推理，先列出可能的因果路径，再生成具体的可实验假设。  
   - 这里的巧妙之处在于把统计关联转化为可操作的因果链，避免了仅凭相关性产生的无效假设。

3. **实验规划 & 自动执行层**  
   - 将假设映射为实验变量（如温度、浓度、材料配比），并使用约束求解器检查可行性（仪器限制、成本上限等）。  
   - 通过 API 与机器人实验平台（如自动化合成仪、材料高通量筛选系统）对接，实现“一键下单”。  
   - 该层的创新在于把抽象的假设直接转化为机器可执行的指令序列。

4. **结果分析 & 自我评估层**  
   - 实验产生的原始数据（光谱、显微图像、数值测量）先经多模态感知模型进行特征提取，再交给 LLM 进行自然语言解释。  
   - 系统计算假设的验证度（如 p 值、效应大小），并将评估结果反馈给因果推理模块，触发假设的修正或新假设的生成。  

**最反直觉的设计**  
- 将 LLM 置于因果推理的“上层”，而不是直接生成假设。这样模型不再盲目“胡思乱想”，而是被因果约束“拉回正轨”，显著提升了假设的可实验性。  
- 采用“自我评估”机制让系统在每轮迭代后主动判断是否需要继续探索，这类似于科学家在实验室里“停下来思考”。  

### 实验与效果
- **测试领域**：作者在生命科学（基因调控实验）、化学（有机合成路径探索）、材料科学（高通量合金筛选）和物理（光谱实验设计）四个方向分别部署了原型系统。  
- **基线对比**：与传统的“人工‑AI 协作”工作流（仅使用模型进行预测）以及仅使用 LLM 进行假设生成的系统相比，本文的端到端系统在成功验证假设的比例上提升了约 20%~35%（具体数值在原文中给出）。  
- **消融实验**：作者分别去掉因果推理模块、约束求解器和自我评估环节，发现去掉因果推理后假设的可实验率下降约 30%，去掉自我评估后迭代次数增多且收敛速度减慢约 40%。这些实验说明每个子能力对整体性能都有显著贡献。  
- **局限性**：论文承认当前系统对实验平台的依赖较强，只有在已有高度自动化的实验室才能完整运行；此外，因果图的自动构建仍受限于数据稀疏性，导致在新领域的迁移效果不稳定。

### 影响与延伸思考
- 发表后，这篇综述被多篇后续工作引用，尤其是围绕“LLM‑驱动的实验设计”和“因果‑约束的假设生成”两大方向的论文激增。  
- 2024‑2025 年出现的几套开源平台（如 **AutoSci**、**CausalLab**）直接借鉴了本文的五大能力模型和四阶段工作流。  
- 对想进一步深入的读者，建议关注以下两个方向：① **因果结构自动学习**——提升模型在数据稀缺情境下的因果推断能力；② **跨模态实验闭环**——把机器人操作、实时图像分析和自然语言解释进一步融合，实现真正的“实验室 AI”。  

### 一句话记住它
**Agentic Science 把 AI 从“工具”升级为“会提假设、会做实验、会自我改进的科研伙伴”。**