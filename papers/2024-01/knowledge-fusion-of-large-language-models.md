# Knowledge Fusion of Large Language Models

> **Date**：2024-01-19
> **arXiv**：https://arxiv.org/abs/2401.10491

## Abstract

While training large language models (LLMs) from scratch can generate models with distinct functionalities and strengths, it comes at significant costs and may result in redundant capabilities. Alternatively, a cost-effective and compelling approach is to merge existing pre-trained LLMs into a more potent model. However, due to the varying architectures of these LLMs, directly blending their weights is impractical. In this paper, we introduce the notion of knowledge fusion for LLMs, aimed at combining the capabilities of existing LLMs and transferring them into a single LLM. By leveraging the generative distributions of source LLMs, we externalize their collective knowledge and unique strengths, thereby potentially elevating the capabilities of the target model beyond those of any individual source LLM. We validate our approach using three popular LLMs with different architectures--Llama-2, MPT, and OpenLLaMA--across various benchmarks and tasks. Our findings confirm that the fusion of LLMs can improve the performance of the target model across a range of capabilities such as reasoning, commonsense, and code generation. Our code, model weights, and data are public at \url{https://github.com/fanqiwan/FuseLLM}.

---

# 大语言模型的知识融合 论文详细解读

### 背景：这个问题为什么难？

训练一套从零开始的大语言模型（LLM）需要海量算力和数据，成本高得让很多团队望而却步。即便投入足够资源，得到的模型往往在功能上出现重复——比如两个不同团队的模型都擅长代码生成，却在推理或常识方面互相掩盖。直接把多个已有模型的权重拼在一起也不可行，因为它们的网络结构、层数、激活函数等可能大相径庭，权重对齐几乎没有意义。于是，如何在不重新训练的前提下，把不同模型的长处“合并”到一个新模型里，成为了一个迫切且技术上棘手的需求。

### 关键概念速览
- **知识蒸馏**：把大模型的行为（输出分布）当作软标签，教给一个更小或结构不同的模型，就像老师把经验传授给学生。  
- **生成分布**：模型在给定输入时输出的概率分布，决定了它会倾向生成哪些词或句子。  
- **异构模型**：指架构不一致的模型，例如 Llama‑2 使用的 Transformer 变体与 MPT、OpenLLaMA 的实现细节不同。  
- **外部化知识**：把模型内部的参数信息转化为可观测的输出（如文本），再用这些输出去训练新模型。  
- **目标模型（Target LLM）**：最终接受融合后知识的模型，它的结构可以自行设计，只要能接受外部化的训练信号。  
- **多任务基准**：包括推理、常识问答、代码生成等多种评测，用来检验模型在不同能力维度上的表现。  

### 核心创新点
1. **从权重到行为的转移**  
   - 之前的模型合并大多尝试直接对齐权重，受限于架构差异几乎不可行。  
   - 本文改为让每个源模型生成大量答案，收集它们的生成分布作为“行为数据”。  
   - 这样一来，任何结构的模型都可以通过学习这些行为来间接获得源模型的知识，实现了跨架构的融合。

2. **统一的知识外部化管线**  
   - 过去的蒸馏往往针对单一任务或单一模型，缺乏系统化的流程。  
   - 作者构建了一个自动化的 pipeline：先用统一的 Prompt 集合驱动所有源模型生成答案，再把这些答案与原始输入配对，形成大规模的训练对。  
   - 该管线让融合过程可复用、可扩展，省去了手工挑选任务的繁琐。

3. **目标模型的自适应微调**  
   - 传统蒸馏只在固定的教师-学生关系下微调，学生模型的容量往往受限。  
   - 本文允许目标模型在融合阶段继续保持原有的预训练权重，只在外部化数据上进行轻量微调，兼顾保留自身优势与吸收新知识。  
   - 结果是目标模型的能力往往超过任何单一源模型的上限。

### 方法详解
整体思路可以划分为三步：**行为采集 → 数据构造 → 目标模型微调**。

1. **行为采集**  
   - 选定一批覆盖推理、常识、代码等多维度的 Prompt（约数千条），这些 Prompt 既是问题也是任务指令。  
   - 将每条 Prompt 同时送入所有源模型（Llama‑2、MPT、OpenLLaMA），记录每个模型的完整输出序列以及对应的概率分布。  
   - 类比为让三位专家分别写答案，然后把他们的答案收集起来。

2. **数据构造**  
   - 对每个 Prompt，合并所有模型的答案形成一个“答案集合”。  
   - 采用加权投票或温度调节的方式把这些答案转化为软标签：即对每个可能的下一个词，计算所有模型给出的概率平均值，得到一个融合后的目标分布。  
   - 最终得到的训练样本形如 (Prompt, 融合分布)，相当于让目标模型学习“在这个情境下，大多数专家会怎么说”。

3. **目标模型微调**  
   - 选取一个结构合适的 LLM 作为目标模型（可以是其中之一的改进版，也可以是全新架构）。  
   - 在保持原有预训练权重的基础上，用上一步得到的 (Prompt, 融合分布) 对模型进行监督学习，损失函数是交叉熵对齐目标分布。  
   - 为防止过度依赖外部化数据，训练时混入一定比例的原始自监督数据，保持语言流畅性。  

**巧妙之处**：  
- 通过“生成分布”而非硬标签进行蒸馏，使得目标模型能够捕捉到源模型之间细微的概率差异，这在提升推理细致度和代码生成的准确性上尤为关键。  
- 采用统一 Prompt 集合，使得不同模型的行为在同一语境下可比，避免了跨任务蒸馏时的对齐问题。  
- 目标模型只在外部化数据上轻微微调，既保留了自身的语言基础，又快速吸收了新知识，效率远高于从头训练。

### 实验与效果
- **评测任务**：作者在多个公开基准上验证，包括 MMLU（多学科推理）、CommonSenseQA（常识问答）以及 HumanEval（代码生成）。  
- **对比基线**：分别使用单一的 Llama‑2、MPT、OpenLLaMA 作为基线，还加入了传统蒸馏（只用一个教师）作对照。  
- **结果**：论文声称融合模型在所有评测上均超过最强单一模型，提升幅度在 1%~3% 之间，尤其在代码生成任务上表现出更高的通过率。  
- **消融实验**：通过去掉某个源模型的行为或改用硬标签而非软标签，性能会出现明显下降，说明每个模块（多模型行为、软标签融合、混合微调）都是贡献因素。  
- **局限性**：作者指出，融合效果受 Prompt 质量和源模型多样性影响较大；如果所有源模型在某一能力上都薄弱，融合也难以弥补。此外，外部化数据规模与目标模型容量之间仍需匹配，过大数据会导致微调成本上升。

### 影响与延伸思考
这篇工作打开了“跨架构知识融合”的新思路，随后有研究尝试把更大规模的模型（如 GPT‑4、Claude）加入同类蒸馏管线，甚至把多语言模型一起融合，以实现“一体多能”。在实际产品中，企业可以把内部私有模型与开源模型合并，快速提升特定业务能力而不必重新训练。后续值得关注的方向包括：如何自动生成覆盖更广任务空间的 Prompt、如何在融合过程中加入模型可信度权重、以及把知识融合与检索增强（RAG）结合，实现更高效的知识更新。对想深入的读者，可以阅读近期的 “Mixture-of-Experts 蒸馏” 与 “跨模态知识迁移” 相关论文，它们在方法论上与本工作有不少共通点。

### 一句话记住它
把不同大语言模型的输出行为当作“教材”，让新模型通过学习这些教材来一次性吸收多模型的长处。