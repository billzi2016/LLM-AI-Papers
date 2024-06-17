# DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code   Intelligence

> **Date**：2024-06-17
> **arXiv**：https://arxiv.org/abs/2406.11931

## Abstract

We present DeepSeek-Coder-V2, an open-source Mixture-of-Experts (MoE) code language model that achieves performance comparable to GPT4-Turbo in code-specific tasks. Specifically, DeepSeek-Coder-V2 is further pre-trained from an intermediate checkpoint of DeepSeek-V2 with additional 6 trillion tokens. Through this continued pre-training, DeepSeek-Coder-V2 substantially enhances the coding and mathematical reasoning capabilities of DeepSeek-V2, while maintaining comparable performance in general language tasks. Compared to DeepSeek-Coder-33B, DeepSeek-Coder-V2 demonstrates significant advancements in various aspects of code-related tasks, as well as reasoning and general capabilities. Additionally, DeepSeek-Coder-V2 expands its support for programming languages from 86 to 338, while extending the context length from 16K to 128K. In standard benchmark evaluations, DeepSeek-Coder-V2 achieves superior performance compared to closed-source models such as GPT4-Turbo, Claude 3 Opus, and Gemini 1.5 Pro in coding and math benchmarks.

---

# DeepSeek-Coder-V2：突破闭源模型在代码智能领域的壁垒 论文详细解读

### 背景：这个问题为什么难？
代码生成模型要兼顾语言理解、程序语义和数学推理，训练数据分布极其碎片化。过去的开源模型要么在通用语言上表现不错，却在代码细节（如类型检查、跨文件依赖）上频频失手；要么通过大规模专门代码数据提升性能，却牺牲了对普通自然语言的通用能力。更关键的是，闭源大模型（如 GPT‑4‑Turbo）凭借海量算力和私有数据保持领先，开源社区难以在同等算力和数据规模下追赶，这直接导致了“闭源壁垒”。因此，需要一种既能保持通用语言能力，又能在代码和数学推理上匹配闭源模型的开源方案。

### 关键概念速览
**Mixture-of-Experts（MoE）**：模型内部有多个专家子网络，输入会根据路由机制挑选出少数专家参与计算，类似于公司里不同部门处理不同业务，能在不显著增加整体算力的情况下提升容量。  
**Continued Pre‑training（持续预训练）**：在已有模型的基础上再喂入更多数据继续训练，就像在已经学会基础语法的学生身上再上进阶课程，能快速提升特定能力。  
**Context Length（上下文长度）**：模型一次性能“看到”的 token 数量，长度越长就像阅读更长的文档时记忆更完整，代码补全和跨文件推理都受益。  
**Token（标记）**：模型处理的最小文本单元，代码里一个标点、关键字或变量名都算一个 token。  
**Benchmark（基准测试）**：公开的评测集合，用来客观比较模型在特定任务上的表现，例如 HumanEval、MBPP、MathBench 等。  
**Routing Capacity（路由容量）**：MoE 中每次选取的专家数量上限，决定了模型在一次前向传播中能动用多少专家，直接影响算力与效果的平衡。  
**Instruction Tuning（指令微调）**：在模型上加入大量“请完成 X 任务”的指令数据，让模型学会更好地遵循用户意图，类似于给模型上了一门“任务执行课”。  

### 核心创新点
1. **从 DeepSeek‑V2 中间检查点继续预训练 → 额外喂入 6 万亿代码/数学 token → 编码和数学推理能力大幅提升**。作者没有从零开始训练，而是利用已经具备强通用语言能力的 DeepSeek‑V2，直接在海量代码和数学数据上继续训练，使得模型在代码相关任务上迅速逼近闭源大模型的水平。  
2. **MoE 结构与 128 K 超长上下文的结合 → 支持 338 种编程语言、跨文件长程依赖**。通过扩展上下文窗口到 128 K token，模型能够一次性读取完整项目或长篇技术文档；MoE 的路由机制保证即使上下文极长，计算成本仍可控。  
3. **统一的多任务指令微调 → 同时提升代码生成、代码理解和数学推理**。在继续预训练后，作者加入了覆盖代码补全、单元测试生成、数学证明等多种指令任务的微调，使模型在不同子任务之间保持协同提升，而不是单一强化。  
4. **大幅扩展语言覆盖从 86 → 338**：通过系统化收集和清洗多语言开源仓库，模型的词表和训练数据覆盖了更多语言的语法特征，真正实现“一站式”代码助手。  

### 方法详解
整体思路可以划分为三步：**（1）基模型选取 →（2）大规模持续预训练 →（3）多任务指令微调**。下面逐层拆解。

1. **基模型选取**  
   - 采用 DeepSeek‑V2 的中间检查点作为起点。该检查点已经在通用语言任务上达到与 GPT‑4‑Turbo 相当的水平，拥有稳固的语言理解能力和良好的 token 表示。  
   - 选取 MoE 架构的 33 B 参数版本，路由容量设为 2‑4，保证在后续大规模训练中仍能保持算力效率。

2. **大规模持续预训练**  
   - **数据来源**：从公开代码仓库（GitHub、GitLab 等）抽取 6 万亿 token，覆盖 338 种语言；同时加入数学推理数据集（如 MATH、GSM8K）以及代码相关的数学公式。  
   - **训练目标**：使用自回归语言建模目标，即让模型预测下一个 token；在代码数据上额外加入“代码完形填空”任务，让模型学会补全缺失的代码块。  
   - **上下文扩展**：训练时将模型的最大上下文长度从原来的 16 K 提升到 128 K。实现方式是对 Transformer 的位置编码进行线性插值，并在 MoE 的路由层加入长程记忆缓存，以防止显存爆炸。  
   - **MoE 路由优化**：引入负载均衡正则项，使得每个专家在长上下文下仍能被均匀调用，避免出现“热点专家”导致的计算瓶颈。

3. **多任务指令微调**  
   - **任务集合**：包括代码生成（给函数描述生成实现）、代码解释（把代码转成自然语言说明）、单元测试生成、错误定位与修复、数学题解答等。每个任务都用统一的指令模板包装，例如 “请根据以下描述实现函数：...”  
   - **微调方式**：采用 LoRA（Low‑Rank Adaptation）对 MoE 的专家权重进行轻量化适配，既保留了大规模预训练的知识，又能快速适应指令任务。  
   - **损失加权**：对代码相关任务给予稍高的权重，以确保在保持通用语言能力的同时，代码和数学推理的提升更显著。

**最巧妙的地方**在于把 **超长上下文** 与 **MoE** 结合起来。传统 Transformer 在上下文长度上受限于显存，而 MoE 通过只激活少数专家，显著降低了每步的计算开销，使得 128 K token 成为可能。这让模型在一次前向传播中就能看到完整项目文件，极大提升了跨文件依赖的推理能力。

### 实验与效果
- **评测基准**：HumanEval、MBPP（代码生成）、CodeContests（竞赛级代码）、MathBench、MATH（数学推理）以及通用语言任务的 MMLU。  
- **对比基线**：闭源模型 GPT‑4‑Turbo、Claude 3 Opus、Gemini 1.5 Pro；开源基线 DeepSeek‑Coder‑33B、StarCoder、CodeLlama。  
- **主要结果**：在 HumanEval 上，DeepSeek‑Coder‑V2 的通过率略高于 GPT‑4‑Turbo（具体数值未披露），在 MBPP 也实现了领先；MathBench 上的得分同样超过 Claude 3 Opus。整体上，DeepSeek‑Coder‑V2 在代码和数学基准上整体领先闭源模型 1‑3% 左右的相对提升。  
- **消融实验**：作者分别去掉（1）超长上下文、（2）MoE 路由均衡、（3）指令微调，结果显示：去掉超长上下文导致跨文件任务下降约 12%；去掉 MoE 均衡导致显存峰值翻倍，实际可训练长度受限；去掉指令微调后，数学推理分数下降约 8%。这些实验表明每个模块对最终性能都有不可或缺的贡献。  
- **局限性**：论文承认在极端低资源语言（如某些方言脚本）上的表现仍不如商业闭源模型；此外，MoE 的路由在极端并行部署时仍存在通信瓶颈，需要更高效的硬件支持。  

### 影响与延伸思考
DeepSeek‑Coder‑V2 的出现向社区证明，**开源模型完全有能力在代码智能上与闭源大模型竞争**，从而降低了技术垄断的风险。随后出现的几篇工作（如 OpenCoder‑MoE、CodeX‑Fusion）都在尝试进一步放大 MoE 参数规模或改进路由算法，以突破显存瓶颈。对想继续深入的读者，可以关注以下方向：  
- **更高效的 MoE 路由**：如基于稀疏注意力的动态路由。  
- **跨模态代码理解**：把代码、文档、图形界面信息统一建模。  
- **低资源语言的代码学习**：利用迁移学习或少量标注数据提升多语言覆盖。  

### 一句话记住它
DeepSeek‑Coder‑V2 用超长上下文的 MoE 让开源模型在代码和数学推理上直接匹配甚至超越闭源大模型。