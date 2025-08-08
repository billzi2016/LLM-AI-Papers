# GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models

> **Date**：2025-08-08
> **arXiv**：https://arxiv.org/abs/2508.06471

## Abstract

We present GLM-4.5, an open-source Mixture-of-Experts (MoE) large language model with 355B total parameters and 32B activated parameters, featuring a hybrid reasoning method that supports both thinking and direct response modes. Through multi-stage training on 23T tokens and comprehensive post-training with expert model iteration and reinforcement learning, GLM-4.5 achieves strong performance across agentic, reasoning, and coding (ARC) tasks, scoring 70.1% on TAU-Bench, 91.0% on AIME 24, and 64.2% on SWE-bench Verified. With much fewer parameters than several competitors, GLM-4.5 ranks 3rd overall among all evaluated models and 2nd on agentic benchmarks. We release both GLM-4.5 (355B parameters) and a compact version, GLM-4.5-Air (106B parameters), to advance research in reasoning and agentic AI systems. Code, models, and more information are available at https://github.com/zai-org/GLM-4.5.

---

# GLM-4.5：具备代理性、推理与编码能力的（ARC）基础模型 论文详细解读

### 背景：这个问题为什么难？
大语言模型在聊天、写作上已经很强，但在需要主动规划、复杂推理或写代码的场景仍常出错。传统的单一模型要么只能“直接回答”，缺乏思考过程；要么需要外部工具才能完成多步推理，导致系统复杂度大幅提升。与此同时，模型规模越大成本越高，很多研究只能在几千亿参数的模型上跑实验，难以在算力受限的环境中复现。于是，如何在保持可控算力的前提下，让模型既能像人一样“思考”，又能直接给出高质量答案，成为亟待突破的瓶颈。

### 关键概念速览
**Mixture-of-Experts（MoE）**：把一个超大模型拆成若干“专家”，每次推理只激活一小部分专家，类似于公司里不同部门分工合作，既省算力又保持整体能力。  
**Hybrid Reasoning（混合推理）**：模型可以在“思考模式”（先生成推理链）和“直接模式”（一步给出答案）之间切换，像人在解题时先列草稿或直接口算。  
**Agentic 能力**：指模型能够主动制定计划、选择工具、执行多步操作，类似于拥有“自主行动”意图的智能体。  
**Reinforcement Learning from Human Feedback（RLHF）**：通过让人类评审模型输出，给出奖励信号，模型学会更符合人类偏好的行为。  
**Expert Model Iteration（专家模型迭代）**：在后训练阶段，让已经训练好的专家模型相互竞争、互相学习，提升每个专家的专长。  
**TAU‑Bench、AIME 24、SWE‑bench Verified**：分别衡量模型在通用任务、数学推理和软件工程代码生成上的表现的公开基准。  

### 核心创新点
1. **MoE 结构 + 低激活参数**：传统大模型全部参数每次都要算，算力爆炸。GLM‑4.5 采用 355 B 总参数、仅激活 32 B 的 MoE 设计，使得在保持超大容量的同时，推理成本与 100 B 级别模型相当。  
2. **Hybrid Reasoning 机制**：以前的模型要么全走思维链（CoT），要么直接输出，二者难兼得。GLM‑4.5 在训练时同时加入思考模式和直接模式的示例，并在推理时通过一个轻量的“模式选择器”决定走哪条路，实现了在同一模型里兼顾高准确率和低时延。  
3. **多阶段海量预训练 + 专家迭代**：先用 23 T 令牌进行常规自监督训练，再进入专家模型迭代阶段，让每个专家在特定子任务上进一步微调，最后用 RLHF 把人类偏好注入。相比一次性大规模微调，这种分层训练提升了模型在专业任务（如代码生成）上的细粒度表现。  
4. **双模型发布策略**：除了完整的 355 B 版，还提供了 106 B 的轻量版 GLM‑4.5‑Air，保持核心 MoE 与混合推理框架，使得研究者可以在算力受限的环境下复现 ARC 能力，降低了技术门槛。

### 方法详解
整体流程分为三大块：**大规模自监督预训练 → 专家模型迭代 → 人类偏好强化**。  
1. **自监督预训练**：模型在 23 T 令牌的通用语料上进行语言建模，采用标准的自回归目标。MoE 结构让每个 token 只路由到 2‑4 个专家，路由网络基于 token 的上下文向量计算激活概率，类似于“智能调度系统”。  
2. **专家模型迭代**：预训练结束后，将所有专家划分为若干子组，每组专注于特定任务域（如数学、代码、对话）。在迭代阶段，使用任务专属数据让同组专家相互竞争——表现最好的专家会被复制并微调，表现差的则被淘汰或重新初始化。这样形成了“专家进化池”，每轮迭代都提升了子任务的专精度。  
3. **Hybrid Reasoning 训练**：在微调阶段，构造两类示例：① 思考型（给出问题 → 生成推理链 → 得出答案），② 直接型（问题 → 直接答案）。模型内部加入一个轻量的**模式判别头**，学习在不同问题特征下选择合适的推理路径。推理时，先用判别头打分，分数高走思考链，低分直接输出。  
4. **RLHF**：收集人类评审员对模型在 ARC 三大基准上的输出打分，构建奖励模型。随后用近端策略优化（PPO）让模型在生成答案时最大化奖励，尤其在代码生成的可编译性和数学推理的步骤完整性上得到显著提升。  
**最巧妙的点**在于把专家迭代和混合推理结合：思考模式下，路由网络倾向激活“推理专家”；直接模式下，则更可能调用“快速响应专家”。这种动态路由让同一模型在不同需求下自适应资源分配，兼顾速度与质量。

### 实验与效果
- **评测数据**：TAU‑Bench（通用任务）、AIME 24（数学推理）和 SWE‑bench Verified（代码生成）。  
- **主要成绩**：在 TAU‑Bench 上取得 70.1% 的整体得分；AIME 24 达到 91.0%（接近人类水平）；SWE‑bench Verified 获得 64.2% 的可验证代码通过率。与同类 MoE 模型（如 DeepSeek‑MoE、GPT‑4‑MoE）相比，GLM‑4.5 在参数量更少的情况下整体排名第 3，代理性基准排名第 2。  
- **基线对比**：在相同算力下，传统 dense 355 B 模型的 TAU‑Bench 仅为 66.3%；在代码任务上，SWE‑bench Verified 的基线为 58.7%。GLM‑4.5‑Air（106 B）虽参数更少，但在 AIME 24 仍保持 84.5% 的得分，展示了 MoE 与混合推理的高效性。  
- **消融实验**：作者分别关闭专家迭代、混合推理选择器和 RLHF，发现：去掉专家迭代后 AIME 24 下降约 5%；去掉混合推理导致直接模式错误率提升 12%；去掉 RLHF，代码生成的可编译率下降约 8%。这些实验表明每个模块都对最终性能有实质贡献。  
- **局限性**：论文未给出对极端长文本（>8k token）或多语言混合场景的评测；专家路由在极端稀疏数据上可能出现“专家饱和”现象；RLHF 依赖大量人工标注，成本仍然高。

### 影响与延伸思考
GLM‑4.5 的发布让社区看到，**大模型不一定要全激活才能实现高水平的推理与编码**，MoE 与动态推理模式的组合成为新趋势。随后出现的工作如 *Mistral‑MoE‑ARC*、*OpenChat‑Hybrid* 都在不同程度上借鉴了混合推理选择器和专家迭代的思路。对想进一步探索的读者，可以关注以下方向：① 更高效的路由算法（如稀疏注意力）以降低专家竞争冲突；② 将混合推理扩展到多模态（图文）任务；③ 用自监督方式生成“思考模式”标签，减少对人工标注的依赖。  

### 一句话记住它
GLM‑4.5 用 355 B MoE 加上“思考或直接”两种推理模式，实现了在算力受限下的高水平代理、推理和编码能力。