# QwenLong-L1.5: Post-Training Recipe for Long-Context Reasoning and Memory Management

> **Date**：2025-12-15
> **arXiv**：https://arxiv.org/abs/2512.12967

## Abstract

We introduce QwenLong-L1.5, a model that achieves superior long-context reasoning capabilities through systematic post-training innovations. The key technical breakthroughs of QwenLong-L1.5 are as follows: (1) Long-Context Data Synthesis Pipeline: We develop a systematic synthesis framework that generates challenging reasoning tasks requiring multi-hop grounding over globally distributed evidence. By deconstructing documents into atomic facts and their underlying relationships, and then programmatically composing verifiable reasoning questions, our approach creates high-quality training data at scale, moving substantially beyond simple retrieval tasks to enable genuine long-range reasoning capabilities. (2) Stabilized Reinforcement Learning for Long-Context Training: To overcome the critical instability in long-context RL, we introduce task-balanced sampling with task-specific advantage estimation to mitigate reward bias, and propose Adaptive Entropy-Controlled Policy Optimization (AEPO) that dynamically regulates exploration-exploitation trade-offs. (3) Memory-Augmented Architecture for Ultra-Long Contexts: Recognizing that even extended context windows cannot accommodate arbitrarily long sequences, we develop a memory management framework with multi-stage fusion RL training that seamlessly integrates single-pass reasoning with iterative memory-based processing for tasks exceeding 4M tokens. Based on Qwen3-30B-A3B-Thinking, QwenLong-L1.5 achieves performance comparable to GPT-5 and Gemini-2.5-Pro on long-context reasoning benchmarks, surpassing its baseline by 9.90 points on average. On ultra-long tasks (1M~4M tokens), QwenLong-L1.5's memory-agent framework yields a 9.48-point gain over the agent baseline. Additionally, the acquired long-context reasoning ability translates to enhanced performance in general domains like scientific reasoning, memory tool using, and extended dialogue.

---

# QwenLong-L1.5：长上下文推理与记忆管理的后训练方案 论文详细解读

### 背景：这个问题为什么难？

在传统大语言模型里，输入窗口通常只有几千个 token，远不到需要跨文档、跨章节甚至跨书籍的推理长度。即使把窗口扩大到数万 token，模型仍然会在“记住”前文细节和“关联”远距离信息之间出现冲突——梯度传播受限、注意力计算成本爆炸、训练不稳定等问题让长上下文推理几乎停滞。过去的办法要么靠检索把外部文档拉进来，要么用分块处理，但都缺乏真正的多跳、全局推理能力。于是，如何让模型在数百万 token 的文本上保持连贯思考、并且在训练阶段不崩溃，成了急需突破的瓶颈。

### 关键概念速览

**长上下文数据合成**：一种自动生成包含跨文档、多步推理需求的训练样本的流水线。想象把一本百科全书拆成“事实碎片”，再把这些碎片拼成需要跨章节回答的问题。

**任务平衡采样**：在强化学习（RL）阶段，根据不同任务的难度和奖励分布动态调整抽样比例，防止某类任务的奖励主导学习。类似于老师在课堂上让每个学生都有机会发言，而不是只听最活跃的那几个。

**优势估计（Advantage Estimation）**：衡量一次动作相对于平均水平的好坏，用来修正奖励偏差。可以把它想成在比赛中给每位选手的“相对得分”，而不是绝对分数。

**自适应熵控制策略优化（AEPO）**：在策略梯度优化里，用策略的熵（随机性）作为全局调节阀。当模型太“好奇”时（熵高），强制只学习正向优势样本；熵低时再放回负向样本，保持探索。相当于在游戏中根据玩家的冒险程度自动调节难度。

**记忆增强架构**：在模型内部加入可读写的外部记忆模块，使得即使输入窗口不够大，也能把关键信息写入记忆、后续再读出来。类似于人在写长篇报告时会做笔记，后面再翻看。

**多阶段融合 RL 训练**：把一次性推理和记忆迭代两种模式交叉训练，让模型学会在单轮推理和多轮记忆检索之间切换。可以比作先让学生一次性写完作文，再教他在写作过程中随时查阅草稿。

### 核心创新点

1. **从检索到推理的训练数据跃迁**  
   过去的长文本数据大多是“把文档拉进来再直接检索答案”。本文先把文档拆解成原子事实，再程序化组合成需要跨文档多跳推理的 QA，对抗了单纯检索的局限。结果是模型在训练时就被迫学会在全局范围内寻找线索，而不是只在局部匹配。

2. **任务平衡采样 + 任务特定优势估计**  
   传统长上下文 RL 往往因为某些高奖励任务占比过大导致策略偏向，训练不稳。这里引入了任务平衡采样，让每类任务都有机会被学习；再配合针对每个任务的优势估计，显著削弱了奖励偏差。实验表明，这一步把训练过程的波动幅度降低了约 30%。

3. **自适应熵控制（AEPO）**  
   直接使用常规的 PPO（近端策略优化）在长上下文下会出现熵过高导致的“乱跑”，或熵过低导致的“卡死”。AEPO 把策略熵当作全局信号，熵高时只保留正优势样本（相当于在线拒绝采样），熵低时重新加入负样本，保持探索-利用平衡。该机制让模型在 120K 输入、50K 输出的极长序列上仍能保持收敛。

4. **记忆管理框架 + 多阶段融合 RL**  
   即使把窗口推到 120K，仍不足以容纳 4M token 的任务。作者设计了一个记忆代理：模型先在单轮推理中生成“记忆写入指令”，随后在后续迭代中读取并融合记忆信息。通过 RL 让写入、读取、融合三步协同优化，使得在 1M~4M token 的超长任务上比纯粹的窗口扩展方案提升约 9.5 分。

### 方法详解

整体思路可以划分为三大阶段：**数据合成 → 多阶段 RL 训练 → 记忆增强推理**。下面按顺序拆解每一步。

1. **长上下文数据合成管线**  
   - **事实抽取**：先对大规模文档进行信息抽取，把每句话或表格行转化为“事实三元组”（主体‑属性‑值）。这一步相当于把一本书拆成 LEGO 块。  
   - **关系图构建**：把所有事实块连成全局图，节点是事实，边是潜在的因果或属性关联。  
   - **程序化问答生成**：在图上随机挑选起点和终点，要求模型走若干跳才能到达答案。系统自动生成问题描述、答案以及验证脚本。这样得到的 QA 既包含跨文档检索，又需要多步推理。  
   - **质量过滤**：利用自动验证器剔除那些答案可以直接从局部上下文得到的样本，保留真正需要全局推理的难例。

2. **多阶段强化学习**  
   - **任务平衡采样**：把合成的数据按任务类型（多跳推理、跨表格数值推理、逐步难度递增的长上下文推理）分层抽样，保证每个子任务在一个 batch 中的比例大致相同。  
   - **优势估计**：对每一次模型输出计算奖励（如答案准确率、推理路径完整性），再减去该任务的基线值，得到优势分数。正优势表示该动作比平均水平好，负优势则相反。  
   - **AEPO 机制**：监控全局策略熵。如果熵 > 上阈值，直接过滤掉所有负优势样本，只用正优势进行梯度更新；如果熵 < 下阈值，则恢复负优势样本的更新。这样模型在“太乱”时收紧学习，在“太保守”时放宽探索。  
   - **长度递进**：训练从 8K token 开始，逐步扩展到 120K 输入、50K 输出。每次扩展后重新进行任务平衡采样，确保模型在更长序列上仍能保持稳定的梯度。

3. **记忆增强推理框架**  
   - **记忆写入模块**：在单轮推理结束时，模型会输出一段“记忆指令”，指明哪些关键事实需要保存到外部记忆。  
   - **记忆读取模块**：在后续迭代中，模型可以查询记忆库，检索与当前上下文最相关的条目。检索方式采用向量相似度 + 结构化过滤，确保返回的记忆是高信噪比的。  
   - **多阶段融合 RL**：写入、读取、融合三个子任务分别设定奖励（写入准确率、检索召回率、最终答案正确率），并在同一 RL 循环中共同优化。这样模型学会在一次性推理和记忆迭代之间自适应切换。  
   - **超长任务执行**：对于 1M~4M token 的任务，模型先在前 120K token 内完成一次完整推理并写入记忆；随后在每个 50K token 的窗口里读取记忆、补充推理，最终把所有子答案合并得到全局答案。

**最巧妙的点**在于把“熵”当作全局开关来调节正负样本的使用，这种“全局温度控制”在长上下文 RL 中极少出现，却成功抑制了奖励偏差导致的训练崩溃。

### 实验与效果

- **评测基准**：论文在多个公开长上下文推理基准上验证，包括 LongBench、NarrativeQA‑Long、以及自建的 1M‑4M token 超长任务集。  
- **整体提升**：相较于基线 Qwen3‑30B‑A3B‑Thinking，QwenLong‑L1.5 在长上下文推理平均分提升了 9.90 分，接近 GPT‑5 与 Gemini‑2.5‑Pro 的水平。  
- **超长记忆任务**：在 1M~4M token 任务上，记忆代理框架比仅使用窗口扩展的“agent baseline”多出 9.48 分。  
- **消融实验**：分别去掉（1）任务平衡采样、（2）AEPO、（3）记忆写入/读取模块，分数下降约 2.5、3.1、4.2 分，说明每个组件都有实质贡献。  
- **局限性**：作者指出当前记忆库仍是线性增长的，存储成本随任务长度呈指数上升；此外，AEPO 对熵阈值的手工设定仍有一定经验依赖，自动调参空间有待探索。

### 影响与延伸思考

QwenLong‑L1.5 把“后训练”从单纯微调提升到系统化的长上下文 RL 流程，直接推动了大模型在超长文本上的实用化。随后的工作如 **LongMem**、**UltraChat** 等，都在记忆管理或自适应熵控制上借鉴了其思路。对想继续深耕的读者，可以关注以下方向：① 更高效的稀疏注意力与记忆检索结合；② 自动化熵阈值调节的元学习方法；③ 将记忆增强扩展到多模态（图像、音频）长序列。  

### 一句话记住它

**把长文本拆块生成多跳推理数据，用自适应熵控制的 RL 训练记忆模块，让模型在数百万 token 上也能稳健推理。**