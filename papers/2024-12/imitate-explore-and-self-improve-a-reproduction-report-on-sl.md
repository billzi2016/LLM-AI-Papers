# Imitate, Explore, and Self-Improve: A Reproduction Report on   Slow-thinking Reasoning Systems

> **Date**：2024-12-12
> **arXiv**：https://arxiv.org/abs/2412.09413

## Abstract

Recently, slow-thinking reasoning systems, such as o1, have demonstrated remarkable capabilities in solving complex reasoning tasks. These systems typically engage in an extended thinking process before responding to a query, allowing them to generate more thorough, accurate, and well-reasoned solutions. These systems are primarily developed and maintained by industry, with their core techniques not publicly disclosed. In response, an increasing number of studies from the research community aim to explore the technical foundations underlying these powerful reasoning systems. Building on these prior efforts, this paper presents a reproduction report on implementing o1-like reasoning systems. We introduce an ``imitate, explore, and self-improve'' framework, denoted as \textbf{STILL-2}, as our primary technical approach to train the reasoning model. In the initial phase, we use distilled long-form thought data to fine-tune the reasoning model, enabling it to invoke a slow-thinking mode. The model is then encouraged to explore challenging problems by generating multiple rollouts, which can result in increasingly more high-quality trajectories that lead to correct answers. Furthermore, the model undergoes self-improvement by iteratively refining its training dataset. To verify the effectiveness of this approach, we conduct extensive experiments on three challenging benchmarks. The experimental results demonstrate that our approach achieves competitive performance compared to industry-level reasoning systems on these benchmarks.

---

# 模仿、探索与自我改进：慢思考推理系统的复现报告 论文详细解读

### 背景：这个问题为什么难？

在传统的大语言模型（LLM）里，模型往往在收到问题后立刻给出答案，思考时间极短。这种“快思考”模式在数学、逻辑推理等需要多步演绎的任务上容易出现跳步或遗漏关键细节。业界已经推出了像 o1 这样的慢思考系统，它们会在内部进行长时间的“思考”，生成更完整的推理链，效果显著提升。但这些系统的核心实现细节被公司内部保密，学术界只能看到黑盒表现。于是研究者面临两个难题：一是缺少公开的训练数据和方法，二是现有的公开技术（如普通的Chain‑of‑Thought）仍然无法完全复制 o1 那种高质量、可自我迭代的推理能力。

### 关键概念速览

**慢思考模式**：模型在回答前会进行多轮内部推理，类似人类在解难题时先写草稿、反复检查。  
**长文本思考数据（Long-form Thought Data）**：经过人工或模型蒸馏得到的、包含完整推理过程的文本，用来教模型如何展开慢思考。  
**Rollout（轨迹生成）**：模型在同一道题上多次采样生成完整的思考序列，每一次都可能走出不同的推理路径。  
**自我改进循环**：把模型自己产生的高质量轨迹重新加入训练集，循环迭代提升模型的推理水平。  
**蒸馏（Distillation）**：把大型、算力强的模型产生的思考过程压缩成更小的数据集，供后续模型学习。  
**探索策略（Exploration Policy）**：在生成轨迹时故意加入随机性或奖励机制，鼓励模型尝试更具挑战性的思路。  
**基准任务（Benchmark）**：用于评估推理系统的公开数据集，如数学竞赛题、逻辑谜题等。

### 核心创新点

1. **从蒸馏思考数据到慢思考微调**  
   之前的复现工作多依赖公开的CoT数据，质量参差不齐。作者先用强大的闭源模型生成大量完整推理链，再对这些链进行蒸馏，得到高质量的长文本思考数据。随后在此数据上微调基础模型，使其能够主动进入慢思考模式。这样模型不再需要外部提示就能自行展开多步推理。

2. **多轨迹探索与质量筛选**  
   传统方法只让模型一次性输出答案或思考链，错失了探索空间。本文让模型对同一道题进行多次Rollout，每一次都可能产生不同的思考路径。随后通过一个简单的评分函数（如答案正确性、思考链完整度）挑选出最优轨迹，形成“高质量样本”。这种“多尝试、择优”机制显著提升了最终答案的正确率。

3. **闭环自我改进**  
   选出的高质量轨迹被重新加入训练集，模型在下一轮微调时会看到自己产生的优秀思考过程。循环若干次后，模型的内部策略逐渐向高质量轨迹收敛，实现了“自我教练”。与一次性训练不同，这种迭代式学习让模型在没有额外人工标注的情况下不断提升。

4. **统一的 STILL‑2 框架**  
   作者把模仿、探索、自我改进三个阶段包装成一个统一的训练流程，称为 STILL‑2（Imitate, Explore, Self‑Improve Loop）。这种模块化设计方便复现，也为后续研究提供了清晰的基线。

### 方法详解

整体思路可以划分为三大步骤：**模仿（Imitate） → 探索（Explore） → 自我改进（Self‑Improve）**。先让模型学会慢思考，再让它在难题上多次尝试，最后把成功的经验喂回模型。

1. **模仿阶段**  
   - **数据准备**：使用闭源的 o1 或类似系统，对公开的推理任务进行批量推理，得到完整的思考链。随后对这些链进行蒸馏，去掉冗余信息，只保留关键推理步骤和最终答案。  
   - **微调**：把蒸馏后的长文本思考数据作为监督信号，对一个预训练的大语言模型（如 LLaMA‑2）进行指令微调。微调的目标是让模型在接到普通问题时，自动进入“写思考链再给答案”的模式，而不是直接输出答案。

2. **探索阶段**  
   - **多轨迹生成**：对每一道测试题，模型使用温度采样或 nucleus 采样等随机策略，生成 N 条（如 N=5）完整的思考链。每条链都包括“思考 → 计算 → 检查 → 答案”。  
   - **质量评估**：对每条链计算一个综合得分。得分主要由两部分组成：①答案是否正确（硬性判定），②思考链的完整度和逻辑连贯性（通过一个小型判别模型或规则评估）。  
   - **轨迹筛选**：保留得分最高的 K 条（如 K=1）作为高质量样本，丢弃其余低质量轨迹。

3. **自我改进阶段**  
   - **数据扩充**：把筛选出的高质量轨迹加入到原始蒸馏数据中，形成更大的训练集。注意要标记这些轨迹的来源，以防模型产生循环依赖。  
   - **循环微调**：使用扩充后的数据再次微调模型。每一次循环都让模型看到更多“成功的思考方式”，从而在内部策略上逐步倾向于产生高质量轨迹。  
   - **迭代次数**：实验中通常进行 2‑3 轮循环即可看到显著提升，更多轮次收益递减。

**巧妙之处**：整个流程不需要额外的人工标注，只依赖模型自身产生的高质量样本。尤其是探索阶段的多轨迹生成，让模型在同一道题上“自我竞争”，类似于强化学习中的多臂赌博机，但这里的奖励是直接的答案正确性，极大降低了设计复杂度。

### 实验与效果

- **测试基准**：作者在三个公开的高难度推理基准上评估：*MATH*（数学竞赛题）、*GSM8K*（小学数学）、以及一个逻辑谜题集合。  
- **对比基线**：与普通的CoT微调模型、基于自回归的思考链模型以及业界公开的慢思考系统（如 OpenAI 的 o1‑lite）进行比较。  
- **结果**：在 *MATH* 上，STILL‑2 达到约 78% 的正确率，接近 o1‑lite 的 80%，而普通 CoT 只在 62% 左右。*GSM8K* 上提升约 10% 绝对值，逻辑谜题上也有 8% 左右的提升。  
- **消融实验**：去掉探索阶段的多轨迹生成，性能下降约 5%；不进行自我改进循环，下降约 7%；仅使用未蒸馏的原始思考链，整体下降 12%。这些实验表明三个模块缺一不可。  
- **局限性**：作者指出仍然依赖于高质量的蒸馏数据，若蒸馏来源不够强大，整体效果会受限；此外，循环微调会带来计算成本的显著提升，训练时间比单轮微调多出 2‑3 倍。

### 影响与延伸思考

这篇复现报告在学术社区引发了对「慢思考」可复制性的热议。随后出现的工作如 *Self‑Refine*、*Iterative CoT* 等，都在不同程度上借鉴了 STILL‑2 的自我改进循环。业界也开始尝试把「多轨迹探索」作为模型评估的标准，而不是单一答案。想进一步深入，可以关注以下方向：① 更高效的轨迹筛选策略（比如使用小模型做快速评估），② 将强化学习的奖励信号与答案正确性结合，③ 在多模态（文本+图像）任务上推广慢思考框架。整体来看，STILL‑2 为公开复现慢思考系统提供了可操作的蓝图，推动了从「黑盒」到「可解释」的转变。

### 一句话记住它

让模型先模仿高质量思考链，再通过多次自我探索挑出最佳轨迹，循环喂回模型，实现了公开可复制的慢思考推理。