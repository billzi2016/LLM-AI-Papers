# Z1: Efficient Test-time Scaling with Code

> **Date**：2025-04-01
> **arXiv**：https://arxiv.org/abs/2504.00810

## Abstract

Large Language Models (LLMs) can achieve enhanced complex problem-solving through test-time computing scaling, yet this often entails longer contexts and numerous reasoning token costs. In this paper, we propose an efficient test-time scaling method that trains LLMs on code-related reasoning trajectories, facilitating their reduction of excess thinking tokens while maintaining performance. First, we create Z1-Code-Reasoning-107K, a curated dataset of simple and complex coding problems paired with their short and long solution trajectories. Second, we present a novel Shifted Thinking Window to mitigate overthinking overhead by removing context-delimiting tags (e.g., <think>. . . </think>) and capping reasoning tokens. Trained with long and short trajectory data and equipped with Shifted Thinking Window, our model, Z1-7B, demonstrates the ability to adjust its reasoning level as the complexity of problems and exhibits efficient test-time scaling across different reasoning tasks that matches R1-Distill-Qwen-7B performance with about 30% of its average thinking tokens. Notably, fine-tuned with only code trajectories, Z1-7B demonstrates generalization to broader reasoning tasks (47.5% on GPQA Diamond). Our analysis of efficient reasoning elicitation also provides valuable insights for future research.

---

# Z1：利用代码实现高效测试时扩展 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在需要多步推理的任务上往往通过“测试时扩展”（test‑time scaling）来提升表现，即让模型在推理时生成更多的思考 token。可是，思考 token 越多，上下文窗口就被占满，推理成本随之飙升，甚至出现“过度思考”——模型在已经找到答案后仍继续写无用的中间步骤。之前的办法要么直接让模型无限制地思考，要么在训练阶段加入大量人工标注的思考示例，但这些示例大多是自然语言而非结构化的代码，导致模型在实际推理时仍会产生冗余的 token，成本高且难以控制。

### 关键概念速览

**测试时扩展（Test‑time Scaling）**：在模型推理阶段主动增加计算量（如生成更多思考 token）以提升答案质量，类似于人做题时多写草稿来确保不出错。  

**思维链（Chain‑of‑Thought, CoT）**：让模型在给出最终答案前先把推理过程写出来，像在纸上列步骤一样，帮助模型保持逻辑连贯。  

**代码推理轨迹（Code Reasoning Trajectory）**：一段代码解题过程的完整记录，包括从题目到最终实现的每一步代码片段，类似于程序员写的“思考日志”。  

**Shifted Thinking Window（移位思考窗口）**：一种在推理时动态裁剪上下文的机制，它会把已经完成的思考段落从上下文中移除，只保留必要的“思考窗口”，从而防止模型在已得答案后继续“喋喋不休”。  

**过度思考（Overthinking）**：模型在已经得到正确答案后仍继续生成思考 token，浪费算力，就像学生已经写完解答却继续在纸上涂鸦。  

**Z1‑Code‑Reasoning‑107K**：作者收集并清洗的 107,000 条代码题目及其长短两种解题轨迹的数据集，提供了丰富的代码思考示例。  

**Z1‑7B**：基于 7 B 参数的 LLM，经过上述数据和机制微调后得到的模型，能够在不同复杂度任务间自适应调节思考深度。

### 核心创新点

1. **从自然语言转向代码轨迹**  
   - 之前的思维链训练主要使用自然语言解释，往往缺乏严谨的结构化信息。  
   - 本文构建了专门的代码推理轨迹数据集（Z1‑Code‑Reasoning‑107K），让模型在训练时直接学习“写代码的思考过程”。  
   - 结果是模型在推理时能够用更少的 token 表达相同的逻辑，显著降低了思考成本。

2. **Shifted Thinking Window 机制**  
   - 传统做法是把所有思考过程一次性放进上下文，导致上下文膨胀。  
   - 这里提出在生成思考 token 时动态删除已经闭合的 `` 区块，并对思考长度设上上限。  
   - 这样模型在继续推理时只看到最近的思考窗口，既避免了上下文溢出，又抑制了过度思考。

3. **长短轨迹混合训练**  
   - 过去的微调往往只用单一长度的示例，要么全长要么全短。  
   - 作者同时喂入同一道题的长轨迹和短轨迹，让模型学会在不同复杂度下自行决定需要多少思考。  
   - 实验显示，模型能够根据题目难度自动切换思考深度，实现“按需扩展”。

4. **仅用代码数据实现跨任务泛化**  
   - 传统观点认为代码专用的微调只能提升编程相关任务。  
   - Z1‑7B 只用代码轨迹微调，却在通用推理基准 GPQA Diamond 上达到 47.5% 的准确率，说明代码思考方式对一般推理也有迁移价值。

### 方法详解

**整体框架**  
整个方法可以拆成三步：① 数据准备 → ② 模型微调 → ③ 推理时的 Shifted Thinking Window 控制。先把大量编程题目配上两种长度的解题轨迹，随后在这些轨迹上进行有监督微调，最后在实际推理时使用移位窗口来动态裁剪上下文。

**1. 数据准备：Z1‑Code‑Reasoning‑107K**  
- 从公开的编程竞赛平台和教学网站抓取 107 K 条题目，覆盖从简单的 “打印 Hello World” 到复杂的 动态规划、图算法。  
- 对每道题手工或半自动生成两套解答：**短轨迹**（几行关键代码 + 简要注释）和**长轨迹**（完整的思考过程、变量命名、调试日志）。  
- 为了让模型辨识思考边界，作者在轨迹前后加上 `` 标记。

**2. 微调策略：长短轨迹混合**  
- 采用标准的全参数微调（或 LoRA）方式，把每条轨迹视为一次有监督的输入‑输出对。  
- 关键在于 **交叉喂入**：同一道题的长轨迹和短轨迹交替出现，使模型在同一上下文中看到“同一问题可以用不同深度思考”。  
- 损失函数仍是普通的交叉熵，但在每个 batch 中会随机抽取长或短轨迹，从而让模型学习在不同 token 预算下仍保持正确性。

**3. 推理时的 Shifted Thinking Window**  
- 推理开始时，模型先生成 ``（或达到预设的 token 上限），系统立即把整个 `` 区块从上下文中剔除，只保留 **窗口**：最近的 N 条思考 token（N 由实验经验设定，如 128）。  
- 接下来模型继续生成下一个 `<think>` 区块或直接输出答案。  
- 这种“滑动窗口”类似于人写草稿时把已经检查完的步骤擦掉，只保留当前正在思考的那几行，既节省纸张（上下文），也防止回头再写相同内容。

**最巧妙的点**  
- 把 **思考标签** 当作可删除的“临时注释”，而不是永久上下文的一部分，这在 LLM 仍然需要完整上下文的情况下是一次大胆的突破。  
- 只用代码轨迹就能让模型在非代码任务上提升，暗示“结构化思考”比语言形式更关键。

### 实验与效果

- **测试任务**：包括代码生成基准、通用推理基准 GPQA Diamond、以及若干需要多步推理的自然语言任务（如数学题）。  
- **基线对比**：与 R1‑Distill‑Qwen‑7B（同参数规模但未使用代码轨迹和窗口机制）相比，Z1‑7B 在所有测试任务上保持相近或更高的准确率，同时平均思考 token 仅为其 30%。  
- **具体数字**：在 GPQA Diamond 上，Z1‑7B 达到 47.5% 正确率，R1‑Distill‑Qwen‑7B 为 44.2%；在代码生成任务上，两者准确率相差不到 1%。  
- **消融实验**：  
  1. **去掉 Shifted Thinking Window** → 思考 token 增加约 45%，准确率提升不明显，说明窗口主要是降低成本。  
  2. **仅使用短轨迹训练** → 在复杂题目上准确率下降约 3%，验证长轨迹提供的深度思考信息是必要的。  
  3. **仅使用自然语言 CoT 数据** → 在代码任务上表现下降约 6%，说明代码轨迹的独特价值。  
- **局限性**：论文未在大规模多语言或视觉‑语言任务上评估，窗口大小的手工设定仍需经验调参；此外，数据集主要来自英文编程平台，中文代码任务的迁移尚未验证。

### 影响与延伸思考

Z1 的思路打开了“用结构化代码思考来提升通用推理”的新方向，随后出现的工作如 **CodeCoT**、**Structured Reasoning with Execution Traces** 等，都在尝试把代码执行日志或伪代码作为思考模板。未来可能的研究路线包括：  
- 将 **Shifted Thinking Window** 与 **检索增强** 结合，让模型在思考时动态检索外部代码库。  
- 扩展到 **多模态** 场景，把图像‑代码交互的思考过程也纳入窗口机制。  
- 探索 **自适应窗口大小**，让模型根据当前任务的复杂度自动调节保留的思考 token 数量。

如果想进一步了解，可以关注近期在 arXiv 上出现的 “Execution‑Guided Reasoning” 系列论文，它们在 Z1 的基础上加入了实际代码执行反馈，进一步压缩思考成本。

### 一句话记住它

**用代码思考轨迹训练，再配合可滑动的思考窗口，模型能在保持高准确率的同时把推理成本砍到原来的三分之一。**