# CWM: An Open-Weights LLM for Research on Code Generation with World Models

> **Date**：2025-09-30
> **arXiv**：https://arxiv.org/abs/2510.02387

## Abstract

We release Code World Model (CWM), a 32-billion-parameter open-weights LLM, to advance research on code generation with world models. To improve code understanding beyond what can be learned from training on static code alone, we mid-train CWM on a large amount of observation-action trajectories from Python interpreter and agentic Docker environments, and perform extensive multi-task reasoning RL in verifiable coding, math, and multi-turn software engineering environments. With CWM, we provide a strong testbed for researchers to explore the opportunities world modeling affords for improving code generation with reasoning and planning in computational environments. We present first steps of how world models can benefit agentic coding, enable step-by-step simulation of Python code execution, and show early results of how reasoning can benefit from the latter. CWM is a dense, decoder-only LLM trained with a context size of up to 131k tokens. Independent of its world modeling capabilities, CWM offers strong performance on general coding and math tasks: it reaches pass@1 scores of 65.8% on SWE-bench Verified (with test-time scaling), 68.6% on LiveCodeBench, 96.6% on Math-500, and 76.0% on AIME 2024. To support further research on code world modeling, we release model checkpoints after mid-training, SFT, and RL.

---

# CWM：用于代码生成与世界模型研究的开源权重大语言模型 论文详细解读

### 这篇论文解决了什么问题？
在传统代码生成模型里，模型只能从静态源码中学习，缺乏对代码运行时行为的真实感知，导致在需要推理、规划或交互的场景表现不佳。作者通过引入“世界模型”，让模型在训练时直接观察代码执行过程，从而突破了仅靠文本学习的瓶颈。

### 关键概念速览
**世界模型（World Model）**：模型内部的“模拟器”，能够预测在特定环境下代码执行后会产生的状态变化。  
**中期训练（Mid‑training）**：在大规模语言模型预训练后，继续用专门的任务数据（如解释器轨迹）进行微调。  
**多任务推理强化学习（Multi‑task Reasoning RL）**：让模型在多个不同的编程/数学任务中通过奖励信号学习规划和推理能力。  
**上下文窗口（Context Window）**：模型一次性能处理的最长序列长度，这里高达 131k token，足以容纳完整的执行日志。  
**pass@1**：在一次生成尝试中，模型输出的代码能否通过全部单元测试的比例，用来衡量代码生成质量。

### 核心创新点
1. **静态代码 → 代码执行轨迹**：以前的模型只看源码，CWM 在中期训练阶段加入了 Python 解释器和 Docker 环境产生的观察‑动作序列，使模型学会“看见”代码运行的每一步。  
2. **单一模型兼顾多任务 RL**：过去需要为每类任务单独设计奖励或微调，CWM 通过统一的多任务强化学习框架，在可验证编码、数学推导和多轮软件工程等场景同时训练，提升了跨域推理和规划能力。  
3. **超长上下文 + 密集解码器**：把上下文窗口扩展到 131k token，并保持 decoder‑only 架构，使模型能够一次性读取完整的执行日志或长对话，进而实现步进式代码仿真。

### 方法怎么做的？
1. **预训练 → 中期训练**：先用通用文本和代码数据训练 32B 参数的语言模型。随后收集大量 Python REPL（交互式解释器）和 Docker 环境下的“状态‑动作”对（比如执行 `a+=1` 后变量值的变化），把这些轨迹当作序列喂入模型进行中期训练。  
2. **多任务 RL 循环**：构建三个任务环境——可验证编码（生成代码并自动跑测试）、数学推理（解题并验证答案）和多轮软件工程（与模拟用户对话完成需求）。在每一步，模型输出下一段代码或对话，环境返回执行结果或奖励信号，模型依据奖励更新参数。整个过程在 131k token 的窗口里完成，保证了完整的上下文信息。

### 效果如何？
- **SWE‑bench Verified**（真实软件工程基准）在使用测试时规模扩展后，pass@1 达到 **65.8%**。  
- **LiveCodeBench**（实时代码生成评测）取得 **68.6%** 的 pass@1。  
- **Math‑500**（数学题库）上正确率为 **96.6%**。  
- **AIME 2024**（美国数学竞赛）得分 **76.0%**。  
这些数字在同等规模的开源模型中处于领先位置，尤其在需要执行验证的任务上表现出显著优势。

### 一句话总结
CWM 用真实代码执行轨迹和超长上下文，让大语言模型在“看见”代码运行的同时学会推理和规划，显著提升了可验证代码生成的实力。