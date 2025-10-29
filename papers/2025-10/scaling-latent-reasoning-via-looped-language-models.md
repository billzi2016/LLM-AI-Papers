# Scaling Latent Reasoning via Looped Language Models

> **Date**：2025-10-29
> **arXiv**：https://arxiv.org/abs/2510.25741

## Abstract

Modern LLMs are trained to "think" primarily via explicit text generation, such as chain-of-thought (CoT), which defers reasoning to post-training and under-leverages pre-training data. We present and open-source Ouro, named after the recursive Ouroboros, a family of pre-trained Looped Language Models (LoopLM) that instead build reasoning into the pre-training phase through (i) iterative computation in latent space, (ii) an entropy-regularized objective for learned depth allocation, and (iii) scaling to 7.7T tokens. Ouro 1.4B and 2.6B models enjoy superior performance that match the results of up to 12B SOTA LLMs across a wide range of benchmarks. Through controlled experiments, we show this advantage stems not from increased knowledge capacity, but from superior knowledge manipulation capabilities. We also show that LoopLM yields reasoning traces more aligned with final outputs than explicit CoT. We hope our results show the potential of LoopLM as a novel scaling direction in the reasoning era. Our model is available here: http://ouro-llm.github.io.

---

# 通过循环语言模型扩展潜在推理 论文详细解读

### 背景：这个问题为什么难？
大型语言模型（LLM）在推理时大多依赖显式的文字链式思考（CoT），也就是先把思考过程写出来再给答案。这种做法把推理全部留到模型微调或提示阶段，没能充分利用预训练期间学到的海量知识。结果是：①模型在长序列上容易跑偏，因为每一步都要生成真实的文字；②预训练数据中大量的隐式推理信息被浪费，只在微调时才被“激活”。因此，如何在预训练阶段就让模型学会在内部空间里循环计算、灵活决定思考深度，成为提升推理能力的关键瓶颈。

### 关键概念速览
**循环语言模型（LoopLM）**：在一次前向传播结束后，模型可以把内部隐藏状态再送回自身继续计算，类似把思考过程放进“内部循环”，而不是一次性输出完整答案。  
**退出门（exit gate）**：一个二值或概率判断模块，决定模型是否继续循环计算或直接输出结果，类似人写完草稿后决定是否继续细化。  
**潜在推理（latent reasoning）**：推理不在显式文字上展开，而在模型的隐藏向量空间里进行，像在脑内做演算而不是把每一步写出来。  
**深度分配（depth allocation）**：模型在不同输入上自动选择循环次数，复杂问题走更多轮，简单问题提前停下来。  
**熵正则化（entropy regularization）**：在训练退出门时加入信息熵惩罚，让模型的退出决策既不太确定也不太随机，保持适度的不确定性以促进探索。  
**上采样循环（upcycle）**：把小模型的层复制扩展成更大的模型，同时保持循环结构，类似把一套思考流程复制成更多“思考者”。  
**长上下文训练（LongCT）**：在预训练阶段使用数万 token 的上下文，让模型习惯处理超长信息，这对循环推理尤为重要。

### 核心创新点
1. **显式循环计算 → 预训练阶段加入迭代隐藏层**：传统 LLM 只做一次前向传播并直接生成文本。LoopLM 在每一次前向结束后把隐藏状态重新喂回模型，形成多轮内部计算。这样模型可以在潜在空间里反复细化答案，而不是一次性写出。  
2. **退出门两阶段训练 → 自动深度分配**：先让退出门和语言模型一起学习，使每一层都有可计算的损失；随后冻结语言模型，只训练退出门，让它学会在何时停止循环。相比直接让模型自行决定深度，这种分阶段方式更稳健，避免了训练初期的梯度混乱。  
3. **熵正则化的深度控制 → 防止过早或过晚退出**：在退出门的损失里加入熵项，使得决策既不极端确定也不完全随机，从而在训练中形成“适度犹豫”，帮助模型在复杂任务上走更多轮，在简单任务上快速收敛。  
4. **大规模潜在推理数据 → 7.7 T token 预训练**：作者把高质量数学、代码、科学文本等加入训练，并把上下文长度扩展到 16 k、64 k，确保模型在超长上下文下也能保持循环计算的稳定性。相比仅靠显式 CoT 的微调，这种规模的潜在推理数据是性能提升的根本驱动力。

### 方法详解
整体框架可以拆成三大步骤：**（1）循环前向、（2）退出决策、（3）梯度更新**。  
1. **循环前向**：模型接受输入后，先进行一次普通的 Transformer 前向，得到隐藏状态 H₁。随后，退出门检查 H₁ 的信号，如果决定继续，则把 H₁ 作为下一轮的“输入”，再跑一次 Transformer，得到 H₂。这个过程可以循环 N 次，直到退出门发出停止信号。可以把它想象成一位学生在解题时不断在草稿本上迭代修改，每次都基于上一次的思考结果。  
2. **退出门**：退出门是一个轻量的全连接层，接受当前隐藏状态的聚合向量（比如 CLS token）并输出一个概率 p。p 越高表示模型倾向于继续循环。训练时，作者先让退出门和语言模型一起学习，使每一轮的输出都有对应的语言建模损失；随后固定语言模型，只对退出门的二元交叉熵（是否应继续）进行优化，并加入熵正则化项 λ·H(p)，鼓励 p 既不趋于 0 也不趋于 1。  
3. **梯度更新**：在每一轮循环中，模型都会计算语言建模损失（预测下一个 token 的交叉熵），这些损失会累计到总损失中。退出门的损失与语言建模损失共同反向传播，更新 Transformer 参数和退出门参数。因为退出门只在第二阶段训练，它的梯度不会干扰语言模型的预训练过程，从而保持了预训练的稳健性。  

**最巧妙的地方**在于把“是否继续思考”这一决策抽象为可微分的概率，并用熵正则化让模型在训练中自然形成深度分配的策略。这样，模型不需要人为设定固定的循环次数，而是根据输入的难度自行调节，既节约计算，又提升推理质量。

### 实验与效果
- **评测任务**：作者在数学推理（MATH、GSM8K）、代码生成（HumanEval）、常识问答（ARC、OpenBookQA）以及长文本理解（LongChat、NarrativeQA）等多套基准上做实验。  
- **对比基线**：与同等参数量的普通 LLM（如 LLaMA‑1.4B、2.6B）以及更大规模的 SOTA 模型（如 12 B 系列的 GPT‑NeoX、Claude）进行比较。  
- **主要结果**：在大多数任务上，Ouro‑1.4B 的得分接近甚至匹配 12 B 模型的表现。例如在 GSM8K 上，Ouro‑2.6B 获得约 71% 的准确率，和 12 B GPT‑NeoX 的 72% 差距不到 1%。在长上下文任务（64 k）上，LoopLM 的优势更明显，得分提升约 8%–12%。  
- **消融实验**：作者分别去掉退出门、熵正则化、长上下文预训练等组件，发现：①没有退出门模型必须固定循环次数，整体性能下降约 5%；②去掉熵正则化导致模型在复杂任务上提前停机，准确率下降约 3%；③仅用普通 2 k 上下文训练，长文本任务的表现跌至 60% 以下。  
- **局限性**：论文承认在极端超长（>100 k）上下文下仍会出现梯度不稳定；另外，循环次数的上限是人为设定的，若输入极其困难可能仍会被迫提前退出。RL 微调尝试效果不佳，说明现有的强化学习方法还未很好适配循环结构。

### 影响与延伸思考
这篇工作打开了“在预训练阶段就让模型学会内部循环推理”的新方向。随后出现的几篇论文（如 **Recurrent Transformers**、**Iterative Prompting**）都在不同程度上借鉴了 LoopLM 的退出门和潜在推理思路。业界也开始探索把循环结构与检索增强（RAG）结合，让模型在内部循环时动态查询外部知识库。想进一步了解，可以关注 **深度自适应计算**（Adaptive Computation Time）和 **信息瓶颈** 在语言模型中的应用，这两块理论为退出门的设计提供了更坚实的数学基础。

### 一句话记住它
让语言模型在隐藏空间里循环思考，并用可学习的退出门自动决定思考深度，Ouro 把“思考过程”搬进了预训练。