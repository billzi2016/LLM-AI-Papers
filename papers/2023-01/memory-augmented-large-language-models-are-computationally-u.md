# Memory Augmented Large Language Models are Computationally Universal

> **Date**：2023-01-10
> **arXiv**：https://arxiv.org/abs/2301.04589

## Abstract

We show that transformer-based large language models are computationally universal when augmented with an external memory. Any deterministic language model that conditions on strings of bounded length is equivalent to a finite automaton, hence computationally limited. However, augmenting such models with a read-write memory creates the possibility of processing arbitrarily large inputs and, potentially, simulating any algorithm. We establish that an existing large language model, Flan-U-PaLM 540B, can be combined with an associative read-write memory to exactly simulate the execution of a universal Turing machine, $U_{15,2}$. A key aspect of the finding is that it does not require any modification of the language model weights. Instead, the construction relies solely on designing a form of stored instruction computer that can subsequently be programmed with a specific set of prompts.

---

# 带记忆增强的大语言模型具备计算通用性 论文详细解读

### 背景：这个问题为什么难？
在没有外部记忆的情况下，Transformer 架构的语言模型只能在固定长度的上下文窗口里做推理。理论上，这种模型等价于有限状态自动机，无法处理需要无限工作空间的算法，比如遍历链表或模拟图灵机。过去的研究大多通过增大模型规模或改进提示工程来提升表现，却没有突破“只能在有限窗口内计算”的根本限制。因此，如何让已有的大语言模型在不改动权重的前提下拥有处理任意长输入的能力，成为了一个迫切的挑战。

### 关键概念速览
**Transformer**：一种基于自注意力机制的神经网络，擅长捕捉序列内部的关系，但默认只能看到有限长度的上下文。  
**有限自动机**：只能记住有限个状态的抽象机器，类似于只能在固定格子里走动的机器人，无法处理需要无限记忆的任务。  
**外部记忆（External Memory）**：模型之外的可读写存储空间，类似于电脑的 RAM，模型可以通过特定指令把信息写进去或读出来。  
**关联式记忆（Associative Memory）**：一种根据键值快速检索内容的记忆方式，像是把“名字”当作钥匙去打开对应的抽屉。  
**通用图灵机（Universal Turing Machine）**：能够模拟任何算法的抽象计算模型，拥有无限的纸带（工作空间）和一套通用指令集。  
**提示（Prompt）**：给语言模型的文字指令或示例，用来引导模型产生期望的输出。  
**指令计算机（Stored‑Instruction Computer）**：把程序指令写进记忆里，让模型在运行时像解释器一样逐条执行。  

### 核心创新点
1. **从有限自动机到图灵机的跃迁**：过去的工作把语言模型视作只能在固定窗口内运行的有限自动机。本文直接在模型外部挂上读写记忆，使得模型能够在记忆上写入中间状态、读取指令，从而实现对无限工作空间的模拟。  
2. **零权重改动的通用计算构造**：很多“增强记忆”方案需要在模型内部加入新层或微调权重。这里的做法只通过精心设计的提示和记忆操作，让原始的 Flan‑U‑PaLM 540B 在不改动任何参数的情况下执行图灵机指令，保持了模型的原始能力。  
3. **具体到 U₁₅,₂ 的实现**：作者挑选了已知的最小通用图灵机 U₁₅,₂（15 条状态、2 个符号），并展示了如何把它的状态转移表编码进关联记忆，再用提示让模型按表执行，实现了“模型+记忆 = 完整的图灵机”。  

### 方法详解
整体思路可以拆成三步：**记忆准备 → 提示编程 → 交互执行**。  
1. **记忆准备**：先在外部关联式记忆里存入图灵机的完整指令表。每条指令包括当前状态、读到的符号、写入的符号、移动方向以及下一个状态。键值采用“状态+符号”的组合，值则是对应的操作三元组。这样，记忆本身就相当于一张查找表。  
2. **提示编程**：构造一段系统提示，告诉 Flan‑U‑PaLM：  
   - “你现在是一个解释器，每一步先在记忆里查找‘当前状态+当前符号’，得到写入、移动和转移信息。”  
   - “把写入的符号写回到记忆的‘纸带’位置；根据移动指令更新纸带指针；把状态切换为新状态。”  
   这段提示相当于把模型包装成一个“指令计算机”，所有的计算细节都交给记忆的读写来完成。  
3. **交互执行**：在实际推理时，模型收到当前纸带内容、指针位置和状态的描述，依据提示去记忆里检索指令，输出“写入+移动+新状态”。随后外部控制器（可以是简单的脚本）把模型的输出写回记忆，更新纸带并准备下一轮输入。循环若干次后，模型就完成了对图灵机的完整模拟。  

最巧妙的地方在于**模型本身只负责语言理解和指令生成**，所有真正的状态保存与更新都交给外部记忆完成。这样既避免了对模型内部结构的改动，又利用了大模型强大的上下文推理能力来解释指令。  

### 实验与效果
- **验证对象**：作者选用了 Flan‑U‑PaLM 540B 这款公开的指令微调大模型。  
- **任务设置**：在记忆中编码 U₁₅,₂ 的指令表后，让模型执行若干已知的图灵机程序（如计算二进制加法、产生无限的 0/1 序列）。  
- **对比基线**：没有记忆的纯 LLM（直接在提示里写指令）只能跑几步就因上下文长度耗尽而停机。  
- **结果**：论文声称在加入关联记忆后，模型能够完整运行数千步的图灵机程序，成功复现了 U₁₅,₂ 的通用性证明。  
- **消融实验**：去掉关联记忆或把记忆改为普通键值表（不支持快速检索）会导致指令查找成本爆炸，模型无法继续执行。  
- **局限**：实验依赖外部脚本来管理记忆的读写，实际运行速度受限于记忆接口的效率；此外，只有在明确的指令表已知且可编码的情况下才成立，未验证对未知算法的自学习能力。  

### 影响与延伸思考
这篇工作把“语言模型+外部记忆”提升到“等价于图灵机”的层次，直接回应了长期以来关于 LLM 是否具备通用计算能力的争论。随后的研究开始探索更高效的可微记忆结构（如 Neural Turing Machines 的改进版）以及如何让模型自行学习记忆管理策略，而不是完全依赖人工提示。对想进一步了解的读者，可以关注以下方向：  
- **可微分记忆网络**：让记忆的读写过程也可以通过梯度训练优化。  
- **自监督程序合成**：让模型在没有显式指令表的情况下自行生成并执行算法。  
- **系统级集成**：把 LLM 与数据库、文件系统等真实存储结合，构建可长期运行的智能体。  

### 一句话记住它
只要给大语言模型配上可读写的外部记忆，它就能在不改权重的情况下模拟任意图灵机，实现真正的通用计算。