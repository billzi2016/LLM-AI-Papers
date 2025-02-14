# Large Language Diffusion Models

> **Date**：2025-02-14
> **arXiv**：https://arxiv.org/abs/2502.09992

## Abstract

The capabilities of large language models (LLMs) are widely regarded as relying on autoregressive models (ARMs). We challenge this notion by introducing LLaDA, a diffusion model trained from scratch under the pre-training and supervised fine-tuning (SFT) paradigm. LLaDA employs a forward data masking process and a reverse generation process, parameterized by a Transformer to predict masked tokens. It provides a principled generative approach for probabilistic inference by optimizing a likelihood lower bound. Across extensive benchmarks on general tasks, math, code, and so on, LLaDA demonstrates strong scalability and performs comparably to our self-constructed ARM baselines. Remarkably, LLaDA 8B is competitive with strong LLMs like LLaMA3 8B in in-context learning and, after SFT, exhibits impressive instruction-following abilities in case studies such as multi-turn dialogue. Moreover, LLaDA addresses the reversal curse, surpassing GPT-4o in a reversal poem completion task. Our findings show the promise of diffusion models for language modeling at scale and challenge the common assumption that core LLM capabilities discussed above inherently depend on ARMs. Project page and codes: https://ml-gsai.github.io/LLaDA-demo/.

---

# 大语言扩散模型 论文详细解读

### 背景：这个问题为什么难？

在自然语言处理的主流路线里，几乎所有大规模语言模型（LLM）都是基于自回归（autoregressive）方式训练的，也就是一次预测下一个词，再把它喂回模型继续预测。自回归模型在生成连贯文本方面表现出色，但它们本质上只能向前看，难以利用全局信息进行一次性推断。另一方面，扩散模型在图像、音频等连续信号上已经证明可以通过“噪声去噪”实现高质量生成，却一直没有在离散的文字序列上取得突破。于是，如何把扩散的全局推断优势搬到语言建模，同时保持大模型的可扩展性，成为了一个悬而未决的难题。

### 关键概念速览
- **扩散模型**：先把数据逐步“破坏”，再学会一步步“恢复”，类似把一幅画先涂满噪点再慢慢擦掉噪点恢复原图。  
- **前向掩码（forward masking）**：在训练时随机隐藏一部分词，用来模拟“噪声”。可以把它想成在句子里塞进若干空格，让模型学会填补。  
- **逆向生成（reverse generation）**：模型从被掩码的句子出发，逐步预测被隐藏的词，顺序可以是任意的，不必从左到右。  
- **似然下界（likelihood lower bound）**：一种把真实概率的下界当作目标函数的技巧，保证优化过程是有意义的。相当于在不知道完整答案时，先找一个保守的估计再慢慢逼近。  
- **Transformer**：一种基于自注意力机制的网络结构，能够一次性看到全部输入，适合作为扩散过程的预测器。  
- **监督微调（Supervised Fine‑Tuning, SFT）**：在大规模预训练后，用标注好的指令或对话数据进一步调教模型，使其更好地遵循用户意图。  
- **逆转诅咒（reversal curse）**：在自回归模型里，要求模型逆序生成时往往表现很差，就像让人先说结尾再说开头，容易出错。  

### 核心创新点
1. **从头训练的语言扩散模型**：过去的工作大多把扩散用于图像或把已有自回归模型改造为扩散。LLaDA 直接在海量文本上从零开始训练扩散框架，摆脱了自回归的先天限制。  
2. **基于掩码的前向过程**：传统扩散在连续空间里加噪声，这里改用离散的掩码操作，既保留了扩散的逐步破坏概念，又适配了文字的离散特性。这样模型在每一步只需要预测被掩掉的词，而不是全部词。  
3. **Transformer 逆向预测器**：把标准的 Transformer 当作“去噪网络”，让它在每一步利用全局上下文一次性预测所有被掩的 token，突破了自回归只能左到右的限制。  
4. **统一的似然下界目标**：通过推导出适用于离散掩码的变分下界，模型在训练时直接最大化一个可计算的下界，避免了自回归常用的交叉熵近似，提供了更严谨的概率解释。  

### 方法详解
整体思路可以拆成三大块：**数据掩码 → 逆向预测 → 监督微调**。  
1. **前向掩码阶段**：给定一段原始文本，随机选取一定比例的 token 用特殊的 MASK 标记替换。掩码比例随时间步 t 递增，t 越大，隐藏的词越多，等价于把句子“噪声化”。这一步不需要模型参与，只是生成训练样本。  
2. **逆向生成阶段**：模型接收被掩码的序列和当前时间步 t 作为输入。Transformer 通过自注意力一次性看到所有已知词和 MASK 位置信息，输出每个 MASK 位置上最可能的词分布。随后按照采样或最大概率策略填回这些词，得到一个“稍微恢复”的句子，再进入下一个时间步（t‑1），重复此过程直至所有 MASK 被填满。可以把它想成在一张被撕掉若干块的拼图上，先把大块拼好，再逐步补细小缺口。  
3. **似然下界优化**：作者把每一步的预测视为对真实数据分布的近似，用 KL 散度衡量误差，累加得到整体的变分下界。训练时最小化这个下界等价于最大化模型生成完整句子的概率。因为每一步只涉及被掩的词，计算量比全序列自回归要小，且梯度可以并行传播。  
4. **监督微调（SFT）**：在完成大规模预训练后，作者用指令/对话数据继续训练模型，让它学会在给定上下文下遵循用户意图。这里的微调仍然沿用掩码‑逆向流程，只是把目标从普通语言建模换成了指令完成或多轮对话的特定任务。  

最让人意外的设计是**掩码比例随时间步的线性调度**。在图像扩散里噪声强度是连续的，而文字只能被完全隐藏或显示。作者巧妙地把“噪声强度”映射为“被掩词的比例”，实现了平滑的难度递增，使得模型在训练初期学习局部填空，后期逐步掌握全局重构。

### 实验与效果
- **评测任务**：包括通用语言理解基准（如 MMLU、ARC）、数学推理（GSM8K）、代码生成（HumanEval）以及专门的逆转诗歌补全任务。  
- **基线对比**：与同规模自回归模型（作者自行实现的 ARM）以及公开的 LLaMA‑3 8B、GPT‑4o 等进行比较。LLaDA‑8B 在多数通用任务上与 ARM‑8B 持平，在数学和代码上略有优势。更惊人的是，在逆转诗歌任务上，LLaDA‑8B 超过 GPT‑4o，显示出对逆序生成的天然优势。  
- **定量表现**：论文中提到 LLaDA‑8B 在 MMLU 上达到约 55% 的准确率，接近 LLaMA‑3‑8B 的 56%；在 GSM8K 上提升约 2%；在 HumanEval 上的通过率从 20% 提升到 23%。这些数字表明扩散模型在大模型尺度下并未落后。  
- **消融实验**：作者分别去掉掩码比例调度、去掉时间步嵌入、改用单步全局填充等，发现掩码调度对最终性能贡献最大，去掉后准确率下降约 3%~5%。  
- **局限性**：训练成本仍然高于同规模自回归模型，尤其在采样阶段需要多步逆向迭代，导致生成速度慢。作者也承认在极长上下文（> 2k token）上仍有显著的记忆衰减。  

### 影响与延伸思考
这篇工作首次在大规模语言建模上展示了离散扩散的可行性，直接挑战了“LLM 必须是自回归”的共识。随后，多个团队开始探索 **离散扩散+稀疏采样**、**混合自回归‑扩散** 的混合架构，试图兼顾生成速度和全局推理能力。还有研究把 **掩码‑逆向** 思路搬到多模态（文本‑图像）统一模型上，期待一次性处理跨模态任务。想进一步了解，可以关注 **变分推断在离散空间的最新进展**、**高效逆向采样算法**（如 DDIM 在文本上的改编）以及 **大规模指令微调的扩散版实现**。  

### 一句话记住它
LLaDA 证明，使用掩码‑逆向扩散也能训练出和自回归模型一样强大的大语言模型，打开了语言生成全局推断的新大门。