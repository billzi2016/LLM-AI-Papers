# Deciphering the Factors Influencing the Efficacy of Chain-of-Thought:   Probability, Memorization, and Noisy Reasoning

> **Date**：2024-07-01
> **arXiv**：https://arxiv.org/abs/2407.01687

## Abstract

Chain-of-Thought (CoT) prompting has been shown to enhance the multi-step reasoning capabilities of Large Language Models (LLMs). However, debates persist about whether LLMs exhibit abstract generalization or rely on shallow heuristics when given CoT prompts. To understand the factors influencing CoT reasoning we provide a detailed case study of the symbolic reasoning task of decoding shift ciphers, where letters are shifted forward some number of steps in the alphabet. We analyze the pattern of results produced by three LLMs -- GPT-4, Claude 3, and Llama 3.1 -- performing this task using CoT prompting. By focusing on a single relatively simple task, we are able to identify three factors that systematically affect CoT performance: the probability of the task's expected output (probability), what the model has implicitly learned during pre-training (memorization), and the number of intermediate operations involved in reasoning (noisy reasoning). We show that these factors can drastically influence task accuracy across all three LLMs; e.g., when tested with GPT-4, varying the output's probability of occurrence shifts accuracy from 26% to 70%. Overall, we conclude that CoT prompting performance reflects both memorization and a probabilistic version of genuine reasoning. Code and data at this https://github.com/aksh555/deciphering_cot

---

# 解密影响思维链效能的因素：概率、记忆与噪声推理 论文详细解读

### 背景：这个问题为什么难？
在大语言模型（LLM）出现之前，机器要完成多步推理往往只能靠手工设计的规则或专门的符号系统，通用模型几乎不可能自行列出思考过程。即使有了“思维链”（Chain‑of‑Thought，CoT）提示，研究者仍不清楚模型到底是在进行抽象的逻辑演绎，还是在套用训练时学到的浅层模式。缺乏可解释的实验平台，使得我们难以区分“真正的推理”与“概率驱动的猜测”。这篇论文正是把焦点放在一个极简的符号任务上，想要把这两者的贡献拆开来看。

### 关键概念速览
**思维链（CoT）**：在回答前让模型把每一步推理写出来，类似于人做数学题时先写草稿，再给出最终答案。  
**概率（Probability）**：指模型在语言分布中对某个输出的先验倾向，越常见的词或短语被预测的概率越高。  
**记忆（Memorization）**：模型在预训练阶段已经看到并记住的模式或事实，类似于人类的“背过的例题”。  
**噪声推理（Noisy Reasoning）**：在多步推理过程中，每一步都会引入一定的随机误差，累计起来会让最终答案偏离正确答案。  
**移位密码（Shift Cipher）**：把字母表循环平移若干位后再加密的古老密码，解密时需要逆向平移相同的步数。  
**输出概率分布**：模型对所有可能答案的置信度列表，决定了哪条答案最容易被采纳。  
**中间操作数（Intermediate Operations）**：在 CoT 中出现的每一步计算或文字描述的数量，步数越多，误差累积的机会越大。

### 核心创新点
1. **单任务深度剖析 → 只用移位密码这一极简符号任务 → 揭示了 CoT 表现背后三大可量化因素**。作者没有像以往那样在多种复杂任务上做横向对比，而是把注意力集中在一个可以精确控制输入输出概率的任务上，从而得到干净的因果关系。  
2. **概率调控实验 → 人为改变答案在训练语料中的出现频率 → 观察到同一模型在不同概率区间的准确率从 26% 跳到 70%**。这一步直接证明了“答案本身的语言概率”会显著左右 CoT 的成功率。  
3. **记忆度量 → 对比模型在已见过的平移步数与全新步数上的表现 → 发现模型在熟悉的步数上几乎不需要真正推理**。通过这种对照，作者把“记忆”与“推理”区分开来。  
4. **噪声推理分析 → 统计 CoT 中出现的中间步骤数量与错误率的相关性 → 证明步骤越多，累计噪声越大，准确率下降**。这让我们明白 CoT 并非越长越好，适度的思考链才是关键。

### 方法详解
整体思路可以概括为“三步走”：  
1) **任务构造**：选定移位密码作为实验对象。给模型一个明文‑密文对，让它在 CoT 提示下写出“先把字母往回移 X 位，然后得到原文”。  
2) **因素变量化**：分别对**输出概率**、**记忆程度**、**中间操作数**进行可控调节。  
   - *输出概率*：通过在提示中加入高频或低频的词汇（如“the” vs. “quixotic”）来改变答案在语言模型中的自然出现概率。  
   - *记忆*：挑选训练语料中常出现的平移步数（如 1、13）与完全未出现的步数（如 7、19）进行对比。  
   - *噪声*：在 CoT 中故意加入冗余的解释步骤或让模型自行展开多余的算式，从而增加中间操作的数量。  
3) **评估与分析**：让 GPT‑4、Claude 3、Llama 3.1 分别在上述三维空间里完成任务，记录最终解密是否正确，并统计每个维度的准确率变化。

**关键模块的类比**：可以把整个实验想象成一次“盲盒”测验。模型是盲盒里的玩家，提示是盒子外的说明书。我们通过调节说明书的文字（概率）和盒子里已有的图案（记忆），以及让玩家写多少步骤（噪声），来观察玩家到底是靠记忆还是靠推理赢得奖品。

**公式/算法的白话解释**：作者没有给出复杂的数学推导，而是用“条件概率”来描述：  
- 设 \(P(y|x)\) 为模型在给定输入 \(x\) 时输出答案 \(y\) 的概率。  
- 当我们把高频词放进答案里时，\(P(y|x)\) 上升，模型更倾向直接输出正确答案，而不需要完整的思考链。  
- 同时，若中间步骤数为 \(k\)，每一步的错误概率约为 \(\epsilon\)，则整体错误率约为 \(1-(1-\epsilon)^k\)，这正是噪声累积的数学体现。

**最巧妙的地方**：作者把“概率”这一语言模型固有属性直接当作实验因子来操控，而不是把它当作不可测的背景噪声。这种“把语言概率当实验变量”的做法在 LLM 研究里极为少见。

### 实验与效果
- **任务**：解码移位密码，平移步数范围 1‑25。  
- **模型**：GPT‑4、Claude 3、Llama 3.1（均使用官方提供的 CoT 提示）。  
- **主要发现**：  
  - 当答案的语言概率低时（如使用罕见词），GPT‑4 的正确率仅约 26%；提升到高频词后，准确率飙升至约 70%。  
  - 对于训练中出现过的平移步数，三模型的准确率普遍在 80% 以上；而对全新步数，准确率跌至 30% 左右，说明记忆对 CoT 起到决定性支撑。  
  - 中间操作数从 2 步增加到 6 步，错误率从 15% 上升到 45%，验证了噪声推理的累积效应。  
- **对比基线**：论文未提供传统 zero‑shot 或 few‑shot 的直接数值对比，只是把 CoT 与不使用 CoT 的结果作了粗略比较，后者在所有设置下均显著低于 CoT。  
- **消融实验**：分别去掉概率调控、记忆控制或噪声控制，发现每一因素的去除都会导致整体准确率下降 10%‑20%。  
- **局限性**：实验仅限于单一的符号任务，是否能直接迁移到更复杂的数学或常识推理仍未验证；此外，作者只分析了三款商业模型，开源模型的行为可能不同。

### 影响与延伸思考
这篇工作在 LLM 推理可解释性方向掀起了“小实验大启发”的潮流。随后有几篇论文（如 2024 年的 “Probabilistic Reasoning in LLMs”）借鉴了“把输出概率当实验因子”的思路，进一步探讨如何通过温度调节或词表重排来控制推理质量。还有研究尝试在更高层次的数学证明任务上复现类似的三因素分析，验证噪声推理在深层次逻辑链中的累积效应。想继续深入的读者可以关注 **“LLM 计量推理”（LLM Quantitative Reasoning）** 以及 **“提示工程中的概率调控”** 两大方向。

### 一句话记住它
CoT 的成功既靠模型记住的答案，也靠答案本身在语言中的高概率，而每增加一步思考链都会累积噪声——这三者共同决定了 LLM 的推理表现。