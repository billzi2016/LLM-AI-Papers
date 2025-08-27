# Diffusion Language Models Know the Answer Before Decoding

> **Date**：2025-08-27
> **arXiv**：https://arxiv.org/abs/2508.19982

## Abstract

Diffusion language models (DLMs) have recently emerged as an alternative to autoregressive approaches, offering parallel sequence generation and flexible token orders. However, their inference remains slower than that of autoregressive models, primarily due to the cost of bidirectional attention and the large number of refinement steps required for high quality outputs. In this work, we highlight and leverage an overlooked property of DLMs early answer convergence: in many cases, the correct answer can be internally identified by half steps before the final decoding step, both under semi-autoregressive and random remasking schedules. For example, on GSM8K and MMLU, up to 97% and 99% of instances, respectively, can be decoded correctly using only half of the refinement steps. Building on this observation, we introduce Prophet, a training-free fast decoding paradigm that enables early commit decoding. Specifically, Prophet dynamically decides whether to continue refinement or to go "all-in" (i.e., decode all remaining tokens in one step), using the confidence gap between the top-2 prediction candidates as the criterion. It integrates seamlessly into existing DLM implementations, incurs negligible overhead, and requires no additional training. Empirical evaluations of LLaDA-8B and Dream-7B across multiple tasks show that Prophet reduces the number of decoding steps by up to 3.4x while preserving high generation quality. These results recast DLM decoding as a problem of when to stop sampling, and demonstrate that early decode convergence provides a simple yet powerful mechanism for accelerating DLM inference, complementary to existing speedup techniques. Our code is publicly available at https://github.com/pixeli99/Prophet.

---

# 扩散语言模型在解码前已知答案 论文详细解读

### 背景：这个问题为什么难？
扩散语言模型（Diffusion LM）通过在噪声空间迭代“去噪”来生成文本，天然支持并行生成和灵活的 token 排序，克服了自回归模型只能一步步往后写的限制。然而，实际推理时需要多轮双向注意力和数十甚至上百次的细化步骤，导致整体速度远慢于自回归模型。研究者一直在尝试压缩迭代次数或改进采样策略，却往往在质量和速度之间陷入权衡：削减步数会让答案变得不完整或产生语法错误。于是，如何在不牺牲生成质量的前提下加速扩散 LM 成了急需突破的瓶颈。

### 关键概念速览
- **扩散语言模型（Diffusion LM）**：把文本看作在噪声中逐步被“恢复”的过程，类似把一张模糊的照片逐层清晰化。  
- **双向注意力（Bidirectional Attention）**：在每一步都让模型同时看到左侧和右侧的上下文，就像在写句子时可以随时回头检查前后文。  
- **细化步骤（Refinement Step）**：每一次去噪的迭代，模型会对所有 token 的分布进行一次更新。  
- **半自回归调度（Semi‑autoregressive Schedule）**：在生成过程中，部分 token 按自回归顺序生成，剩余 token 同时预测，兼顾并行和顺序的优点。  
- **随机重掩调度（Random Remasking Schedule）**：每轮迭代随机挑选一部分已生成的 token 重新掩码，让模型重新估计，提升鲁棒性。  
- **置信度间隙（Confidence Gap）**：模型在当前步对最高概率候选和第二高概率候选的概率差距，用来衡量答案是否已经“确定”。  
- **提前提交解码（Early Commit Decoding）**：当置信度间隙足够大时，直接一次性输出剩余 token，省去后续细化。  

### 核心创新点
1. **早期答案收敛的现象发现 → 通过实验在 GSM8K、MMLU 等任务上观察到超过 95% 的样本在完整迭代的一半之前已经出现唯一正确答案 → 这为加速提供了理论依据，说明模型内部已经“知道”答案，只是等着把它写出来。**  
2. **Prophet 判停机制 → 在每一步计算 top‑2 token 的概率差，如果差值超过预设阈值，就触发“全力一次解码”，即把所有未完成的 token 在同一步骤一次性生成 → 这种判停不需要额外训练，只是利用已有的概率分布做决策。**  
3. **训练无关的实现方式 → Prophet 直接嵌入现有的扩散 LM 推理流程，几乎不增加计算开销，也不改变模型参数 → 这使得它可以立刻在 LLaDA‑8B、Dream‑7B 等已有模型上复现。**  
4. **与现有加速手段互补 → 传统的加速方法（如步数压缩、稀疏注意力）主要在模型结构或采样上做文章，而 Prophet 关注“何时停”，两者可以叠加使用，进一步提升整体吞吐。**  

### 方法详解
整体思路可以概括为三步：**（1）常规扩散迭代 →（2）置信度评估 →（3）提前提交**。  
1. **常规迭代**：模型按照既定的调度（半自回归或随机重掩）进行若干细化步骤。每一步都使用双向注意力，对所有 token 的分布进行去噪。  
2. **置信度评估**：在每一步结束后，遍历当前每个位置的概率分布，取最高概率 p₁ 和第二高概率 p₂，计算 gap = p₁ - p₂。若所有位置的 gap 均大于一个经验阈值 τ，则认为答案已经确定。这里的阈值可以在验证集上手动调参，通常在 0.1~0.2 之间。  
3. **提前提交**：一旦满足判停条件，模型不再继续细化，而是直接把每个位置的最高概率 token 作为最终输出。这一步相当于把剩余的细化步骤“一键完成”，因此称为 “all‑in”。如果条件未满足，则继续下一轮细化，循环回到第 2 步。  

**关键细节**  
- **动态阈值**：作者指出固定阈值在不同任务上表现差异较大，实际使用时可以根据任务难度或模型大小自适应调节。  
- **全局 vs 局部判停**：Prophet 采用全局判停（所有位置都满足 gap 条件），而不是局部提前输出部分 token，这样可以避免出现“半成品”导致的语义不连贯。  
- **无额外训练**：置信度 gap 完全来源于模型本身的输出分布，不需要再训练一个判别网络或修改损失函数，这也是实现成本极低的关键。  
- **与采样策略兼容**：即使使用 nucleus sampling（核采样）或 top‑k 采样，gap 仍然可以直接计算，只是概率分布可能被截断，需要在完整分布上评估。  

最巧妙的地方在于把“模型已经知道答案”这一本质信息转化为一个 **何时停止** 的决策问题，而不是去改动模型的生成机制。这样既保留了扩散 LM 的高质量特性，又大幅削减了不必要的迭代。

### 实验与效果
- **数据集与任务**：作者在数学推理 GSM8K、通用知识测验 MMLU、以及开放式对话任务上评估 LLaDA‑8B 与 Dream‑7B 两个主流扩散 LM。  
- **基线对比**：与原始不做提前停止的模型相比，Prophet 在保持相同 BLEU/Exact‑Match 分数的前提下，将平均迭代步数削减了 2.1×~3.4×。在 GSM8K 上，Exact‑Match 下降不到 0.3%，MMLU 上的准确率下降约 0.2%。  
- **早期收敛比例**：实验显示，在 GSM8K 中约 97% 的样本在完整迭代的一半之前已经出现唯一正确答案；MMLU 中比例更高，达到 99%。这直接验证了论文的核心假设。  
- **消融实验**：作者分别关闭置信度判停、改用固定步数压缩、以及使用更宽松的阈值进行对比。结果表明，置信度 gap 判停是提升速度的主要驱动因素，阈值过宽会导致质量明显下降，过窄则削减效果不明显。  
- **局限性**：在极端长文本或需要细粒度控制的生成任务（如代码生成）上，提前提交的成功率下降，作者承认需要更细致的局部判停策略或结合外部校验。  

### 影响与延伸思考
Prophet 把“何时停”提升为扩散 LM 推理的核心设计点，开启了后续研究的一个新方向。随后有几篇工作尝试将类似的置信度判停与 **自适应步数调度** 结合，甚至在多模态扩散模型（如文本‑图像）中引入跨模态置信度共享，以进一步压缩推理时间。对想深入的读者，可以关注 **自适应采样**、**动态噪声调度** 以及 **跨任务置信度校准** 等方向，这些都是在 Prophet 思路上自然延伸的研究热点。

### 一句话记住它
Prophet 通过检测模型输出的置信度间隙，直接在答案已经确定时“一键完成”剩余生成，从而把扩散语言模型的推理速度提升数倍而几乎不损失质量。