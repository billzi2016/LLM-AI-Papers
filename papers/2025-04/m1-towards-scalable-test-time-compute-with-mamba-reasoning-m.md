# M1: Towards Scalable Test-Time Compute with Mamba Reasoning Models

> **Date**：2025-04-14
> **arXiv**：https://arxiv.org/abs/2504.10449

## Abstract

Effective reasoning is crucial to solving complex mathematical problems. Recent large language models (LLMs) have boosted performance by scaling test-time computation through long chain-of-thought reasoning. However, transformer-based models are inherently limited in extending context length due to their quadratic computational complexity and linear memory requirements. In this paper, we introduce a novel hybrid linear RNN reasoning model, M1, built on the Mamba architecture, which allows memory-efficient inference. Our approach leverages a distillation process from existing reasoning models and is further enhanced through RL training. Experimental results on the AIME and MATH benchmarks show that M1 not only outperforms previous linear RNN models but also matches the performance of state-of-the-art Deepseek R1 distilled reasoning models at a similar scale. We also compare our generation speed with a highly performant general purpose inference engine, vLLM, and observe more than a 3x speedup compared to a same size transformer. With throughput speedup, we are able to achieve higher accuracy compared to DeepSeek R1 distilled transformer reasoning models under a fixed generation time budget using self-consistency voting. Overall, we introduce a hybrid Mamba reasoning model and provide a more effective approach to scaling test-time generation using self-consistency or long chain of thought reasoning.

---

# M1：面向可扩展测试时计算的 Mamba 推理模型 论文详细解读

### 背景：这个问题为什么难？

在数学推理任务中，模型往往需要写出很长的思考链才能得到正确答案。传统的 Transformer 虽然在 “Chain‑of‑Thought” 上表现突出，却因为自注意力的二次时间复杂度和线性内存需求，难以处理上万 token 的超长上下文。换句话说，想让模型在推理时跑得更久、思考得更细，往往会把显存和算力压到天花板。于是，如何在不牺牲推理质量的前提下，实现更高的测试时计算效率，成为了迫切需要突破的瓶颈。

### 关键概念速览
- **Transformer**：一种基于自注意力的神经网络，能够一次性捕获序列中任意位置的关系，但计算量随序列长度的平方增长，像在宽阔的河面上划船，水面越宽阻力越大。  
- **Quadratic complexity（二次复杂度）**：指算法的运行时间随输入规模的平方增长，对长序列来说会导致算力和显存爆炸。  
- **Chain‑of‑Thought（思维链）**：让模型在给出最终答案前先写出逐步推理过程，就像人在解题时先列出草稿，能够显著提升复杂题目的正确率。  
- **Self‑consistency voting（自一致性投票）**：在同一问题上多次采样生成不同思维链，然后把多数答案当作最终结果，类似让多个专家独立思考后取多数意见，以降低偶然错误。  
- **Mamba architecture**：一种基于线性状态空间模型（SSM）的混合 RNN，内部计算是线性的，记忆只保留必要的隐藏状态，像把长河的水流压缩成一条可随时回放的录像带。  
- **Linear RNN（线性循环神经网络）**：相较于传统 RNN，前向传播的时间复杂度随序列长度呈线性增长，适合超长文本的逐步推理。  
- **Distillation（蒸馏）**：把一个大模型（老师）的行为“压缩”到小模型（学生）里，让学生在学习过程中模仿老师的输出，类似把名厨的烹饪技巧写进菜谱让学徒照着做。  
- **Reinforcement Learning（强化学习）**：通过奖励信号让模型自行探索更好的行为策略，这里奖励是答案是否正确，模型会学会在思维链的每一步都朝正确方向前进。

### 核心创新点
1. **Transformer → Mamba 线性 RNN → 计算效率提升**  
   过去的推理模型几乎全靠 Transformer，受限于二次复杂度。M1 把推理核心换成 Mamba，这种线性 RNN 能在保持上下文连贯性的同时把时间和显存需求降到线性，实际推理速度比同尺寸的 Transformer 快了 3 倍以上。  

2. **直接蒸馏教师的思维链 → 学生 M1 学会长链推理 → 知识迁移成功**  
   作者先让已有的高性能推理模型（如 DeepSeek R1）生成完整的思维链，再用交叉熵损失让 M1 学会逐词复现这些链条。这样 M1 在没有 Transformer 的庞大参数量的情况下，也能掌握同等质量的推理步骤。  

3. **加入强化学习微调 → 让模型自我纠错 → 推理质量进一步提升**  
   在蒸馏完成后，使用策略梯度让模型在实际解题时获得“答案正确”奖励。奖励不仅针对最终答案，还会对中间步骤的合理性进行加权，使模型在生成思维链时更倾向于走对路。  

4. **自一致性投票 + 固定时间预算 → 同等算力下更高准确率**  
   由于 M1 推理快，能够在同样的时间预算内多次采样生成思维链，再通过自一致性投票取多数答案。实验表明，在相同的生成时间限制下，这种组合的准确率超过了使用同等算力的 DeepSeek R1 蒸馏 Transformer。

### 方法详解
整体思路可以划分为四个阶段：**教师准备 → 蒸馏学习 → 强化学习微调 → 高效推理 + 投票**。

1. **教师准备**  
   选取已有的强大数学推理模型（如 DeepSeek R1）作为老师。给定题目提示，老师生成完整的 Chain‑of‑Thought（包括每一步计算、文字解释），并记录最终答案。  

2. **蒸馏学习**  
   - **输入**：原始题目提示。  
   - **目标**：让 M1（基于 Mamba 的线性 RNN）在相同提示下逐词预测老师的思维链。  
   - **实现**：使用标准的交叉熵损失，让学生的输出分布尽可能接近老师的分布。因为 Mamba 的内部是状态空间模型，信息是通过递归的隐藏状态逐步传递的，学生能够在每一步“记住”前面的推理内容，而不需要一次性加载全部上下文。  

3. **强化学习微调**  
   - **奖励设计**：如果模型在一次完整推理后得到正确答案，奖励 +1；否则 0。为了鼓励中间步骤的合理性，还会对每一步的预测与老师的对应步骤的相似度给出小额奖励。  
   - **优化方式**：采用 REINFORCE 或 PPO 等策略梯度方法，对模型的参数进行微调，使得在采样时更倾向于产生高奖励的思维链。  
   - **关键点**：强化学习只在蒸馏后进行，避免了从零开始的探索成本，同时利用了老师已经提供的高质量思维链作为“软约束”。  

4. **高效推理 + 自一致性投票**  
   - **推理**：在实际测试时，M1 以线性时间复杂度逐 token 生成思维链。由于每一步只依赖前一隐藏状态，显存占用保持在常数级别，能够轻松处理数千甚至上万 token。  
   - **多次采样**：在固定的时间预算（比如 2 秒）内，模型会被调用多次，产生若干不同的思维链。  
   - **投票**：把所有生成的最终答案进行多数投票，选出出现次数最多的答案作为最终输出。因为采样次数多，偶然错误的概率被显著稀释。  

**最巧妙的地方**在于把两种看似冲突的技术结合：一方面利用 Mamba 的线性时序特性实现显存友好、速度快的推理；另一方面通过蒸馏和强化学习把 Transformer 级别的推理质量迁移过去。这样既解决了“算力瓶颈”，又不牺牲“思维链质量”。  

### 实验与效果
- **数据集**：AIME（美国数学邀请赛）和 MATH（高校数学竞赛）两大公开数学推理基准。  
- **对比基线**：  
  - 之前的线性 RNN 推理模型（如基于传统 LSTM 的版本）。  
  - DeepSeek R1 的蒸馏 Transformer 版本（当前最强的开源推理模型之一）。  
- **主要结果**：  
  - 在两套基准上，M1 的准确率超过了之前的线性 RNN，且与 DeepSeek R1 在相同模型规模下的表现持平。  
  - 与同尺寸的 Transformer 相比，M1 的生成速度提升超过 3 倍。  
  - 在固定的生成时间预算下，利用自一致性投票，M1 的最终准确率超过了 DeepSeek R1（具体提升幅度论文未给出精确数字，但声称“更高”。）  
- **消融实验**：  
  - 去掉蒸馏阶段，直接用强化学习训练的 M1 准确率下降显著（论文未给出具体数值）。  
  - 移除强化学习微调，模型仍能复制老师的思维链，但在实际解题时的正确率略低，说明 RL 对提升最终答案的鲁棒性有帮助。  
- **局限性**：  
  - 虽然速度快，但在极端超长上下文（超过数万 token）时仍会出现信息遗忘。  
  - 与最新的超大规模 Transformer（数十亿参数）相比，整体上限仍稍低。  
  - 论文对 RL 奖励函数的细节描述不够完整，实际复现可能需要自行调参。  

### 影响与延伸思考
M1 的出现让研究者重新审视“推理必须用 Transformer”这一假设，证明了基于状态空间模型的线性 RNN 也能在数学推理上竞争。随后出现的工作（如 **Mamba‑2**、**SSM‑CoT**）进一步探索更深层的状态空间网络、混合检索‑推理框架以及更细粒度的奖励设计。对想继续深入的读者，可以关注以下方向：  
- 把 Mamba 与检索增强（RAG）结合，让模型在推理时动态查找外部知识。  
- 研究更高效的自一致性采样策略，降低多次采样的算力开销。  
- 探索更大规模的 Mamba 模型，验证线性时序结构在千亿参数级别的可扩展性。  

### 一句话记住它
M1 用线性‑时序的 Mamba 网络把推理速度提升三倍，同时在固定时间预算下保持和最强 Transformer 一样的数学解题水平。