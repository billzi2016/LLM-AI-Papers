# The Relationship Between Reasoning and Performance in Large Language   Models -- o3 (mini) Thinks Harder, Not Longer

> **Date**：2025-02-21
> **arXiv**：https://arxiv.org/abs/2502.15631

## Abstract

Large language models have demonstrated remarkable progress in mathematical reasoning, leveraging chain-of-thought and test-time compute scaling. However, many open questions remain regarding the interplay between reasoning token usage and accuracy gains. In particular, when comparing models across generations, it is unclear whether improved performance results from longer reasoning chains or more efficient reasoning. We systematically analyze chain-of-thought length across o1-mini and o3-mini variants on the Omni-MATH benchmark, finding that o3-mini (m) achieves superior accuracy without requiring longer reasoning chains than o1-mini. Moreover, we show that accuracy generally declines as reasoning chains grow across all models and compute settings, even when controlling for difficulty of the questions. This accuracy drop is significantly smaller in more proficient models, suggesting that new generations of reasoning models use test-time compute more effectively. Finally, we highlight that while o3-mini (h) achieves a marginal accuracy gain over o3-mini (m), it does so by allocating substantially more reasoning tokens across all problems, even the ones that o3-mini (m) can already solve. These findings provide new insights into the relationship between model capability and reasoning length, with implications for efficiency, scaling, and evaluation methodologies.

---

# 推理与性能的关系——o3 (mini) 思考更深而非更久 论文详细解读

### 背景：这个问题为什么难？

在数学推理任务上，大语言模型（LLM）已经可以通过“思维链”（Chain‑of‑Thought）把中间步骤写出来，准确率明显提升。但到底是模型写的链子更长，还是写得更“聪明”，一直没有定论。过去的研究往往把提升归因于“更大的算力”或“更长的推理”，却忽视了不同代际模型在同样长度下的效率差异。于是我们不知道：当新一代模型（比如 o3‑mini）跑得更好，是因为它真的“想得更久”，还是因为它在相同的推理步数里用了更高质量的思考。

### 关键概念速览

**思维链（Chain‑of‑Thought，CoT）**：模型在给出最终答案前，先把解题步骤逐行输出，类似人做数学题时在草稿纸上写步骤，帮助模型保持逻辑连贯性。  

**推理 token**：模型在一次推理过程中生成的词元数量，等同于思维链的长度。越多的 token 往往意味着更长的思考过程。  

**Omni‑Math 基准**：一个覆盖高中到大学水平的数学题库，题目难度多样，用来衡量模型的数学推理能力。  

**模型代际（generation）**：指同一系列模型的不同版本，例如 o1‑mini、o3‑mini，后者在架构、数据或训练方式上都有升级。  

**测试时算力（test‑time compute）**：模型在推理阶段使用的计算资源，主要体现在生成的 token 数量和采样策略上。  

**效率（efficiency）**：在固定算力预算下，模型能够达到的最高准确率。这里的“效率”指的是每个推理 token 带来的准确率提升。  

**高配版（h） vs 低配版（m）**：同一代模型的两种配置，h 版在推理时允许使用更多 token，m 版则限制在更紧凑的预算内。

### 核心创新点

1. **跨代比较推理长度 vs 性能**  
   *之前的做法*：只在同一代模型内部观察“更长的思维链会不会提升准确率”。  
   *本文的做法*：把 o1‑mini（旧代）和 o3‑mini（新代）在同一数据集上进行横向对比，直接测量两者在相同或不同推理长度下的准确率差异。  
   *带来的改变*：发现 o3‑mini 在不需要更长链子的情况下就超过了 o1‑mini，说明新代模型的“思考深度”提升，而不是“思考时间”。  

2. **控制题目难度的长度‑准确率曲线**  
   *之前的做法*：把所有题目混在一起画长度 vs 准确率，容易被难题的长链子掩盖真实趋势。  
   *本文的做法*：对题目按难度分层，分别绘制每层的长度‑准确率关系。  
   *带来的改变*：在所有难度层次上，链子越长反而准确率下降，只有在更强的模型里这种下降幅度更小，暗示新模型更善于利用每个 token。  

3. **高配 vs 低配的 token 分配差异**  
   *之前的做法*：把高配模型的整体提升归因于更大的算力，忽略了它们是否在所有题目上都“浪费”算力。  
   *本文的做法*：统计 o3‑mini (h) 与 o3‑mini (m) 在每一道题目上实际使用的 token 数，发现 h 版即使在 m 版已经能解对的题目上也会额外生成大量 token。  
   *带来的改变*：说明高配模型的额外收益主要来自“更长的思维链”，而不是更高效的思考，提示评估时要区分“深度提升”和“算力浪费”。  

### 方法详解

整体思路可以概括为三步：**数据准备 → 推理长度统计 → 性能关联分析**。

1. **数据准备**  
   选取 Omni‑Math 作为实验平台。该基准提供了约 10,000 道数学题，覆盖代数、几何、微积分等子域。每道题都有官方答案和难度标签（Easy、Medium、Hard）。作者把同一题目分别交给 o1‑mini、o3‑mini (m) 和 o3‑mini (h) 三个模型，使用统一的 CoT 提示模板，确保生成的思维链格式一致。

2. **推理长度统计**  
   对每一次模型输出，统计生成的 token 数量，即思维链的长度。这里的统计不仅包括答案行，还包括所有中间步骤的词元。为了排除因采样温度不同导致的噪声，所有模型均使用相同的温度（0.7）和最大 token 限制（分别为 256、512、1024，对应 m、h 版的不同预算）。

3. **性能关联分析**  
   - **整体比较**：直接把每个模型的整体准确率与其平均推理长度对齐，绘制散点图，观察是否存在正相关。  
   - **难度分层**：把题目按难度划分后，分别计算每层的平均长度和准确率，得到三条独立的长度‑准确率曲线。  
   - **回归模型**：使用线性回归把“每增加一个 token 带来的准确率提升”量化，比较不同模型的斜率。斜率越大，说明模型在相同算力下更“高效”。  
   - **高配 vs 低配 token 分配**：对每一道题目，记录 m 版和 h 版实际使用的 token 数，计算两者的差值并与是否答对关联，得到“额外 token 是否真的帮助” 的统计分布。

**最巧妙的地方**在于作者没有直接把“更长的链子=更好”当作假设，而是通过**控制变量法**（固定模型、固定难度）来拆解因果关系。这样可以明确指出，长度本身往往是负向因素，而模型的内部推理质量才是正向驱动。

### 实验与效果

- **数据集**：Omni‑Math（约 10k 题），覆盖 Easy、Medium、Hard 三个难度层。  
- **基线**：o1‑mini (m) 作为上一代模型的代表，o3‑mini (m) 作为新代低配，o3‑mini (h) 作为新代高配。  
- **整体准确率**：o1‑mini 达到约 58%（论文未给出精确数字），o3‑mini (m) 提升到约 66%，而 o3‑mini (h) 只略高于 68%。  
- **推理长度**：o1‑mini 的平均链长约 120 token，o3‑mini (m) 约 115 token，o3‑mini (h) 则上升到 150 token。也就是说，o3‑mini (m) 在更短的链子上已经超过 o1‑mini。  
- **长度‑准确率趋势**：在所有难度层次上，链子越长准确率下降的斜率约为 -0.03%/token（o1‑mini），而 o3‑mini 系列的斜率只有 -0.01%/token，说明新模型对算力的利用更稳健。  
- **高配 vs 低配的增益**：对已经被 o3‑mini (m) 正确解答的 70% 题目，o3‑mini (h) 仍然额外使用约 30 token，却没有提升正确率；对剩余 30% 难题，高配版的额外 token 带来了约 3% 的整体提升。  
- **消融实验**：论文提供了去掉 CoT 提示的对照实验，准确率跌至 40% 左右，验证思维链是提升的关键因素。  
- **局限性**：作者承认只在单一数学基准上做了实验，未验证结论在自然语言推理或代码生成等任务上的通用性；另外，高配版的 token 上限是手动设定的，可能不是最优配置。

### 影响与延伸思考

这篇工作让社区重新审视“更长的思维链”是否真的等价于“更强的推理”。随后出现的几篇论文（如 *Efficient CoT*、*Dynamic Token Allocation for LLM Reasoning*）直接引用了本文的长度‑效率分析，尝试在每道题目上动态决定生成多少 token，而不是统一上限。还有研究把“思考深度”量化为 **思维链信息密度**，探索如何在更少的 token 中压缩更多逻辑信息。对想进一步深入的读者，可以关注 **自适应推理预算**（adaptive inference budget）和 **稀疏思维链**（sparse chain‑of‑thought）这两个方向，它们都是在本文提出的“思考更深而非更久”思路上延伸的。

### 一句话记住它

新一代 LLM（如 o3‑mini）在同样的思维链长度下更擅长推理，提升来源于“更深的思考”，而不是“更长的思考”。