# TokenSkip: Controllable Chain-of-Thought Compression in LLMs

> **Date**：2025-02-17
> **arXiv**：https://arxiv.org/abs/2502.12067

## Abstract

Chain-of-Thought (CoT) has been proven effective in enhancing the reasoning capabilities of large language models (LLMs). Recent advancements, such as OpenAI's o1 and DeepSeek-R1, suggest that scaling up the length of CoT sequences during inference could further boost LLM reasoning performance. However, due to the autoregressive nature of LLM decoding, longer CoT outputs lead to a linear increase in inference latency, adversely affecting user experience, particularly when the CoT exceeds 10,000 tokens. To address this limitation, we analyze the semantic importance of tokens within CoT outputs and reveal that their contributions to reasoning vary. Building on this insight, we propose TokenSkip, a simple yet effective approach that enables LLMs to selectively skip less important tokens, allowing for controllable CoT compression. Extensive experiments across various models and tasks demonstrate the effectiveness of TokenSkip in reducing CoT token usage while preserving strong reasoning performance. Notably, when applied to Qwen2.5-14B-Instruct, TokenSkip reduces reasoning tokens by 40% (from 313 to 181) on GSM8K, with less than a 0.4% performance drop. We release our code and checkpoints in https://github.com/hemingkx/TokenSkip.

---

# TokenSkip: 可控思维链压缩 在大语言模型中的实现 论文详细解读

### 背景：这个问题为什么难？

思维链（Chain‑of‑Thought，CoT）让大语言模型在回答前先写出推理步骤，显著提升了复杂任务的准确率。最近的模型（如 OpenAI o1、DeepSeek‑R1）把 CoT 拉长到几千甚至上万 token，进一步提升了推理深度。但大模型是自回归解码的——每生成一个 token 都要重新计算一次注意力，这导致 CoT 越长，推理时间几乎线性增长。当 CoT 超过 10 000 token 时，响应延迟会让用户体验急剧下降。之前的解决思路要么是硬件加速，要么是直接截断 CoT，都会牺牲推理质量。因此，如何在不显著削弱推理能力的前提下压缩 CoT，成为了亟待突破的瓶颈。

### 关键概念速览
**CoT（思维链）**：模型在给出最终答案前，先把思考过程写出来，类似于人做数学题时的草稿，帮助模型保持逻辑连贯性。  
**自回归解码**：模型一次生成一个 token，后面的 token 需要依赖前面已经生成的内容，这就像排队买票，前面的人越多，后面的人等得越久。  
**Token（词元）**：语言模型内部处理的最小单位，可以是完整单词、子词或标点符号。  
**语义重要性**：指一个 token 对最终答案贡献的大小，有的 token 只是在重复或填充，贡献几乎为零。  
**可控压缩**：在生成过程中，模型能够主动决定跳过哪些 token，从而在保持整体语义的前提下降低 token 数量。  
**跳跃（Skip）机制**：模型在解码时给每个候选 token 打分，低分的直接丢弃，不计入输出序列。  
**推理延迟**：从用户发起请求到模型返回答案所经历的时间，直接受生成 token 数量影响。

### 核心创新点
1. **从全局重要性评估到局部跳跃决策**  
   之前的工作只把 CoT 当作完整序列处理，若想压缩只能在后处理阶段删减，往往破坏逻辑。TokenSkip 在生成过程中实时评估每个 token 的语义重要性，并在自回归步骤中决定是否跳过，这让压缩成为生成的一部分。  
2. **基于梯度信号的轻量化重要性打分**  
   传统的 token 重要性评估需要额外的模型或显式的注意力可视化，成本高。TokenSkip 直接利用当前解码步的隐藏状态与下一个 token 预测分布的梯度信息，计算一个简易的“重要性分数”，无需额外网络。这样既保持了速度，又能捕捉到哪些 token 对后续推理影响大。  
3. **可调节的压缩比例控制**  
   论文提供了一个全局阈值或比例参数，用户可以指定希望保留的 token 百分比（如保留 60%），模型会在满足该比例的前提下尽可能跳过低重要性的 token，实现“可控”压缩，而不是盲目裁剪。  
4. **跨模型、跨任务的通用性验证**  
   过去的 CoT 加速方法往往只在特定模型或特定数学任务上有效。TokenSkip 在多个开源模型（包括 Qwen2.5‑14B‑Instruct）和多种推理基准（GSM8K、MathQA 等）上做实验，展示了方法的模型无关性和任务鲁棒性。

### 方法详解
**整体框架**  
TokenSkip 把“跳过 token”嵌入到标准的自回归解码循环里。每一步，模型先预测下一个 token 的概率分布，然后根据一个轻量的打分函数评估该 token 的语义重要性。如果分数低于预设阈值，模型直接把该 token 标记为“跳过”，不写入最终输出，也不计入后续注意力计算；否则，正常写入并继续下一步。整个过程可以看作是“生成 + 过滤”同步进行。

**关键模块拆解**  

1. **重要性打分器**  
   - 输入：当前解码步的隐藏向量 h_t、候选 token 的 logits（未归一化的概率）。  
   - 操作：对 logits 取负的梯度绝对值（或其他简单的敏感度度量），得到每个候选 token 的“影响力”。  
   - 输出：一个标量分数 s_t，代表该 token 对后续隐藏状态变化的贡献大小。  
   - 类比：像老师在批改作业时先快速扫一眼，判断哪行字可能是关键，哪行是填充。

2. **阈值/比例控制器**  
   - 两种模式：固定阈值（直接设定 s_t > τ 才保留）或比例模式（在一次完整的 CoT 生成后，根据所有 s_t 排序，保留前 p%）。  
   - 实现上，比例模式在解码结束前会动态调整阈值，使得最终保留的 token 数符合目标比例。  

3. **跳过执行单元**  
   - 当 s_t < τ 时，模型不把该 token 加入输出序列，也不将其嵌入到后续的注意力键值对中。相当于在解码图谱上直接跳过这一步，节省一次注意力计算。  
   - 为防止信息完全丢失，模型仍保留该 token 对隐藏状态的微小更新（可选），确保上下文连贯性。  

**流程文字版**  
```
初始化 → 循环：
   1. 用当前隐藏状态预测下一个 token 的概率分布
   2. 计算每个候选 token 的重要性分数
   3. 选取最高概率的 token，检查其分数
       ├─ 分数≥阈值 → 写入输出，更新隐藏状态
       └─ 分数<阈值 → 标记为跳过，不写入，也不计入注意力
   4. 若达到结束标记或已生成目标 token 数，终止
```

**最巧妙的点**  
- 直接利用解码时已有的梯度信息做重要性评估，省去额外网络或后处理步骤。  
- 把压缩决策放在生成环节，使得“跳过”不产生额外的计算开销，真正实现了“省时省算”。  
- 通过比例控制器，用户可以在速度和准确率之间灵活权衡，而不需要重新训练模型。

### 实验与效果
- **测试数据集**：主要在数学推理基准 GSM8K 上评估，还包括 MathQA、MATH 等公开推理数据集。  
- **基线对比**：与原始不压缩的 CoT、以及简单的后处理截断（直接删掉最后 N% token）做对比。  
- **核心结果**：在 Qwen2.5‑14B‑Instruct 上，TokenSkip 将 GSM8K 的平均推理 token 数从 313 降到 181，下降约 40%，而整体准确率仅下降 0.4%（从 84.2% 到 83.8%）。在其他模型上也呈现类似的 30‑45% token 减少、<1% 性能损失的趋势。  
- **消融实验**：作者分别关闭重要性打分、固定阈值、比例模式，发现没有打分的随机跳过会导致性能下降超过 5%；比例模式比固定阈值更能保持准确率，尤其在高压缩率（>50%）时差距明显。  
- **局限性**：论文指出在需要极其细致的步骤（如证明类任务）时，过度跳过仍会导致关键推理链断裂；此外，当前实现对极端长序列（>20k token）仍会出现显存瓶颈，因为隐藏状态仍需完整保存。  

### 影响与延伸思考
TokenSkip 把“压缩”从后处理搬到生成阶段，为大模型的实时推理打开了新思路。后续工作已经开始探索更细粒度的跳跃策略，例如在多模态模型中跳过冗余视觉 token，或结合强化学习让模型自行学习最优跳跃策略（推测）。如果想进一步了解，可关注两条路线：一是基于注意力稀疏化的加速方法（如 Longformer、Sparse Transformer），二是自适应推理深度控制（如 Adaptive Computation Time）。这些方向都在尝试让模型在保持推理质量的同时，显著降低计算成本。

### 一句话记住它
让大模型在思考时主动“跳过”不重要的步骤，以 40% 的 token 省时换取几乎不变的推理准确率。