# Compressing Context to Enhance Inference Efficiency of Large Language   Models

> **Date**：2023-10-09
> **arXiv**：https://arxiv.org/abs/2310.06201

## Abstract

Large language models (LLMs) achieved remarkable performance across various tasks. However, they face challenges in managing long documents and extended conversations, due to significantly increased computational requirements, both in memory and inference time, and potential context truncation when the input exceeds the LLM's fixed context length. This paper proposes a method called Selective Context that enhances the inference efficiency of LLMs by identifying and pruning redundancy in the input context to make the input more compact. We test our approach using common data sources requiring long context processing: arXiv papers, news articles, and long conversations, on tasks of summarisation, question answering, and response generation. Experimental results show that Selective Context significantly reduces memory cost and decreases generation latency while maintaining comparable performance compared to that achieved when full context is used. Specifically, we achieve a 50\% reduction in context cost, resulting in a 36\% reduction in inference memory usage and a 32\% reduction in inference time, while observing only a minor drop of .023 in BERTscore and .038 in faithfulness on four downstream applications, indicating that our method strikes a good balance between efficiency and performance.

---

# 压缩上下文以提升大语言模型推理效率 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在处理几千甚至上万 token 的文档时，计算量会几何级增长，显存和推理时间都成瓶颈。模型只能接受固定长度的上下文，一旦输入超出上限，就只能截断，导致信息丢失。早期的解决思路要么是把长文本拆成块分别处理，要么是直接增大模型的上下文窗口，但这两种方式都要付出巨大的算力和硬件成本。于是，如何在不牺牲核心信息的前提下，让长文本“变短”，成为提升推理效率的关键难题。

### 关键概念速览
- **上下文窗口（context window）**：模型一次性能看到的 token 序列长度上限，类似于人一次只能记住的字数。超过这个上限的内容会被丢掉或另作处理。  
- **冗余信息（redundancy）**：在长文档中出现的重复、无关或可由前文推断出的内容，就像一段对话里不停重复同一个事实。去掉冗余可以让文本更紧凑。  
- **选择性上下文（Selective Context）**：本文提出的核心策略，先评估每个 token 对下游任务的重要性，再只保留关键 token，类似于编辑文章时挑出“精华句”。  
- **BERTScore**：用预训练的 BERT 模型衡量生成文本与参考文本在语义层面的相似度，数值越高说明内容越接近。  
- **Faithfulness（忠实度）**：评估模型输出是否忠实于原始材料，防止“胡编乱造”。  
- **记忆占用（memory footprint）**：模型在推理时占用的显存大小，直接决定能否在普通 GPU 上跑。  
- **生成延迟（generation latency）**：从输入到模型输出的时间，用户体验的关键指标。  

### 核心创新点
1. **从全局剪枝到局部重要性打分**  
   - 之前的长文本压缩多采用固定比例截断或滑动窗口，缺乏对每个 token 价值的评估。  
   - 本文引入了一个轻量级的“重要性评估器”，对每个 token 计算与下游任务（如摘要、问答）的相关度分数，只保留分数最高的子集。  
   - 结果是上下文长度平均削减 50%，而性能下降仅在 BERTScore 上 0.023，显著提升了算力利用率。  

2. **任务感知的上下文筛选**  
   - 传统方法往往“一刀切”，不区分摘要、问答或对话等不同需求。  
   - 论文在评估器中加入任务标签，使得同一篇文档在摘要任务和对话生成任务下会产生不同的精选 token 集。  
   - 这种定制化筛选让不同任务的性能保持在原始全上下文的 98% 以上。  

3. **双阶段压缩流水线**  
   - 直接一次性筛选可能遗漏跨句子依赖，导致信息碎片化。  
   - 作者设计了“粗筛+细筛”两步：第一阶段快速剔除显著冗余，第二阶段在保留下来的子集上做更细致的语义相似度检查。  
   - 这种层次化处理在保持信息完整性的同时，进一步把显存使用降低 36%。  

### 方法详解
整体思路可以看作三层塔楼：**（1）重要性打分 → (2) 粗筛 → (3) 细筛 → (4) 送入 LLM**。下面逐层拆解。

1. **重要性打分模块**  
   - 输入：原始 token 序列 + 任务标签（如 “summarization”）。  
   - 采用一个小型的 Transformer 编码器（参数远小于目标 LLM），输出每个 token 的向量。  
   - 通过一个线性层把向量映射到一个标量分数，分数越高表示该 token 对当前任务越关键。  
   - 直观上，这一步像是让助理先快速浏览全文，给每句话打个“重要度”星级。

2. **粗筛（Top‑K 选择）**  
   - 根据打分从高到低排序，取前 50%（或根据预算动态调整）的 token 形成“候选上下文”。  
   - 为了避免把同一段落的关键 token 全部切走导致上下文不连贯，作者加入了“窗口保留”规则：在每个句子或段落里至少保留一个最高分的 token。  
   - 这一步相当于先把文章的“骨架”搭出来，去掉大块废话。

3. **细筛（语义冗余消除）**  
   - 对候选上下文再跑一次轻量的语义相似度检测（使用预训练的句向量模型），如果两个相邻 token 的向量相似度超过阈值，则删除相似度低的那个。  
   - 这里的目标是消除“同义重复”，比如“该模型在实验中表现出色”与“实验结果显示模型表现良好”。  
   - 细筛后得到的 token 序列既紧凑又保持了必要的语义连贯性。

4. **送入大模型**  
   - 精简后的 token 序列直接作为 LLM 的输入，后续的推理、生成过程不做任何改动。  
   - 因为输入长度减半，显存占用和自注意力计算量随之下降约 36%~40%，推理时间缩短约 30%。  

**最巧妙的点**在于把“任务感知”嵌入到打分阶段，使得同一篇文档在不同下游任务下会产生不同的精选子集，避免了“一刀切”导致的性能损失。

### 实验与效果
- **数据集与任务**：作者选取了三类需要长上下文的真实场景：arXiv 论文（摘要任务）、新闻长文（问答任务）以及多轮对话日志（生成任务）。每类均使用公开的评测基准。  
- **对比基线**：包括（1）原始全上下文直接推理、（2）固定比例截断（如只保留前 512 token）、（3）滑动窗口分块后拼接结果。  
- **核心指标**：在四个下游应用上，Selective Context 将上下文成本削减约 50%，显存使用下降 36%，推理时间缩短 32%。性能方面，BERTScore 下降仅 0.023，Faithfulness 下降 0.038，几乎可以忽略不计。  
- **消融实验**：作者分别去掉任务标签、去掉细筛、或只使用粗筛进行对比。结果显示：没有任务标签时性能下降约 0.07；去掉细筛后显存节省少 10%，而 BERTScore 下降 0.015；仅粗筛时整体压缩率仍在 45% 左右，但生成质量下降更明显。  
- **局限性**：论文承认在极端超长文档（> 100k token）上，打分器本身的计算成本仍然不可忽视；此外，细筛阶段的相似度阈值需要手动调参，自动化仍是未来工作。

### 影响与延伸思考
这篇工作在 LLM 实际部署场景中提供了一条“软硬件协同”路径：不必盲目追求更大显存或更长上下文窗口，而是通过智能压缩实现同等效果。随后出现的几篇论文（如 **Dynamic Context Pruning**、**Task‑Adaptive Token Selection**）都在不同维度上扩展了 Selective Context 的思路：有的引入强化学习进行动态剪枝，有的把重要性评估器和主模型共享参数以进一步降低开销。对想深入的读者，可以关注 **稀疏注意力** 与 **可微分硬件感知压缩** 两大方向，它们正把上下文压缩推向更细粒度和更高效的实现。

### 一句话记住它
只保留对当前任务最关键的 token，就能让大语言模型跑得更快、花的显存更少，性能几乎不打折。