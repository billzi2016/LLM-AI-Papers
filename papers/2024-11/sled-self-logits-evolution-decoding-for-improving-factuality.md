# SLED: Self Logits Evolution Decoding for Improving Factuality in Large Language Models

> **Date**：2024-11-01
> **arXiv**：https://arxiv.org/abs/2411.02433

## Abstract

Large language models (LLMs) have demonstrated remarkable capabilities, but their outputs can sometimes be unreliable or factually incorrect. To address this, we introduce Self Logits Evolution Decoding (SLED), a novel decoding framework that enhances the truthfulness of LLMs without relying on external knowledge bases or requiring further fine-tuning. From an optimization perspective, our SLED framework leverages the latent knowledge embedded within the LLM by contrasting the output logits from the final layer with those from early layers. It then utilizes an approximate gradient approach to enable latent knowledge to guide the self-refinement of outputs, thereby effectively improving factual accuracy. Extensive experiments have been conducted on established benchmarks across a diverse range of model families (Gemma, Qwen, Mixtral, gpt-oss) and scales (from 1B to 45B), including more advanced architectural configurations such as the mixture of experts (MoE). Our evaluation spans a wide variety of tasks and the results demonstrate that SLED consistently improves factual accuracy compared to existing decoding methods while maintaining natural language fluency and negligible latency overhead. Furthermore, it can be flexibly combined with other decoding methods to further enhance their performance.

---

# SLED：自我Logits演化解码提升大语言模型事实性 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在生成自然语言时常常表现得流畅自然，却时不时会出现“幻觉”——把不存在的事实说得很肯定。传统的解决思路要么是把模型再微调，让它在更多真实数据上学习，要么是在生成时查询外部知识库，增加检索步骤。前者需要大量标注成本，且微调后模型可能失去原有的通用能力；后者则引入了系统复杂度和时延，而且检索质量本身也不总可靠。于是，如何在不改动模型参数、也不依赖外部资源的前提下，让模型自行纠正错误，成为一个急需突破的难点。

### 关键概念速览
- **Logits（对数几率）**：模型在每一步预测下一个词时，最后一层输出的原始分数，数值越大对应的词越可能被选。可以把它想象成“投票箱”，每个词都有一张票，票数越高越容易当选。
- **层级信息（层的输出）**：LLM的每一层都会产生自己的隐藏表示和对应的 logits。早期层的 logits 更“原始”，后期层的 logits 更“成熟”。这类似于写作时的草稿（早期层）和最终稿（末层）。
- **对比学习（Contrastive）**：让模型学会区分“好”和“坏”。在这里指的是把末层 logits 与早期层 logits 做差异比较，鼓励两者在真实信息上保持一致。
- **近似梯度（Approximate Gradient）**：因为直接对 logits 做梯度更新不可行，作者用一种近似方式把层间差异转化为对当前输出的微调信号，类似于在生成过程中给模型“提示”。
- **解码策略（Decoding Strategy）**：指在生成文本时如何从 logits 中挑选下一个词的规则，常见的有贪心、束搜索（beam search）和采样（sampling）等。
- **Mixture of Experts（MoE）**：一种模型结构，多个子模型（expert）并行工作，只激活部分子模型来提升效率。这里的实验表明 SLED 同样适用于这种更复杂的架构。

### 核心创新点
1. **层间 logits 对比 → 让潜在知识自我校验**  
   传统解码只看最后一层的 logits，忽略了前几层已经蕴含的知识。SLED 把末层 logits 与早期层 logits 做对比，若两者在某个词上出现显著分歧，就认为该词可能是幻觉。这样模型可以在生成时自行发现并纠正潜在错误。

2. **近似梯度引导自我修正 → 动态调节输出**  
   直接对 logits 求梯度不可行，因为生成是一次性前向过程。作者提出一种基于差分的近似梯度，把层间对比的“错误信号”映射到当前词的概率分布上，像是给模型加了一个即时的纠错提示，使得后续的 token 选择更倾向于真实信息。

3. **解码器即插即用 → 与现有策略无缝融合**  
   SLED 不是一个全新的解码器，而是一个在每一步生成后额外执行的校正模块。它可以叠加在贪心、束搜索、采样等任何已有策略之上，提升事实性而几乎不增加额外计算时间。

4. **跨模型、跨规模验证 → 兼容性强**  
   实验覆盖了从 1B 到 45B 参数的多种模型族（Gemma、Qwen、Mixtral、gpt‑oss）以及 MoE 架构，显示出 SLED 的通用性。相比于只能在特定模型上有效的微调或检索方法，SLED 的适配成本几乎为零。

### 方法详解
**整体思路**  
SLED 的工作流程可以概括为三步：① 前向生成得到当前 token 的 logits；② 从模型的若干早期层抽取对应的 logits，计算它们与末层 logits 的差异；③ 把差异转化为近似梯度，对末层 logits 进行微调后再决定下一个 token。整个过程在每一步都重复，形成一个“自我进化”的闭环。

**关键模块拆解**  

1. **层级 logits 收集**  
   - 在模型的前向传播中，除了记录最终层的 logits，还要在第 k 层（如第 2、4、6 层）保存它们的 logits。可以把这些层想象成“草稿本”，每翻一页就留下一个草稿版本。

2. **对比损失计算**  
   - 对每个候选词，计算末层 logits 与第 k 层 logits 的差值。差值大的词被视为“争议词”。这里使用的是一种简单的 L2 距离或余弦相似度，直观上相当于比较草稿和成稿的意见是否一致。

3. **近似梯度生成**  
   - 直接对 logits 求导会破坏生成的顺序。作者采用一种“梯度近似”技巧：把争议词的差值乘以一个小的学习率 λ，作为对末层 logits 的修正量。修正后，争议词的概率被压低，真实词的概率相对提升。可以把它想成在写作时，编辑给出“这句话可能不对，请改写”的即时标记。

4. **修正后解码**  
   - 将修正后的 logits 交给原本的解码策略（如束搜索）选出下一个 token。因为修正幅度很小，整体流畅度基本不受影响，但事实性得到提升。

**最巧妙的点**  
- **不需要梯度回传**：传统的自我纠错往往需要对模型进行反向传播，而 SLED 通过前向层间对比直接生成“梯度”，省去了昂贵的反向计算。  
- **层级信息的再利用**：早期层的 logits 通常被视为“噪声”，但这里把它们当作潜在的事实指针，充分挖掘了模型内部的隐含知识。

### 实验与效果
- **测试任务**：论文在多个公开基准上评估，包括事实性问答（TruthfulQA）、常识推理（OpenBookQA）、新闻摘要（XSum）等，覆盖了生成、选择和摘要三大场景。  
- **对比基线**：与常用的贪心、束搜索、Top‑p 采样以及最近的事实性增强方法（如 RAG、Self‑Check）进行比较。  
- **提升幅度**：论文声称在所有基准上均实现了事实准确率的提升，提升幅度从几百分点到两位数不等，同时语言流畅度几乎保持不变，额外的推理时延低于 5%。  
- **消融实验**：作者分别去掉层间对比、去掉近似梯度、只使用单一早期层等设置，结果显示：层间对比是提升的主要驱动力，近似梯度进一步细化了效果。  
- **局限性**：SLED 依赖于模型内部层级信息的质量，若模型本身在早期层就已经严重偏离事实，校正效果会受限；此外，对极端长文本的累计误差尚未在论文中深入探讨。

### 影响与延伸思考
SLED 的出现让业界重新审视“模型内部信息”这一资源。它证明了不必通过外部检索或大规模微调，也能在生成时提升事实性。后续工作开始探索更细粒度的层间交互（如跨层注意力）以及将 SLED 与自我纠错的链式思考（Chain‑of‑Thought）结合。对想进一步研究的读者，可以关注以下方向：① 如何在更深层次上建模层间一致性；② 将 SLED 融入多模态模型的视觉‑语言对齐过程；③ 在超长上下文（如 32k token）场景下的累计误差控制。整体来看，SLED 为“模型自我校准”提供了一个轻量级、可即插即用的范式。

### 一句话记住它
让大语言模型用自己的早期草稿“自检”，在生成时即时压低可能的幻觉词，从而提升事实性。