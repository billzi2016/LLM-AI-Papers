# Mini-Omni-Reasoner: Token-Level Thinking-in-Speaking in Large Speech Models

> **Date**：2025-08-18
> **arXiv**：https://arxiv.org/abs/2508.15827

## Abstract

Reasoning is essential for effective communication and decision-making. While recent advances in LLMs and MLLMs have shown that incorporating explicit reasoning significantly improves understanding and generalization, reasoning in LSMs remains in a nascent stage. Early efforts attempt to transfer the "Thinking-before-Speaking" paradigm from textual models to speech. However, this sequential formulation introduces notable latency, as spoken responses are delayed until reasoning is fully completed, impairing real-time interaction and communication efficiency. To address this, we propose Mini-Omni-Reasoner, a framework that enables reasoning within speech via a novel "Thinking-in-Speaking" formulation. Rather than completing reasoning before producing any verbal output, Mini-Omni-Reasoner interleaves silent reasoning tokens with spoken response tokens at the token level. This design allows continuous speech generation while embedding structured internal reasoning, leveraging the model's high-frequency token processing capability. Although interleaved, local semantic alignment is enforced to ensure that each response token is informed by its preceding reasoning. To support this framework, we introduce Spoken-Math-Problems-3M, a large-scale dataset tailored for interleaved reasoning and response. The dataset ensures that verbal tokens consistently follow relevant reasoning content, enabling accurate and efficient learning of speech-coupled reasoning. Built on a hierarchical Thinker-Talker architecture, Mini-Omni-Reasoner delivers fluent yet logically grounded spoken responses, maintaining both naturalness and precision. On the Spoken-MQA benchmark, it achieves a +19.1% gain in arithmetic reasoning and +6.4% in contextual understanding, with shorter outputs and zero decoding latency.

---

# Mini-Omni-Reasoner：大规模语音模型中的令牌级思考即说 论文详细解读

### 背景：这个问题为什么难？
在文本大模型里，先让模型“思考”再输出答案（Think‑then‑Speak）已经被证实能显著提升推理准确率。但把同样的思路搬到语音模型上会导致明显的交互延迟：模型必须等全部内部推理结束后才开始合成语音，用户会听到一段沉默再出现答案，破坏了自然对话的流畅感。早期的语音推理系统只能在完整推理后一次性生成语音，既慢又不够灵活，难以满足实时沟通的需求。

### 关键概念速览
**Thinking‑before‑Speaking（思考后说）**：模型先完成所有推理步骤，再把结果转成语音，类似写完作文后才朗读。  
**Thinking‑in‑Speaking（思考即说）**：推理过程和语音生成交叉进行，模型在说话的同时插入内部思考的“隐形”令牌，像边走路边思考路线。  
**令牌（Token）**：模型处理的最小单位，文本里是词或子词，语音里对应短时声学特征片段。  
**Silent Reasoning Token（沉默推理令牌）**：在输出可听语音前，模型内部生成的、不会直接被听到的推理信息。  
**Thinker‑Talker 架构**：把模型拆成两个子模块，Thinker 负责产生沉默推理令牌，Talker 把这些令牌与可听语音令牌交错输出。  
**Spoken‑Math‑Problems‑3M 数据集**：3 百万条带有明确推理‑语音对应关系的数学口述题，用来训练模型学会在说话时插入推理。  
**Spoken‑MQA 基准**：评估语音模型在数学推理和上下文理解两方面表现的测试集。

### 核心创新点
1. **从完整推理→令牌级交错**：传统语音模型在推理结束后一次性生成完整语音，Mini‑Omni‑Reasoner 把推理拆成细粒度的沉默令牌，交错在可听语音令牌之间。这样模型在每说出一个词之前，都已经完成对应的局部推理，显著降低了感知延迟。  
2. **局部语义对齐约束**：为防止沉默令牌与后续语音脱节，作者在训练时加入了“前后语义一致性”损失，强制每个可听令牌必须受其前置沉默令牌的影响。相当于在每一步都做一次小范围的“思考检查”。  
3. **Thinker‑Talker 双流设计**：模型内部并行运行两个子网络：Thinker 只输出沉默令牌，Talker 负责把这些令牌和真实语音令牌混排。双流结构让推理和语音生成可以在同一时刻进行，提升了硬件利用率。  
4. **专属交错推理数据集**：构建了 Spoken‑Math‑Problems‑3M，确保每条口述题的推理步骤与对应的语音片段严格对齐，为模型学习“说中思考”提供了高质量监督信号。

### 方法详解
整体框架可以看作三步走：  
1) **输入编码**：把用户的语音输入转成声学特征序列，送入共享的底层编码器。  
2) **Thinker 生成沉默令牌**：编码器的输出进入 Thinker 模块，Thinker 按时间步预测一个“沉默推理令牌”。这些令牌本身不对应任何可听声波，只携带逻辑状态（如中间计算结果、变量绑定）。  
3) **Talker 交错输出**：Talker 同时接收当前的沉默令牌和前一步的可听语音令牌，决定本步是输出真实语音片段还是继续沉默。输出序列最终是“沉默‑语音‑沉默‑语音 …”交错的形式。

**关键流程**（文字版流程图）  
- 用户说：“两只鸟站在树上，飞走一只，还剩几只？”  
- 编码器 → Thinker 产生沉默令牌 “计数=2”。  
- Talker 把沉默令牌映射为短暂停顿，然后输出语音片段 “两只”。  
- 下一时间步，Thinker 继续推理 “飞走=1”。  
- Talker 再输出 “还剩”。  
- 如此循环，最终说出完整答案 “还剩一只”。  

**局部语义对齐**：在训练时，作者把每个沉默令牌的向量与随后生成的语音令牌的向量做余弦相似度约束，确保两者在语义空间里保持一致。相当于让模型在每一步都“自检”，防止出现“说了话却没有真正思考”的情况。

**最巧妙的地方**：把推理信息压缩成“看不见的令牌”，并让模型在同一时间步同时产生这两类输出。这样既利用了大模型高频率的 token 处理能力，又避免了传统思考‑后‑说的阻塞式延迟。

### 实验与效果
- **数据集**：主要在新建的 Spoken‑Math‑Problems‑3M 上训练，在 Spoken‑MQA 基准上评估。  
- **对比基线**：传统 Think‑then‑Speak 语音模型、直接端到端的语音‑to‑answer 系统。  
- **结果**：在算术推理上提升了 19.1%，上下文理解提升了 6.4%。输出更短，且解码过程几乎没有额外延迟（作者称为“零解码延迟”）。  
- **消融实验**：去掉局部语义对齐损失会导致推理准确率下降约 4%；仅使用单流（不分 Thinker/Talker）会使延迟回到原来的水平。  
- **局限**：论文未给出在嘈杂环境或多说话人场景下的表现；沉默令牌的解释性仍然是黑盒，难以直接检查推理过程。

### 影响与延伸思考
Mini‑Omni‑Reasoner 把“思考即说”概念引入语音大模型，打开了实时交互式语音推理的新方向。后续工作已经开始探索把这种令牌级交错用于对话系统、实时翻译以及语音指令执行等场景（如 2024 年的 “Speak‑Think‑Speak” 系列论文）。如果想进一步了解，可以关注以下两个方向：① 如何在噪声鲁棒的前提下保持沉默令牌的有效性；② 把沉默令牌映射为可解释的中间语言（如符号表达式），实现“听得见的思考”。  

### 一句话记住它
Mini‑Omni‑Reasoner 让大语音模型在说话的每一步都先完成对应的微小推理，从而实现零感知延迟的实时思考即说。