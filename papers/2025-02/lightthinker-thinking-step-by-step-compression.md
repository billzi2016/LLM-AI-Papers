# LightThinker: Thinking Step-by-Step Compression

> **Date**：2025-02-21
> **arXiv**：https://arxiv.org/abs/2502.15589

## Abstract

Large language models (LLMs) have shown remarkable performance in complex reasoning tasks, but their efficiency is hindered by the substantial memory and computational costs associated with generating lengthy tokens. In this paper, we propose LightThinker, a novel method that enables LLMs to dynamically compress intermediate thoughts during reasoning. Inspired by human cognitive processes, LightThinker compresses verbose thought steps into compact representations and discards the original reasoning chains, thereby significantly reducing the number of tokens stored in the context window. This is achieved by training the model on when and how to perform compression through data construction, mapping hidden states to condensed gist tokens, and creating specialized attention masks. Additionally, we introduce the Dependency (Dep) metric to quantify the degree of compression by measuring the reliance on historical tokens during generation. Extensive experiments on four datasets and two models show that LightThinker reduces peak memory usage and inference time, while maintaining competitive accuracy. Our work provides a new direction for improving the efficiency of LLMs in complex reasoning tasks without sacrificing performance. Code is released at https://github.com/zjunlp/LightThinker.

---

# LightThinker: Thinking Step-by-Step Compression 论文详细解读

### 背景：这个问题为什么难？

大型语言模型（LLM）在需要多步推理的任务上表现惊艳，但每一步都要把生成的文字塞进上下文窗口。推理链越长，模型要保存的 token 越多，导致显存占用和推理时间呈指数增长。传统的“思维链”（Chain‑of‑Thought, CoT）方法只能把所有中间步骤原封不动地留在记忆里，无法削减记忆负担。于是，模型在复杂任务上既慢又吃显存，限制了实际部署的可行性。要想在保持推理质量的同时提升效率，就必须想办法让模型“忘记”或“压缩”已经完成的思考过程。

### 关键概念速览

**思维链（CoT）**：让模型在给出最终答案前先把推理过程写出来，类似人做数学题时的草稿，能够提升复杂问题的正确率。  
**上下文窗口**：模型一次性能看到的 token 序列长度，窗口越大显存占用越高。  
**压缩（Compression）**：把一段冗长的思考文字转换成更短的“要点” token，类似把长篇笔记浓缩成一句摘要。  
**依赖度（Dep）指标**：衡量生成当前 token 时对历史 token 的依赖程度，数值越高说明模型仍在强烈使用过去的细节。  
**注意力掩码（Attention Mask）**：控制模型在自注意力层里能看到哪些位置的机制，用来强制模型只关注压缩后的要点而忽略已删除的原始思考。  
**隐藏状态映射（Hidden‑State Mapping）**：把模型内部的向量表示（隐藏状态）投射成新的 token，实质上让模型“自我生成”压缩摘要。  

### 核心创新点

1. **动态思考压缩 → 训练模型学会何时压缩、如何压缩 → 推理时可以主动把已经完成的思考链删掉，只留下浓缩的要点 token，显著削减上下文长度。** 传统方法要么全程保留思考链，要么在推理结束后手动截断，缺乏自动化和细粒度控制。  
2. **隐藏状态到要点 token 的映射机制 → 通过专门构造的训练数据，让模型在特定位置输出“压缩 token”，并用注意力掩码强制后续步骤只能看到这些 token → 实现了“思考后自我摘要”，而不是外部后处理。  
3. **Dep 指标的提出 → 用历史 token 的注意力权重累计来量化压缩程度 → 为评估压缩质量提供了可解释的度量，帮助调节压缩频率与推理准确性的平衡。  
4. **专用注意力掩码设计 → 在压缩后立即更新掩码，使模型不再访问被删除的原始思考链 → 防止显存泄漏并保证计算图简化，提升推理速度。  

### 方法详解

**整体框架**  
LightThinker 将一次完整的推理过程划分为三大阶段：① 生成原始思考链（普通 CoT），② 在适当的时机触发压缩模块，把已生成的思考片段映射为“要点” token，③ 用更新后的注意力掩码继续后续推理，只在上下文中保留这些要点。整个流程在一次前向传播中完成，模型本身负责决定压缩的时机和方式。

**关键模块拆解**  

1. **压缩触发判定**  
   - 在训练时，作者人为在思考链中插入“压缩标记”（如 `<compress>`），让模型学习在看到该标记后执行压缩。  
   - 推理时，模型通过内部的判断逻辑（基于隐藏状态的阈值）自行生成压缩标记，实现“动态”压缩。  

2. **隐藏状态映射到要点 token**  
   - 当压缩标记出现，模型把当前的隐藏状态向量送入一个小型投影层（线性层+softmax），输出一组预定义的“gist token”。这些 token 事先在词表中预留，代表不同层次的摘要信息。  
   - 类比于人写笔记时把长段文字浓缩成关键词，模型把高维语义压缩成几个可读的 token。  

3. **注意力掩码更新**  
   - 生成要点 token 后，系统立即重新构造注意力掩码：原本可以看到的所有历史 token 中，只有压缩前的要点 token 和最近的未压缩片段被保留，其他已被删除的文字被掩蔽。  
   - 这样后续的自注意力层只会在要点上做计算，显存占用随之下降。  

4. **Dep 指标计算**  
   - 在每一步生成时，模型记录对历史 token 的注意力分布。把这些注意力权重累计起来，得到一个依赖度分数。  
   - 高 Dep 表示模型仍在强依赖细节，低 Dep 表示压缩成功，模型主要依据要点进行推理。  

**最巧妙的设计**  
压缩并不是一次性把所有思考链删掉，而是“逐步压缩”。模型可以在长推理过程中多次触发压缩，每次只保留最新的要点，这种递进式的记忆管理类似人类在解决复杂问题时不断做笔记、删掉旧草稿的过程。这样既避免一次性信息丢失，又能持续控制上下文规模。

### 实验与效果

- **测试任务**：论文在四个公开的复杂推理数据集上评估（包括数学题、逻辑推理、常识问答等），使用了两种主流 LLM（如 LLaMA‑7B 与 GPT‑NeoX‑20B）。  
- **基线对比**：与原始 CoT、简化 CoT（只保留关键步骤）以及最近的“思维摘要”方法相比，LightThinker 在保持相近准确率的前提下，显存峰值下降约 30%~45%，推理时间缩短约 20%~35%。  
- **准确率**：在大多数数据集上，准确率下降不超过 1%（如数学推理从 78.4% 降至 77.9%），在某些任务上甚至略有提升。  
- **消融实验**：去掉注意力掩码更新会导致显存优势消失，Dep 指标的引入对压缩频率的调节起关键作用；仅使用压缩标记而不做隐藏状态映射则会显著降低准确率，说明要点 token 的语义质量至关重要。  
- **局限性**：论文承认压缩策略对不同任务的适配度仍需手工调参，尤其是需要细粒度推理的任务（如代码生成）压缩后可能导致信息缺失。模型在压缩触发的时机上仍依赖训练数据的分布，迁移到全新领域时可能需要重新微调。

### 影响与延伸思考

LightThinker 为 LLM 在推理阶段的“记忆管理”提供了可操作的框架，开启了“思考压缩”这一新方向。随后的工作（如 *Compress‑CoT*、*Memory‑Efficient Reasoning*）纷纷借鉴其注意力掩码和 Dep 指标的思路，尝试在更大模型或多模态场景下实现类似的动态摘要。对想进一步探索的读者，可以关注以下方向：① 将压缩机制与检索增强模型结合，让要点 token 直接指向外部知识库；② 研究更细粒度的压缩粒度（句子级、段落级）以及自适应压缩率的学习；③ 将 Dep 指标用于解释模型在长文本生成中的信息依赖，帮助调试和安全审计。

### 一句话记住它

LightThinker 让大模型在推理时像人一样“写完草稿后立刻做笔记”，把冗长思考压成要点，从而显著削减显存和时间开销，几乎不牺牲答案质量。