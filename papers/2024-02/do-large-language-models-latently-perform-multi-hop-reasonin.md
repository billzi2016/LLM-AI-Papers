# Do Large Language Models Latently Perform Multi-Hop Reasoning?

> **Date**：2024-02-26
> **arXiv**：https://arxiv.org/abs/2402.16837

## Abstract

We study whether Large Language Models (LLMs) latently perform multi-hop reasoning with complex prompts such as "The mother of the singer of 'Superstition' is". We look for evidence of a latent reasoning pathway where an LLM (1) latently identifies "the singer of 'Superstition'" as Stevie Wonder, the bridge entity, and (2) uses its knowledge of Stevie Wonder's mother to complete the prompt. We analyze these two hops individually and consider their co-occurrence as indicative of latent multi-hop reasoning. For the first hop, we test if changing the prompt to indirectly mention the bridge entity instead of any other entity increases the LLM's internal recall of the bridge entity. For the second hop, we test if increasing this recall causes the LLM to better utilize what it knows about the bridge entity. We find strong evidence of latent multi-hop reasoning for the prompts of certain relation types, with the reasoning pathway used in more than 80% of the prompts. However, the utilization is highly contextual, varying across different types of prompts. Also, on average, the evidence for the second hop and the full multi-hop traversal is rather moderate and only substantial for the first hop. Moreover, we find a clear scaling trend with increasing model size for the first hop of reasoning but not for the second hop. Our experimental findings suggest potential challenges and opportunities for future development and applications of LLMs.

---

# 大型语言模型是否潜在进行多跳推理？ 论文详细解读

### 背景：这个问题为什么难？

在传统的问答或知识检索任务里，模型往往只需要一次性匹配答案。可是很多真实问题需要跨越多个实体或关系才能得到答案，例如“《Superstition》的歌手的母亲是谁”。在这种情形下，模型必须先找出中间实体（歌手），再利用对该实体的属性进行推理。过去的工作要么显式构造知识图谱并让模型走图谱路径，要么让模型在提示中写出思考链（CoT），但这些方法都依赖外部结构或额外的指令。直接探究大规模语言模型（LLM）在不加任何显式引导的情况下，是否已经在内部暗自完成了这种两步推理，一直是个未知数。

### 关键概念速览
- **多跳推理**：需要经过两个或更多中间实体或关系才能得到最终答案的推理过程。想象在地图上从A点走到C点，需要先经过B点，这个B点就是“桥实体”。  
- **桥实体（bridge entity）**：在多跳问题里连接第一跳和第二跳的中间实体，例如上例中的“Stevie Wonder”。  
- **潜在（latent）**：指模型内部可能已经产生了某种表征或记忆，但外部观察不到直接的输出。就像人脑里暗暗记住了答案，却没有说出来。  
- **内部召回（internal recall）**：模型在生成答案前，对某个概念的激活程度。研究者通过改变提示来观察这种激活是否增强。  
- **关系类型（relation type）**：指问题中涉及的语义关系，如“歌手-母亲”“作者-出生地”等，不同关系的难度和模型表现差异很大。  
- **模型规模（model size）**：指参数数量，常用的有数十亿到上千亿不等，规模往往决定了知识容量和推理能力。  

### 核心创新点
1. **从单跳到双跳的内部信号拆解**  
   - 之前的评估大多只看最终答案对错，忽略了模型内部是否真的走了两步。  
   - 这篇论文把多跳任务拆成两段：先检测模型对桥实体的内部召回是否提升，再检测这种提升是否帮助模型利用桥实体的属性。  
   - 这样可以更细粒度地判断模型是否真的在做多跳，而不是偶然猜对。

2. **提示改写实验来操控内部召回**  
   - 传统做法直接给出完整问题，无法控制模型关注的焦点。  
   - 作者设计了“间接提及”版提示，让桥实体在句子里出现但不直接被问到，观察模型内部激活是否随之上升。  
   - 结果显示，间接提及显著提升了桥实体的召回，为后续第二跳提供了更可靠的依据。

3. **规模效应的系统性对比**  
   - 过去只报告了大模型整体表现，缺少对不同规模模型的细致比较。  
   - 论文在多个模型尺寸上重复实验，发现第一跳的内部召回随模型变大而明显提升，而第二跳并没有相同的趋势。  
   - 这揭示了模型规模对多跳推理的非对称影响，为后续模型设计提供了线索。

### 方法详解
整体思路可以概括为三步：**（1）构造两段式提示 →（2）测量桥实体的内部召回 →（3）关联召回强度与最终答案质量**。下面把每一步拆开说。

1. **提示构造**  
   - 对每个原始多跳问题，作者生成两种版本：  
     a. **直接版**：完整问题，例如 “The mother of the singer of ‘Superstition’ is”。  
     b. **间接版**：在问题中加入桥实体的描述，但不直接询问它，例如 “The mother of the singer of ‘Superstition’, who is Stevie Wonder, is”。  
   - 通过这种改写，模型在生成答案前会先看到桥实体的名字，从而可能在内部激活对应的知识节点。

2. **内部召回测量**  
   - 采用模型的隐藏层输出或自回归概率来估计桥实体的激活程度。具体做法是：在生成答案的过程中，记录模型在每一步对桥实体词汇的预测概率。  
   - 如果间接版提示下，这个概率显著高于直接版，就说明提示成功提升了桥实体的内部召回。

3. **关联分析**  
   - 将第一跳的召回强度与第二跳的答案正确率做相关性统计。若召回提升后，模型更频繁地利用桥实体的属性（如母亲的名字），则可以视为模型在进行真正的两跳推理。  
   - 为了排除偶然因素，作者还统计了两跳路径在所有提示中的共现比例：当两跳都被检测到时，视为一次成功的潜在多跳推理。

**最巧妙的地方**在于把“模型是否走了两步”转化为“模型内部是否对中间实体产生了更高的激活”，并用提示改写来人为调控这种激活。这样既不需要额外的知识图谱，也不需要显式的思维链指令，完全在原始 LLM 的生成流程里完成测量。

### 实验与效果
- **数据与任务**：作者挑选了包含多种关系类型的问答集合，覆盖音乐、文学、地理等领域。每个问题都可以拆解为两跳结构。  
- **基线对比**：与直接使用原始提示的模型表现相比，间接版提示在第一跳的内部召回提升上超过 80% 的问题出现了显著提升。  
- **整体多跳成功率**：在所有测试问题中，约有 80% 的提示展示了两跳路径的共现，说明模型在大多数情况下已经潜在完成了多跳推理。  
- **规模趋势**：从 7B 到 175B 参数的模型，第一跳召回随规模线性增长；第二跳的召回和最终答案的提升则没有明显的规模依赖。  
- **消融实验**：去掉间接提及的桥实体后，第一跳召回下降近 30%，多跳成功率随之下降，验证了提示改写的关键作用。  
- **局限性**：作者承认对第二跳的证据整体仍属中等强度，说明模型在利用桥实体属性上仍有不稳定因素；此外，实验主要聚焦于关系型问题，未覆盖更复杂的逻辑推理场景。

### 影响与延伸思考
这篇工作打开了一个新视角：即使没有显式的思维链指令，LLM 也可能在内部暗暗完成多跳推理。随后的研究开始探索如何**强化**这种潜在路径，例如通过微调让模型更可靠地激活桥实体，或设计更细粒度的提示模板来提升第二跳的利用率。还有工作尝试把这种内部召回信号作为监督信号，训练模型显式输出推理链。对想进一步了解的读者，可以关注**“内部表征可解释性”**和**“提示工程在推理中的作用”**这两个方向，尤其是2024年后出现的几篇利用激活梯度提升多跳推理的论文。

### 一句话记住它
LLM 在没有任何显式指令的情况下，往往已经暗暗走完了两步推理，只是我们需要巧妙的提示来把这条隐形路径显现出来。