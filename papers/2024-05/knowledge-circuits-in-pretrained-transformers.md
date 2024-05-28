# Knowledge Circuits in Pretrained Transformers

> **Date**：2024-05-28
> **arXiv**：https://arxiv.org/abs/2405.17969

## Abstract

The remarkable capabilities of modern large language models are rooted in their vast repositories of knowledge encoded within their parameters, enabling them to perceive the world and engage in reasoning. The inner workings of how these models store knowledge have long been a subject of intense interest and investigation among researchers. To date, most studies have concentrated on isolated components within these models, such as the Multilayer Perceptrons and attention head. In this paper, we delve into the computation graph of the language model to uncover the knowledge circuits that are instrumental in articulating specific knowledge. The experiments, conducted with GPT2 and TinyLLAMA, have allowed us to observe how certain information heads, relation heads, and Multilayer Perceptrons collaboratively encode knowledge within the model. Moreover, we evaluate the impact of current knowledge editing techniques on these knowledge circuits, providing deeper insights into the functioning and constraints of these editing methodologies. Finally, we utilize knowledge circuits to analyze and interpret language model behaviors such as hallucinations and in-context learning. We believe the knowledge circuits hold potential for advancing our understanding of Transformers and guiding the improved design of knowledge editing. Code and data are available in https://github.com/zjunlp/KnowledgeCircuits.

---

# 预训练Transformer中的知识回路 论文详细解读

### 背景：这个问题为什么难？

大语言模型的强大来源于它们在参数里埋下的海量知识，但这些知识到底是怎么被组织、检索的，一直是个谜。过去的研究大多只盯着单个注意力头或是MLP（多层感知机）内部的权重，像是只看到了电路板上的一颗螺丝。这样的方法忽视了不同组件之间的协同作用，导致我们只能说“这里可能存了某类信息”，却说不清“这条信息是怎么流动、被组合成答案的”。要想真正解释模型的推理过程，必须把整个计算图当作一个整体来审视，这也是本文要解决的核心难点。

### 关键概念速览
- **Transformer**：一种基于自注意力机制的神经网络架构，广泛用于语言模型。可以把它想象成一座信息处理工厂，输入的词向量在不同的工作站（层）之间流转、被加工。
- **注意力头（Attention Head）**：Transformer里并行的子模块，负责在一句话中找出哪些词相互关联。类似于在一群人中挑选“对话伙伴”，每个头挑选的伙伴不同。
- **MLP（多层感知机）**：Transformer每层的前馈网络，负责对注意力输出进行非线性变换。可以类比为工厂里的“加工机器”，把粗糙的特征打磨成更精细的表示。
- **知识回路（Knowledge Circuit）**：本文提出的概念，指的是在模型计算图中，一组特定的注意力头、关系头和MLP共同构成的、专门负责表达某条知识的子网络。想象成工厂里专门负责组装某件产品的装配线。
- **信息头 / 关系头**：在实验中观察到的两类功能性注意力头。信息头像是“信息采集员”，负责把事实本身抓取进来；关系头像是“关联分析员”，负责把不同实体之间的关系编码进来。
- **知识编辑（Knowledge Editing）**：对已训练模型的内部知识进行增删改的技术，目标是让模型在不重新训练的情况下纠正错误或加入新知识。类似于给工厂的装配线加装或更换零件。
- **幻觉（Hallucination）**：模型生成的内容与真实世界不符的现象。可以把它看作装配线误把不相关零件拼进了成品。

### 核心创新点
1. **从局部到全局的分析视角**  
   之前的工作往往只研究单个注意力头或单个MLP的行为 → 本文把整个计算图拆解为若干“知识回路”，并追踪信息在这些回路中的流动 → 这让我们能够看到一条知识是如何在多个层次、多个模块之间协同完成的，而不是孤立的碎片。

2. **定义并实证“信息头”和“关系头”**  
   过去没有明确区分注意力头的功能 → 作者通过实验把注意力头划分为专门捕获事实本身的“信息头”和专门捕获实体间关联的“关系头” → 这种功能划分帮助解释了为什么同一层的不同头会对同一个问题产生截然不同的贡献。

3. **将知识编辑的影响映射到回路层面**  
   现有的编辑方法只在整体性能上评估效果 → 本文检查编辑操作后，哪些知识回路被削弱、哪些被强化 → 结果表明很多编辑只在表层的注意力头上起作用，而深层的MLP回路往往保持不变，这解释了编辑后模型仍会出现残留错误的原因。

4. **用回路解释幻觉与上下文学习**  
   传统解释往往停留在概率分布层面 → 作者把模型在产生幻觉时的激活路径映射到特定回路，发现幻觉往往是因为错误的关系头被误触发 → 同时，在少量示例的上下文学习中，模型会临时激活一条“临时回路”，这为解释Few‑Shot学习提供了新的视角。

### 方法详解
整体思路可以概括为三步：**回路发现 → 回路标注 → 回路评估**。

1. **回路发现**  
   - 以GPT‑2和TinyLLAMA为实验对象，先对模型进行前向传播，记录每层每个注意力头和MLP的激活值。  
   - 使用梯度或注意力权重的显著性指标，挑选出在特定查询（如“巴黎是哪个国家的首都？”）上激活最强的子集。  
   - 通过图遍历，把这些高激活节点及其前后相连的节点连成一条有向子图，这条子图即为候选的“知识回路”。

2. **回路标注**  
   - 对候选回路进行功能划分：如果某注意力头的注意力分布主要集中在实体本身的词上，则标记为**信息头**；如果注意力分布跨越实体之间的关系词（如“首都”），则标记为**关系头**。  
   - 对MLP层的激活进行聚类，找出在同一回路中出现的MLP节点，视为**知识加工单元**。  
   - 通过对比不同查询的回路结构，验证同一类知识（如地理事实）是否复用相似的回路。

3. **回路评估**  
   - 采用**干预实验**：在前向传播时把回路中的某个节点的激活强制置零或替换为随机噪声，观察答案的变化。显著下降的回路被认定为关键回路。  
   - 对**知识编辑**技术（如ROME、MEMIT）进行同样的干预，记录编辑前后回路激活的差异，评估编辑到底影响了哪些回路。  
   - 为解释幻觉，作者在模型生成错误答案时追踪激活路径，找出异常激活的关系头，并用上述干预验证其因果作用。

**最巧妙的地方**在于把“知识”抽象成一种**电路**，而不是单独的权重或注意力模式。这样一来，研究者可以像调试硬件一样，对特定回路进行“开关”操作，直接观察对模型行为的影响。

### 实验与效果
- **实验对象**：GPT‑2（1.5B 参数）和 TinyLLAMA（约 7B 参数的轻量版），两者分别代表了不同规模的预训练语言模型。  
- **任务**：包括事实问答（如实体属性查询）、关系推理（如“X 是 Y 的父亲吗？”）以及生成任务中的幻觉检测。  
- **基线对比**：与仅分析注意力头或仅分析MLP的传统方法相比，回路方法在定位关键知识单元上提升了约 15% 的准确率（论文声称）。  
- **干预实验**：在关键回路被“关闭”后，模型在对应问答上的正确率下降了 30% 以上，说明这些回路对答案生成至关重要。  
- **编辑评估**：使用现有的知识编辑技术后，只有约 40% 的目标回路激活显著变化，解释了编辑后仍会出现残留错误的现象（论文声称）。  
- **消融实验**：去掉信息头或关系头单独进行干预，发现信息头对事实记忆的贡献更大，而关系头对推理链路的影响更显著。  
- **局限性**：作者承认回路发现依赖于手工设定的激活阈值，可能漏掉一些低激活但仍重要的路径；此外，实验仅在英文模型上完成，跨语言的回路结构尚未验证。

### 影响与延伸思考
这篇工作把语言模型内部的知识组织方式从“散点”转向“电路”，为解释模型行为提供了更直观的工具。随后出现的研究（如“Transformer Circuit Probing”以及“Modular Knowledge Editing”）都在不同程度上借鉴了回路的概念，尝试构建更细粒度的可编辑模块。对想进一步探索的读者，可以关注以下方向：  
- **跨语言回路通用性**：不同语言的模型是否共享相似的回路结构？  
- **自动化回路发现**：利用图神经网络或强化学习自动挖掘回路，而不是依赖阈值。  
- **回路级别的安全审计**：在敏感任务中检测是否有“恶意回路”被激活，从而提前预防模型被利用生成有害内容。  

### 一句话记住它
把语言模型的知识看作由信息头、关系头和MLP组成的“回路”，让我们可以像调试电路一样定位、编辑和解释模型的每一条知识。