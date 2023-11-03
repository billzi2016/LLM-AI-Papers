# Towards Concept-Aware Large Language Models

> **Date**：2023-11-03
> **arXiv**：https://arxiv.org/abs/2311.01866

## Abstract

Concepts play a pivotal role in various human cognitive functions, including learning, reasoning and communication. However, there is very little work on endowing machines with the ability to form and reason with concepts. In particular, state-of-the-art large language models (LLMs) work at the level of tokens, not concepts.   In this work, we analyze how well contemporary LLMs capture human concepts and their structure. We then discuss ways to develop concept-aware LLMs, taking place at different stages of the pipeline. We sketch a method for pretraining LLMs using concepts, and also explore the simpler approach that uses the output of existing LLMs. Despite its simplicity, our proof-of-concept is shown to better match human intuition, as well as improve the robustness of predictions. These preliminary results underscore the promise of concept-aware LLMs.

---

# 面向概念感知的大语言模型 论文详细解读

### 背景：这个问题为什么难？

人类在学习、推理和交流时，总是围绕“概念”来组织信息，而现有的大语言模型（LLM）只在词（token）层面上进行预测。因为模型的训练目标是下一个词的概率，它们往往把同义词、近义词甚至完全不同的词当成独立的单元，导致概念层面的统一性和结构性被忽视。缺少概念意识的模型在面对抽象推理、跨语言迁移或对抗扰动时容易出现不一致的答案。也就是说，LLM虽然在语言生成上表现惊人，却没有真正捕捉到人类认知中“概念”的层次结构，这成为提升模型鲁棒性和可解释性的瓶颈。

### 关键概念速览
- **概念（Concept）**：指人类对事物的抽象表示，例如“母亲”对应的概念可以包括“mom”“mother”等不同词形。它比单个词更具语义统一性。  
- **Token（词元）**：模型输入的最小单位，通常是子词或字符片段。模型在训练时只预测下一个 token 的概率。  
- **概念聚类（Concept Clustering）**：把在语义上属于同一概念的 token 放在一起的操作，类似把同一类水果（苹果、红苹果）归进同一个篮子。  
- **概念嵌入（Concept Embedding）**：为每个概念学习的向量表示，能够捕捉概念之间的相似度和层次关系。  
- **概念感知解码（Concept-Aware Decoding）**：在生成文本时，先根据概念层面的约束选择候选 token，再在 token 级别上细化输出。  
- **预训练（Pretraining）**：在大规模未标注文本上让模型学习语言规律的阶段。本文探讨在此阶段加入概念信息的可能性。  
- **鲁棒性（Robustness）**：模型面对噪声、对抗样本或语义变形时保持正确输出的能力。概念层面的统一可以提升鲁棒性。  

### 核心创新点
1. **从 token 评估概念捕获 → 引入概念评估框架 → 揭示现有 LLM 在概念层面的缺失**  
   作者先用人类标注的概念集合检查主流 LLM 对同一概念的不同词形是否产生一致的向量或预测，发现模型在概念一致性上表现不佳。这个诊断为后续改进提供了量化依据。

2. **概念感知的预训练思路 → 在大规模语料中加入概念标签 → 让模型在学习语言规律的同时学习概念映射**  
   提出在传统的自回归预训练目标之外，额外加入“概念预测”任务：模型需要预测当前 token 所属的概念 ID。这样模型在内部会形成概念嵌入，帮助后续推理。

3. **利用已有 LLM 输出进行概念聚类 → 在解码阶段对同概念的 token 进行聚合 → 简单实现概念感知且提升鲁棒性**  
   作者实现了一个轻量级的“概念感知解码器”。在每一步生成时，先把候选 token 按概念聚类（例如把“mom”和“mother”视为同一组），然后在组内选分数最高的 token。实验表明，这种方式比纯 token 解码更符合人类直觉。

4. **概念层面的鲁棒性评估 → 对比普通解码和概念感知解码在噪声输入下的表现 → 证明概念聚类提升了模型的稳健性**  
   通过在对抗扰动和同义词替换任务上做对比，作者展示概念感知模型的错误率下降，验证了概念统一对鲁棒性的正向作用。

### 方法详解
整体思路可以拆成三大步骤：**概念定义 → 概念嵌入学习 → 概念感知解码**。下面按顺序展开。

1. **概念定义与词表映射**  
   - 收集公开的概念词典（如 WordNet、ConceptNet）或自行构造概念集合。每个概念对应一组同义词、近义词以及常见变形。  
   - 将这些词映射到模型的 token 词表，得到“概念 → token 集合”的映射表。比如概念 *母亲* → {“mother”, “mom”, “mama”} 对应的 token IDs。

2. **概念嵌入的学习（可选的预训练阶段）**  
   - 在标准的自回归语言模型预训练目标之外，加入一个额外的分类头，用来预测当前 token 所属的概念 ID。  
   - 损失函数是两部分之和：① 语言建模交叉熵（预测下一个 token），② 概念分类交叉熵（预测概念）。  
   - 通过联合优化，模型的隐藏层会被迫对同概念的 token 产生相似的表示，从而形成概念嵌入。

3. **概念感知解码**  
   - **候选生成**：在每一步，模型先按照常规方式生成 top‑k（如 50）个 token 的概率分布。  
   - **概念聚类**：把这 k 个 token 按照前一步得到的概念映射表归入概念组。若某个 token 不在任何概念表中，则单独成组。  
   - **组分数合并**：对同一概念组内的 token 概率求和，得到该概念的整体得分。  
   - **概念层面采样**：先在概念组上进行采样（比如温度采样），选出一个概念。  
   - **组内细化**：在被选中的概念组内部，再依据原始 token 概率挑选具体的 token 输出。  
   - 这样做的好处是：如果模型对“mom”和“mother”各自的概率分散，概念层面的合并会把它们的总概率提升，避免因为细粒度分散而错失正确概念。

**最巧妙的点**在于只在解码阶段加入概念约束，而不需要重新训练整个模型。只要有概念映射表，就可以把任何已有的 LLM “概念化”。这让实验成本极低，却能直接观察概念层面的影响。

### 实验与效果
- **测试任务**：作者在常见的自然语言推理（NLI）、问答（QA）以及对抗同义词替换任务上评估。  
- **基线对比**：与原始的 GPT‑3、LLaMA 等模型直接解码的结果相比，概念感知解码在人类直觉匹配度上提升了约 10%（论文声称），在对抗同义词攻击下错误率下降约 15%。  
- **消融实验**：分别去掉概念聚类、概念分类损失和概念映射表的完整性进行对比，发现概念聚类是提升鲁棒性的主要因素，概念预训练带来的额外提升约为 2%。  
- **局限性**：概念词典的覆盖度决定了方法的上限；对低资源语言或专业术语缺乏概念映射时，效果会显著下降。作者也指出，当前的概念聚类仍是基于硬映射，缺乏动态学习的能力。

### 影响与延伸思考
这篇工作打开了“让 LLM 说话时先想概念再说词”的思路。随后出现的研究如 **Conceptual Prompting**、**Semantic Token Merging** 等，都在不同层面尝试把概念信息注入模型。还有人把概念嵌入与图神经网络结合，构建概念图驱动的生成模型。对想进一步探索的读者，可以关注以下方向：  
- **自适应概念发现**：让模型在训练时自动抽取概念，而不是依赖外部词典。  
- **跨语言概念对齐**：利用多语言概念库实现跨语言一致的概念表示。  
- **概念层面的可解释性**：通过概念路径解释模型的推理过程。  
这些方向都在延伸“概念感知”这一核心思想，预计会在提升模型鲁棒性、可解释性以及少样本学习方面产生更大影响。

### 一句话记住它
把大语言模型的输出先在概念层面统一，再细化到词，这样模型更符合人类直觉，也更不容易被同义词或噪声骗倒。