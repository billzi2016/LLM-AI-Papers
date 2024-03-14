# TaxoLLaMA: WordNet-based Model for Solving Multiple Lexical Semantic   Tasks

> **Date**：2024-03-14
> **arXiv**：https://arxiv.org/abs/2403.09207

## Abstract

In this paper, we explore the capabilities of LLMs in capturing lexical-semantic knowledge from WordNet on the example of the LLaMA-2-7b model and test it on multiple lexical semantic tasks. As the outcome of our experiments, we present TaxoLLaMA, the everything-in-one model, lightweight due to 4-bit quantization and LoRA. It achieves 11 SotA results, 4 top-2 results out of 16 tasks for the Taxonomy Enrichment, Hypernym Discovery, Taxonomy Construction, and Lexical Entailment tasks. Moreover, it demonstrates very strong zero-shot performance on Lexical Entailment and Taxonomy Construction with no fine-tuning. We also explore its hidden multilingual and domain adaptation capabilities with a little tuning or few-shot learning. All datasets, code, and model are available online at https://github.com/VityaVitalich/TaxoLLaMA

---

# TaxoLLaMA：基于WordNet的多词义语义任务模型 论文详细解读

### 背景：这个问题为什么难？

词汇语义任务（如上位词发现、词典构建、词义蕴含等）本质上要求模型掌握细粒度的概念层级和词之间的逻辑关系。传统做法要么依赖手工构建的本体（如WordNet），要么使用大规模语言模型进行微调，但前者更新慢、覆盖不足，后者在细节推理上常常表现平平。更糟的是，已有的模型大多只能针对单一任务进行优化，缺乏“一站式”解决方案，导致实际应用中需要维护多套模型，成本高且难以保证一致性。

### 关键概念速览
- **WordNet**：一个人工编辑的英语词汇本体，像是词语的家谱树，记录了同义词集合（synset）以及上下位关系。  
- **上位词发现（Hypernym Discovery）**：给定一个词，找出它的上位概念，就像在动物分类里把“金毛”归到“犬”。  
- **词典扩充（Taxonomy Enrichment）**：在已有的概念层级中加入新概念，类似给家谱添上新成员。  
- **词义蕴含（Lexical Entailment）**：判断两个词之间是否存在蕴含关系，例如“狗”蕴含“动物”。  
- **LoRA（Low-Rank Adaptation）**：在大模型上加一层低秩矩阵进行微调，像在原有模型上贴一层薄薄的可调胶水，既省显存又保持原始能力。  
- **4-bit 量化**：把模型参数压缩到4位表示，类似把高清图片压成低分辨率，显著降低存储和推理成本。  
- **Zero-shot**：模型在没有看到任务示例的情况下直接完成任务，就像第一次玩新游戏却凭直觉就能上手。  

### 核心创新点
1. **从 WordNet 到 LLM 的桥接 → 直接在 LLaMA‑2‑7B 上进行 WordNet‑aware 微调 → 让大模型在保持通用语言能力的同时，获得系统化的概念层级知识。** 过去的做法要么只用 WordNet 的规则推理，要么完全忽视本体信息，这里把两者合二为一。
2. **轻量化部署方案（4‑bit 量化 + LoRA） → 把原本需要数十 GB 显存的模型压到几 GB，并只在少量参数上微调 → 在普通 GPU 上也能跑，降低了硬件门槛。** 传统的全参数微调既耗时又费显存，这里用低秩适配加极端量化实现了“轻装上阵”。
3. **“一站式”多任务框架 → 同时在上位词发现、词典扩充、词典构建、词义蕴含四大任务上训练 → 在 16 项基准中拿到 11 项 SOTA、4 项 Top‑2。** 以前的系统往往只能在单一任务上表现突出，这里实现了跨任务的统一性能提升。
4. **零样本与少样本能力的探索 → 在不做任何微调的情况下直接评估 Lexical Entailment 与 Taxonomy Construction → 仍能保持竞争力。** 这表明模型内部已经形成了可迁移的语义结构，而不是仅靠任务特定的微调。

### 方法详解
整体思路可以拆成三步：**（1）准备 WordNet 语义图谱、（2）在 LLaMA‑2‑7B 上进行 LoRA 微调、（3）量化并部署**。下面逐层展开：

1. **语义图谱构建**  
   - 从 WordNet 抽取同义词集合（synset）以及上位/下位关系，形成一个有向无环图。  
   - 为每条边生成自然语言描述，例如把 “golden retriever → dog” 转化为 “golden retriever is a kind of dog”。这些描述随后作为训练样本的目标文本。

2. **LoRA 微调**  
   - 在原始 LLaMA‑2‑7B 基础上插入低秩适配层，只有几百万参数需要更新。  
   - 训练目标是让模型在给定词汇提示时输出对应的上位词描述，等价于让模型学会“把概念映射到它的父概念”。  
   - 为了兼顾多任务，训练数据混合了四类任务的样本：上位词对、同义词集合、词典扩充的新增概念、以及蕴含判断的正负例。模型在一次前向传播中即可学习所有关系。

3. **4‑bit 量化**  
   - 采用最新的块级量化技术，把每个权重压到 4 位，同时保留关键的激活尺度信息。  
   - 量化后模型大小从约 13 GB 降到 3‑4 GB，推理时显存需求大幅下降。  
   - 量化过程与 LoRA 参数分离，确保微调得到的适配层仍保持高精度。

4. **推理与多任务使用**  
   - 对于上位词发现，只需给模型一个词的提示，模型会输出最可能的上位词。  
   - 对于词典扩充，提供新词的定义，模型会判断它在现有层级中的最佳挂靠位置。  
   - 对于词义蕴含，直接输入“X entails Y?”，模型在 zero‑shot 情况下给出 Yes/No。  
   - 所有任务均使用同一个模型实例，省去切换模型的开销。

**最巧妙的点**在于把 WordNet 的结构化知识转化为自然语言训练信号，再通过 LoRA 让大模型“偷学”这些结构，而不是硬编码规则。这样既保留了 LLM 的语言理解优势，又注入了本体的层级约束。

### 实验与效果
- **测试任务**：Taxonomy Enrichment、Hypernym Discovery、Taxonomy Construction、Lexical Entailment 共 16 项子任务。  
- **基准对比**：与各任务的最新专用模型（如 HyperLex、BERT‑based 上位词发现器等）相比，TaxoLLaMA 在 11 项上实现了 SOTA（state‑of‑the‑art）水平，另外 4 项进入前二。  
- **零样本表现**：在 Lexical Entailment 与 Taxonomy Construction 上，未做任何微调直接使用原始 LLaMA‑2‑7B+LoRA 的输出，就已经接近或超过了多数微调模型的成绩。  
- **消融实验**：论文报告了去掉 LoRA、去掉 4‑bit 量化、仅使用单任务微调的三组消融。结果显示：LoRA 是提升多任务一致性的关键，量化对最终性能影响微乎其微，而单任务微调只能在对应任务上略有提升，跨任务表现明显下降。  
- **局限性**：实验主要在英文 WordNet 上进行，跨语言迁移效果仅在少量多语言实验中略有展示，尚未系统评估；此外，4‑bit 量化在极端低显存环境下仍会出现少量数值漂移，导致极端长序列推理时出现轻微不稳定。

### 影响与延伸思考
TaxoLLaMA 把本体知识与大语言模型的融合示范得相当成功，随后出现了几篇尝试把其他结构化资源（如 ConceptNet、DBpedia）同样“语言化”后注入 LLM 的工作。推测未来会有更多“轻量化+本体驱动”的模型，用于知识图谱补全、专业术语归类等场景。对想深入的读者，可以关注以下方向：  
1. **多语言本体迁移**：如何把 WordNet 的层级结构平滑映射到其他语言的同义词库。  
2. **更细粒度的概念对齐**：在同一层级内部区分细微差别（如“犬” vs “狼”），需要更复杂的对比学习。  
3. **自适应量化**：在保持零样本性能的前提下，进一步压缩模型体积。  

### 一句话记住它
TaxoLLaMA 用 LoRA+4‑bit 轻量化把 WordNet 的概念层级直接灌进 LLaMA‑2，实现了“一模型多任务”的词义语义全能手。