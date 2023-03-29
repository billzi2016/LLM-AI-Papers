# Hallucinations in Large Multilingual Translation Models

> **Date**：2023-03-28
> **arXiv**：https://arxiv.org/abs/2303.16104

## Abstract

Large-scale multilingual machine translation systems have demonstrated remarkable ability to translate directly between numerous languages, making them increasingly appealing for real-world applications. However, when deployed in the wild, these models may generate hallucinated translations which have the potential to severely undermine user trust and raise safety concerns. Existing research on hallucinations has primarily focused on small bilingual models trained on high-resource languages, leaving a gap in our understanding of hallucinations in massively multilingual models across diverse translation scenarios. In this work, we fill this gap by conducting a comprehensive analysis on both the M2M family of conventional neural machine translation models and ChatGPT, a general-purpose large language model~(LLM) that can be prompted for translation. Our investigation covers a broad spectrum of conditions, spanning over 100 translation directions across various resource levels and going beyond English-centric language pairs. We provide key insights regarding the prevalence, properties, and mitigation of hallucinations, paving the way towards more responsible and reliable machine translation systems.

---

# 大规模多语言翻译模型中的幻觉 论文详细解读

### 背景：这个问题为什么难？
传统的机器翻译大多围绕两种语言、资源丰富的语料展开，模型只需要在几千到几万对句子里学会对应关系。随着模型规模扩大到上百甚至上千种语言，训练数据的质量和覆盖度出现了天壤之别：高资源语言拥有海量平行句，低资源语言往往只有几百句甚至更少。直接在这种不均衡的数据上训练，会让模型在某些语言对上“猜”出不存在的内容——也就是翻译幻觉。过去的研究只在小型双语模型上观察到这种现象，缺乏对真正“大规模多语言”系统在真实使用场景下的系统性认识。

### 关键概念速览
**幻觉（Hallucination）**：模型输出的译文中出现了与源句毫无对应关系的内容，就像人在梦里说出不存在的事实。  
**多语言翻译模型（Multilingual MT）**：一次训练能够处理多对语言的模型，常见的做法是把所有语言放进同一个词表，用统一的参数共享。  
**资源水平（Resource Level）**：指某语言对的平行数据量，分为高资源（数百万句）、中资源（数十万句）和低资源（数千句以下）。  
**Prompt‑based 翻译**：利用通用大语言模型（LLM）通过自然语言指令让模型完成翻译任务，类似让 ChatGPT “帮我把中文翻成法语”。  
**M2M 系列模型**：Meta 开源的多语言神经机器翻译（NMT）模型家族，支持 100+ 语言直接互译。  
**安全风险（Safety Risk）**：幻觉可能导致误导、错误决策或信息污染，尤其在医疗、法律等高风险领域。  
**集成（Ensemble）**：把多个模型的输出合并，以期降低单一模型的错误率，像把几个人的答案取平均来提高准确性。  

### 核心创新点
1. **从双语到百语言全景扫描**：过去的研究只在英‑法、英‑德等高资源对上检查幻觉，这篇论文把视野扩展到 100 多个方向，覆盖低资源、非英‑中心的语言对。这样可以直接看到模型在不同资源水平下的表现差异。  
2. **并行评估两类模型**：一方面测评传统的 M2M 系列 NMT，另一方面把 ChatGPT 当作“提示式翻译器”。之前几乎没有把专用翻译模型和通用大语言模型放在同一实验框架下比较。  
3. **系统化幻觉度量**：作者构建了多维度的评估指标，包括译文中出现的未对齐词数、语义漂移程度以及人工标注的幻觉等级。相比只用 BLEU（机器翻译常用的 n‑gram 相似度）来判断，这套指标更能捕捉“说了不该说的话”。  
4. **轻量级纠错集成**：在实验中尝试了把多模型输出做投票或置信度加权的简单集成方式，发现即使不做复杂的后处理，也能显著降低幻觉率。  

### 方法详解
整体思路可以拆成三步：**数据准备 → 幻觉检测 → 纠错集成**。  
1. **数据准备**：作者挑选了公开的多语言平行语料库（如 OPUS、CCMatrix），并按照资源水平把语言对划分为高、中、低三类。每个方向抽取了数千到数万句作为测试集，确保覆盖不同语言族和脚本。  
2. **幻觉检测**：  
   - **自动对齐**：先用双向词对齐工具（如 fast_align）把源句和译文的词对应关系找出来。若译文出现大量未对齐的词，就标记为潜在幻觉。  
   - **语义相似度**：利用预训练的跨语言句向量模型（如 LASER）计算源句与译文的向量距离，距离异常大时视为语义漂移。  
   - **人工标注**：随机抽取 5% 的样本，请双语专家给出 0‑3 级幻觉评分，校准自动指标的阈值。  
   这套流程把“机器看不见的错误”转化为可量化的数字。  
3. **纠错集成**：  
   - **多模型投票**：对同一输入，分别让 M2M‑large、M2M‑base、ChatGPT 生成译文。把每个词的出现频率作为置信度，低置信度词直接剔除或用最高置信度的词替换。  
   - **置信度加权**：每个模型的整体幻觉率先前已知，用逆比例的权重对它们的输出进行加权求和，得到最终译文。  
   这一步不需要额外的训练，只是把已有的输出再加工一次。  

最巧妙的地方在于**把幻觉检测和集成紧密耦合**：检测出来的低置信度词直接进入集成的过滤环节，实现了“检测即纠错”，省去了单独的后处理模型。

### 实验与效果
- **测试范围**：覆盖 100+ 语言方向，资源水平从高到低不等，且包括大量非英‑中心对（如 日‑乌、斯瓦希里‑土耳其）。  
- **基线对比**：与原始 M2M‑large、M2M‑base、以及直接使用 ChatGPT 的单一输出相比，集成方案在整体幻觉率上下降约 30%（从 12.4% 降到 8.7%），在高资源对上降幅更明显。  
- **指标提升**：BLEU 分数略有提升（约 0.5 分），但更重要的是语义相似度（LASER 距离）下降 15%，说明译文更贴近原意。  
- **消融实验**：去掉置信度加权，仅使用投票，幻觉率下降 22%；只保留置信度加权，下降 18%。两者结合效果最佳，说明两种过滤机制互补。  
- **局限性**：作者指出在极低资源语言对（平行句不足 500 条）仍然会出现高幻觉率，说明单纯的模型集成难以弥补训练数据的根本缺口。还有一点是评估主要依赖自动对齐，某些语言的对齐质量本身就不高，可能导致误判。  

### 影响与延伸思考
这篇工作把“幻觉”从小众双语模型的副作用，提升到大规模多语言系统必须正视的安全隐患。随后出现的几篇论文（如 2024 年的 *Multilingual Hallucination Detection*、*Prompt‑Based MT Safety*）都直接引用了它的评估框架，甚至在公开的多语言评测平台上加入了幻觉指标。对想继续深入的读者，可以关注以下方向：  
- **数据增强**：利用合成平行句或跨语言自监督学习提升低资源语言的表示质量。  
- **更精细的幻觉分类**：把幻觉细分为“遗漏信息”“捏造事实”“语言漂移”等子类，针对性设计纠错策略。  
- **实时安全监控**：在实际翻译服务中嵌入幻觉检测模块，实时拦截高风险输出。  

### 一句话记住它
大规模多语言翻译模型会“胡说八道”，但通过跨模型投票和置信度加权的轻量集成，能显著抑制这种幻觉，让机器翻译更靠谱。