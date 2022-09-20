# Learn to Explain: Multimodal Reasoning via Thought Chains for Science   Question Answering

> **Date**：2022-09-20
> **arXiv**：https://arxiv.org/abs/2209.09513

## Abstract

When answering a question, humans utilize the information available across different modalities to synthesize a consistent and complete chain of thought (CoT). This process is normally a black box in the case of deep learning models like large-scale language models. Recently, science question benchmarks have been used to diagnose the multi-hop reasoning ability and interpretability of an AI system. However, existing datasets fail to provide annotations for the answers, or are restricted to the textual-only modality, small scales, and limited domain diversity. To this end, we present Science Question Answering (ScienceQA), a new benchmark that consists of ~21k multimodal multiple choice questions with a diverse set of science topics and annotations of their answers with corresponding lectures and explanations. We further design language models to learn to generate lectures and explanations as the chain of thought (CoT) to mimic the multi-hop reasoning process when answering ScienceQA questions. ScienceQA demonstrates the utility of CoT in language models, as CoT improves the question answering performance by 1.20% in few-shot GPT-3 and 3.99% in fine-tuned UnifiedQA. We also explore the upper bound for models to leverage explanations by feeding those in the input; we observe that it improves the few-shot performance of GPT-3 by 18.96%. Our analysis further shows that language models, similar to humans, benefit from explanations to learn from fewer data and achieve the same performance with just 40% of the data. The data and code are available at https://scienceqa.github.io.

---

# 学会解释：通过思维链进行多模态推理的科学问答 论文详细解读

### 背景：这个问题为什么难？

科学类选择题往往涉及文字、图表、实验图片等多种信息源，答案需要把这些碎片拼凑成完整的推理链。传统的机器阅读理解模型大多只看文本，缺乏跨模态的知识整合能力；即使是大规模语言模型（LLM），在多跳推理时也常直接给出答案，内部过程不可见，导致错误难以追溯。现有的科学问答数据集要么没有提供答案背后的解释，要么规模太小、只覆盖单一学科，无法检验模型的解释能力和跨模态推理水平。因此，如何让模型像人一样先“思考”、再给出答案，并且在多模态环境下保持一致性，成为亟待突破的瓶颈。

### 关键概念速览
- **多模态（Multimodal）**：指同时涉及文字、图片、图表等不同感官信息的任务，就像我们在课堂上既看教材又听老师讲解。  
- **思维链（Chain of Thought，CoT）**：模型在输出最终答案前，先把推理步骤写出来，类似解数学题时的草稿，帮助模型分解复杂问题。  
- **ScienceQA**：作者新建的约 21,000 条多模态选择题数据集，每题配有对应的课堂讲义和详细解释，覆盖多学科科学内容。  
- **Few‑shot Prompting**：在不做额外微调的情况下，只给模型少量示例就让它完成任务的技巧，常用于 GPT‑3 等大模型。  
- **UnifiedQA**：一种在多种问答数据上统一训练的模型，能够直接 fine‑tune 来适配新任务。  
- **解释注入（Explanation Injection）**：把人工或模型生成的解释直接拼到输入里，让模型“先读解释再答题”，相当于给它提供了思考的线索。  
- **数据效率（Data Efficiency）**：在相同性能下使用更少的标注数据，类似用更少的练习题就能掌握同样的知识点。

### 核心创新点
1. **构建大规模、多模态、带解释的科学问答基准**  
   之前的科学 QA 数据集要么只有文字，要么规模不足。作者收集并清洗了约 21k 条包含图片、图表的选择题，并为每题配上课堂讲义和逐步解释，形成了完整的“题‑讲‑解”三元组。这样模型既能练习跨模态信息检索，又能学习生成可解释的思维链。  

2. **让语言模型主动生成讲义和解释作为思维链**  
   传统的 CoT 只要求模型输出推理步骤，忽略了背景知识的组织。本文让模型先生成一段“lecture”（相当于提取或重述相关科学概念），再写出“explanation”（具体的推理过程），最后给出答案。相当于让模型先“上课”，再“做题”。  

3. **系统评估解释对模型性能的上限**  
   作者把人工标注的解释直接拼进输入，测出 GPT‑3 在 few‑shot 设置下性能提升了约 19%。这提供了一个“解释上限”，说明如果模型能完美利用解释，性能提升空间巨大。  

4. **展示解释提升数据效率的实验**  
   在只使用 40% 标注数据的情况下，加入生成的思维链后模型仍能达到使用全部数据时的表现，证明解释帮助模型更快学会概念和推理模式。  

### 方法详解
**整体框架**  
模型的输入是一道多模态选择题（文字+可能的图片），输出分三步：① 生成与题目相关的课堂讲义；② 基于讲义写出逐步解释；③ 给出最终答案。训练时使用教师强制（teacher forcing）把真实讲义和解释喂进去，让模型学习这条完整的思维链；推理时模型自行生成完整链条，再从中抽取答案。

**关键模块拆解**  

1. **多模态编码**  
   - 文本部分使用标准的词向量或 BERT‑style 编码。  
   - 图片通过预训练的视觉模型（如 ResNet）提取特征。  
   - 两种特征在一个跨模态 Transformer 中融合，得到统一的上下文表示。  
   类比：把文字和图片分别翻译成同一种语言，再交给同一个翻译官一起处理。

2. **思维链生成器**  
   - 基于统一的跨模态表示，模型先生成“lecture”。这一步相当于让模型检索或重述题目涉及的科学概念。  
   - 接着模型在同一解码器里继续生成“explanation”，每一步都可以参考前一步的输出，形成递进式推理。  
   - 最后模型输出答案选项的字母或文字。  
   这里的技巧是把三个子任务串成一个长序列，让模型在一次前向传播中一次性完成，避免多次调用导致信息丢失。

3. **解释注入实验**  
   - 为了测上限，作者把真实的 lecture+explanation 直接拼到输入前缀，模型只需要读取并给出答案。  
   - 这相当于给模型提供了“思考材料”，检验它是否能利用已有解释提升性能。

**最巧妙的设计**  
把讲义和解释视作思维链的两层结构，使模型在生成答案前先完成“知识准备”和“推理演练”。这种层次化的生成比一次性输出完整推理更易学习，因为模型可以先专注于概念抽取，再专注于逻辑推演，类似人类先复习教材再做练习。

### 实验与效果
- **数据集**：ScienceQA，约 21k 条多模态选择题，覆盖物理、化学、生物、地理等多个科学子领域，每题配有课堂讲义和解释。  
- **基线模型**：few‑shot GPT‑3（直接让模型回答），以及在 ScienceQA 上 fine‑tuned 的 UnifiedQA。  
- **核心结果**：  
  - 在 few‑shot GPT‑3 上加入模型生成的思维链，准确率提升 1.20%。  
  - 在 UnifiedQA fine‑tune 后加入思维链，提升 3.99%。  
  - 把人工解释直接注入输入，few‑shot GPT‑3 的性能跃升约 18.96%，展示了解释的潜在上限。  
  - 使用仅 40% 标注数据并加入思维链，模型仍能达到使用全部数据时的表现，说明解释显著提升了数据效率。  
- **消融实验**：原文未详细描述每个模块的独立贡献，但通过对比仅使用 lecture、仅使用 explanation、两者皆用的设置，作者证明两层思维链共同作用时效果最佳。  
- **局限性**：解释的质量仍受限于模型本身的生成能力；在极其专业或需要深层实验数据的题目上，生成的讲义可能不够准确。作者也指出，当前实验只在选择题上验证，开放式问答仍需进一步探索。

### 影响与延伸思考
ScienceQA 为多模态、可解释的科学问答提供了首个大规模基准，随后出现的工作多围绕“解释驱动的推理”展开，例如在医学问答中加入病例报告作为解释，或在法律文书分析中使用判例摘要作为思维链。未来的研究方向可能包括：  
- **自监督的解释生成**：让模型在没有人工解释的情况下自行学习生成高质量讲义。  
- **跨语言多模态解释**：把思维链扩展到非英语教材，实现多语言科学教育。  
- **解释可视化与交互**：把生成的解释以图文并茂的形式展示，帮助人机协同学习。  
如果想进一步了解，可关注近期在 ACL、EMNLP 上出现的 “Explainable QA” 系列论文，它们大多受 ScienceQA 思路启发。

### 一句话记住它
让模型先“上课再做题”，用生成的讲义和解释构建思维链，显著提升多模态科学问答的准确率和数据效率。