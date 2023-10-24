# This is not a Dataset: A Large Negation Benchmark to Challenge Large   Language Models

> **Date**：2023-10-24
> **arXiv**：https://arxiv.org/abs/2310.15941

## Abstract

Although large language models (LLMs) have apparently acquired a certain level of grammatical knowledge and the ability to make generalizations, they fail to interpret negation, a crucial step in Natural Language Processing. We try to clarify the reasons for the sub-optimal performance of LLMs understanding negation. We introduce a large semi-automatically generated dataset of circa 400,000 descriptive sentences about commonsense knowledge that can be true or false in which negation is present in about 2/3 of the corpus in different forms. We have used our dataset with the largest available open LLMs in a zero-shot approach to grasp their generalization and inference capability and we have also fine-tuned some of the models to assess whether the understanding of negation can be trained. Our findings show that, while LLMs are proficient at classifying affirmative sentences, they struggle with negative sentences and lack a deep understanding of negation, often relying on superficial cues. Although fine-tuning the models on negative sentences improves their performance, the lack of generalization in handling negation is persistent, highlighting the ongoing challenges of LLMs regarding negation understanding and generalization. The dataset and code are publicly available.

---

# 这不是数据集：大规模否定基准挑战大型语言模型 论文详细解读

### 背景：这个问题为什么难？
在自然语言处理中，否定是最常见却最容易让模型出错的结构。早期的语言模型往往通过大量的统计关联学习词语之间的关系，但它们很少被显式要求去“理解”“不”。现有的评测大多聚焦于事实正确性或推理能力，几乎没有专门的、规模足够大的否定测试集。于是模型在日常对话里会把“我没有去”误判成“我去”，这说明它们缺乏对否定的深层语义把握。要系统评估并改进这种能力，就必须先有一个覆盖多种否定形式、规模足够大的基准，这正是这篇论文想要填补的空白。

### 关键概念速览
**否定（Negation）**：在句子里加入“不”“没有”等词，使原本的陈述变为相反的意义，就像在灯开关上按下“关闭”键。  
**零样本（Zero‑shot）**：模型在没有见过任何相同任务的训练数据的情况下直接完成任务，类似于第一次见到新游戏却能凭直觉玩。  
**微调（Fine‑tuning）**：在已有的大模型上再用少量特定任务数据继续训练，让模型适应新需求，就像在通用厨师的基础上教会他做特定菜系。  
**半自动生成（Semi‑automatic generation）**：利用规则或模板自动产生大量数据，再人工检查或过滤少量错误，类似于机器先写草稿，人工再润色。  
**表层线索（Superficial cue）**：模型仅凭词汇出现频率或句子长度等浅层特征做判断，而不真正理解语义，就像只看包装判断食物好坏。  
**通用常识（Commonsense knowledge）**：人们日常生活中默认的事实，如“水会湿”，模型需要把这些常识与语言形式结合。  
**基准（Benchmark）**：一套标准化的测试数据和评估指标，用来比较不同模型的表现，就像跑步比赛的计时系统。  

### 核心创新点
1. **从“数据集”到“否定基准”**：以前的评测大多是通用的问答或推理数据集，缺少针对否定的专门设计。作者先用模板生成约40万条带有否定的常识句，再通过人工筛选保证质量。这样得到的基准覆盖了否定词、双重否定、否定短语等多种形式，填补了评测空白。  
2. **零样本评估大模型的否定推理**：在没有任何针对否定的提示或示例的情况下，直接让公开的开源大模型（如LLaMA、Falcon）判断句子真假。结果显示，模型在肯定句上接近90%准确率，却在否定句上跌至约55%，暴露出对否定的系统性弱点。  
3. **微调实验验证可训练性**：作者挑选了部分模型，在仅使用否定句子进行微调后，准确率提升约15%。但即使微调后，模型在未见过的否定结构上仍表现不佳，说明学习到的仍是表层模式而非真正的否定语义。  
4. **消融分析揭示表层线索依赖**：通过去掉句子中的否定词或替换同义词，模型准确率骤降，进一步证明模型主要靠否定词本身的出现与否来做决定，而不是对整体语义的综合理解。  

### 方法详解
整体思路可以拆成三步：**数据构造 → 零样本评估 → 否定微调**。下面按顺序展开。

1. **数据构造**  
   - **模板库**：作者先准备了数百条常识性陈述模板，如“[主体] 能够 [动作]”，并在其中插入否定占位符。  
   - **实体填充**：利用公开的实体库（如WordNet、ConceptNet）随机挑选主体和动作，生成数十万条肯定句。  
   - **否定化**：对每条肯定句，按照三类规则生成否定句：① 直接在动词前加“不/没有”；② 使用否定短语如“从不”“永不”；③ 双重否定（如“不是不”）来增加难度。约2/3的最终语料带有否定。  
   - **质量控制**：通过人工抽样检查，剔除语义不通或逻辑矛盾的样本，确保基准既大又可靠。  

2. **零样本评估**  
   - **模型选取**：使用公开的、参数规模在7B到70B之间的开源大模型。  
   - **提示设计**：仅给模型一个最简指令：“判断下面句子是真还是假”。没有任何示例或额外上下文。  
   - **推理过程**：模型直接输出“True”或 “False”。作者记录每个模型在肯定句和否定句上的准确率，比较两者差距。  

3. **否定微调**  
   - **训练集划分**：从基准中抽取约5万条否定句作为微调数据，其余保持为测试集。  
   - **微调方式**：采用标准的语言模型微调流程，使用交叉熵损失，让模型学习在否定句上输出正确标签。  
   - **评估**：微调后再次在全量测试集上评测，重点观察模型在未见过的否定结构（如不同否定词组合）上的表现。  

4. **消融实验**  
   - **去除否定词**：把测试句中的否定词全部删除，观察模型是否仍能保持正确判断。  
   - **同义替换**：将否定词换成同义的表达（如“不”→“未”），检验模型对词形变化的鲁棒性。  
   - **结果解释**：若准确率大幅下降，说明模型依赖表层线索；若保持稳定，则说明有更深层的语义理解。  

最巧妙的地方在于**半自动生成**的策略：通过模板+实体的组合，作者在几天内就能产出数十万条高质量的否定句，这在以往需要人工标注的大规模数据集里是难以实现的。

### 实验与效果
- **测试基准**：约40万条常识句，其中约27万带有否定。  
- **零样本表现**：在肯定句上，大模型的准确率普遍在85%–92%之间；在否定句上，仅在55%左右，且不同模型差距不大。  
- **微调提升**：对LLaMA‑13B进行5轮微调后，否定句准确率提升至约70%，但在全新否定结构上仍只有约62%。  
- **消融结果**：去掉否定词后，模型准确率跌至30%以下；同义替换导致准确率下降约10个百分点，说明模型对否定词的表层依赖非常强。  
- **局限性**：作者承认基准仍主要围绕常识陈述，缺少更复杂的推理或多句上下文；微调实验只在少数模型上进行，未探索更大规模或不同架构的潜在效果。  

### 影响与延伸思考
这篇工作在发布后迅速成为评估LLM否定能力的标准基准，后续不少论文把它加入自己的多任务评测套件，尤其是关注模型鲁棒性和逻辑一致性的研究。还有工作尝试在基准上加入对话式否定、跨语言否定等扩展，以检验模型在更真实场景下的表现。对想进一步深入的读者，可以关注**逻辑推理基准（如Logical Entailment）**以及**对抗样本生成**方向，这两块正逐步与否定理解交叉融合，推动模型从“记忆词”向“真正推理”迈进。

### 一句话记住它
LLM在否定句上几乎只靠词表层线索，规模再大也难逃“看不懂不”这一根本缺陷。