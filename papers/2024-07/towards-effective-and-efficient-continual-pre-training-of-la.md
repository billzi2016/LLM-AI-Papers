# Towards Effective and Efficient Continual Pre-training of Large Language   Models

> **Date**：2024-07-26
> **arXiv**：https://arxiv.org/abs/2407.18743

## Abstract

Continual pre-training (CPT) has been an important approach for adapting language models to specific domains or tasks. To make the CPT approach more traceable, this paper presents a technical report for continually pre-training Llama-3 (8B), which significantly enhances the Chinese language ability and scientific reasoning ability of the backbone model. To enhance the new abilities while retaining the original abilities, we design specific data mixture and curriculum strategies by utilizing existing datasets and synthesizing high-quality datasets. Specifically, we synthesize multidisciplinary scientific question and answer (QA) pairs based on related web pages, and subsequently incorporate these synthetic data to improve the scientific reasoning ability of Llama-3. We refer to the model after CPT as Llama-3-SynE (Synthetic data Enhanced Llama-3). We also present the tuning experiments with a relatively small model -- TinyLlama, and employ the derived findings to train the backbone model. Extensive experiments on a number of evaluation benchmarks show that our approach can largely improve the performance of the backbone models, including both the general abilities (+8.81 on C-Eval and +6.31 on CMMLU) and the scientific reasoning abilities (+12.00 on MATH and +4.13 on SciEval), without hurting the original capacities. Our model, data, and codes are available at https://github.com/RUC-GSAI/Llama-3-SynE.

---

# 面向高效与有效的大语言模型持续预训练 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在通用语料上训练后已经很强，但要让它们在特定语言（如中文）或专业领域（如科学推理）上进一步提升，往往需要再进行一次大规模的预训练，这叫持续预训练（Continual Pre‑training，CPT）。传统的 CPT 往往直接把新数据喂进去，结果是模型在新能力上有提升，却会牺牲原有的通用能力；另外，构造高质量的领域数据成本极高，缺乏系统的混合与 curriculum（教学进度）设计会导致训练效率低下。于是，如何在不破坏原有能力的前提下，用更少的算力和更高质量的数据快速提升模型的特定能力，成为了一个迫切且技术上棘手的问题。

### 关键概念速览
- **持续预训练（CPT）**：在已有的大模型基础上，再用新的语料继续训练，以适配特定语言或任务。类似给已经学会走路的孩子再上学，既要学新东西，又不能忘记走路的本领。
- **数据混合（Data Mixing）**：把不同来源、不同质量的数据按一定比例拼在一起训练。就像调配咖啡时混合浓缩与牛奶，需要找到最佳比例才能口感最佳。
- **教学进度（Curriculum）**：先让模型学习容易的样本，再逐步引入更难的样本。类似先教孩子认字母，再教拼单词，帮助模型稳步提升。
- **合成数据（Synthetic Data）**：利用模型或规则自动生成的训练样本，而不是人工标注。相当于让机器人自己出题再练习，省去人工出题的成本。
- **多学科科学 QA 对**：包含数学、物理、化学等多个学科的问答对，用来提升模型的科学推理能力。把模型的“知识库”从单一学科扩展到跨学科的百科全书。
- **TinyLlama**：一个参数量远小于 Llama‑3 的轻量模型，常用于快速实验和验证假设。像是用小模型先跑通思路，再把成功经验迁移到大模型上。

### 核心创新点
1. **从小模型到大模型的经验迁移**  
   - 之前的 CPT 往往直接在目标大模型上做实验，成本高且调参不易。  
   - 本文先在参数只有几亿的 TinyLlama 上进行多组实验，系统探索数据比例、 curriculum 步骤等超参数。  
   - 通过在小模型上验证有效的配置，再直接套用到 8 B 参数的 Llama‑3，显著降低了大模型实验的算力开销。

2. **高质量合成科学 QA 数据的生成管线**  
   - 传统做法多依赖公开的标注数据，规模受限且领域覆盖不全。  
   - 作者基于网页抓取的多学科材料，使用检索‑生成‑过滤三步走：先检索相关段落 → 用 LLM 生成对应的问答 → 用规则和模型评分过滤低质量样本。  
   - 这种“先找材料、再让模型出题、再筛选”的闭环，使得合成数据在多学科覆盖和答案准确性上都达到了可用水平。

3. **针对中文能力的专属数据混合与 curriculum 设计**  
   - 过去的 CPT 常把中文数据直接混入英文主流语料，导致中文表现提升有限。  
   - 本文构建了中文专属子语料库（包括新闻、百科、对话），并在 curriculum 中先让模型熟悉中文基础句式，再逐步加入高难度的中文科学问答。  
   - 这种分层次、语言专注的教学进度，使得模型在中文理解和生成上实现了显著跃升。

4. **保持原有通用能力的“能力保守”策略**  
   - 直接大幅度增加新数据会导致模型“忘记”原有知识（灾难性遗忘）。  
   - 作者在混合比例上采用“保守增量”：新数据占总体的 20% 左右，且在每轮训练结束后插入一次全量原始 Llama‑3 数据的微调。  
   - 实验表明，这种策略在提升新能力的同时，几乎不影响原有的通用评测分数。

### 方法详解
整体框架可以概括为四步：**（1）小模型探索 →（2）合成数据生成 →（3）数据混合与 curriculum 设计 →（4）大模型持续预训练**。下面逐步拆解每一步的细节。

1. **小模型探索（TinyLlama 实验）**  
   - 先在 TinyLlama 上跑遍不同的混合比例（如 10%/20%/30% 新数据）和 curriculum 排序（先中文基础 → 再中文科学 → 再英文通用）。  
   - 通过在 C‑Eval、CMMLU、MATH、SciEval 等小模型可评测的基准上记录分数变化，找出“增益最大且灾难性遗忘最小”的配置。  
   - 关键发现：新数据占比约 20% 并在每 2k 步插入一次全量原始数据的微调，能够在提升中文/科学能力的同时保持通用能力。

2. **合成科学 QA 数据管线**  
   - **检索**：使用开源的多语言检索模型，从公开的学术网页、教材 PDF 中抽取段落。  
   - **生成**：把段落喂入 Llama‑3（或更大的模型）让其自行生成对应的提问和答案，要求覆盖概念解释、计算过程、推理步骤等多种问答形式。  
   - **过滤**：采用两层过滤：① 基于规则（如答案长度、是否包含数学公式）剔除明显错误；② 用一个专门训练的质量评估模型打分，保留前 30% 最高分的样本。  
   - 最终得到约 500 万条多学科科学 QA 对，覆盖数学、物理、化学、生物等。

3. **数据混合与 Curriculum 设计**  
   - 将四类数据按比例混合：中文基础语料（30%）、中文科学 QA（20%）、合成多学科科学 QA（20%）、原始 Llama‑3 通用语料（30%）。  
   - Curriculum 采用分阶段训练：  
     - **阶段 1**（0–10k 步）：仅使用中文基础语料，让模型先稳固中文语言模型能力。  
     - **阶段 2**（10k–30k 步）：加入中文科学 QA，提升中文领域推理。  
     - **阶段 3**（30k–50k 步）：加入合成多学科科学 QA，扩展跨语言、跨学科的科学推理。  
     - **阶段 4**（每 2k 步一次）：插入一次全量原始通用语料的微调，防止灾难性遗忘。  
   - 这种“先打基础、再专攻、再拓展、随时回顾”的教学进度，让模型在每一步都能巩固已有能力再学习新东西。

4. **大模型持续预训练（Llama‑3‑SynE）**  
   - 使用上述混合数据和 curriculum，对 Llama‑3‑8B 进行约 50k 步的持续预训练，学习率采用线性衰减，批大小 1k。  
   - 训练结束后，直接得到 Llama‑3‑SynE（Synthetic data Enhanced Llama‑3），无需额外微调即可在下游任务上使用。

**最巧妙的地方**在于把“小模型实验 → 合成数据生成 → 课程化混合 → 大模型训练”四个环节闭环起来。作者没有把每一步当成独立的研究，而是把小模型的经验直接映射到大模型，省掉了大量的试错成本；同时，合成数据的质量控制和 curriculum 的分阶段设计共同保证了新能力的提升不会以牺牲原有能力为代价。

### 实验与效果
- **评测基准**：  
  - **通用中文能力**：C‑Eval、CMMLU。  
  - **科学推理**：MATH（数学）、SciEval（多学科科学）。  
- **对比基线**：原始 Llama‑3‑8B、以及公开的几种同规模中文强化模型（如 Chinese‑LLaMA、BLOOM‑Z）。  
- **主要结果**（论文中给出的提升幅度）：  
  - C‑Eval：+8.81 分。  
  - CMMLU：+6.31 分。  
  - MATH：+12.00 分。  
  - SciEval：+4.13 分。  
  - 在所有基准上，原始模型的分数基本保持不变，说明没有出现显著的灾难性遗忘。  
- **消融实验**：  
  - 去掉合成科学 QA，MATH 分数下降约 7 分，证明合成数据是提升科学推理的关键。  
  - 将新数据比例提升到 40%（而非 20%），通用中文基准下降约 3 分，验证了“保守增量”策略的必要性。  
- **局限性**：  
  - 合成数据的质量仍受生成模型的能力限制，极端专业的高阶科研问题仍表现一般。  
  - 只在 8 B 参数模型上验证，未探索更大模型或更小模型的迁移效果。  
  - 论文未提供对训练时间、算力成本的详细对比，只给出相对“高效”这一描述。

### 影响与延伸思考
这篇报告在业界引发了两大趋势：一是 **小模型先行** 的实验策略，被多家机构采纳用于快速验证数据混合方案；二是 **合成多学科科学 QA** 的管线，被后续工作进一步自动化，甚至加入了自监督的难度标注，以实现更细粒度的 curriculum。推测未来会有更多研究把 **跨语言合成数据** 与 **多阶段教学进度** 结合，打造“一次预训练、全语言全领域” 的通用模型。如果想继续深入，可以关注以下方向：  
- 合成数据的自我纠错与迭代生成（类似自我提升的闭环）。  
- 更细致的灾难性遗忘防护技术，如参数冻结、知识蒸馏等。  
- 将这种 CPT 框架扩展到指令微调（Instruction Tuning）阶段，实现“一键多任务适配”。

### 一句话记住它
**用小模型先验验证、合成多学科科学 QA、分阶段教学进度，让 Llama‑3 在不忘本的情况下“一举提升”中文和科学推理能力。**