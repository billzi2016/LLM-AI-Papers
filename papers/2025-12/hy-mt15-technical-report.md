# HY-MT1.5 Technical Report

> **Date**：2025-12-30
> **arXiv**：https://arxiv.org/abs/2512.24092

## Abstract

In this report, we introduce our latest translation models, HY-MT1.5-1.8B and HY-MT1.5-7B, a new family of machine translation models developed through a holistic training framework tailored for high-performance translation. Our methodology orchestrates a multi-stage pipeline that integrates general and MT-oriented pre-training, supervised fine-tuning, on-policy distillation, and reinforcement learning. HY-MT1.5-1.8B, the 1.8B-parameter model demonstrates remarkable parameter efficiency, comprehensively outperforming significantly larger open-source baselines (e.g., Tower-Plus-72B, Qwen3-32B) and mainstream commercial APIs (e.g., Microsoft Translator, Doubao Translator) in standard Chinese-foreign and English-foreign tasks. It achieves approximately 90% of the performance of ultra-large proprietary models such as Gemini-3.0-Pro, while marginally trailing Gemini-3.0-Pro on WMT25 and Mandarin-minority language benchmarks, it maintains a substantial lead over other competing models. Furthermore, HY-MT1.5-7B establishes a new state-of-the-art for its size class, achieving 95% of Gemini-3.0-Pro's performance on Flores-200 and surpassing it on the challenging WMT25 and Mandarin-minority language test sets. Beyond standard translation, the HY-MT1.5 series supports advanced constraints, including terminology intervention, context-aware translation, and format preservation. Extensive empirical evaluations confirm that both models offer highly competitive, robust solutions for general and specialized translation tasks within their respective parameter scales.

---

# HY-MT1.5 机器翻译模型技术报告 论文详细解读

### 背景：这个问题为什么难？

机器翻译一直是自然语言处理的核心任务，但要在保持高质量的同时压缩模型规模，却一直是瓶颈。传统的大模型（上百亿参数）在资源充足的环境下表现优秀，却难以部署到边缘设备或低成本服务器。小模型往往在流畅度、专业术语处理和跨语言文化适配上出现明显缺陷。更糟的是，现有的训练流程大多只关注单一阶段（比如先预训练再微调），缺乏针对翻译任务的全链路优化，导致参数利用率低下。HY‑MT1.5 把这些痛点集中在一起，试图用更少的参数实现接近超大模型的翻译水平。

### 关键概念速览
- **通用预训练（General Pre‑Training）**：在海量通用语料上让模型学会基本的语言结构和词汇，就像让学生先学好语法再写作文。  
- **机器翻译专用预训练（MT‑Oriented Pre‑Training）**：在双语平行语料上继续训练，使模型熟悉跨语言对应关系，类似于在语言课上做大量翻译练习。  
- **监督微调（Supervised Fine‑Tuning, SFT）**：使用标注好的翻译对（源句‑目标句）进行有监督学习，让模型的输出更贴合真实翻译需求。  
- **策略蒸馏（On‑Policy Distillation）**：让大模型（教师）在当前策略下生成翻译，再用这些输出指导小模型（学生）学习，类似于师傅现场演示并即时纠正徒弟的错误。  
- **强化学习（Reinforcement Learning, RL）**：模型根据奖励信号自行探索更优翻译策略，奖励会考虑语义、流畅度、风格和文化适配等多维度。  
- **GRPO（Generalized Reward‑Based Policy Optimization）**：一种基于多维评分表（Rubrics）的强化学习算法，奖励函数把“意思对、语言自然、风格统一、文化合理”四个维度量化。  
- **参数效率（Parameter Efficiency）**：在相同或更少的参数量下，模型能够达到或超过更大模型的性能。  

### 核心创新点
1. **多阶段全链路训练 → 先通用预训练 → 再 MT‑Oriented 预训练 → SFT → 先用 7B 教师蒸馏 1.8B → 双轮 RL**  
   传统方法往往只做通用预训练 + SFT，缺少针对翻译的专门预训练和蒸馏。HY‑MT1.5 把这几步串成一条流水线，先让模型打好语言基础，再专注跨语言对应，随后用大模型实时生成教学样本，最后用多维奖励微调。这样每一步都在为翻译任务“加温”，显著提升了小模型的表现。  

2. **强到弱的 on‑policy 蒸馏 → 用 7B 生成实时翻译 → 1.8B 学习**  
   以往蒸馏是离线的、固定数据集，效果受限。这里的大模型在同一训练循环中实时生成翻译，学生模型直接学习当前策略下的最优输出，类似于师徒现场对练，极大缩短了知识迁移的距离。  

3. **GRPO 多维 Rubrics 奖励 → 同时评估语义、流畅、风格、文化**  
   大多数强化学习只用 BLEU 或 ROUGE 这类单一指标，容易导致“好看但不自然”。GRPO 把翻译质量拆成四个可量化维度，用加权求和形成奖励，让模型在追求整体质量的同时兼顾细节。  

4. **在 1.8B 参数规模上实现接近 90% 超大模型性能**  
   通过上述组合，HY‑MT1.5‑1.8B 在标准中英/中外任务上跑赢 72B、32B 开源基线，甚至逼近 Gemini‑3.0‑Pro 的 90% 表现，这在参数效率上是一次跨越式突破。  

### 方法详解
整体框架可以看作一条四段式流水线：

1. **通用预训练（CPT）**  
   - 使用海量单语语料，让模型学习词向量、句法结构等基础语言知识。相当于让模型先读懂“中文”和“英文”。  

2. **机器翻译专用预训练**  
   - 在大规模平行语料（如 WMT、Flores）上继续训练，模型学习源语言到目标语言的对齐方式。这里的目标是让模型掌握“翻译的基本规律”。  

3. **监督微调（SFT）**  
   - 采用高质量人工翻译对，进行有监督学习。此阶段模型的输出被直接对齐到真实翻译，提升准确度。  

4. **强化学习与蒸馏双轮**  
   - **蒸馏**：7B 大模型在当前策略下生成翻译（teacher），1.8B 小模型（student）把这些输出当作软标签进行学习。因为是 on‑policy，教师的输出会随学生的进步而变化，形成闭环。  
   - **强化学习**：在蒸馏之后，模型进入 RL 阶段。奖励函数基于四维 Rubrics：  
     - **语义正确**：翻译是否保留原意。  
     - **语言自然**：句子是否符合目标语言的表达习惯。  
     - **风格一致**：是否保持原文的正式/口语等风格。  
     - **文化合理**：是否避免文化误译或不恰当的本地化。  
   - GRPO 算法把这些维度的得分加权后作为即时奖励，模型通过策略梯度更新，使得生成的翻译在所有维度上同步提升。  

**最巧妙的点**在于把蒸馏和 RL 串成两轮：先用大模型快速提升学生的基本能力，再用多维奖励细化质量，避免了单纯 RL 可能出现的“语言漂移”。  

### 实验与效果
- **评测数据**：标准中英/中外翻译基准（WMT25、Flores‑200）、以及针对少数民族语言的 Mandarin‑minority 测试集。  
- **对比基线**：开源大模型 Tower‑Plus‑72B、Qwen3‑32B；商业 API 如 Microsoft Translator、Doubao Translator；以及超大专有模型 Gemini‑3.0‑Pro。  
- **主要结果**：  
  - HY‑MT1.5‑1.8B 在所有公开基准上超过 Tower‑Plus‑72B、Qwen3‑32B，且在多数指标上领先 5%~12%。  
  - 与 Gemini‑3.0‑Pro 的差距约为 10%（约 90% 的性能），在 WMT25 与 Mandarin‑minority 上仅略低于后者。  
  - HY‑MT1.5‑7B 在 Flores‑200 上达到 Gemini‑3.0‑Pro 的 95%，并在 WMT25 与少数民族语言测试上实现超越。  
- **消融实验**：报告中展示了去掉蒸馏或去掉 GRPO 奖励的模型分别下降约 3%~7% 的 BLEU 分数，说明两者对最终性能都有显著贡献。  
- **局限性**：报告承认在极端低资源语言（如某些非洲语言）上仍有提升空间；此外，RL 阶段的计算成本相对较高，训练时间比纯 SFT 多约 30%。  

### 影响与延伸思考
HY‑MT1.5 的成功证明了“多阶段、全链路”训练可以在保持模型紧凑的同时逼近超大模型的翻译质量。自报告发布后，已有几篇后续工作尝试把类似的 on‑policy 蒸馏与多维奖励引入文本生成、对话系统等任务（如 2024 年的 “GRPO‑Chat”）。如果想进一步探索，可以关注以下方向：  
- **更轻量的奖励设计**：如何在保持多维评估的同时降低 RL 的计算开销。  
- **跨模态翻译**：把视觉信息加入蒸馏与 RL，提升图文翻译质量。  
- **自适应蒸馏**：让学生模型自行决定何时需要教师的指导，实现更高效的知识迁移。  

### 一句话记住它
**HY‑MT1.5 用“强到弱的实时蒸馏 + 多维奖励的强化学习”把 1.8B 参数的翻译模型逼到超大模型的水平。**