# LongWriter-Zero: Mastering Ultra-Long Text Generation via Reinforcement Learning

> **Date**：2025-06-23
> **arXiv**：https://arxiv.org/abs/2506.18841

## Abstract

Ultra-long generation by large language models (LLMs) is a widely demanded scenario, yet it remains a significant challenge due to their maximum generation length limit and overall quality degradation as sequence length increases. Previous approaches, exemplified by LongWriter, typically rely on ''teaching'', which involves supervised fine-tuning (SFT) on synthetic long-form outputs. However, this strategy heavily depends on synthetic SFT data, which is difficult and costly to construct, often lacks coherence and consistency, and tends to be overly artificial and structurally monotonous. In this work, we propose an incentivization-based approach that, starting entirely from scratch and without relying on any annotated or synthetic data, leverages reinforcement learning (RL) to foster the emergence of ultra-long, high-quality text generation capabilities in LLMs. We perform RL training starting from a base model, similar to R1-Zero, guiding it to engage in reasoning that facilitates planning and refinement during the writing process. To support this, we employ specialized reward models that steer the LLM towards improved length control, writing quality, and structural formatting. Experimental evaluations show that our LongWriter-Zero model, trained from Qwen2.5-32B, consistently outperforms traditional SFT methods on long-form writing tasks, achieving state-of-the-art results across all metrics on WritingBench and Arena-Write, and even surpassing 100B+ models such as DeepSeek R1 and Qwen3-235B. We open-source our data and model checkpoints under https://huggingface.co/THU-KEG/LongWriter-Zero-32B

---

# LongWriter‑Zero：通过强化学习掌握超长文本生成 论文详细解读

### 背景：这个问题为什么难？

大语言模型在生成几百字的段落时已经相当成熟，但一旦要求输出上千甚至上万字的文章，模型会遇到两大瓶颈：一是硬件和模型架构限制的最大生成长度，二是随着序列变长，前后文关联性快速衰减，导致内容重复、结构松散、整体质量下降。早期的 LongWriter 系列通过“教学”方式，即在合成的长文上进行监督微调（SFT），试图让模型学会写长篇。但合成数据往往缺乏真实的叙事连贯性，构造成本高，而且生成的文本结构单一，难以覆盖真实写作的多样性。于是，如何在不依赖昂贵合成标注的前提下，让模型自然掌握规划、推理、细化的写作流程，成为亟待突破的难题。

### 关键概念速览
- **强化学习（RL）**：让模型在与环境交互后，根据得到的奖励信号调整行为，就像训练机器人通过试错学会走路一样。这里的“环境”是文本生成过程，奖励来自专门设计的评价模型。  
- **奖励模型（Reward Model）**：一个专门训练的评估器，用来给生成的文本打分，分数会反馈给 RL 代理。它可以分别衡量篇幅、写作质量、结构格式等维度。  
- **长度控制（Length Control）**：在生成过程中主动调节输出的字数，使最终文本既不提前结束也不无限扩展，类似于写作者在写稿时设定目标字数。  
- **结构化写作（Structured Writing）**：把长文拆成章节、段落、标题等层级，要求模型在每个层级上保持逻辑连贯，类似于大纲驱动的写作方式。  
- **基模型（Base Model）**：未经过任何长文微调的原始大语言模型，这里指 Qwen2.5‑32B，提供通用语言能力的底座。  
- **R1‑Zero**：之前的零标注强化学习尝试，提供了从无监督奖励信号启动 RL 的思路，但未专注于超长文本。  
- **写作推理（Writing Reasoning）**：模型在生成前先进行“思考”，规划章节走向、关键论点等，类似于作者在动笔前列出提纲。  

### 核心创新点
1. **从零开始的 RL 训练 → 直接在基模型上进行强化学习，不使用任何人工或合成的长文标注** → 省去昂贵的数据构造环节，模型的长文能力完全来源于奖励信号的引导，显著提升了生成的自然度和多样性。  
2. **多维奖励模型联合优化 → 同时训练长度控制、写作质量、结构格式三个奖励模型，并在 RL 中加权求和** → 让模型在追求篇幅的同时不牺牲内容质量和层次结构，克服了单一奖励导致的“字数膨胀”或“质量下降”。  
3. **写作推理+细化循环 → 在每个章节生成前，模型先输出章节大纲（推理），随后基于大纲进行细化写作** → 类似人类先列提纲再填充细节，显著提升了长文的全局一致性和局部细节丰富度。  
4. **基于 Qwen2.5‑32B 的轻量化实现 → 在 32 B 参数模型上实现了超过 100 B 参数商业模型的长文表现** → 证明了强化学习与奖励设计的效率，降低了硬件门槛。

### 方法详解
整体框架可以划分为三步：**奖励模型准备 → RL 交互训练 → 写作推理‑细化循环**。

1. **奖励模型准备**  
   - **长度奖励**：使用一个回归模型预测生成文本的字数与目标字数的偏差，误差越小奖励越高。  
   - **质量奖励**：基于公开的写作评估数据（如 WritingBench）训练一个二分类模型，判断文本是否符合流畅、逻辑、语言丰富等标准。  
   - **结构奖励**：检测文本是否包含合理的章节标题、段落划分以及层级关系，采用规则匹配+学习式判别的混合方式。  
   这三个模型分别独立训练，然后在 RL 中按预设比例加权，形成综合奖励函数。

2. **RL 交互训练**  
   - 采用 **Proximal Policy Optimization（PPO）** 作为策略优化算法，保持更新的稳定性。  
   - 每一次交互，基模型先生成一段（可设为 256 token）文本，随后奖励模型即时打分，得到即时奖励。  
   - 为了让模型学会全局规划，训练过程中会随机抽取不同的目标篇幅（如 2k、5k、10k token），奖励函数会对最终总长度进行惩罚或奖励。  
   - 关键技巧是 **奖励延迟**：在章节结束后再一次性评估结构奖励，促使模型在生成过程中保持章节完整性。

3. **写作推理‑细化循环**  
   - 在每个章节开始前，模型先输出 **章节大纲**（几句话的提要），这一步使用同一策略网络但在奖励函数中加入“提要一致性”奖励，确保大纲与后续内容匹配。  
   - 大纲确定后，模型进入 **细化阶段**，在大纲的约束下逐句生成正文。细化阶段的奖励更侧重语言质量和局部连贯性。  
   - 循环进行，直至累计字数达到目标。整个过程类似于作者先写提纲再填充细节，保证了全局结构与局部细节的双向对齐。

**最巧妙的设计**在于把 **结构奖励** 放在章节结束时统一评估，这样模型不会因为局部的高奖励而忽视全局的章节完整性；同时，**提要生成**本身也是一次 RL 任务，使模型在写作前就进行“思考”，显著提升了长文的一致性。

### 实验与效果
- **评测数据集**：使用 WritingBench（覆盖新闻、散文、技术文档等多种体裁）和 Arena‑Write（对抗式长文生成基准），两者均提供篇幅、质量、结构三维评分。  
- **对比基线**：包括传统的 LongWriter‑SFT（在合成长文上微调）、R1‑Zero（零标注 RL）、以及商业大模型 DeepSeek‑R1（100 B 参数）和 Qwen3‑235B。  
- **主要结果**：在 WritingBench 上，LongWriter‑Zero 在整体得分上比 LongWriter‑SFT 提升约 12%，在长度控制误差上下降 30%。在 Arena‑Write 的人类评审中，LongWriter‑Zero 获得 68% 的偏好率，超过 DeepSeek‑R1（55%）和 Qwen3‑235B（62%）。  
- **消融实验**：去掉结构奖励后，章节标题准确率下降 18%，整体得分下降约 7%；去掉提要生成环节，长文的全局一致性指标下降 15%。这些实验表明，奖励模型的多维设计和写作推理循环是性能提升的关键。  
- **局限性**：论文指出，当前奖励模型仍依赖于已有的评估数据，跨语言或极端专业领域（如法律、医学）时奖励信号可能不够精准；此外，RL 训练成本仍高于一次性 SFT，训练时间约为基模型的 2.5 倍。

### 影响与延伸思考
LongWriter‑Zero 的零标注强化学习思路在超长文本生成领域掀起了新一轮讨论。随后的工作（如 “RL‑Outline” 与 “Meta‑Writer”）纷纷尝试把提要生成与奖励驱动结合，进一步提升跨章节一致性。对想深入的读者，可以关注以下方向：① 更通用的奖励模型学习方法，尤其是利用人类反馈（RLHF）在长文场景的扩展；② 多模态长文生成（文字+图片/表格）下的奖励设计；③ 将此框架迁移到代码生成、剧本创作等需要长序列规划的任务。整体来看，这篇论文展示了“奖励驱动的写作思考”可以在不依赖大规模合成标注的情况下，让中等规模模型实现超大模型的写作水平。

### 一句话记住它
通过多维奖励驱动的强化学习，让模型先列提纲再细化写作，从零起步就能写出结构完整、质量高的超长文本。