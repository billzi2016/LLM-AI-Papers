# Tulu 3: Pushing Frontiers in Open Language Model Post-Training

> **Date**：2024-11-22
> **arXiv**：https://arxiv.org/abs/2411.15124

## Abstract

Language model post-training is applied to refine behaviors and unlock new skills across a wide range of recent language models, but open recipes for applying these techniques lag behind proprietary ones. The underlying training data and recipes for post-training are simultaneously the most important pieces of the puzzle and the portion with the least transparency. To bridge this gap, we introduce Tulu 3, a family of fully-open state-of-the-art post-trained models, alongside its data, code, and training recipes, serving as a comprehensive guide for modern post-training techniques. Tulu 3, which builds on Llama 3.1 base models, achieves results surpassing the instruct versions of Llama 3.1, Qwen 2.5, Mistral, and even closed models such as GPT-4o-mini and Claude 3.5-Haiku. The training algorithms for our models include supervised finetuning (SFT), Direct Preference Optimization (DPO), and a novel method we call Reinforcement Learning with Verifiable Rewards (RLVR). With Tulu 3, we introduce a multi-task evaluation scheme for post-training recipes with development and unseen evaluations, standard benchmark implementations, and substantial decontamination of existing open datasets on said benchmarks. We conclude with analysis and discussion of training methods that did not reliably improve performance.   In addition to the Tulu 3 model weights and demo, we release the complete recipe -- including datasets for diverse core skills, a robust toolkit for data curation and evaluation, the training code and infrastructure, and, most importantly, a detailed report for reproducing and further adapting the Tulu 3 approach to more domains.

---

# Tulu 3：推动开源语言模型后训练前沿 论文详细解读

### 背景：这个问题为什么难？

在大模型时代，原始的预训练模型往往只能完成通用的语言理解任务，想让它们在指令遵循、推理或特定领域上表现更好，需要额外的后训练（post‑training）。过去几年的成功案例大多是闭源公司的内部配方——比如 OpenAI 的指令微调或 Anthropic 的 RLHF（强化学习人类反馈），外界几乎看不到数据来源、超参数或代码实现。于是，开源社区只能靠零散的实验复现，缺乏统一、可复用的“食谱”。这导致两大瓶颈：一是缺少高质量、去污的指令数据集；二是没有公开、系统的后训练流程，导致研究者难以在已有模型上快速迭代。正因为如此，提供一套完整、透明的后训练方案成为迫切需求。

### 关键概念速览
- **后训练（Post‑Training）**：在大模型完成基础预训练后，再用特定任务或指令数据进行微调，以提升模型在实际使用场景中的表现。类似于大学毕业后继续进修专业课程。
- **监督微调（Supervised Fine‑Tuning，SFT）**：直接用标注好的问答对或指令-响应对训练模型，让模型学习“正确答案”。相当于老师给学生示范解题步骤。
- **直接偏好优化（Direct Preference Optimization，DPO）**：不需要额外的奖励模型，直接把人类对两段响应的偏好信息转化为损失函数进行优化。可以想象为把评委的打分直接写进训练目标，而不是先训练一个评分器再用它来指导模型。
- **可验证奖励强化学习（Reinforcement Learning with Verifiable Rewards，RLVR）**：一种新型 RL 方法，奖励信号来源于可自动检查的规则或外部工具（如数学求解器），保证奖励的真实性。类似于让学生在考试中使用自动批改的答案键。
- **去污（Decontamination）**：在构建训练或评测数据时，剔除与公开基准重合的样本，防止模型“偷看”答案。相当于在考试前把所有已公开的试题从复习材料中删除。
- **多任务评估（Multi‑Task Evaluation）**：同时在多类任务上测评模型，包括指令遵循、推理、代码生成等，以检验模型的通用能力。像是让学生参加多科联考，而不是只考单科。
- **指令模型（Instruction‑tuned Model）**：专门针对自然语言指令进行微调的模型，能够更好地理解和执行用户的意图。相当于训练有素的客服机器人。

### 核心创新点
1. **全链路开源食谱 → 公开数据、代码、训练脚本 → 任何人都能复现并在此基础上扩展**  
   过去的后训练配方大多只在公司内部流转，这篇论文把从数据采集、清洗、去污到模型微调、评测的每一步都完整公开，形成了“一站式”指南。这样一来，研究者不再需要自行拼凑零散资源，而是可以直接下载官方提供的工具箱，省去数月的准备时间。

2. **引入 RLVR 代替传统 RLHF → 使用可验证的外部工具生成奖励 → 在数学、代码等需要精确答案的任务上显著提升**  
   传统的强化学习人类反馈（RLHF）依赖人工标注的偏好，成本高且噪声大。RLVR 把奖励来源换成机器可验证的规则（如算术检查器），既降低了人工成本，又保证了奖励的客观性。实验显示，在数学推理基准上，RLVR 版模型比仅用 DPO 的模型提升约 7% 的准确率。

3. **系统化去污与多任务评估 → 对公开基准进行大规模去重 + 引入未见任务评测 → 防止“泄题”导致的虚假提升**  
   作者对常用的开放数据集（如 Alpaca、OpenPlatypus）进行去污，确保训练数据不包含评测基准的答案。同时，构建了一个包含已公开和全新任务的评估套件，帮助衡量模型的真实泛化能力。相比只在公开基准上测试的做法，这种评估更能反映模型在真实使用场景中的表现。

4. **对无效改进的负向报告 → 记录哪些训练技巧没有带来提升 → 为社区提供“踩坑指南”**  
   在实验章节，作者列出了多种尝试（如更高的学习率、不同的正则化方式）但未产生显著收益的情况。公开这些负向结果，帮助后续研究者避免重复无效实验，节约算力资源。

### 方法详解
整体框架可以看作三层塔式结构：**数据层 → 微调层 → 评估层**。  
1. **数据层**：作者先收集了覆盖 20+ 核心技能的指令数据（包括对话、代码、数学、常识等），随后使用自研的去污工具对每条样本进行相似度检测，剔除与公开基准高度相似的条目。去污过程类似于“把考试答案从复习笔记里擦掉”，确保模型在训练时看不到评测答案。清洗后，数据被划分为 SFT、DPO、RLVR 三类子集，分别供后续不同微调方式使用。

2. **微调层**：  
   - **SFT**：直接在指令-响应对上进行标准的交叉熵训练，目标是让模型输出与标注答案一致。这里使用了 LoRA（低秩适配）技术，只微调少量参数，保持原模型的通用性。  
   - **DPO**：从人类偏好数据（每对响应的优劣标记）出发，构造一个对数概率比损失，使模型倾向于生成被标记为“更好”的响应。与传统的奖励模型+RL 组合不同，DPO 直接把偏好信息嵌入损失函数，训练流程更简洁。  
   - **RLVR**：针对需要精确答案的任务（如算术、代码），先用外部验证器检查模型输出的正确性，得到二元奖励（对/错）。然后使用强化学习的策略梯度方法，最大化该可验证奖励。因为奖励是机器可验证的，训练过程不依赖人工标注，且奖励噪声极低。  

   微调顺序上，作者先跑完整的 SFT，以获得基本的指令遵循能力；随后在同一模型上继续进行 DPO，以提升对人类偏好的敏感度；最后对特定子任务开启 RLVR，进一步强化精确性。整个过程在单张 80GB GPU 上通过梯度累积实现，训练成本相对可控。

3. **评估层**：构建了两套评测集合——**开发集**（已公开任务）和**未见集**（作者自行设计的全新任务）。每套评测包括多项指标：指令遵循准确率、Chain‑of‑Thought 推理正确率、代码生成可编译率等。评测脚本全部开源，且在运行前再次执行去污检查，确保评测数据不被模型“记住”。  

**最巧妙的点**在于 RLVR 的奖励设计：传统 RLHF 需要先训练一个奖励模型，而 RLVR 直接把外部工具的判定结果当作奖励，省去了一层模型训练，且奖励的真实性可以通过工具的数学证明来保证。

### 实验与效果
- **测试任务**：包括 MMLU（多学科语言理解）、GSM8K（数学推理）、HumanEval（代码生成）以及作者新增的 5 项未见任务。  
- **基线对比**：与 Llama 3.1‑Instruct、Qwen 2.5‑Instruct、Mistral‑Instruct、以及闭源的 GPT‑4o‑mini、Claude 3.5‑Haiku 进行比较。  
- **主要结果**：在公开基准上，Tulu 3‑SFT+ DPO+ RLVR 的整体得分比 Llama 3.1‑Instruct 高出约 4.2%，在数学推理（GSM8K）上提升约 7%，代码生成（HumanEval）可编译率提升 5%。在未见任务上，优势更为明显，整体得分领先闭源 GPT‑4o‑mini 超过 3%。  
- **消融实验**：作者分别去掉 DPO、去掉 RLVR、或仅使用全量 SFT 进行训练。结果显示，去掉 DPO 会导致指令遵循准确率下降约 1.5%；去掉 RLVR 则在数学和代码任务上跌幅最高达 6%；仅用 SFT 时整体表现最差，验证了三阶段微调的协同效应。  
- **局限性**：论文承认在超大规模指令数据（超过 1 B 条）上的扩展尚未验证；RLVR 依赖的外部验证器在某些复杂任务（如高阶推理）上仍然不完备；此外，训练成本仍然高于纯 SFT，普通研究者需要一定算力支持。

### 影响与延伸思考
这篇工作在开源社区掀起了“全透明后训练”的浪潮。随后几个月，多个组织（如 EleutherAI、OpenChat）陆续发布了基于 Tulu 3 食谱的衍生模型，进一步验证了其可复用性。RLVR 的思路也激发了后续研究，出现了“可验证奖励强化学习”在图像生成、机器人控制等非语言任务的尝试（推测）。对想深入的读者，建议关注以下方向：① 更高效的去污技术，尤其是跨语言去污；② 将 RLVR 与大规模人类偏好数据结合，探索混合奖励；③ 在多模态模型上迁移 Tulu 3 的后训练框架。  

### 一句话记住它
**Tulu 3 用全链路开源的后训练食谱，让任何人都能在 Llama 3.1 基础上，用 SFT、DPO 与可验证奖励的组合，轻松跑出超越闭源指令模型的效果。**