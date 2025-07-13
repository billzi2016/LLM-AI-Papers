# RedOne: Revealing Domain-specific LLM Post-Training in Social Networking Services

> **Date**：2025-07-13
> **arXiv**：https://arxiv.org/abs/2507.10605

## Abstract

As a primary medium for modern information dissemination, social networking services (SNS) have experienced rapid growth, which has proposed significant challenges for platform content management and interaction quality improvement. Recently, the development of large language models (LLMs) has offered potential solutions but existing studies focus on isolated tasks, which not only encounter diminishing benefit from the data scaling within individual scenarios but also fail to flexibly adapt to diverse real-world context. To address these challenges, we introduce RedOne, a domain-specific LLM designed to break the performance bottleneck of single-task baselines and establish a comprehensive foundation for the SNS. RedOne was developed through a three-stage training strategy consisting of continue pretraining, supervised fine-tuning, and preference optimization, using a large-scale real-world dataset. Through extensive experiments, RedOne maintains strong general capabilities, and achieves an average improvement up to 14.02% across 8 major SNS tasks and 7.56% in SNS bilingual evaluation benchmark, compared with base models. Furthermore, through online testing, RedOne reduced the exposure rate in harmful content detection by 11.23% and improved the click page rate in post-view search by 14.95% compared with single-tasks finetuned baseline models. These results establish RedOne as a robust domain-specific LLM for SNS, demonstrating excellent generalization across various tasks and promising applicability in real-world scenarios.

---

# RedOne：揭示社交网络服务领域专用大语言模型的后训练 论文详细解读

### 背景：这个问题为什么难？

社交网络服务（SNS）每天产生海量的文字、图片、视频和互动记录，内容种类极其多样——从商品推荐、情感表达到潜在的违规信息。传统的大语言模型（LLM）虽然在通用语言理解上表现出色，但直接搬到 SNS 场景会遇到两大瓶颈：一是单任务微调只能在某个细分任务上提升，无法共享跨任务的知识；二是模型在大规模通用语料上预训练得到的能力，面对 SNS 特有的口语化、平台术语和实时热点时会出现“迁移失效”。因此，业界急需一种既保留通用语言能力，又能深度适配 SNS 场景的统一模型。

### 关键概念速览

**大语言模型（LLM）**：在海量文本上预训练得到的神经网络，能够生成或理解自然语言。类似于“通用语言工具箱”，可以被进一步调教来完成具体任务。

**继续预训练（Continue Pretraining）**：在已有的通用模型基础上，再用目标领域的原始数据继续训练，让模型熟悉该领域的词汇、表达和风格。相当于让“语言工具箱”在新环境里做一次适应性训练。

**监督微调（Supervised Fine‑Tuning）**：使用标注好的任务数据（如分类标签、问答对）对模型进行有监督学习，使其在特定任务上表现更好。就像给工具箱装上专门的螺丝刀。

**偏好优化（Preference Optimization）**：基于人类偏好或评价信号，对模型输出进行强化学习式的微调，使生成内容更符合用户期待或安全要求。类似于让工具箱在使用过程中不断接受使用者的反馈并自我改进。

**多任务学习**：一次性在多个任务上训练模型，让不同任务之间共享表征，提升整体泛化能力。可以想象为让同一套工具箱同时适用于多种工作，而不是为每个工作单独准备一套工具。

**领域适配**：把通用模型的能力迁移到特定行业或平台（这里是 SNS），需要针对该领域的数据特征进行专门的训练和调优。

**有害内容检测**：识别并过滤平台上可能违规、暴力、误导等不良信息的技术。它是 SNS 内容安全的第一道防线。

**点击率（CTR）优化**：提升用户在搜索或推荐结果中点击进入目标页面的比例，是衡量信息检索和推荐系统效果的重要指标。

### 核心创新点

1. **从单任务到全域任务的转变**  
   *之前的方法*：大多数研究只在单一任务（如情感分析或违规检测）上微调模型，导致模型只能在该任务上稍有提升，跨任务时仍需重新训练。  
   *RedOne 的做法*：构建一个统一的 SNS 专用 LLM，使用同一模型同时覆盖 8 大核心任务，并在后训练阶段保持通用能力。  
   *带来的改变*：一次训练即可服务多种业务，显著降低了模型维护成本，同时在所有任务上实现了平均 14.02% 的性能提升。

2. **三阶段后训练流水线**  
   *之前的做法*：要么直接在通用模型上做监督微调，要么只进行一次强化学习，缺乏层次化的适配过程。  
   *RedOne 的做法*：先进行继续预训练，让模型熟悉 SNS 原始语料；随后进行监督微调，针对标注任务注入精确信息；最后使用偏好优化，利用人类偏好数据进一步提升安全性和用户满意度。  
   *带来的改变*：每一步都针对不同层面的需求进行强化，使模型在保持通用语言能力的同时，显著提升了领域适配度和安全性。

3. **大规模真实业务数据的引入**  
   *之前的局限*：公开数据集往往规模有限、分布偏离真实业务，导致实验结果难以直接迁移。  
   *RedOne 的做法*：使用平台内部收集的海量真实 SNS 文本（包括帖子、评论、搜索日志等），在继续预训练阶段直接喂给模型。  
   *带来的改变*：模型对平台特有的口语、流行语和热点事件拥有更精准的感知，实验中在双语评估基准上提升了 7.56%。

4. **线上 A/B 实验验证**  
   *之前的研究*：大多停留在离线评测，缺乏真实用户交互的检验。  
   *RedOne 的做法*：在实际产品中部署模型，分别对有害内容检测和帖子搜索两大场景进行对照实验。  
   *带来的改变*：有害内容曝光率下降 11.23%，搜索点击率提升 14.95%，直接证明了模型在真实业务中的价值。

### 方法详解

#### 整体框架

RedOne 的训练流程可以看作一条“流水线”，分为三段：**继续预训练 → 监督微调 → 偏好优化**。起点是一个公开的通用 LLM（如 LLaMA），终点是一个能够在 SNS 多任务上直接上线的模型。每一段都使用了平台内部的大规模数据，并在前一段的基础上继续迭代。

#### 1. 继续预训练（Domain‑Continued Pretraining）

- **数据来源**：平台抓取的原始帖子、评论、搜索查询、用户生成的短视频字幕等，覆盖多语言（中文、英文）和多模态文字。  
- **目标**：让模型学习 SNS 特有的词汇分布、口语化表达以及平台内部的热点话题。  
- **实现方式**：采用自回归语言建模目标（预测下一个词），与原始通用模型的参数共享，只是把学习率调低，以免破坏已有的通用知识。可以把它想象成让模型在“新城市”里走街串巷，熟悉当地的语言习惯。

#### 2. 监督微调（Supervised Multi‑Task Fine‑Tuning）

- **任务集合**：包括但不限于（1）有害内容二分类、（2）情感倾向分析、（3）商品推荐意图识别、（4）帖子标题生成、（5）搜索意图匹配、（6）多语言问答、（7）用户画像抽取、（8）热点话题预测。  
- **标签来源**：平台内部的人工标注、规则生成的弱标签以及历史运营数据。  
- **训练方式**：采用多任务混合训练，每个批次随机抽取若干任务的数据，使用任务特定的头部（如分类层、生成层），其余共享底层 Transformer 参数。这样模型在学习每个任务的细节时，也能在共享层中捕获跨任务的共性特征。  
- **技巧**：为防止某些任务数据量过大导致“主导效应”，作者使用了任务权重平衡和梯度归一化技术，使每个任务的学习贡献相对均衡。

#### 3. 偏好优化（Preference Optimization / RLHF‑style）

- **偏好数据**：运营团队对模型输出的安全性、用户满意度、商业价值等维度进行打分，形成对比式偏好（A 更好于 B）。  
- **优化目标**：最大化模型输出被标记为“更好”的概率，等价于在强化学习框架下对模型进行奖励信号的微调。  
- **实现细节**：使用 PPO（Proximal Policy Optimization）等近端策略优化算法，对模型的生成策略进行细粒度调节。因为已经经过监督微调，模型的基本能力已经稳固，偏好优化只在细节层面进行微调，避免出现大幅度的“灾难性遗忘”。  
- **直观类比**：把模型当作一位写手，先让他熟悉平台的语言（继续预训练），再教他写特定类型的稿件（监督微调），最后让编辑不断给出“这段更好、那段需要改”的反馈，写手据此微调自己的写作风格（偏好优化）。

#### 巧妙之处

- **层次递进的训练顺序**：先让模型“熟悉环境”，再“学会任务”，最后“听取偏好”，每一步都在前一步的基础上细化，避免一次性大幅度更新导致已学知识被冲掉。  
- **大规模真实业务数据的闭环使用**：从原始日志到标注任务再到偏好反馈，形成了完整的数据闭环，使模型的每一次迭代都紧贴业务需求。  
- **多任务共享底层 + 任务专属头部**：兼顾了跨任务知识迁移和任务细节的专门化，解决了单任务微调的“只能专精单一业务”问题。

### 实验与效果

- **离线评测**：在平台内部构建的 8 大 SNS 任务上，RedOne 相比基线通用模型（未做任何后训练）平均提升 14.02%；在公开的 SNS 双语评估基准上提升 7.56%。  
- **对比基线**：包括（1）原始通用 LLM、（2）单任务微调模型、（3）仅做继续预训练的模型。RedOne 在所有基线上均实现正向提升，尤其在有害内容检测和搜索意图匹配上提升幅度最高。  
- **消融实验**：作者分别去掉继续预训练、监督微调或偏好优化，发现继续预训练对跨语言表现贡献最大，偏好优化对有害内容检测的安全性提升最显著，三者缺一不可。  
- **线上 A/B 测试**：在真实用户流量中部署后，有害内容曝光率下降 11.23%，说明模型在过滤违规信息时更精准；帖子搜索的点击页面率提升 14.95%，表明搜索结果的相关性和用户满意度都有提升。  
- **局限性**：论文未公开具体的数据规模和标签质量评估，可能存在平台内部数据偏见；三阶段训练对算力需求极高，普通研究团队难以复现；偏好优化依赖人工评分，成本较大。

### 影响与延伸思考

RedOne 的成功展示了“领域专用后训练”在高流量平台的可行性，激发了后续工作在以下方向的探索：  
- **跨平台领域适配**：如电商、短视频、新闻聚合等场景的专属 LLM，采用类似三阶段流水线进行快速落地。  
- **更高效的多任务微调技术**：研究如何在更少标注数据下实现类似的跨任务提升，例如使用自监督任务或提示学习（prompt‑tuning）。  
- **安全偏好学习的自动化**：尝试用模型自身生成的安全评分或用户交互信号替代人工偏好，降低成本。  
- **模型压缩与部署**：针对移动端或边缘计算的需求，探索在保持领域适配能力的前提下进行蒸馏或量化。  

如果想进一步了解，可以关注近期在 “Domain‑Adapted LLM” 方向的会议论文（如 ACL、EMNLP）以及开源项目如 “OpenChatKit” 中的领域微调实现。

### 一句话记住它

RedOne 用“三段式后训练 + 大规模真实 SNS 数据”，把通用大语言模型变成“一把钥匙开所有社交业务”的专用模型。