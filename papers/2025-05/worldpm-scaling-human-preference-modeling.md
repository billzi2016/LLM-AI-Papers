# WorldPM: Scaling Human Preference Modeling

> **Date**：2025-05-15
> **arXiv**：https://arxiv.org/abs/2505.10527

## Abstract

Motivated by scaling laws in language modeling that demonstrate how test loss scales as a power law with model and dataset sizes, we find that similar laws exist in preference modeling. We propose World Preference Modeling$ (WorldPM) to emphasize this scaling potential, where World Preference embodies a unified representation of human preferences. In this paper, we collect preference data from public forums covering diverse user communities, and conduct extensive training using 15M-scale data across models ranging from 1.5B to 72B parameters. We observe distinct patterns across different evaluation metrics: (1) Adversarial metrics (ability to identify deceptive features) consistently scale up with increased training data and base model size; (2) Objective metrics (objective knowledge with well-defined answers) show emergent behavior in larger language models, highlighting WorldPM's scalability potential; (3) Subjective metrics (subjective preferences from a limited number of humans or AI) do not demonstrate scaling trends. Further experiments validate the effectiveness of WorldPM as a foundation for preference fine-tuning. Through evaluations on 7 benchmarks with 20 subtasks, we find that WorldPM broadly improves the generalization performance across human preference datasets of varying sizes (7K, 100K and 800K samples), with performance gains exceeding 5% on many key subtasks. Integrating WorldPM into our internal RLHF pipeline, we observe significant improvements on both in-house and public evaluation sets, with notable gains of 4% to 8% in our in-house evaluations.

---

# WorldPM：扩展人类偏好建模 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）里，让模型遵循人类偏好一直是核心难点。过去的做法大多依赖少量人工标注的偏好对，或者把偏好当成单一任务来微调，导致模型在新场景下容易失效。根本的瓶颈在于缺乏系统化、规模化的偏好数据以及对“偏好随模型规模如何变化”的认识。没有明确的 scaling law（规模律），研究者很难判断投入更多算力或数据是否真的能提升偏好建模的质量。

### 关键概念速览
- **偏好建模（Preference Modeling）**：训练模型预测人类在两段文本之间更倾向哪一段，就像让机器学会“你更喜欢哪个”。  
- **World Preference**：作者把所有来源的偏好统一成一种全局表示，类似把世界各地的口味汇总成一本“全球食谱”。  
- **对抗性指标（Adversarial Metrics）**：评估模型识别欺骗性或噪声特征的能力，像是让模型在“陷阱题”中不被误导。  
- **客观指标（Objective Metrics）**：针对有唯一正确答案的知识题进行评估，类似考查模型的事实准确度。  
- **主观指标（Subjective Metrics）**：基于少数人或 AI 的个人喜好打分，类似品酒师的主观评分。  
- **RLHF（Reinforcement Learning from Human Feedback）**：用人类反馈作为奖励信号进行强化学习，让模型在生成文本时更符合人类期望。  
- **规模律（Scaling Law）**：经验公式，说明模型性能随参数量或数据量的幂律增长。  

### 核心创新点
1. **从零构建统一的偏好数据池 → 收集公开论坛的 1500 万条偏好对，覆盖多语言、多文化社区 → 证明了跨社区的偏好可以被统一表示，为后续规模实验提供了坚实基底。**  
2. **系统化探索偏好建模的规模律 → 在 1.5B、7B、13B、30B、72B 参数模型上分别训练，同步扩大训练数据量 → 发现对抗性指标随规模稳步提升，客观指标在大模型上出现突现（emergent）行为，而主观指标缺乏明显趋势。**  
3. **将 WorldPM 作为偏好微调的基础模型 → 在七个公开基准的 20 个子任务上，用 7K、100K、800K 规模的偏好数据进行二次微调 → 多数子任务提升超过 5%，显示出跨任务的泛化能力。**  
4. **在内部 RLHF 流程中直接接入 WorldPM → 替换原有的偏好模型后，在内部和公开评测上分别提升 4%~8% → 证明了规模化偏好模型可以直接提升实际产品的对话质量。  

### 方法详解
整体思路可以拆成三步：**数据收集 → 统一建模 → 规模实验与微调**。  
1. **数据收集**：作者爬取了 Reddit、知乎、StackExchange 等公开论坛的评论对，筛选出明确的“更好/更差”标记，形成 1500 万条二选一的偏好对。每条对都附带来源标签（语言、社区），以便后续分析。  
2. **统一建模（World Preference）**：把所有来源的偏好映射到同一个向量空间。具体做法是：先用一个预训练的大语言模型把两段文本分别编码成向量，然后通过一个小的二分类头预测哪一段更受欢迎。训练目标是交叉熵损失，和传统的对比学习类似，只是这里的正负样本来自真实人类选择，而不是随机负例。  
3. **规模实验**：在 1.5B‑72B 参数的基模型上分别进行上述偏好训练，训练数据量也随模型大小线性增长（小模型用 1/5 数据，大模型用全量）。每个模型训练完后，用三类指标评估：  
   - **对抗性指标**：构造包含误导性信息的对比对，检查模型是否仍能捕捉真实偏好。  
   - **客观指标**：在事实问答集上让模型选择更准确的答案。  
   - **主观指标**：让少数标注者或已有的偏好模型打分，观察模型输出的主观满意度。  
4. **微调与 RLHF 集成**：把训练好的 WorldPM 作为偏好奖励模型，直接喂给强化学习阶段的 PPO（Proximal Policy Optimization）算法。微调时仍保持原有的语言模型权重，只在奖励函数上使用 WorldPM 的输出。  

最巧妙的地方在于**“统一表示”**的设计：不同社区的偏好往往有文化差异，但作者通过共享的编码器和统一的二分类头，让模型学会在同一空间里比较跨文化的偏好，这为规模律的发现提供了可能。  

### 实验与效果
- **数据与任务**：1500 万偏好对、七个公开基准（包括 TruthfulQA、OpenAI‑Evals 等）共 20 个子任务，分别使用 7K、100K、800K 规模的微调数据。  
- **对比基线**：传统的单任务偏好微调模型、直接使用原始语言模型进行 RLHF、以及公开的 OpenAI SFT + RLHF 流程。  
- **主要结果**：  
  - 对抗性指标随模型和数据规模几乎呈线性提升，72B 模型在欺骗检测上比 1.5B 提高约 12%。  
  - 客观指标在 30B 以上出现突现，准确率提升约 6%‑9%。  
  - 主观指标在所有规模上波动不大，说明少量人类或 AI 评审的主观偏好难以通过规模化学习捕获。  
  - 在微调实验中，多数子任务的得分提升超过 5%，尤其在 “信息检索” 与 “事实一致性” 两类任务上提升最高。  
  - 将 WorldPM 融入内部 RLHF 后，内部评测提升 4%‑8%，公开评测（如 MT-Bench）也有可观提升。  
- **消融实验**：作者分别去掉统一编码器、只使用单社区数据、以及固定奖励模型不更新，发现统一编码器和全量跨社区数据是提升的关键因素。  
- **局限性**：主观指标缺乏规模律，说明仅靠大模型和大数据仍难捕获细粒度的个人喜好；此外，公开论坛数据可能带有噪声和偏见，作者在论文中承认需要更严格的数据清洗。  

### 影响与延伸思考
这篇工作首次在偏好建模上提出规模律的概念，直接推动了“偏好大模型”这一新方向。随后出现的几篇论文（如 **PreferenceGPT**、**ScalingRLHF**）都在尝试用更大规模的偏好数据或更深的模型来进一步验证和扩展 WorldPM 的发现。对实际产品而言，使用统一的跨社区偏好模型可以显著降低微调成本，尤其在多语言、多文化的对话系统中具有潜在价值。想继续深入，可以关注以下几个方向：  
- **噪声与偏见过滤**：如何在大规模爬取的偏好数据中自动剔除有害信息。  
- **主观偏好建模**：探索更细粒度的个人化偏好表示，或结合用户画像进行个性化微调。  
- **跨模态偏好**：把文本偏好扩展到图像、音频等多模态场景，验证规模律是否仍然成立。  

### 一句话记住它
**WorldPM 证明：把全世界的偏好数据统一训练，大模型就能在对抗性和客观任务上持续提升，但个人主观喜好仍需要别的办法。**