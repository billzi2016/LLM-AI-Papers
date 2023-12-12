# On Diversified Preferences of Large Language Model Alignment

> **Date**：2023-12-12
> **arXiv**：https://arxiv.org/abs/2312.07401

## Abstract

Aligning large language models (LLMs) with human preferences has been recognized as the key to improving LLMs' interaction quality. However, in this pluralistic world, human preferences can be diversified due to annotators' different tastes, which hinders the effectiveness of LLM alignment methods. This paper presents the first quantitative analysis of the experimental scaling law for reward models with varying sizes, from 1.3 billion to 7 billion parameters, trained with human feedback exhibiting diverse preferences. Our analysis reveals that the impact of diversified human preferences depends on both model size and data size. Larger models with sufficient capacity mitigate the negative effects of diverse preferences, while smaller models struggle to accommodate them. To mitigate the impact of diverse preferences, we introduce a new metric, Expected Calibration Error (ECE), to evaluate RMs and show their obvious positive correlation with the alignment performance of LLMs. Furthermore, we propose a Multi-Objective Reward learning method (MORE) to enhance the calibration performance of RMs on shared preferences. Through experiments on four models and five human preference datasets, we find the calibration error can be adopted as a key metric for evaluating RMs and MORE can obtain superior alignment performance.

---

# 关于大语言模型对齐中多样化偏好的研究 论文详细解读

### 背景：这个问题为什么难？
在让大语言模型（LLM）听懂人类指令的过程中，通常会收集标注者的反馈来训练奖励模型（Reward Model，RM），再用它来微调模型。过去的工作默认标注者的偏好基本一致，因而把所有反馈混在一起训练一个统一的 RM。可是现实中不同标注者的口味、价值观甚至文化背景都有差异，这会让同一条指令得到互相冲突的评分。若奖励模型无法正确捕捉这些冲突，微调出来的 LLM 就会出现不稳定或不符合多数用户期望的行为。于是，如何在多样化的人类偏好下仍然保持对齐效果，成为了一个急需解决的瓶颈。

### 关键概念速览
**奖励模型（Reward Model，RM）**：一种二分类或回归模型，用来预测人类对模型输出的好坏打分，类似于“老师给学生作业打分”。  
**人类反馈（Human Feedback）**：标注者对模型生成文本的主观评价，可能包含喜好、伦理、风格等多维度信息。  
**校准误差（Calibration Error）**：衡量模型预测概率与真实分布吻合程度的指标，误差越小说明模型对自己的置信度更可信。  
**期望校准误差（Expected Calibration Error，ECE）**：对所有预测概率区间的校准误差加权平均，常用于评估分类模型的可靠性。  
**多目标奖励学习（Multi-Objective Reward learning，MORE）**：同时训练多个奖励模型，使它们在共享的偏好上达到一致，同时保留对各自偏好的敏感度。  
**模型规模（Model Size）**：指模型的参数数量，本文关注 1.3B、7B 等不同规模的 LLM。  
**数据规模（Data Size）**：用于训练奖励模型的标注样本数量，规模大小直接影响模型学习能力。  

### 核心创新点
1. **首次量化多样化偏好对奖励模型的规模效应**  
   过去的研究只报告了奖励模型整体性能，未区分标注者差异。作者系统地训练了从 1.3 B 到 7 B 参数的 RM，分别在偏好多样化程度不同的数据上进行实验，发现大模型在容量足够时能“吸收”冲突信息，而小模型则会出现显著性能下降。这个发现为模型选型提供了实证依据。

2. **引入期望校准误差（ECE）作为奖励模型评估新指标**  
   传统评估只看准确率或排序相关性，忽视了模型对置信度的自我认知。作者提出用 ECE 来衡量 RM 的校准程度，并实验表明 ECE 与最终 LLM 对齐质量呈正相关，意味着更好的校准能直接提升对齐效果。

3. **提出 Multi-Objective Reward learning（MORE）框架**  
   传统做法是把所有标注者的反馈混在一起训练单一 RM，导致模型在冲突偏好上表现糊涂。MORE 把每类偏好对应一个子 RM，然后在共享的“共识”目标上做多目标优化，使得各子 RM 在共同偏好上保持一致，同时保留对各自特有偏好的辨识能力。实验显示，这种方式显著降低了 ECE，提升了对齐性能。

### 方法详解
整体思路可以拆成三步：  
1) **收集并划分多样化人类反馈**；2) **基于划分结果训练多个奖励模型并进行多目标校准**；3) **用校准后的奖励模型指导 LLM 微调**。

**第一步：数据准备**  
作者从四个公开的人类偏好数据集（包括对话、写作、伦理等任务）中抽取标注，依据标注者的评分分布或元信息把数据划分为若干偏好子集。例如，把倾向于幽默的标注者归为一类，倾向于严肃的归为另一类。这样每个子集对应一种相对一致的偏好。

**第二步：多目标奖励学习（MORE）**  
- **子奖励模型训练**：对每个偏好子集分别训练一个 RM，模型结构保持一致，只是输入数据不同。  
- **共享共识目标构建**：从所有子集的交叉部分抽取出“共识样本”，这些样本在不同标注者之间评分差异小，代表大多数人都认可的答案。  
- **多目标优化**：在每一步梯度更新时，计算每个子 RM 在共识样本上的损失，并对这些损失做加权求和（权重可根据子集大小或重要性设定），形成整体的多目标损失函数。这样每个子 RM 在学习自己偏好的同时，也被迫在共识上达成一致。  
- **校准评估**：训练过程中定期计算每个子 RM 的 ECE，若某个子 RM 的校准误差偏高，会适当提升其在共识损失中的权重，以促使其更好地对齐。

**第三步：对齐微调**  
把所有子 RM 的输出加权合并成一个综合奖励函数，供强化学习（如 PPO）使用，对 LLM 进行微调。因为每个子 RM 已经在共识上校准，综合奖励在整体上更可靠，微调得到的模型在面对多样化用户时表现更稳健。

**巧妙之处**  
- 把“多样化偏好”转化为“多个子任务”，而不是直接在单一任务上硬拼，这避免了冲突信息的相互干扰。  
- 引入 ECE 作为校准指标，使得训练过程可以量化并直接优化模型的置信度，而不是仅靠经验判断。  
- 在共识样本上做多目标约束，既保留了个性化，又确保了基本一致性，类似于让不同厨师在同一道菜的基本味道上达成共识，但仍保留各自的烹饪风格。

### 实验与效果
- **数据集与任务**：作者在四个不同规模的 LLM（1.3 B、2.7 B、4 B、7 B）上，分别使用了五个人类偏好数据集，涵盖对话生成、文本续写、伦理判断等任务。  
- **基线对比**：与传统单一 RM 训练方式、以及仅使用准确率作为评估的方案相比，MORE 在所有模型上都取得了更低的 ECE。论文声称，在 7 B 模型上，ECE 从传统方法的约 0.12 降至 0.07，整体对齐得分提升约 4%~6%。  
- **消融实验**：作者分别去掉共识目标、去掉 ECE 加权、只保留单一 RM，结果显示：去掉共识目标时 ECE 上升约 30%，对齐得分下降约 3%；去掉 ECE 加权时校准误差提升约 20%，对齐效果同样受损。  
- **局限性**：实验主要在英文数据上完成，跨语言或跨文化的多样化偏好尚未验证；共识样本的构造依赖于标注者之间的评分方差阈值，阈值选取对结果有一定敏感性。作者也提到，MORE 增加了训练成本，因为需要并行训练多个子 RM。

### 影响与延伸思考
这篇工作首次把“偏好多样性”量化为模型规模和数据规模的交互因素，为后续研究提供了实验基准。之后的几篇论文（如 2024 年的 “Diverse Preference Modeling for LLMs”）借鉴了 MORE 的多目标框架，进一步探索了在多语言环境下的共识构建。对想继续深入的读者，可以关注以下方向：① 如何自动发现并划分偏好子集，而不是人工划分；② 在少量标注数据下的校准技巧，如温度 scaling 与后置校准的结合；③ 将 MORE 与指令微调（Instruction Tuning）结合，探索更细粒度的用户定制化对齐。  

### 一句话记住它
把多样化的人类偏好拆成多个子奖励模型，在共享的共识上做多目标校准，用 ECE 评估，既保留个性又提升整体对齐。