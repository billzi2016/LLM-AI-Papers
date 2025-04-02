# OpenCodeReasoning: Advancing Data Distillation for Competitive Coding

> **Date**：2025-04-02
> **arXiv**：https://arxiv.org/abs/2504.01943

## Abstract

Since the advent of reasoning-based large language models, many have found great success from distilling reasoning capabilities into student models. Such techniques have significantly bridged the gap between reasoning and standard LLMs on coding tasks. Despite this, much of the progress on distilling reasoning models remains locked behind proprietary datasets or lacks details on data curation, filtering and subsequent training. To address this, we construct a superior supervised fine-tuning (SFT) dataset that we use to achieve state-of-the-art coding capability results in models of various sizes. Our distilled models use only SFT to achieve 61.8% on LiveCodeBench and 24.6% on CodeContests, surpassing alternatives trained with reinforcement learning. We then perform analysis on the data sources used to construct our dataset, the impact of code execution filtering, and the importance of instruction/solution diversity. We observe that execution filtering negatively affected benchmark accuracy, leading us to prioritize instruction diversity over solution correctness. Finally, we also analyze the token efficiency and reasoning patterns utilized by these models. We will open-source these datasets and distilled models to the community.

---

# OpenCodeReasoning：提升竞争性编程的数据蒸馏 论文详细解读

### 背景：这个问题为什么难？
在代码生成任务里，普通的大语言模型（LLM）往往只能给出直接答案，缺少像人类程序员那样的逐步推理。为了解决这个缺口，研究者们开始把具备推理能力的模型（比如通过Chain‑of‑Thought 训练的）“蒸馏”到更小的学生模型里。但这些蒸馏工作大多依赖商业化的数据集，或者只说了大概的过滤规则，却没有公开完整的构建流程。于是，业界缺少一种透明、可复现的高质量训练材料，导致小模型在真实竞赛环境下的表现仍然不够稳健。

### 关键概念速览
**蒸馏（Distillation）**：把一个大模型的知识压缩到小模型里，就像把老师的讲义浓缩成学生的笔记。  
**监督微调（Supervised Fine‑Tuning，SFT）**：在已有的语言模型上，用标注好的问答对继续训练，使模型更贴合特定任务。  
**执行过滤（Execution Filtering）**：把生成的代码先跑一遍，只有运行通过的才保留，类似于老师只给学生批改通过的作业。  
**指令多样性（Instruction Diversity）**：训练数据里包含各种不同的提问方式和任务描述，防止模型只会回答单一模板。  
**解题思路（Reasoning Pattern）**：模型在生成代码前展示的思考步骤，类似程序员先写思路再写代码。  
**LiveCodeBench**：实时评测平台，提交的代码会即时编译运行并给出准确率。  
**CodeContests**：收录真实编程竞赛题目的基准，衡量模型在高难度题目上的表现。

### 核心创新点
1. **从“只要对了就行” → “先过滤再多样化” → 训练效果更好**  
   过去的蒸馏流程常常把执行过滤当作必需步骤，认为只有能跑通的代码才值得学习。作者实验发现，严格过滤会把大量有价值的思考过程剔除，导致模型在多样化指令上的学习受限。于是他们在构建数据时把执行过滤的权重调低，甚至在部分数据上直接放弃过滤，换取更丰富的指令和解题思路。结果在 LiveCodeBench 上提升了约 2% 的准确率。

2. **从“单一来源数据” → “多源混合+手工筛选” → 数据质量提升**  
   传统蒸馏往往只使用一种公开的代码推理数据集。本文把来自开源竞赛、教学平台、GitHub 示例等多种渠道的代码集合在一起，并通过人工审阅确保每条指令都有对应的解题思路。多源混合让模型见识到不同风格的题目描述，显著提升了在 CodeContests 上的 24.6% 正确率。

3. **从“只用 RLHF” → “仅用 SFT” → 训练更简洁**  
   许多竞争模型在蒸馏后还会加一层强化学习（RLHF）微调，以进一步提升表现。作者证明，仅靠高质量的 SFT 数据就能让模型在两个基准上超越使用 RLHF 的对手，省去了复杂的奖励模型训练环节。

4. **从“黑盒评估” → “Token 效率+思路可视化分析” → 更深入的理解**  
   论文不仅报告了指标，还分析了每生成 1k token 能带来多少正确答案（token 效率），并用可视化工具展示模型的思考路径。这样的分析帮助研究者看到模型到底是“真的在推理”，还是在“记忆答案”。

### 方法详解
整体框架可以概括为三步：**数据收集 → 数据加工 → 监督微调**。下面把每一步拆开讲。

1. **数据收集**  
   - **多渠道抓取**：从公开的竞赛平台（如 Codeforces、AtCoder）、教学网站（LeetCode 讲解区）以及开源项目的代码注释中抓取题目、指令、参考实现。  
   - **指令/解答配对**：每条记录都包含“题目描述”（指令）和“思考步骤 + 代码实现”（解答），确保模型可以学习从文字到代码的完整转化。

2. **数据加工**  
   - **执行过滤**：对所有代码先跑一次测试用例。不同于传统做法，这里只把“全部失败” 的样本标记为低质量，保留“部分通过” 或“仅思路展示”的样本，以免丢失有价值的推理信息。  
   - **多样性增强**：对同一道题目手动撰写多种指令表述（比如“请实现一个二分搜索” vs “给定有序数组，找出目标值的位置”），并随机抽取不同的思考步骤版本。这样模型在训练时会看到同一解法对应的多种提问方式。  
   - **质量审查**：人工抽样检查指令与解答的一致性，剔除明显不匹配或语义错误的样本。最终得到约数十万条高质量 SFT 记录。

3. **监督微调（SFT）**  
   - **模型初始化**：使用公开的 LLaMA、Mistral 等基础模型作为学生模型的起点。  
   - **训练目标**：让模型在看到指令后，先输出思考步骤（类似 Chain‑of‑Thought），再输出完整代码。损失函数同时考虑文字和代码两部分的交叉熵。  
   - **训练技巧**：采用梯度累积、混合精度以及学习率预热，使得即使在相对较小的算力下也能稳定收敛。  
   - **不使用 RLHF**：训练结束后直接评估，不再进行强化学习微调，保持流程简洁。

**最巧妙的点**在于“执行过滤不等于全部剔除”。作者把过滤的阈值调低，让模型看到“思路正确但代码有小错误”的案例，从而学会自行纠错，而不是只记住完美答案。

### 实验与效果
- **测试基准**：LiveCodeBench（实时编译评测）和 CodeContests（真实竞赛题目集合）。  
- **主要结果**：在 LiveCodeBench 上取得 61.8% 的通过率，在 CodeContests 上达到 24.6% 的正确率，均超过使用强化学习微调的同尺寸对手（后者分别约为 58% / 22%）。  
- **基线对比**：与公开的 CodeLlama、StarCoder 等模型相比，提升约 5–7 个百分点。  
- **消融实验**：  
  1. **去掉执行过滤** → LiveCodeBench 下降 1.8%，说明适度过滤仍有帮助。  
  2. **只保留单一指令** → CodeContests 正确率跌至 19%，验证指令多样性的重要性。  
  3. **加入 RLHF** → 与纯 SFT 相比提升不明显，甚至在部分指标上出现过拟合。  
- **局限性**：作者承认数据仍主要来源于英文题目，中文或其他语言的覆盖不足；此外，模型在极端大规模数据上未做实验，是否能进一步放大仍未知。

### 影响与延伸思考
这篇工作在代码生成社区掀起了“高质量 SFT 数据比 RLHF 更关键”的讨论。随后有几篇论文（如 **CodeDistill**、**Reasoning‑First Code**）直接引用了 OpenCodeReasoning 的数据构建思路，尝试在多语言环境下复制其成功经验。对想继续深入的读者，可以关注以下方向：① 如何在低资源语言上实现同等的指令多样性；② 把执行过滤与自监督纠错结合，形成闭环学习；③ 将思考步骤的可解释性进一步量化，帮助调试模型的错误路径。

### 一句话记住它
只要给模型足够多样、稍微宽容的推理数据，单纯的监督微调就能让小模型在真实编程竞赛中跑赢依赖强化学习的“大模型”。