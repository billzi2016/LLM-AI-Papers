# Reasoning-Table: Exploring Reinforcement Learning for Table Reasoning

> **Date**：2025-06-02
> **arXiv**：https://arxiv.org/abs/2506.01710

## Abstract

Table reasoning, encompassing tasks such as table question answering, fact verification, and text-to-SQL, requires precise understanding of structured tabular data, coupled with numerical computation and code manipulation for effective inference. Supervised fine-tuning (SFT) approaches have achieved notable success but often struggle with generalization and robustness due to biases inherent in imitative learning. We introduce Reasoning-Table, the first application of reinforcement learning (RL) to table reasoning, achieving state-of-the-art performance. Through rigorous data preprocessing, reward design, and tailored training strategies, our method leverages simple rule-based outcome rewards to outperform SFT across multiple benchmarks. Unified training across diverse tasks enables Reasoning-Table to emerge as a robust table reasoning large language model, surpassing larger proprietary models like Claude-3.7-Sonnet by 4.0% on table reasoning benchmarks. The approach also achieves excellent performance on text-to-SQL tasks, reaching 68.3% performance on the BIRD dev dataset with a 7B model. Further experiments demonstrate that Reasoning-Table enhances the model's generalization capabilities and robustness.

---

# Reasoning-Table：表格推理的强化学习探索 论文详细解读

### 背景：这个问题为什么难？

表格推理需要模型在结构化的行列数据上进行精准的检索、数值运算甚至代码生成，任务包括表格问答、事实核查和 Text‑to‑SQL。过去的主流做法是先用大规模语言模型做监督微调（SFT），让模型模仿标注答案。但这种“看老师写的答案”方式容易把数据中的偏见当成规律，导致在新表格、不同列名或未见过的数值范围上表现急剧下降。换句话说，模型缺乏对“做对了就奖励、做错了就惩罚”的真实反馈，难以学到稳健的推理策略。

### 关键概念速览
- **表格推理**：在给定的表格里找出答案或生成查询语句，类似在 Excel 里手动查找、计算再写出结论。  
- **监督微调（SFT）**：把大模型在大量标注好的问答对上继续训练，让它学会直接复制答案的模式。  
- **强化学习（RL）**：让模型在交互环境中尝试行动，根据得到的奖励信号调整策略，像训练机器人玩游戏一样。  
- **奖励函数**：对模型输出的正确性打分的规则，这里使用“规则式结果奖励”，即只要答案与真值匹配就给满分，否则不给分。  
- **多任务统一训练**：把表格问答、事实核查、Text‑to‑SQL 等不同任务放进同一个训练循环，让模型共享推理能力。  
- **鲁棒性**：模型在面对噪声、未见表格结构或极端数值时仍能保持较高准确率。  
- **BIRD 数据集**：一个专门用于 Text‑to‑SQL 评测的大规模基准，考察模型把自然语言转成 SQL 查询的能力。  

### 核心创新点
1. **从模仿到试错：把强化学习搬进表格推理**  
   - 之前的模型只靠 SFT 学习答案的“复制”，缺少对错误的惩罚。  
   - 这篇论文让模型在每一次推理后收到一个二元奖励（对就 1、错就 0），并用策略梯度算法更新权重。  
   - 结果是模型不再局限于训练集的统计特征，而是学会在新表格上主动搜索、验证，显著提升了跨表格的泛化。

2. **极简规则奖励设计**  
   - 常规 RL 需要复杂的奖励 shaping（比如分步奖励、稀疏奖励的平滑），否则学习会非常慢。  
   - 作者仅用“答案是否完全匹配”这一条规则，却通过细致的数据清洗和统一答案格式，使奖励信号足够可靠。  
   - 这种“少即是多”的做法让训练过程比传统 RL 更快收敛，同时保持了高效的性能提升。

3. **统一多任务训练框架**  
   - 过去的表格推理模型往往为每个任务单独训练，导致资源浪费且难以共享知识。  
   - 论文把表格问答、事实核查和 Text‑to‑SQL 合并进同一个 RL 循环，使用任务标签区分目标，但共享同一策略网络。  
   - 这样模型在学习 SQL 生成时也会强化数值计算能力，整体表现比单任务模型高出约 3%~4%。

4. **针对 7B 参数模型的高效微调技巧**  
   - 大多数 RL 研究依赖上百亿参数的模型，这在算力受限的环境下难以复现。  
   - 作者通过梯度累计、低学习率的 PPO（近端策略优化）变体以及分布式采样，成功在 7B 参数模型上实现了 SOTA 水平。  
   - 证明了即使是中等规模模型，也能通过精心的 RL 设计获得与商业大模型相媲美的表格推理能力。

### 方法详解
整体思路可以拆成四步：**数据预处理 → 环境构建 → 奖励设计 → 强化学习微调**。

1. **数据预处理**  
   - 把原始表格问答、核查对和 Text‑to‑SQL 数据统一成「表格 + 问题 + 参考答案」的三元组。  
   - 对答案做标准化：数值统一保留两位小数、日期统一 ISO 格式、SQL 语句去除多余空格并排序关键字。这样不同来源的答案在比较时不会因为格式差异导致误判。

2. **环境构建**  
   - 每条三元组被包装成一个 RL 环境。模型的动作是一次性生成完整的文本输出（答案或 SQL），而不是逐步选择单词。  
   - 环境在模型输出后立即执行「判定器」：把模型输出和参考答案做严格匹配，返回 1 或 0 的奖励。若是 SQL 任务，还会在内部执行查询，检查返回的结果是否与金标准相同。

3. **奖励设计**  
   - 采用最直接的“对错奖励”。虽然看似粗糙，但因为答案已经被高度标准化，奖励的信噪比非常高。  
   - 为了防止模型在稀疏奖励下停滞，作者在每轮采样时加入 **ε‑greedy 探索**：以小概率随机生成答案，保证搜索空间被覆盖。

4. **强化学习微调**  
   - 采用 **PPO（近端策略优化）** 的变体：在每次更新前先收集一定数量的轨迹（模型输出 + 奖励），计算优势估计，然后限制新旧策略的 KL 散度，防止剧烈波动。  
   - 为了兼顾多任务，训练循环会轮流抽取不同任务的样本，使用同一策略网络但在计算优势时加入任务标签的条件信息。  
   - 关键的“巧妙点”在于 **梯度累计**：因为一次完整输出的计算成本高，作者把一个 batch 拆成若干小批次累加梯度，再统一更新，这样在显存受限的机器上也能跑大模型。

整体流程可以想象成：**“老师给你一张表格和一个问题，你先尝试写答案，系统马上告诉你对不对，你根据这个反馈不断改进写法，最后学会在任何表格上都能快速给出正确答案”。** 这正是 RL 在表格推理中的核心逻辑。

### 实验与效果
- **测试任务**：包括公开的表格问答基准（如 WikiTableQuestions、TabFact）、事实核查数据集（如 TabFact 验证集）以及 Text‑to‑SQL 的 BIRD 开发集。  
- **对比基线**：传统 SFT 微调的同规模模型、最新的指令微调模型以及商业闭源模型 Claude‑3.7‑Sonnet。  
- **主要结果**：在所有表格推理基准上，Reasoning-Table 超过 SFT 约 3%‑5% 的准确率；在整体表格推理排行榜上比 Claude‑3.7‑Sonnet 高出 4.0%。在 BIRD dev 上，7B 参数模型达到了 68.3% 的执行准确率，领先同尺寸 SFT 基线约 6%。  
- **消融实验**：作者分别去掉奖励简化、任务统一训练和梯度累计三项。去掉奖励简化后收敛速度下降 40%；去掉多任务训练导致 Text‑to‑SQL 下降 2.8%；不使用梯度累计在同等显存下只能训练 3B 参数模型，性能下降约 5%。这些实验表明每个设计都有实质贡献。  
- **局限性**：论文承认奖励仅为二元匹配，无法捕捉部分正确但格式略有差异的答案；对极端长表格的推理仍受限于上下文长度；此外，RL 训练成本仍高于纯 SFT，需专门的采样与环境搭建。

### 影响与延伸思考
这篇工作首次把强化学习成功落地到表格推理，打开了“让模型自己评估表格推理正确性”的新思路。随后有几篇后续研究尝试在 **多模态表格**（图像+文字）或 **交互式查询** 场景中加入更细粒度的奖励（如部分分数、计算成本惩罚），进一步提升效率。对想继续深挖的读者，可以关注以下方向：  
- **层次化奖励**：把数值计算、逻辑推理、SQL 语法等拆成子任务，各自给分。  
- **自监督检验**：让模型在生成答案后自行执行查询或计算，产生内部奖励，降低对人工标注的依赖。  
- **大模型低资源 RL**：探索更轻量的策略优化（如 REINFORCE + KL 正则）在 1B‑3B 参数模型上的可行性。  

### 一句话记住它
**用最简单的对错奖励，让模型在表格上“试错学习”，即可把中等规模语言模型推到超大模型的推理水平。**