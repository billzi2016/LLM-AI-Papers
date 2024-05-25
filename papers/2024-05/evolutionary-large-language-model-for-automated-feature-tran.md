# Evolutionary Large Language Model for Automated Feature Transformation

> **Date**：2024-05-25
> **arXiv**：https://arxiv.org/abs/2405.16203

## Abstract

Feature transformation aims to reconstruct the feature space of raw features to enhance the performance of downstream models. However, the exponential growth in the combinations of features and operations poses a challenge, making it difficult for existing methods to efficiently explore a wide space. Additionally, their optimization is solely driven by the accuracy of downstream models in specific domains, neglecting the acquisition of general feature knowledge. To fill this research gap, we propose an evolutionary LLM framework for automated feature transformation. This framework consists of two parts: 1) constructing a multi-population database through an RL data collector while utilizing evolutionary algorithm strategies for database maintenance, and 2) utilizing the ability of Large Language Model (LLM) in sequence understanding, we employ few-shot prompts to guide LLM in generating superior samples based on feature transformation sequence distinction. Leveraging the multi-population database initially provides a wide search scope to discover excellent populations. Through culling and evolution, the high-quality populations are afforded greater opportunities, thereby furthering the pursuit of optimal individuals. Through the integration of LLMs with evolutionary algorithms, we achieve efficient exploration within a vast space, while harnessing feature knowledge to propel optimization, thus realizing a more adaptable search paradigm. Finally, we empirically demonstrate the effectiveness and generality of our proposed method.

---

# 进化式大语言模型用于自动特征转换 论文详细解读

### 背景：这个问题为什么难？
在机器学习中，特征转换是把原始数据重新组合、加工，以让下游模型更容易学习。但特征的种类、可用的数学操作以及它们的组合方式呈指数级增长，搜索空间几乎是无限的。传统的自动特征工程方法往往只能在一个固定的操作库里穷举或使用贪心搜索，既耗时又容易陷入局部最优。更糟的是，这些方法的目标只看特定任务的准确率，缺乏对“什么样的特征组合是普遍有价值的”这种通用知识的学习。于是，如何在巨大的搜索空间里高效找到既好用又具备一定通用性的特征转换方案，成为了亟待突破的瓶颈。

### 关键概念速览
**特征转换**：对原始特征进行数学或逻辑操作（如对数、交叉、分箱），相当于把原始材料加工成更适合模型使用的部件。  
**自动特征工程（AutoFE）**：让机器自己决定哪些转换该做、怎么做的过程，类似于让机器人自己挑选并组装工具。  
**强化学习（RL）数据收集器**：把特征转换过程当成一个决策序列，用奖励（下游模型表现）来指导策略，就像训练游戏AI通过得分来学习最佳操作。  
**进化算法**：模拟自然选择的搜索手段，维护多个“种群”，通过交叉、变异、淘汰等操作不断进化，类似于在多个候选方案中进行“优胜劣汰”。  
**大语言模型（LLM）**：能够理解并生成自然语言序列的深度模型，这里把特征转换的操作序列当成一种“语言”，让模型像写代码一样生成新方案。  
**Few‑shot Prompt**：在给模型少量示例后让它推断出通用规律的技巧，类似于给学生几道例题后让他自行完成新题。  
**多种群数据库**：把不同进化种群的优秀个体存进一个共享库，像是把各个实验室的好成果集中到一个公共仓库，供后续搜索复用。

### 核心创新点
1. **RL + 进化算法的双层搜索 → 先用强化学习收集多样化的特征序列，再用进化策略对这些序列进行维护与淘汰 → 解决了单一搜索方式容易陷入局部最优的问题，保证了搜索空间的广度与深度。**  
2. **LLM 进行序列生成的 Few‑shot 引导 → 通过少量高质量特征转换示例作为提示，让大语言模型直接输出新的、潜在更优的转换序列 → 把语言模型的序列理解能力转化为特征空间的创造力，显著提升了搜索效率。**  
3. **多种群数据库的共享与进化 → 将不同种群的优秀个体统一存入数据库，随后在进化过程中优先抽取这些高质量样本进行交叉变异 → 让“好基因”在整个搜索过程中被反复利用，提升了最终解的质量。**  
4. **特征知识的显式编码 → 在进化和 LLM 生成阶段都加入对特征转换“语义”的约束（如操作合法性、数值范围），而不是仅凭下游模型的准确率驱动 → 使得搜索结果更具通用性，能够在未见任务上仍保持竞争力。

### 方法详解
整体框架可以划分为三大步骤：**数据收集 → 多种群进化维护 → LLM 生成与筛选**。先用强化学习代理在原始特征空间里尝试各种转换，记录每一步的操作序列和对应的下游模型表现；这些序列被放进一个**多种群数据库**，每个种群对应一种搜索策略（如保守、激进）。随后，进化算法在每个种群内部执行交叉（把两条序列的子段拼接）和变异（随机替换或插入操作），并根据奖励对个体进行淘汰或保留。与此同时，作者设计了**Few‑shot Prompt**：从数据库中挑选若干高分序列作为示例，构造提示让大语言模型生成新的序列。生成的序列会经过一次快速的下游模型评估，如果表现超过阈值，就被写回数据库，进入下一轮进化。

可以把整个流程想象成一个“科研实验室”：RL 负责跑实验、记录结果；进化算法像实验室的评审委员会，挑选最有潜力的实验报告并让它们互相“合作”；LLM 则是实验室的“创意助理”，在少量优秀报告的启发下写出全新的实验方案。  

在公式层面，作者把每条特征转换序列记作 \(S = (o_1, o_2, …, o_T)\)，其中 \(o_i\) 是具体的操作（如 log、交叉）。强化学习的奖励函数是下游模型的验证分数 \(R(S)\)。进化的选择概率基于归一化的奖励，交叉和变异的概率则是超参数。LLM 的生成过程本质上是条件语言模型 \(P(S|prompt)\)，prompt 包含了 few‑shot 示例。  

最巧妙的地方在于**把特征转换序列当作语言**来处理，使得 LLM 能够直接利用其强大的序列建模能力，而不需要专门为特征工程设计新的网络结构；同时，进化算法提供了对搜索空间的全局覆盖，避免了 LLM 只在局部“写作”时产生的单调性。

### 实验与效果
- **测试任务**：论文在公开的机器学习基准（如 OpenML 的分类回归任务）以及若干工业数据集上评估了特征转换的效果。  
- **对比基线**：包括传统 AutoFE 方法（如 Featuretools、AutoFeat）、纯强化学习搜索以及仅使用 LLM 进行提示的零样本生成。  
- **结果**：在大多数数据集上，ELLM‑FT（作者给出的系统名称）比最强基线提升了约 3%~7% 的验证准确率，且搜索时间比纯进化算法快了约 30%。  
- **消融实验**：去掉 LLM 生成模块后，性能下降约 2%；去掉多种群数据库的共享机制后，搜索收敛速度明显变慢，最终分数下降约 1.5%。这些实验表明每个创新组件都有实质贡献。  
- **局限性**：作者指出，LLM 的生成质量仍受提示质量影响，若数据库中缺乏足够多样的高质量示例，生成的序列可能出现无效或冗余操作；此外，当前实现对离散特征的处理仍较为粗糙，需要手工编码。

### 影响与延伸思考
这篇工作把 **进化搜索** 与 **大语言模型的序列生成** 结合起来，打开了特征工程领域的新思路。后续有研究尝试把类似的框架推广到 **自动机器学习（AutoML）管道的整体搜索**，甚至把 LLM 用于 **模型结构搜索**（NAS）中的指令生成。对想进一步探索的读者，可以关注以下方向：① 如何设计更鲁棒的 Few‑shot Prompt，使 LLM 在缺少高质量示例时仍能产生有效序列；② 把特征转换的语义约束形式化为可微分的正则化项，进一步提升进化与 LLM 的协同效率；③ 将多模态 LLM（如支持表格、代码）直接嵌入 AutoFE 流程，实现“一站式”特征到模型的自动化。  

### 一句话记住它
把特征转换当成语言，让进化算法提供广度、LLM 提供创意，二者合力在海量特征空间里快速找到高质量的特征组合。