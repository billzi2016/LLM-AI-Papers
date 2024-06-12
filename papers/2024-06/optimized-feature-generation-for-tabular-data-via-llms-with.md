# Optimized Feature Generation for Tabular Data via LLMs with Decision   Tree Reasoning

> **Date**：2024-06-12
> **arXiv**：https://arxiv.org/abs/2406.08527

## Abstract

In tabular prediction tasks, tree-based models combined with automated feature engineering methods often outperform deep learning approaches that rely on learned representations. While these feature engineering techniques are effective, they typically depend on a pre-defined search space and primarily use validation scores for feature selection, thereby missing valuable insights from previous experiments. To address these limitations, we propose a novel tabular learning framework that utilizes large language models (LLMs), termed Optimizing Column feature generator with decision Tree reasoning (OCTree). Our key idea is to leverage the reasoning capabilities of LLMs to identify effective feature generation rules without manually specifying the search space and provide language-based reasoning information highlighting past experiments as feedback for iterative rule improvements. We use decision trees to convey this reasoning information, as they can be easily represented in natural language, effectively providing knowledge from prior experiments (i.e., the impact of the generated features on performance) to the LLMs. Our empirical results demonstrate that OCTree consistently enhances the performance of various prediction models across diverse benchmarks, outperforming competing automated feature engineering methods. Code is available at https://github.com/jaehyun513/OCTree.

---

# 利用决策树推理的 LLM 优化表格特征生成 论文详细解读

### 背景：这个问题为什么难？

在结构化表格预测任务里，传统的树模型（如随机森林、梯度提升树）配合手工或自动特征工程往往跑得比深度学习模型好。可是现有的自动特征工程工具大多只能在一个预先设定好的搜索空间里挑特征，搜索过程只看验证分数，根本不利用之前实验产生的经验。于是搜索效率低、生成的特征质量受限，导致在实际业务中很难快速得到最优特征集合。

### 关键概念速览
- **表格特征工程**：对原始列进行组合、变换、编码等操作，以产生更有预测力的特征。类似于厨师在原材料上切、调、混合，做出更好吃的菜。
- **大语言模型（LLM）**：能够理解并生成自然语言的深度模型，如 GPT 系列。这里把它当成“会写特征生成规则的智能助理”。
- **决策树**：一种把特征空间划分成若干叶子节点的模型，结构像一棵树，根节点是判断条件，叶子给出预测。因为每个分支都可以用一句话描述，它是把模型行为翻译成自然语言的好桥梁。
- **搜索空间**：特征生成规则的候选集合。传统方法需要人工列出所有可能的组合方式，就像提前写好所有菜谱。
- **迭代式反馈**：把上一次实验的结果（哪些特征提升了性能，哪些没有）以语言形式告诉 LLM，让它在下一轮生成更有针对性的规则。相当于厨师尝完一道菜后，根据口味反馈调整配方。

### 核心创新点
1. **搜索空间不再手工定义 → 用 LLM 直接生成特征规则**  
   传统方法必须先列出所有可能的算子组合，搜索时只能在这些固定选项里挑。OCTree 把特征生成任务交给 LLM，让它在自然语言提示下自行创作规则，省去人工列举的步骤。

2. **把实验结果转化为自然语言 → 决策树充当“解释器”**  
   直接把验证分数喂给 LLM 会让模型难以理解。作者用决策树把每一次特征生成的效果映射成一段可读的描述（例如“如果新特征 A 与 B 的乘积提升了 2%”，就写成一句话），再作为 LLM 的上下文。这样 LLM 能像人类一样“听”到过去的经验。

3. **闭环迭代 → LLM 根据反馈不断改进规则**  
   每生成一批特征后，用决策树总结其贡献，再把总结喂回 LLM，指导它在下一轮生成更有价值的特征。相当于“尝—评—改”三部曲的自动化。

4. **跨模型提升 → 生成的特征可直接供任意预测模型使用**  
   与只针对特定树模型的特征工程不同，OCTree 产生的特征在实验中对多种模型（包括线性模型、神经网络）都有提升效果，展示了方法的通用性。

### 方法详解
**整体框架**  
OCTree 的工作流可以概括为四步：① 初始化目标列；② LLM 生成候选特征规则；③ 用决策树评估这些规则并生成自然语言反馈；④ 将反馈重新喂给 LLM，进入下一轮迭代。循环若干次后，收敛得到一套高质量特征集合。

**关键模块拆解**  

1. **目标列生成**  
   首先让 LLM 根据任务描述（比如“预测客户是否流失”）输出一个或多个目标列的名称，这一步确保后续特征生成围绕正确的预测目标展开。

2. **特征规则生成**  
   LLM 接收到目标列和上一轮的语言反馈后，依据提示生成若干自然语言描述的特征构造规则，例如“创建新特征 = `age` 与 `income` 的比值”。这些规则随后被解析成可执行的代码（如 pandas 操作），并加入到数据表中。

3. **决策树评估与解释**  
   将新生成的特征加入训练集，训练一棵浅层决策树（深度一般不超过 3），观察每个新特征在树的分裂中出现的频率和对验证分数的贡献。因为树的每个分支都是“如果‑则”形式，作者把它翻译成一句自然语言，例如：“当 `age_income_ratio` > 0.3 时，预测概率提升 1.8%”。这些句子构成了本轮的反馈文本。

4. **反馈回环**  
   反馈文本与原始提示一起送回 LLM，LLM 在生成新规则时会倾向于强化已被证明有效的特征模式，抑制无效或冗余的组合。循环结束的判定依据可以是验证分数不再显著提升，或达到预设的迭代次数。

**最巧妙的设计**  
把决策树的结构直接映射成自然语言，让 LLM 能“读懂”模型的经验，这一步把黑盒的数值反馈变成了人类可解释的知识。相比直接把分数喂进去，这种语言化的反馈更符合 LLM 的训练目标，也让迭代过程更透明。

### 实验与效果
- **数据集与任务**  
  论文在公开的表格基准（如 OpenML、Kaggle 中的回归与分类任务）上做实验，覆盖金融、医疗、营销等多个领域的真实业务数据。

- **对比基线**  
  与传统自动特征工程工具（如 Featuretools、AutoFeat）以及最新的基于强化学习的搜索方法进行比较。论文声称在大多数数据集上，OCTree 提升了 1%~4% 的验证 AUC（或 RMSE），并且在一些高维稀疏数据上超过 6%。

- **消融实验**  
  作者分别去掉（1）决策树反馈、（2）LLM 生成规则的语言提示，发现性能下降约 0.8%~2%，说明两者都是提升的关键因素。

- **局限性**  
  由于依赖大型语言模型，计算成本比传统特征工程高；在特征空间极其庞大的数据集上，LLM 生成的规则仍可能遗漏一些细粒度的交叉特征。论文也提到对 LLM 的提示工程仍有改进空间。

### 影响与延伸思考
这篇工作把自然语言推理引入表格特征工程，打开了“语言驱动特征搜索”的新思路。后续有研究尝试把其他可解释模型（如规则集、贝叶斯网络）同样转化为语言反馈，进一步提升 LLM 与结构化任务的协同效率。想继续深入，可以关注以下方向：① 更高效的提示优化方法；② 将 LLM 与强化学习结合，实现更细粒度的搜索；③ 在资源受限环境下的轻量化 LLM 替代方案。

### 一句话记住它
让决策树把特征效果说成话，LLM 听了再写新特征——语言闭环驱动的表格特征生成。