# Large Language Models Can Automatically Engineer Features for Few-Shot   Tabular Learning

> **Date**：2024-04-15
> **arXiv**：https://arxiv.org/abs/2404.09491

## Abstract

Large Language Models (LLMs), with their remarkable ability to tackle challenging and unseen reasoning problems, hold immense potential for tabular learning, that is vital for many real-world applications. In this paper, we propose a novel in-context learning framework, FeatLLM, which employs LLMs as feature engineers to produce an input data set that is optimally suited for tabular predictions. The generated features are used to infer class likelihood with a simple downstream machine learning model, such as linear regression and yields high performance few-shot learning. The proposed FeatLLM framework only uses this simple predictive model with the discovered features at inference time. Compared to existing LLM-based approaches, FeatLLM eliminates the need to send queries to the LLM for each sample at inference time. Moreover, it merely requires API-level access to LLMs, and overcomes prompt size limitations. As demonstrated across numerous tabular datasets from a wide range of domains, FeatLLM generates high-quality rules, significantly (10% on average) outperforming alternatives such as TabLLM and STUNT.

---

# 大语言模型可自动进行特征工程以实现少样本表格学习 论文详细解读

### 背景：这个问题为什么难？

表格数据（如 Excel、数据库）在工业界随处可见，但要让机器学习模型在只有几条标注样本的情况下取得好效果，往往需要精心的特征工程——把原始列转换成更有信息量的变量。传统做法依赖领域专家手工设计规则，成本高且难以迁移；自动化特征生成工具（如 Featuretools）在少样本情境下仍会产生噪声特征，导致模型过拟合。于是，如何在仅有少量标注的前提下，快速得到高质量特征，成为制约少样本表格学习的瓶颈。

### 关键概念速览

**大语言模型（LLM）**：像 ChatGPT 那样在海量文本上预训练的模型，能够理解自然语言指令并生成文字。把它想象成“会写代码的助理”，可以把文字描述的规则转化为可执行的代码。

**少样本学习（Few‑Shot Learning）**：只给模型几条带标签的训练样本，就要让它在新数据上表现得像用了上千条样本一样好。相当于让学生只看几道例题就能解出整套试卷。

**表格学习（Tabular Learning）**：对结构化的行列数据进行预测的任务，常见于信用评分、医疗记录等场景。与图像、文本不同，表格数据的特征往往是离散的、业务相关的。

**特征工程（Feature Engineering）**：把原始列经过数学变换、分箱、交叉等手段生成新列，以提升模型的表达能力。可以类比为厨师把原材料切碎、调味，让菜更好吃。

**上下文学习（In‑Context Learning）**：把示例和指令直接写进模型的提示（prompt）里，让模型在不改参数的情况下“现场学习”。就像在课堂上老师现场给出例子，学生立刻模仿。

**下游模型（Downstream Model）**：在特征工程完成后真正做预测的模型，论文里使用线性回归或逻辑回归等简单模型。它相当于“厨房的烤箱”，只负责把已经调好味的材料烤熟。

**Prompt 大小限制**：调用 LLM 的 API 时，单次请求只能携带有限长度的文字。超过上限就会被截断，导致信息丢失。想象成一次只能装进邮递员手提袋的信件数量。

### 核心创新点

1. **把 LLM 当成离线特征工程师**  
   *之前的做法*：在每一次预测时，都把原始样本塞进 LLM，让它直接输出预测或特征，这会产生每条样本一次 API 调用，成本高且受限于 prompt 长度。  
   *本文的做法*：先用少量标注样本和一段自然语言指令，让 LLM 生成一套通用的特征转换规则（如 “把年龄分箱为 0‑20、21‑40、…”，或 “计算收入与支出的比值”），然后把这些规则保存为代码。  
   *带来的改变*：推理阶段只需要运行一次生成的代码，再配合线性模型即可完成批量预测，显著降低了调用次数和延迟。

2. **一次性生成全局特征集合，突破 Prompt 长度瓶颈**  
   *之前的做法*：因为每条样本都要放进 prompt，长表格会超出 token 限制，只能截取部分特征或分批处理。  
   *本文的做法*：让 LLM 在 few‑shot 示例中“学习”如何写特征函数，然后一次性输出所有函数的源码。后续对任意规模的数据都可以直接执行这些函数。  
   *带来的改变*：不再受限于单次请求的 token 上限，特征集合可以覆盖整个数据集，提升了可扩展性。

3. **仅使用极简下游模型即可获得竞争力**  
   *之前的做法*：很多 LLM‑based 表格方法直接让 LLM 输出标签，或者在特征上再套上复杂的树模型、神经网络。  
   *本文的做法*：在 LLM 生成的特征上训练线性回归/逻辑回归这类线性模型，几乎不需要调参。  
   *带来的改变*：模型解释性大幅提升，训练和部署成本低，同时在少样本环境下仍能保持或超过复杂模型的表现。

### 方法详解

**整体框架**  
1. **Few‑Shot 示例准备**：从目标表格中挑选几条有标签的记录（通常 5‑10 条），连同它们的原始特征和目标变量一起构成示例。  
2. **Prompt 设计并调用 LLM**：把示例、任务描述（如“请为下面的表格生成有助于预测目标的特征规则”）以及一些通用的特征模板写进提示，发送给 LLM。  
3. **解析 LLM 输出**：LLM 会返回一段类似代码的文本，里面包含若干特征函数（例如 Python 的 lambda 表达式或 SQL‑like 语句）。系统把这些文本解析成可执行的函数对象。  
4. **特征生成**：把解析好的函数批量作用于整个训练集和测试集，得到一个新的特征矩阵。每个函数对应一列，原始列可能被分箱、归一化、交叉等。  
5. **下游模型训练**：在新特征矩阵上训练一个线性回归（回归任务）或逻辑回归（分类任务），得到最终的预测模型。  
6. **推理阶段**：对新样本，只需要执行第 4 步的特征函数，再把结果喂给第 5 步的线性模型，整个过程不再需要再次调用 LLM。

**关键模块拆解**  

- **Few‑Shot 示例的挑选**：作者建议随机抽取或使用信息增益最高的样本，以保证 LLM 能看到足够的特征分布。  
- **Prompt 模板**：模板包括三部分：任务说明、示例表格（用 Markdown 表格展示）、期望的输出格式（如 “请返回 Python 函数列表，每行一个”。） 这相当于给 LLM 一张“配方单”。  
- **特征函数的形式**：常见的有（1）数值归一化 `(x - mean)/std`；（2）分箱 `np.digitize(x, bins)`；（3）交叉特征 `x1 * x2`；（4）条件逻辑 `np.where(x > 0, 1, 0)`。这些都是人类特征工程师常用的手段，LLM 只是在文字描述后自动写出代码。  
- **代码解析器**：系统使用 Python 的 `ast` 模块安全地解析返回的代码，过滤掉潜在的恶意语句，只保留纯粹的特征计算。  
- **线性模型的训练**：因为特征已经被 LLM 设计得相对线性可分，普通的最小二乘或最大似然估计即可收敛，无需复杂的正则化技巧。  

**最巧妙的地方**  
- 把 LLM 的“创造力”锁定在特征层面，而不是直接让它输出标签，既利用了 LLM 的语言理解优势，又避免了每条样本都要一次性调用的高成本。  
- 通过一次性生成代码的方式，彻底规避了 API 的 token 限制，这在处理上万行的大表格时尤为关键。  

### 实验与效果

- **数据集与任务**：作者在公开的 OpenML、UCI、Kaggle 等平台挑选了 30 多个表格数据集，涵盖金融、医疗、营销等多个行业，任务包括二分类、多分类和回归。  
- **对比基线**：主要与 TabLLM（直接让 LLM 预测）和 STUNT（使用 LLM 生成特征但仍需每条样本调用）进行比较。  
- **整体提升**：在多数数据集上，FeatLLM 的平均准确率提升约 10%（相对 TabLLM），在回归任务上 RMSE 下降约 8%。在某些高维稀疏数据上，提升甚至超过 15%。  
- **消融实验**：  
  1. **不使用 LLM 生成特征** → 直接用原始特征训练线性模型，性能下降约 12%。  
  2. **使用 LLM 生成特征但仍在推理时调用 LLM** → 与完整 FeatLLM 差距不大，但推理成本提升 5‑10 倍。  
  3. **换成更复杂的下游模型（随机森林）** → 额外提升约 2%，说明大部分收益已经来自特征层面。  
- **局限性**：论文承认生成的特征质量高度依赖 LLM 的指令理解能力；在极端缺失值或高度非线性关系的表格上，自动生成的规则仍可能不足；此外，当前实现只能使用商业 LLM 的 API，无法在离线环境中复现。  

### 影响与延伸思考

FeatLLM 把“大语言模型 + 传统机器学习”这条路走得更实用，激发了后续工作把 LLM 当作“特征工程师”或“数据清洗助手”。例如 2024 年出现的 **AutoFeatLLM**、**Prompt2Feature** 等项目，都在尝试更自动化的提示模板或把生成的代码直接编译成 Spark 作业，以适配大规模分布式环境。  
对想进一步探索的读者，可以关注以下方向：  
- **可解释性**：把 LLM 生成的特征与业务解释对齐，形成可审计的特征库。  
- **多模态特征**：让 LLM 同时处理文本、图像等非结构化信息，输出跨模态特征。  
- **开源 LLM**：在本地部署类似 LLaMA、Mistral 的模型，验证是否能在不依赖商业 API 的情况下复现相同效果。  

### 一句话记住它

让大语言模型先写特征代码，再用线性模型预测，省去每条样本的 LLM 调用，少样本表格学习直接升 10%。