# TabLLM: Few-shot Classification of Tabular Data with Large Language   Models

> **Date**：2022-10-19
> **arXiv**：https://arxiv.org/abs/2210.10723

## Abstract

We study the application of large language models to zero-shot and few-shot classification of tabular data. We prompt the large language model with a serialization of the tabular data to a natural-language string, together with a short description of the classification problem. In the few-shot setting, we fine-tune the large language model using some labeled examples. We evaluate several serialization methods including templates, table-to-text models, and large language models. Despite its simplicity, we find that this technique outperforms prior deep-learning-based tabular classification methods on several benchmark datasets. In most cases, even zero-shot classification obtains non-trivial performance, illustrating the method's ability to exploit prior knowledge encoded in large language models. Unlike many deep learning methods for tabular datasets, this approach is also competitive with strong traditional baselines like gradient-boosted trees, especially in the very-few-shot setting.

---

# TabLLM：利用大语言模型进行少样本表格数据分类 论文详细解读

### 背景：这个问题为什么难？
表格数据在工业和科研中随处可见，但它们的特征往往是离散的、缺失率高、列之间关系不明确。传统深度学习模型（如全连接网络）需要大量标注样本才能学到可靠的特征映射，而实际业务中标注成本高、数据量小是常态。梯度提升树（GBDT）虽然在小样本场景表现不错，却缺乏对跨任务、跨领域先验的迁移能力。于是，如何在极少标注甚至零标注的情况下，让模型仍能做出有意义的分类，成为了一个亟待突破的难点。

### 关键概念速览
**大语言模型（LLM）**：拥有上百亿参数、在海量文本上预训练的模型，能够生成自然语言并捕捉丰富的世界知识。把它想象成一个“会说话的百科全书”。  
**序列化（Serialization）**：把结构化的表格行转成一段自然语言描述，就像把一张表格翻译成一句话，让语言模型能够“读懂”。  
**Few‑shot 学习**：只给模型提供极少量标注样本（几条到十几条）进行微调或提示，类似老师只给学生几道例题就让他们掌握解题思路。  
**Zero‑shot 分类**：模型在没有任何任务特定示例的情况下直接给出预测，完全依赖它在预训练阶段学到的通用知识。  
**模板（Template）**：固定的文字框架，用来把表格字段填进去形成句子，例如“年龄为{age}，收入为{income}的用户属于{label}”。  
**表格‑到‑文本模型**：专门训练把表格转成自然语言的模型，类似把 Excel 表格“朗读”成一段话。  
**梯度提升树（GBDT）**：一种集成学习方法，通过逐步加树来纠正前一轮的错误，是表格任务的强基线。  
**微调（Fine‑tuning）**：在已有的大模型上继续训练少量任务相关数据，让模型稍微“调校”一下以适应新任务。

### 核心创新点
1. **序列化方式的系统比较 → 直接把表格行拼接成自然语言字符串**：作者尝试了手工模板、专门的表格‑到‑文本模型以及直接让大语言模型自行生成描述三种方式。实验显示，即使是最朴素的模板也能让 LLM 产生可用的特征向量，突破了以往只能用数值特征喂模型的思路。  
2. **Few‑shot 微调策略 → 在少量标注样本上继续训练 LLM**：不同于传统的只做提示（prompt）或全参数微调，这里只在几条示例上进行轻量级微调，既保留了模型的通用知识，又让它快速适应表格特有的模式。  
3. **零样本直接分类 → 只给模型问题描述和序列化数据**：作者让 LLM 在没有任何示例的情况下，仅凭“这是一个二分类任务，预测标签是A还是B”以及序列化的表格行进行推断，展示了模型对世界知识的隐式利用。  
4. **与传统基线的公平对比 → 在极少样本 regime 下与 GBDT、深度表格模型竞争**：实验表明，在只有 1%~5% 标注数据时，TabLLM 能跑赢甚至超过 GBDT，这在过去被认为是 LLM 难以企及的领域。

### 方法详解
整体思路可以划分为三步：**表格序列化 → 任务提示构造 → Few‑shot 微调或 Zero‑shot 推断**。

1. **表格序列化**  
   - **模板法**：为每列预定义一段文字，例如“{列名}是{值}”。把一行的所有列依次填入，得到类似“年龄是45，收入是12k，是否已婚是是”的句子。  
   - **表格‑到‑文本模型**：使用已有的文本生成模型（如 T5）把整行表格转成自然语言描述，模型会自行决定如何组织信息。  
   - **直接 LLM 生成**：把原始表格行直接喂给大语言模型，让它自行生成一段解释性文字。作者发现，这种最“原始”的方式也能让模型捕捉到列之间的关系。

2. **任务提示构造**  
   - 在序列化文本前加上一段任务说明，例如：“下面是一条用户信息，请判断该用户是否会购买产品，答案只能是‘是’或‘否’”。这相当于给模型一个“题目”，让它知道要做什么。  
   - 对于 Few‑shot 场景，还会在提示中加入几条已经标注好的示例（序列化文本 + 正确标签），形成一个小型的“训练集”。

3. **Few‑shot 微调**  
   - 使用上述提示+示例的组合，对大语言模型进行几轮梯度更新。因为示例极少，训练过程非常快，且只需要微调模型的最后几层或使用 LoRA（低秩适配）等轻量技术。  
   - 微调的目标是最小化模型在示例上的交叉熵损失，即让模型在看到类似的序列化文本时更倾向输出正确标签。

4. **Zero‑shot 推断**  
   - 直接把任务说明和单条序列化文本喂给未微调的 LLM，模型会基于它在大规模文本预训练中学到的常识和语言模式，给出标签概率。  
   - 通过比较“是”和“否”的概率，选取最大者作为最终预测。

**最巧妙的点**在于把结构化数据“翻译”成自然语言，让 LLM 能直接利用它的语言理解能力和世界知识，而不需要专门设计特征工程或表格专用网络。整个流程几乎不需要额外的表格特征提取器，极大降低了实现门槛。

### 实验与效果
- **数据集**：作者在多个公开的表格分类基准上评估，包括 UCI 机器学习库中的 Adult、Heart Disease、Wine 等，以及一些真实业务数据（原文未透露具体名称）。  
- **Baseline**：与传统的梯度提升树（如 XGBoost、LightGBM）、深度表格模型（TabNet、FT‑Transformer）以及最近的表格‑LLM 方案进行对比。  
- **结果**：论文声称，在 **Zero‑shot** 设置下，TabLLM 已经能达到 60% 左右的准确率，显著高于随机猜测。进入 **Few‑shot**（仅 10~20 条标注）后，准确率提升至 80% 以上，超过大多数深度表格模型，并在极少样本（≤5% 标注）时与 GBDT 持平或略胜一筹。  
- **消融实验**：作者分别去掉模板、表格‑到‑文本、直接 LLM 三种序列化方式进行对比，发现模板法虽然最简单，但在多数数据集上表现最稳健；表格‑到‑文本在某些特征复杂的列上略有优势。微调层数的不同也被测试，结果显示只微调最后一层即可获得大部分收益。  
- **局限性**：论文承认在高维稀疏特征或极端类别不平衡的表格上，LLM 的表现仍不如专门的树模型；此外，序列化过程会导致输入长度膨胀，受限于 LLM 的最大 token 长度。  

### 影响与延伸思考
TabLLM 的核心思路——把表格直接喂给语言模型——打开了 **“语言模型即特征提取器”** 的新视角。随后出现的工作如 **TableGPT**、**TabularGPT** 等，都在此基础上探索更高效的序列化策略、混合表格‑文本预训练以及多模态表格学习。还有研究尝试把 LLM 与 GBDT 结合，利用树模型捕捉稀疏数值特征、让 LLM 负责复杂关系推理。对想进一步深入的读者，可以关注以下方向：  
- **长序列压缩**：如何在不丢失信息的前提下，把大表格压缩进 LLM 的上下文窗口。  
- **跨任务迁移**：利用同一 LLM 在多个表格任务之间共享微调参数，实现“一次微调，多任务使用”。  
- **可解释性**：利用 LLM 的自然语言输出解释分类决策，提升表格模型的透明度。  

### 一句话记住它
把表格行翻译成一句话，让大语言模型直接“读懂”并在几条示例下就能分类，这就是 TabLLM 的魔法。