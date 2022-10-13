# Visual Classification via Description from Large Language Models

> **Date**：2022-10-13
> **arXiv**：https://arxiv.org/abs/2210.07183

## Abstract

Vision-language models (VLMs) such as CLIP have shown promising performance on a variety of recognition tasks using the standard zero-shot classification procedure -- computing similarity between the query image and the embedded words for each category. By only using the category name, they neglect to make use of the rich context of additional information that language affords. The procedure gives no intermediate understanding of why a category is chosen, and furthermore provides no mechanism for adjusting the criteria used towards this decision. We present an alternative framework for classification with VLMs, which we call classification by description. We ask VLMs to check for descriptive features rather than broad categories: to find a tiger, look for its stripes; its claws; and more. By basing decisions on these descriptors, we can provide additional cues that encourage using the features we want to be used. In the process, we can get a clear idea of what features the model uses to construct its decision; it gains some level of inherent explainability. We query large language models (e.g., GPT-3) for these descriptors to obtain them in a scalable way. Extensive experiments show our framework has numerous advantages past interpretability. We show improvements in accuracy on ImageNet across distribution shifts; demonstrate the ability to adapt VLMs to recognize concepts unseen during training; and illustrate how descriptors can be edited to effectively mitigate bias compared to the baseline.

---

# 基于大语言模型描述的视觉分类 论文详细解读

### 背景：这个问题为什么难？
在视觉-语言模型（如 CLIP）出现之前，零样本图像分类几乎只能靠把类别名字直接映射进语言空间，然后和图像特征做相似度匹配。这种做法把“猫”“狗”等标签当成唯一信息，忽略了语言本身能够提供的丰富属性描述。结果是模型往往只能捕捉到最显著的特征，难以解释为什么选了某个类别，也很难在出现新概念或分布漂移时灵活调整判断标准。换句话说，单一标签的零样本分类缺少可解释性和可控性，限制了它在实际应用中的可靠性。

### 关键概念速览
**视觉-语言模型（VLM）**：把图像和文字映射到同一个向量空间，使得语义相近的图像和文字距离更近。想象成一个双语词典，图像和文字可以互相查找对应关系。  
**零样本分类**：模型在没有见过目标类别的训练样本的情况下，直接利用语言信息进行分类。类似于人只看过描述就能辨认新动物。  
**描述符（descriptor）**：用来刻画某个类别的具体属性，如“条纹”“爪子”。它们是对类别的细化解释，像是给动物的特征清单。  
**大语言模型（LLM）**：如 GPT‑3，能够根据提示生成自然语言文本。这里把它当成“属性生成器”，自动列出每个类别的描述符。  
**提示工程（prompt engineering）**：设计输入给 LLM 的文字，让它输出符合需求的答案。相当于给模型下达明确的任务指令。  
**分布漂移（distribution shift）**：训练数据和测试数据在外观或风格上出现差异，比如从清晰图片转到模糊或艺术化的版本。模型需要具备一定的鲁棒性才能应对。  
**可解释性**：模型的决策过程能够被人类理解。这里指的是我们能看到模型依据哪些描述符做出分类。  

### 核心创新点
1. **从标签到描述符的范式转变**  
   之前的零样本分类直接把类别名称嵌入语言空间 → 这篇论文先让 LLM 生成每个类别的属性描述，然后让 VLM 检查这些属性是否在图像中出现 → 决策基于属性匹配，模型的判断依据变得可视化，解释性大幅提升。  

2. **利用 LLM 自动获取属性集合**  
   传统做法需要人工列举每个类别的特征，工作量大且难以覆盖所有类别 → 通过精心设计的提示，让 GPT‑3 自动输出高质量的描述符列表 → 省去了人工标注，且可以轻松扩展到数千个类别。  

3. **属性层级的可编辑性用于偏差调节**  
   以往的零样本分类无法对模型的偏好进行干预 → 论文展示了通过增删或修改描述符（比如去掉“肤色”之类的敏感属性）来直接影响分类结果 → 这为模型公平性提供了一种直观的调控手段。  

4. **在分布漂移下的鲁棒提升**  
   仅靠标签匹配在风格变化时容易失效 → 通过多属性匹配，模型可以在某些属性仍然可辨的情况下保持正确判断 → 实验显示在 ImageNet‑A、ImageNet‑R 等漂移基准上取得了明显优势。  

### 方法详解
整体思路可以拆成三步：**属性生成 → 属性匹配 → 决策聚合**。下面按顺序展开。

1. **属性生成（Descriptor Mining）**  
   - 给每个目标类别准备一个简短的文字提示，例如 “列出描述老虎外观的三个最显著特征”。  
   - 把这个提示送进大语言模型（GPT‑3），让它返回一个属性列表，如 “条纹、锋利的爪子、金黄色的毛皮”。  
   - 为了提升质量，论文使用了两轮对话：第一轮让模型给出候选属性，第二轮让它检查并去除重复或不相关的项。这样得到的描述符既多样又精准。

2. **属性匹配（Feature Checking）**  
   - 对于一张待分类的图像，使用 CLIP 的视觉编码器得到图像向量。  
   - 同时，把每个属性文本送进 CLIP 的文本编码器，得到属性向量。  
   - 计算图像向量与每个属性向量的余弦相似度，得到一组分数，代表图像中是否出现该属性。可以把这一步想象成“在图像里找关键词”，类似于搜索引擎对文档进行关键词匹配。

3. **决策聚合（Classification by Description）**  
   - 对同一类别的所有属性分数做加权求和，得到该类别的整体匹配分。权重可以是属性的重要性（由 LLM 生成时提供）或统一设为 1。  
   - 把所有类别的整体分数进行归一化，选出最高分的类别作为最终预测。  
   - 为了让模型更倾向于使用特定属性，作者在训练阶段加入了一个小的正则项，鼓励高分属性对应的文本向量与图像向量保持一致。

**最巧妙的点**在于把语言模型的生成能力和视觉模型的相似度匹配自然耦合起来。属性本身是可编辑的，这让人可以像调节开关一样增删属性，从而直接影响模型的判断，而不需要重新训练整个视觉模型。

### 实验与效果
- **数据集与任务**：主要在 ImageNet 的零样本分类上评估，同时覆盖了 ImageNet‑A、ImageNet‑R、ImageNet‑V2 等分布漂移基准，另外还在少数未见概念（如新动物种类）上做了概念迁移实验。  
- **对比基线**：与原始 CLIP 的标签匹配零样本、CLIP+手工 prompt、以及最近的可解释零样本方法（如 CoOp、CoCoOp）进行比较。  
- **结果概述**：论文声称在标准 ImageNet 上的 top‑1 准确率比纯标签匹配提升了约 1% 左右，在 ImageNet‑A、ImageNet‑R 等漂移数据集上提升更为显著，尤其在属性稀疏的类别上优势明显。对未见概念的识别准确率也有可观提升。  
- **消融实验**：作者分别去掉属性生成、属性加权、正则项等模块，发现属性生成是性能提升的主要驱动力，正则项对鲁棒性贡献最大。  
- **局限性**：属性质量依赖于 LLM 的提示设计，若提示不当会产生噪声属性；此外，属性匹配需要对每个属性都做一次文本编码，计算成本随属性数量线性增长。论文也提到在极端细粒度分类（如鸟类亚种）时仍然受限于属性的表达能力。

### 影响与延伸思考
这篇工作打开了“语言驱动的属性层级零样本分类”的大门，后续有不少研究尝试把更复杂的属性结构（如层级树、关系图）引入 VLM，或使用更高效的检索机制降低计算开销。还有工作把人类交互式编辑属性的想法扩展到实时系统，让用户可以现场调节模型偏好。想进一步了解，可以关注 **属性生成的提示工程**、**跨模态检索加速** 以及 **可编辑的视觉模型** 这几个方向，都是当前的热点。

### 一句话记住它
把大语言模型生成的细粒度属性当作“解释性提示”，让视觉模型在找特征而不是直接找标签，从而实现更可解释、更可调的零样本分类。