# Model Cards for Model Reporting

> **Date**：2018-10-05
> **arXiv**：https://arxiv.org/abs/1810.03993

## Abstract

Trained machine learning models are increasingly used to perform high-impact tasks in areas such as law enforcement, medicine, education, and employment. In order to clarify the intended use cases of machine learning models and minimize their usage in contexts for which they are not well suited, we recommend that released models be accompanied by documentation detailing their performance characteristics. In this paper, we propose a framework that we call model cards, to encourage such transparent model reporting. Model cards are short documents accompanying trained machine learning models that provide benchmarked evaluation in a variety of conditions, such as across different cultural, demographic, or phenotypic groups (e.g., race, geographic location, sex, Fitzpatrick skin type) and intersectional groups (e.g., age and race, or sex and Fitzpatrick skin type) that are relevant to the intended application domains. Model cards also disclose the context in which models are intended to be used, details of the performance evaluation procedures, and other relevant information. While we focus primarily on human-centered machine learning models in the application fields of computer vision and natural language processing, this framework can be used to document any trained machine learning model. To solidify the concept, we provide cards for two supervised models: One trained to detect smiling faces in images, and one trained to detect toxic comments in text. We propose model cards as a step towards the responsible democratization of machine learning and related AI technology, increasing transparency into how well AI technology works. We hope this work encourages those releasing trained machine learning models to accompany model releases with similar detailed evaluation numbers and other relevant documentation.

---

# 模型卡片：模型报告指南 论文详细解读

### 背景：这个问题为什么难？
机器学习模型已经渗透到警务、医疗、教育、招聘等高风险场景。过去，研究者往往只在论文里给出整体的准确率或损失，却很少说明模型在不同人群、不同地区、不同光照等细分情境下的表现。于是模型被直接部署，结果在某些子群体上出现偏差、误判甚至危害安全。缺少系统化、可比的性能报告，使得使用者难以判断模型是否适合自己的业务，也让监管机构难以追责。

### 关键概念速览
- **模型卡片（Model Card）**：随模型一起发布的简短文档，像产品说明书一样列出模型的性能、适用范围和使用限制。  
- **基准评估（Benchmark Evaluation）**：在公开的、标准化的数据集上跑实验，得到可复现的分数，类似于汽车的油耗测试。  
- **人口统计切片（Demographic Slice）**：把测试数据按种族、性别、年龄等属性拆分，分别计算指标，帮助发现“某类人群上表现差”。  
- **交叉切片（Intersectional Slice）**：进一步组合两个或多个属性（如“老年黑人女性”），捕捉更细粒度的偏差。  
- **预期使用场景（Intended Use）**：模型作者声明模型设计的具体任务和业务边界，类似于药品说明书的适应症。  
- **透明度（Transparency）**：公开模型训练细节、评估方法和局限，让使用者能够“看见”模型的真实能力。  
- **负责任的民主化（Responsible Democratization）**：在让更多人使用 AI 的同时，确保技术不会被误用或产生不公平后果。  

### 核心创新点
1. **从单一指标到多维切片报告**  
   - 之前：大多数模型只报告整体准确率或 F1。  
   - 本文：在模型卡片中系统列出在不同人口统计和交叉切片上的指标。  
   - 改变：使用者可以快速判断模型在自己关注的子群体上是否可靠，避免“一刀切”部署。

2. **明确声明模型的适用范围**  
   - 之前：模型发布时很少提及“只能用于 X 场景”。  
   - 本文：卡片专门设立“预期使用”章节，说明模型适合的任务、数据来源和不推荐的情境。  
   - 改变：降低模型被误用于高风险领域的概率，提供法律合规的参考依据。

3. **提供统一模板促进行业标准化**  
   - 之前：每个团队自行撰写“模型说明”，格式千差万别。  
   - 本文：提出一套结构化的卡片模板（模型概述、数据、评估方法、切片结果、局限等），便于不同组织之间的对比。  
   - 改变：推动社区形成统一的报告规范，后续工具可以自动生成或校验卡片。

4. **通过真实案例展示可操作性**  
   - 之前：概念层面的讨论居多，缺少落地示例。  
   - 本文：分别为“笑脸检测模型”和“有害评论检测模型”制作完整卡片，展示如何收集切片、报告指标并解释局限。  
   - 改变：为后续研究者提供可复制的实践路径，降低采用门槛。

### 方法详解
整体思路可以概括为三步：**（1）定义报告结构 → （2）收集多维评估数据 → （3）填充卡片并公开**。

1. **定义报告结构**  
   作者先列出卡片的必备章节：模型概述（任务、架构、训练数据概况）、预期使用、评估方法、整体性能、切片性能、局限与风险、伦理与公平性说明。这个结构像一本“产品手册”，每一页都有固定的标题和要点，确保信息不遗漏。

2. **收集多维评估数据**  
   - **基准测试**：在公开的标准数据集上跑完整训练‑测试流程，得到整体指标（如准确率、召回率）。  
   - **切片划分**：根据数据集提供的元信息（种族、性别、年龄、皮肤类型等），把测试集拆成若干子集。若元信息不足，作者会自行标注或使用外部标签。  
   - **交叉切片**：对两个属性做笛卡尔积，形成更细的子集，例如“女性+深色皮肤”。  
   - **指标计算**：对每个子集重复基准测试，记录同样的指标。这样得到一张矩阵，行是属性组合，列是指标。

3. **填充卡片并公开**  
   - 将整体指标和切片矩阵填入对应章节。  
   - 在“局限”章节解释哪些切片表现不佳、可能的原因（数据不平衡、标签噪声等），并给出使用建议（如“仅在光照良好的室内环境使用”。）  
   - 最后把卡片以 PDF、Markdown 或 JSON 等机器可读格式随模型代码一起发布，方便开发者下载查看。

**巧妙之处**：作者没有把卡片当成“附录”，而是把切片评估作为核心内容。这样一来，卡片本身就能直接回答“模型在我关心的群体上表现如何？”而不是让使用者去自行切片再跑实验。

### 实验与效果
- **实验对象**：两种监督学习模型——（1）基于卷积神经网络的笑脸检测模型，使用公开的人脸数据集；（2）基于文本分类的有害评论检测模型，使用公开的在线评论数据。  
- **评估维度**：分别在性别、种族、年龄、皮肤类型以及它们的交叉组合上报告准确率、召回率等指标。  
- **对比基线**：原文没有提供与其他报告方式的数值对比，只是展示了卡片能够揭示的性能差异。例如，笑脸模型在深色皮肤女性子集上的召回率比整体低约 10%。  
- **消融实验**：论文未给出专门的消融实验，因为核心贡献是报告框架本身，而非模型改进。  
- **局限性**：作者承认卡片质量依赖于标注的完整性和切片的选取；如果训练数据本身缺少某些属性，卡片只能报告“缺失”。此外，卡片本身不解决模型偏差，只是让问题可见。

### 影响与延伸思考
自从这篇论文提出模型卡片后，业界和学术界都开始把“模型报告”当作交付物的必备环节。后续出现了 **Datasheets for Datasets**（数据集说明书）和 **FactSheets**（服务化模型的合规报告），形成了“模型‑数据‑服务”三位一体的透明化生态。大公司（Google、Microsoft、IBM）在模型库（如 TensorFlow Hub、Model Zoo）里直接附带模型卡片，开源社区也推出了自动生成卡片的工具（如 `model-card-toolkit`）。如果想进一步了解，可以关注 **AI Model Governance**、**公平性评估平台**（如 IBM AI Fairness 360）以及 **监管政策**（欧盟 AI 法案）对模型文档的要求——这些都是模型卡片的自然延伸。

### 一句话记住它
模型卡片把模型的“性能报告”做成结构化的说明书，让每个子群体的表现一目了然，从而帮助大家负责任地使用 AI。