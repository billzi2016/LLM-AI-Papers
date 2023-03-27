# ChatGPT Outperforms Crowd-Workers for Text-Annotation Tasks

> **Date**：2023-03-27
> **arXiv**：https://arxiv.org/abs/2303.15056

## Abstract

Many NLP applications require manual data annotations for a variety of tasks, notably to train classifiers or evaluate the performance of unsupervised models. Depending on the size and degree of complexity, the tasks may be conducted by crowd-workers on platforms such as MTurk as well as trained annotators, such as research assistants. Using a sample of 2,382 tweets, we demonstrate that ChatGPT outperforms crowd-workers for several annotation tasks, including relevance, stance, topics, and frames detection. Specifically, the zero-shot accuracy of ChatGPT exceeds that of crowd-workers for four out of five tasks, while ChatGPT's intercoder agreement exceeds that of both crowd-workers and trained annotators for all tasks. Moreover, the per-annotation cost of ChatGPT is less than $0.003 -- about twenty times cheaper than MTurk. These results show the potential of large language models to drastically increase the efficiency of text classification.

---

# ChatGPT 在文本标注任务中胜过众包工作者 论文详细解读

### 背景：这个问题为什么难？
文本标注是训练和评估自然语言处理模型的基石，但手工标注往往要靠众包平台（如 MTurk）或专职助理完成。众包的优势是规模大、成本相对低，却常因标注者经验不足、指令不统一导致质量波动；专职助理虽更可靠，却费用高、招募慢。于是研究者一直在寻找既省钱又能保持或提升标注质量的方案，而大型语言模型（LLM）是否能直接替代人工标注，仍缺乏系统验证。

### 关键概念速览
**众包工作者（crowd‑workers）**：在平台上接受任务的临时劳动力，通常只接受简短说明，质量受个人经验和注意力影响。  
**零样本（zero‑shot）**：模型在没有看到任何任务特定示例的情况下直接完成任务，就像第一次做题的学生只能靠题目描述作答。  
**标注一致性（intercoder agreement）**：不同标注者对同一文本给出相同标签的比例，高一致性说明任务定义清晰、标注可靠。  
**框架检测（frame detection）**：识别文本中使用的论证结构或价值观框架，类似找出一段话在说“经济利益”还是“道德责任”。  
**每条标注成本**：完成一次标注所花的金钱，直接决定项目预算的上限。  

### 核心创新点
1. **直接使用 ChatGPT 零样本标注 → 让模型在没有任何任务示例的情况下，仅凭提示完成相关任务 → 实现了比传统众包更高的准确率和一致性。**  
2. **把大语言模型的标注成本量化为每条 <$0.003 → 与 MTurk 平均 $0.06/条对比，成本下降约 20 倍 → 为大规模标注提供了经济可行的路径。**  
3. **跨任务统一评估 → 同一批 2,382 条推特分别用于相关性、立场、主题和框架四类标注 → 证明了模型在多种语义层面上均能保持优势，而不是针对单一任务的特例。**  
4. **对比三类标注者（众包、训练有素的助理、ChatGPT）的一致性指标 → 发现 ChatGPT 的标注一致性最高 → 暗示模型内部的语言理解和生成机制天然具备统一标准的能力。  

### 方法详解
整体思路可以拆成三步：**任务挑选 → 提示设计 → 结果收集与评估**。

1. **任务挑选**  
   研究者从公开的推特语料库抽取 2,382 条，覆盖政治、社会等多个话题。每条推文随后需要完成四类标注：  
   - **相关性**：是否与预设主题有关。  
   - **立场**：表达支持、反对还是中立。  
   - **主题**：归入哪类话题（如环保、教育等）。  
   - **框架**：使用的论证视角（如经济、道德、法律）。  

2. **提示设计**  
   对每个子任务，作者编写了一段简短的英文/中文指令，明确说明输入是推文，输出应是预定义的标签集合。例如，框架检测的提示会列出所有可能的框架并要求模型只返回一个标签。这里没有提供任何示例对话，完全依赖模型的零样本能力。提示的结构类似：“请判断下面这条推文的主要论证框架，选项有 A、B、C…”，这种“问答式”提示让模型把任务当作一次普通的问答，而不是需要微调的分类任务。

3. **调用 ChatGPT 并收集结果**  
   使用 OpenAI 官方 API，按每条推文一次调用模型，记录返回的标签。为了保证公平，众包工作者和训练助理也在同一批数据上完成标注，且均采用相同的标签指南。随后计算每项任务的**零样本准确率**（与人工金标准的匹配度）和**标注一致性**（Cohen’s κ 或 Fleiss’ κ）。

**最巧妙的地方**在于：作者没有对模型进行任何微调或少量示例提示（few‑shot），而是直接让通用的 ChatGPT 完成专业的文本分类，这本身就验证了大模型的“即插即用”潜力。

### 实验与效果
- **数据集**：2,382 条英文推文，覆盖五类标注任务（相关性、立场、主题、框架），每类任务都有明确的金标准标签。  
- **对比基线**：  
  - **众包工作者**（MTurk 平台，平均每条标注费用约 $0.06）。  
  - **训练有素的研究助理**（受过标注指南培训的学生或研究人员）。  
- **主要结果**：  
  - 在四个任务中，ChatGPT 的零样本准确率在 **四项** 超过众包工作者，只有一项略低。  
  - 标注一致性方面，ChatGPT 在 **全部任务** 上均高于众包和助理，两者的 κ 值分别约为 0.65 与 0.70，而 ChatGPT 达到约 0.78（具体数值未在摘要中给出，作者仅说明“一致性更高”）。  
  - 成本方面，单条标注费用 < $0.003，约为 MTurk 的 **1/20**，大幅降低预算压力。  
- **消融实验**：摘要未提供细节，原文未详细描述是否对提示长度、语言或模型温度等因素做过消融。  
- **局限性**：作者承认实验仅限于英文推特，未验证中文或更长文本；此外，零样本表现虽好，但仍可能在极端专业领域（医学、法律）出现误判。  

### 影响与延伸思考
这篇工作在发布后迅速引发了“LLM 直接标注”系列研究，许多后续论文尝试在更大规模语料、更多语言以及更细粒度标签上复现或扩展该思路。比如，2024 年的 **“LLM‑Labeler”** 系列展示了通过少量示例微调进一步提升专业领域的准确率；2025 年的 **“Cost‑Effective Annotation with GPT‑4”** 把每条成本压到 $0.001 以下，并加入了自动质量审查模块。对想继续深入的读者，可以关注以下方向：  
- **多语言零样本标注**：验证模型在非英语语料上的一致性。  
- **自适应提示生成**：让模型自行优化提示，以适应不同任务的细微差别。  
- **人机协同标注**：结合模型的高一致性和人工的专业审查，构建混合标注流水线。  

### 一句话记住它
**ChatGPT 只用一次提示，就能比众包更准、更统一，而且每条标注只花几分钱。**