# Evaluating the Logical Reasoning Ability of ChatGPT and GPT-4

> **Date**：2023-04-07
> **arXiv**：https://arxiv.org/abs/2304.03439

## Abstract

Harnessing logical reasoning ability is a comprehensive natural language understanding endeavor. With the release of Generative Pretrained Transformer 4 (GPT-4), highlighted as "advanced" at reasoning tasks, we are eager to learn the GPT-4 performance on various logical reasoning tasks. This report analyses multiple logical reasoning datasets, with popular benchmarks like LogiQA and ReClor, and newly-released datasets like AR-LSAT. We test the multi-choice reading comprehension and natural language inference tasks with benchmarks requiring logical reasoning. We further construct a logical reasoning out-of-distribution dataset to investigate the robustness of ChatGPT and GPT-4. We also make a performance comparison between ChatGPT and GPT-4. Experiment results show that ChatGPT performs significantly better than the RoBERTa fine-tuning method on most logical reasoning benchmarks. With early access to the GPT-4 API we are able to conduct intense experiments on the GPT-4 model. The results show GPT-4 yields even higher performance on most logical reasoning datasets. Among benchmarks, ChatGPT and GPT-4 do relatively well on well-known datasets like LogiQA and ReClor. However, the performance drops significantly when handling newly released and out-of-distribution datasets. Logical reasoning remains challenging for ChatGPT and GPT-4, especially on out-of-distribution and natural language inference datasets. We release the prompt-style logical reasoning datasets as a benchmark suite and name it LogiEval.

---

# 评估ChatGPT和GPT-4的逻辑推理能力 论文详细解读

### 背景：这个问题为什么难？
逻辑推理要求模型在阅读理解时能够捕捉前提之间的隐含关系，而不是仅靠词频或常识匹配。过去的模型大多在事实型问答上表现不错，却在需要严密推理的多选题和自然语言蕴含（NLI）任务上频频失误。尤其是当测试数据与训练语料分布不一致时，模型往往会“猜”答案而不是演绎出正确的结论。于是，评估最新的大语言模型（LLM）在真实逻辑推理场景下的表现，成为检验它们通用理解能力的关键。

### 关键概念速览
- **多选阅读理解（Multiple‑choice RC）**：给出一段文章和若干备选答案，模型要选出唯一正确的选项，类似考试中的阅读理解题。  
- **自然语言蕴含（Natural Language Inference, NLI）**：判断两句话之间是“蕴含”“矛盾”还是“中性”，考察模型对句子逻辑关系的把握。  
- **Out‑of‑Distribution（OOD）数据**：指分布与模型训练时看到的文本不同的测试集，用来检验模型的鲁棒性。  
- **Prompt‑style 数据集**：把题目和选项包装成自然语言提示（prompt），让模型直接在对话框里回答，而不是做额外的微调。  
- **LogiQA / ReClor**：两个公开的逻辑推理基准，题目来源于真实考试，难度较高。  
- **AR‑LSAT**：新发布的逻辑推理数据集，模仿美国法学院入学考试（LSAT）中的推理题。  
- **LogiEval**：作者公开的完整评测套件，集合了上述所有数据并提供统一的 Prompt 格式。  

### 核心创新点
1. **从微调转向 Prompt 评测**：传统做法会把模型微调到特定数据集上，然后报告准确率。这里直接使用 ChatGPT 与 GPT‑4 的原始 API，给出统一的提示词，让模型在“零样本”条件下作答。这样可以更公平地比较不同模型的原始推理能力，而不被微调技巧掩盖。  
2. **构建 OOD 逻辑推理集**：作者自行设计了一批与现有基准风格截然不同的题目，专门用来探测模型在分布漂移时的表现下降幅度。相比仅在公开基准上打分，这一步揭示了模型的真实鲁棒性。  
3. **系统化对比 ChatGPT 与 GPT‑4**：利用早期获取的 GPT‑4 API，分别在同一套 Prompt 上跑两者，得到直接的性能差距。此前大多数工作只报告单一模型的成绩，缺少横向对比。  
4. **发布 LogiEval 基准套件**：把所有 Prompt‑style 数据整理成统一仓库，方便后续研究者复现和扩展评测。  

### 方法详解
整体思路可以概括为三步：**准备数据 → 设计 Prompt → 调用模型 → 统计结果**。  
1. **数据准备**：作者挑选了四类公开基准（LogiQA、ReClor、AR‑LSAT、NLI 数据）以及自行构造的 OOD 题库。每道题都被转化为“问题 + 选项 A/B/C/D”的文本形式，确保所有模型看到的输入完全一致。  
2. **Prompt 设计**：为了让模型理解任务，提示词采用了类似“下面是一段阅读材料，请根据材料选择最合适的答案”。在 NLI 场景下，提示词会明确要求判断“蕴含、矛盾或中性”。这种结构类似老师在课堂上给学生的指令，避免模型产生歧义。  
3. **模型调用**：使用官方提供的 ChatGPT（基于 GPT‑3.5）和 GPT‑4 API，分别对每个 Prompt 发送请求。返回的文本被解析为选项字母，然后与金标准答案比对。  
4. **结果统计**：计算每个数据集的准确率，并对比两模型的差异。为了评估鲁棒性，还统计了在 OOD 数据上的跌幅。  
**巧妙之处**在于完全不做任何微调或后处理，所有推理过程都交给模型本身完成，这样的“黑箱”评测更能反映真实使用场景下的能力。  

### 实验与效果
- **测试数据**：LogiQA、ReClor、AR‑LSAT、多个 NLI 基准以及作者自建的 OOD 题库。  
- **基线对比**：主要与 RoBERTa（经过任务微调）的成绩作比较。论文指出，ChatGPT 在大多数逻辑推理基准上显著超越 RoBERTa，尤其在 LogiQA 与 ReClor 上提升了数个百分点。  
- **ChatGPT vs GPT‑4**：在同一套 Prompt 上，GPT‑4 的准确率整体高于 ChatGPT，尤其在 AR‑LSAT 和 NLI 数据上优势更明显。  
- **OOD 结果**：两模型在自建的分布外数据上准确率均出现明显下降，说明当前的大语言模型仍对训练分布敏感。  
- **消融实验**：论文未提供细粒度的消融分析，因为方法本身只有 Prompt 设计这一可变因素。作者仅说明不同提示词的微调（如是否加入“请思考一步再回答”）对结果影响不大。  
- **局限性**：评测仅限于 API 调用的零样本场景，未探索微调或检索增强的潜力；此外，Prompt 的语言风格可能对不同模型产生不均等的影响。  

### 影响与延伸思考
这篇工作在社区里引发了对“大模型原始推理能力”的关注，促使后续研究者在评测时更倾向于使用统一 Prompt 而非微调后模型。随后出现的 **LogiEval** 基准被多篇论文引用，用来检验新模型的逻辑推理鲁棒性。推测，未来会有更多围绕 **Prompt Engineering**（提示工程）和 **自检验（self‑verification）** 的方法出现，以弥补模型在 OOD 场景下的短板。想进一步了解，可以关注 **Chain‑of‑Thought**（思维链）在逻辑推理中的应用，以及 **Retrieval‑augmented Generation**（检索增强生成）如何帮助模型获取外部知识来提升推理准确率。  

### 一句话记住它
GPT‑4 在零样本 Prompt 下比 ChatGPT 更擅长逻辑推理，但面对分布外题目，两者仍会显著掉分，说明逻辑推理仍是大模型的软肋。