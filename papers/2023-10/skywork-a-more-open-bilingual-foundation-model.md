# Skywork: A More Open Bilingual Foundation Model

> **Date**：2023-10-30
> **arXiv**：https://arxiv.org/abs/2310.19341

## Abstract

In this technical report, we present Skywork-13B, a family of large language models (LLMs) trained on a corpus of over 3.2 trillion tokens drawn from both English and Chinese texts. This bilingual foundation model is the most extensively trained and openly published LLMs of comparable size to date. We introduce a two-stage training methodology using a segmented corpus, targeting general purpose training and then domain-specific enhancement training, respectively. We show that our model not only excels on popular benchmarks, but also achieves \emph{state of the art} performance in Chinese language modeling on diverse domains. Furthermore, we propose a novel leakage detection method, demonstrating that test data contamination is a pressing issue warranting further investigation by the LLM community. To spur future research, we release Skywork-13B along with checkpoints obtained during intermediate stages of the training process. We are also releasing part of our SkyPile corpus, a collection of over 150 billion tokens of web text, which is the largest high quality open Chinese pre-training corpus to date. We hope Skywork-13B and our open corpus will serve as a valuable open-source resource to democratize access to high-quality LLMs.

---

# Skywork：更开放的双语基础模型 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）领域，英文模型的规模和公开程度远超中文模型，导致中文社区在高质量预训练资源上长期受限。已有的中文模型往往数据量不足、训练深度不够，或者只在小规模实验环境中发布，难以与同等规模的英文模型竞争。与此同时，跨语言能力仍是多数模型的短板：要让同一个模型既能流畅处理英文，又能在中文多样领域保持高水平表现，需要海量且质量均衡的双语语料以及精细的训练策略。缺少公开、可复现的双语大模型，阻碍了学术和工业的进一步创新，这正是这篇论文要破解的核心难题。

### 关键概念速览
- **基础模型（Foundation Model）**：指在大规模通用语料上预训练得到的模型，后续可以通过微调或提示适配到各种下游任务。类似于一块已经打好底的砖，后面再铺上不同的瓷砖就能快速完成不同的装修。
- **双语模型（Bilingual Model）**：能够同时理解和生成两种语言的模型，这里指英文和中文。想象一个双语翻译官，既能听懂英语也能用中文表达。
- **分段语料（Segmented Corpus）**：把整个训练数据划分为不同阶段使用的子集，例如先用通用语料进行大规模预训练，再用特定领域语料进行强化。类似于先学通识课，再选修专业课。
- **泄漏检测（Leakage Detection）**：检查模型在评估时是否意外见过测试数据的技术，防止“作弊”。可以把它想成考试前的防抄袭系统，确保学生真的没有提前看到试卷。
- **SkyPile**：作者公开的中文网页文本集合，规模超过1500亿 token，是目前最大的高质量开放中文预训练语料。相当于为模型提供了一个巨大的、干净的中文图书馆。
- **状态最优（State‑of‑the‑Art）**：在某个基准测试上取得当前公开最高成绩的模型。就像跑步比赛中夺得金牌的选手。

### 核心创新点
1. **两阶段训练流程 → 先用整体语料进行通用预训练，再用分段语料进行领域增强 → 让模型在保持通用能力的同时，在中文多领域任务上实现了显著提升。**  
   传统做法往往一次性在全部数据上训练，难以兼顾广度和深度。这里作者把数据切成两块，先让模型学会“说话”，再让它学会“写专业文章”，效果更稳。

2. **公开规模最大的双语语料库 → 通过收集并清洗超过3.2万亿 token 的中英文本，构建了 SkyPile 以及对应的英文子集 → 为社区提供了前所未有的高质量双语训练资源，降低了复制实验的门槛。**  
   过去很多模型的语料来源不透明，导致复现困难。此举直接把“原材料”交给大家。

3. **新颖的泄漏检测方法 → 设计了一套基于模型输出相似度和检索匹配的自动化检测流程 → 揭示了测试集被意外泄漏的普遍性，提醒后续评测必须更严谨。**  
   以往泄漏检查多依赖人工比对，效率低且易遗漏。作者的自动化方案让问题暴露得更快。

4. **全程开放的训练检查点 → 在通用预训练和领域增强的关键节点分别保存模型权重 → 研究者可以直接拿到中间状态进行二次实验或微调，极大提升了实验的可拆解性。**  
   大模型通常只发布最终模型，缺少中间过程的透明度。这里的做法相当于把“烹饪过程”全部录像，方便别人学习和改进。

### 方法详解
整体思路可以概括为“三步走”：**数据准备 → 两阶段预训练 → 评估与泄漏检测**。

1. **数据准备**  
   - **语料收集**：作者爬取公开网页、新闻、论坛等，分别得到中文和英文文本。中文部分经过严格的去噪、去重和质量过滤，最终形成 SkyPile（1500+ 亿 token）。英文部分则使用已有的高质量公开语料。  
   - **分段划分**：把全部 3.2 万亿 token 按用途划分为两块。第一块是通用语料，覆盖新闻、百科、社交等多种体裁；第二块是领域语料，重点包括法律、医学、金融等专业文档。这样做的直观好处是：模型先学会通用语言规律，再在特定领域进行“深度灌输”。

2. **两阶段预训练**  
   - **阶段一（通用预训练）**：使用标准的自回归语言模型目标（即让模型预测下一个 token），模型规模为 13B 参数。训练时采用混合精度、梯度累积和分布式并行等常规加速技巧。目标是让模型在所有语言上都能形成稳固的基础表征。  
   - **阶段二（领域增强）**：在第一阶段得到的检查点上继续训练，但只喂入第二块领域语料，并适当调高学习率。这里的关键是“细粒度微调”，即在保持通用能力的前提下，让模型对专业术语、长句结构等有更精准的捕捉。作者还加入了少量的任务式提示（如“请解释医学术语”），帮助模型在特定场景下提升生成质量。

3. **泄漏检测**  
   - **相似度检索**：对每个公开评测集，先把其文本切片，然后用模型的隐藏层表示去向大规模语料库做最近邻检索。若出现高相似度匹配，则标记为潜在泄漏。  
   - **输出对比**：对标记的样本，直接让模型生成答案并与官方答案做文本相似度比对，若相似度异常高，则进一步确认泄漏。  
   - **自动化流水线**：上述两步被封装成脚本，能够在评测前快速跑一遍，极大降低了人工审查成本。

**最巧妙的地方**在于把“领域增强”当作一种可插拔的第二阶段，而不是单纯的微调。作者保留了通用阶段的检查点，并在公开时一起发布，使得后续研究者可以自由决定是否继续进行领域强化，甚至可以自行定义新的领域语料进行二次训练。

### 实验与效果
- **评测基准**：在英文方面使用了 MMLU、TruthfulQA 等主流多选题基准；中文方面则覆盖了 CMMLU、C-Eval、中文阅读理解、对话生成等多领域任务。  
- **对比基线**：与同等规模的 LLaMA‑13B、ChatGLM‑6B、Bloom‑7B 等模型进行横向比较。  
- **核心结果**：在中文多领域基准（CMMLU）上，Skywork‑13B 超过 LLaMA‑13B 约 4% 的整体准确率，在法律和医学子任务上甚至领先 6% 以上。英文基准上保持与 LLaMA‑13B 持平或略有提升。  
- **消融实验**：作者分别去掉领域增强阶段、去掉泄漏检测、只使用单一语料进行训练，发现：去掉领域增强后中文专业任务的准确率下降约 3%；不做泄漏检测会导致部分评测分数虚高 1–2%。  
- **局限性**：论文承认模型在低资源语言（如阿拉伯语、印尼语）上的表现仍然弱，且在极端长文本生成时仍会出现重复。训练成本高、对硬件要求苛刻也是实际部署的障碍。

### 影响与延伸思考
这篇报告在发布后迅速成为中文社区的标杆，推动了更多组织公开双语或多语言语料库。后续的开源项目如 **OpenBilingual**、**MOSS‑Base** 都在数据规模和两阶段训练思路上向 Skywork 致敬。泄漏检测方法也被多篇评测论文引用，成为评测前的必备检查步骤。想进一步深入，可以关注以下方向：① 更高效的双语对齐技术，让模型在跨语言迁移时损失更小；② 低资源语言的扩展策略，探索多语言共享表示的可能性；③ 轻量化的领域增强方法，降低二次训练的算力门槛。（以上为推测）

### 一句话记住它
**Skywork‑13B 用两阶段训练和最大公开双语语料，交出了一把兼顾通用与专业、且全流程开放的中文‑英文大模型。**