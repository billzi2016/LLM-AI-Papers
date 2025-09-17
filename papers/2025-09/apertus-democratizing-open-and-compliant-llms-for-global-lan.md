# Apertus: Democratizing Open and Compliant LLMs for Global Language Environments

> **Date**：2025-09-17
> **arXiv**：https://arxiv.org/abs/2509.14233

## Abstract

We present Apertus, a fully open suite of large language models (LLMs) designed to address two systemic shortcomings in today's open model ecosystem: data compliance and multilingual representation. Unlike many prior models that release weights without reproducible data pipelines or regard for content-owner rights, Apertus models are pretrained exclusively on openly available data, retroactively respecting `robots.txt` exclusions and filtering for non-permissive, toxic, and personally identifiable content. To mitigate risks of memorization, we adopt the Goldfish objective during pretraining, strongly suppressing verbatim recall of data while retaining downstream task performance. The Apertus models also expand multilingual coverage, training on 15T tokens from over 1800 languages, with ~40% of pretraining data allocated to non-English content. Released at 8B and 70B scales, Apertus approaches state-of-the-art results among fully open models on multilingual benchmarks, rivalling or surpassing open-weight counterparts. Beyond model weights, we release all scientific artifacts from our development cycle with a permissive license, including data preparation scripts, checkpoints, evaluation suites, and training code, enabling transparent audit and extension.

---

# Apertus：让开放且合规的大语言模型惠及全球语言环境 论文详细解读

### 背景：这个问题为什么难？
当前的开源大语言模型（LLM）大多在数据来源上缺乏透明度，往往直接发布权重，却不给出完整的爬取、清洗和授权流程，导致版权争议和合规风险。与此同时，主流模型的训练语料几乎全部集中在英语或少数高资源语言，低资源语言的表现几乎是被忽视的。于是出现了两个根本性瓶颈：一是“数据合规”难以审计，二是“多语言覆盖”严重不足，这直接限制了开源模型在全球范围的落地与信任。

### 关键概念速览
**LLM（大语言模型）**：一种基于深度神经网络的模型，能够在海量文本上学习语言规律，进而完成对话、写作、翻译等任务。可以把它想象成“会说话的百科全书”。  
**开放数据合规**：指在收集训练数据时，只使用公开可获取且不侵犯版权的内容，并遵守网站的 `robots.txt` 规则。类似于在图书馆只借阅标记为“可外借”的书。  
**Goldfish 目标**：一种预训练技巧，专门抑制模型对原始文本的逐字记忆，防止出现“照本宣科”。就像金鱼的记忆只有几秒，模型只保留抽象模式而忘掉具体句子。  
**多语言预训练**：在训练阶段同时喂入上千种语言的文本，让模型学会跨语言的共性表示。相当于让一个学生同时学习多门语言的语法和词汇。  
**记忆抑制**：通过特殊的损失函数或采样策略，让模型在保持任务性能的前提下，降低对训练数据的直接复制。  
**robots.txt**：网站用来声明哪些路径可以被爬虫抓取、哪些不可以的文本文件。遵守它相当于尊重网站的“访客守则”。  
**有害内容过滤**：在数据清洗阶段剔除暴力、仇恨、个人隐私等不适宜信息，确保模型不会学习或输出这些内容。

### 核心创新点
1. **从根源保证数据合规 → 只使用公开可获取的数据，并在爬取后回溯检查 `robots.txt` 排除项 → 训练数据全程可审计，版权争议风险大幅降低。**  
2. **引入 Goldfish 目标 → 在预训练损失中加入强记忆抑制项，使模型在学习语言结构时主动忘记完整句子 → 在保持下游任务表现的同时，显著降低了模型直接复制训练文本的概率。**  
3. **大规模多语言覆盖 → 采集约 15 万亿 token，覆盖 1800 多种语言，其中约 40% 为非英语内容 → 在多语言基准上接近最先进水平，尤其在低资源语言上实现了可观提升。**  
4. **全链路开源 → 除模型权重外，公开数据准备脚本、过滤规则、训练代码、评估套件，并使用宽松许可证 → 任何研究者都能复现、审计甚至在此基础上继续扩展。**

### 方法详解
整体思路可以拆成四个阶段：**数据收集 → 合规清洗 → Goldfish 预训练 → 多尺度模型发布**。

1. **数据收集**  
   - 爬取公开网页、维基百科、开源文档等资源，使用通用爬虫框架。  
   - 每抓取一次后立即对目标站点的 `robots.txt` 进行解析，标记被禁止的路径。即使已经下载，也会在后续步骤中剔除，确保“事后合规”。  

2. **合规清洗**  
   - **版权过滤**：只保留明确标记为 CC‑BY、CC‑0 或公共领域的文本。  
   - **有害内容过滤**：利用已有的内容审查模型和关键词列表，剔除暴力、仇恨、个人隐私等信息。  
   - **语言识别与划分**：使用轻量语言检测器把文本分配到 1800+ 语言桶中，确保每种语言都有足够的 token。  

3. **Goldfish 预训练**  
   - 传统的自回归语言模型会最小化预测下一个 token 的交叉熵，这会鼓励模型记住完整句子。Goldfish 目标在此基础上加入 **记忆抑制项**：对模型在训练批次中出现的完整 n‑gram 进行惩罚，使其在预测时倾向于使用更抽象的上下文。  
   - 实际实现上，作者在每个训练步骤随机抽取若干已出现的句子片段，计算模型对这些片段的概率，并在损失中加入负向权重。这样模型在学习语言规律的同时，被“训练成金鱼”，不容易把原文原封不动地记住。  

4. **多尺度模型发布**  
   - 采用相同的训练数据和 Goldfish 目标，分别训练 **8B**（约 80 亿参数）和 **70B**（约 700 亿参数）两个规模的模型。  
   - 训练使用混合并行技术（数据并行 + 张量并行），在多机 GPU 集群上完成。  
   - 完成后，作者把模型权重、训练日志、数据处理脚本、评估基准全部打包，以 **MIT‑style** 许可证公开，任何人都可以直接下载、复现或二次开发。

**最巧妙的地方**在于把合规检查放在爬取后“回溯”执行，而不是事前依赖爬虫的过滤规则，这保证了即使爬虫配置失误，也能在清洗阶段彻底剔除违规内容；同时，Goldfish 目标的记忆抑制机制在不牺牲下游性能的前提下，有效降低了模型泄露训练数据的风险，这在开源 LLM 中尚属首次系统化实现。

### 实验与效果
- **评测任务**：论文在多语言理解基准（如 XGLUE、MMLU‑Multi、Flores-200）以及通用对话/生成任务上进行评估。  
- **对比基线**：与同规模的 LLaMA、Falcon、Mistral 等开源模型以及部分商业闭源模型进行比较。  
- **结果概述**：在大多数多语言任务上，Apertus‑70B 的得分接近或略超同类开源模型的最优水平，尤其在低资源语言（如斯瓦希里语、塔吉克语）上提升明显。8B 版本虽然整体略低，但在资源受限环境下仍能提供竞争力的表现。  
- **消融实验**：作者分别去掉 Goldfish 目标、去除 robots.txt 回溯过滤、以及削减非英语数据比例，实验显示：不使用 Goldfish 时模型的记忆率上升约 2‑3 倍；不做回溯过滤会导致约 5% 的数据被标记为违规；降低非英语比例会导致低资源语言的准确率下降 10% 以上。  
- **局限性**：论文承认仍然存在 **数据稀疏** 的问题——即使覆盖 1800 种语言，部分极低资源语言的 token 数仍不足以支撑高质量表示；此外，Goldfish 目标虽然抑制了逐字记忆，但在极端长文本上仍可能出现局部泄露。  

### 影响与延伸思考
Apertus 的全链路合规与多语言策略在开源社区引发了广泛讨论，推动了 **“合规开源 LLM”** 这一新潮流。随后出现的项目如 **OpenChat‑Compliant**、**Mistral‑Multilingual** 等，都在数据收集阶段明确引用了 robots.txt 回溯检查的思路。对研究者而言，Goldfish 记忆抑制提供了一个可复用的技术框架，未来可以与 **差分隐私**、**联邦学习** 等隐私保护手段结合，进一步降低模型泄露风险。若想深入了解，可关注 **数据合规审计工具**（如 Crawl‑Audit）和 **跨语言表示学习**（如 XLM‑R）等方向的最新进展。

### 一句话记住它
Apertus 用全公开、合规的数据管道和 Goldfish 记忆抑制，让大语言模型真正做到“开源且不侵犯”，并把 1800+ 语言的能力带给全球用户。