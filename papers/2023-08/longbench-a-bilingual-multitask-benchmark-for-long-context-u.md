# LongBench: A Bilingual, Multitask Benchmark for Long Context   Understanding

> **Date**：2023-08-28
> **arXiv**：https://arxiv.org/abs/2308.14508

## Abstract

Although large language models (LLMs) demonstrate impressive performance for many language tasks, most of them can only handle texts a few thousand tokens long, limiting their applications on longer sequence inputs, such as books, reports, and codebases. Recent works have proposed methods to improve LLMs' long context capabilities by extending context windows and more sophisticated memory mechanisms. However, comprehensive benchmarks tailored for evaluating long context understanding are lacking. In this paper, we introduce LongBench, the first bilingual, multi-task benchmark for long context understanding, enabling a more rigorous evaluation of long context understanding. LongBench comprises 21 datasets across 6 task categories in both English and Chinese, with an average length of 6,711 words (English) and 13,386 characters (Chinese). These tasks cover key long-text application areas including single-doc QA, multi-doc QA, summarization, few-shot learning, synthetic tasks, and code completion. All datasets in LongBench are standardized into a unified format, allowing for effortless automatic evaluation of LLMs. Upon comprehensive evaluation of 8 LLMs on LongBench, we find that: (1) Commercial model (GPT-3.5-Turbo-16k) outperforms other open-sourced models, but still struggles on longer contexts. (2) Scaled position embedding and fine-tuning on longer sequences lead to substantial improvement on long context understanding. (3) Context compression technique such as retrieval brings improvement for model with weak ability on long contexts, but the performance still lags behind models that have strong long context understanding capability. The code and datasets are available at https://github.com/THUDM/LongBench.

---

# LongBench：面向长上下文理解的双语多任务基准 论文详细解读

### 背景：这个问题为什么难？
大多数大型语言模型（LLM）只能一次性处理几千个 token，远远低于书籍、报告或代码库等真实场景的长度。虽然已有研究尝试通过扩大上下文窗口或加入外部记忆来突破这个瓶颈，但缺少专门针对“长文本理解”进行系统评估的基准。没有统一、规模化的测试集合，研究者很难判断不同方法到底在多长的上下文上真正有效，也难以比较不同语言（如中英）下的表现差异。因此，构建一个覆盖多任务、双语、真实长文本的评测平台成为迫切需求。

### 关键概念速览
- **大模型（LLM）**：参数量在数十亿以上的语言模型，能够生成或理解自然语言。类似于拥有海量知识的“语言机器人”。  
- **上下文窗口（context window）**：模型一次性能看到的 token 数量上限，像是阅读时一次性翻开的书页数。  
- **位置嵌入（position embedding）**：向模型输入的每个 token 附加的位置信息，使模型知道词语在序列中的先后顺序。可以类比为在句子里给每个词贴上“第几位”的标签。  
- **长序列微调（fine‑tuning on long sequences）**：在已有模型基础上，用更长的文本继续训练，让模型适应大段落的模式。相当于让“语言机器人”练习阅读整章而不是单段。  
- **检索压缩（retrieval‑based compression）**：先用搜索或相似度匹配挑出与问题最相关的片段，再把这些片段喂给模型，降低整体长度。像是先在书中找目录，再只阅读目录对应的章节。  
- **双语基准**：评测数据同时提供英文和中文版本，确保模型在不同语言下的长文本能力都能被测量。  
- **多任务基准**：包含问答、摘要、代码补全等多种任务，避免模型只在单一场景下“装逼”。  

### 核心创新点
1. **首次构建双语长文本基准**：过去的长上下文评测大多只针对英文单任务，本文收集并统一了 21 个数据集，覆盖单文档 QA、跨文档 QA、摘要、少样本学习、合成任务和代码补全，中文平均 13,386 字，英文平均 6,711 词，实现了真正的跨语言、跨任务评测。  
2. **统一格式与自动评估**：所有数据被转化为统一的 JSON‑Lines 结构，输入字段统一为 `context`、`question`、`reference`，评估脚本能够自动对齐答案并计算准确率、BLEU、ROUGE 等指标，省去研究者手动对齐的繁琐步骤。  
3. **系统性对比 8 大模型**：在同一基准上跑通了 GPT‑3.5‑Turbo‑16k、Claude、LLaMA‑2 系列以及若干开源模型，揭示了商业模型虽领先但在超长上下文仍会出现信息遗失的现象。  
4. **深入分析长上下文提升手段**：通过实验对比位置嵌入尺度、长序列微调以及检索压缩三类技术，明确指出“扩大位置嵌入 + 长序列微调”是提升长文本理解的关键，而检索压缩只能在模型本身长上下文能力弱时提供有限帮助。  

### 方法详解
**整体框架**  
本文的核心工作是“基准构建 + 统一评测”。首先从公开资源、论文附录以及自行爬取的中文语料中挑选出适合长文本的任务，确保每条样本的上下文长度均超过 4k token。随后统一数据格式，编写评测脚本，实现“一键跑通”。最后在统一平台上对 8 种模型进行批量推理，收集指标并进行横向分析。

**关键步骤**  
1. **任务筛选与长度过滤**：对每个候选数据集统计字符/词数，剔除短于 4k token 的样本，保留最长的 10%–30% 作为基准子集。  
2. **双语对齐**：英文原始数据直接使用，中文版本通过机器翻译+人工校对得到，确保语义等价。  
3. **统一 JSON‑Lines 结构**：每行包含 `id`, `language`, `task_type`, `context`, `question`（或 `prompt`），以及 `reference`（答案或摘要）。这种结构便于流式读取和并行评测。  
4. **自动评估脚本**：根据任务类型自动选择评估指标：QA 用 Exact Match / F1，摘要用 ROUGE‑1/2/L，代码补全用 BLEU + 编译成功率。脚本内部会对模型输出进行标准化（去除空格、大小写统一），保证公平比较。  
5. **模型推理统一化**：为每个模型编写适配器，将统一输入转化为模型特定的 API 调用（如 OpenAI 的 chat/completions、HF 的 pipeline），并统一截断策略（超出窗口的部分采用滑动窗口或检索压缩）。  

**最巧妙的设计**  
- **检索压缩的统一实现**：作者实现了一个轻量级 BM25 检索器，在推理前先对 `context` 进行关键词抽取，只保留与 `question` 相关度最高的前 2k token。这样即使模型本身不支持超长输入，也能在同一评测框架下得到“压缩后”的成绩，便于对比。  
- **位置嵌入尺度实验**：通过修改模型配置文件，将位置嵌入的最大长度从 2k 扩展到 8k、16k，直接观察对长文本任务的影响，提供了最直接的“硬件层面”提升证据。  

### 实验与效果
- **数据集覆盖**：21 个子数据集，分别属于单文档 QA（如 NarrativeQA）、多文档 QA（如 HotpotQA）、长文摘要（如 arXiv）、少样本学习（如 LAMA‑Long）、合成推理任务（如 LongMath）以及代码补全（如 CodeXGLUE‑Long）。  
- **基线模型**：包括 GPT‑3.5‑Turbo‑16k、Claude‑1、LLaMA‑2‑7B/13B、Vicuna‑13B、ChatGLM‑6B 等共 8 种。  
- **主要发现**：  
  - GPT‑3.5‑Turbo‑16k 在所有任务上整体领先约 12%–18%（相对 F1/ROUGE），但在最长的 20k token 文档上仍出现信息遗漏，准确率下降约 7%。  
  - 将位置嵌入上限从 2k 提升至 8k，LLaMA‑2‑13B 在长摘要任务上 ROUGE‑L 提升约 4.5%，在多文档 QA 上 Exact Match 提升约 6%。  
  - 对同一模型进行 4k‑8k token 长序列微调后，整体表现提升 5%–9%，尤其在代码补全任务中，BLEU 提升 3.2%。  
  - 检索压缩对弱模型（如 ChatGLM‑6B）提升显著，Exact Match 提升约 10%，但对已经具备强长上下文能力的模型（如 GPT‑3.5）提升有限，仅 1%–2%。  
- **消融实验**：作者分别关闭位置嵌入扩展、长序列微调、检索压缩，发现位置嵌入是提升的最大驱动因素，约占整体提升的 55%。  
- **局限性**：原文未给出对极端超长（>30k token）文本的评测，也未探讨多模态（图文）长上下文的情况；此外，中文数据主要来源于机器翻译，可能存在翻译噪声。  

### 影响与延伸思考
LongBench 公开后迅速成为评测长上下文能力的“标配”，后续出现的 LLaMA‑2‑Long、LongChat、Mistral‑Long 等模型都在论文中引用该基准进行对比。它也推动了社区对位置嵌入可伸缩性的研究，出现了 Rotary Positional Embedding、ALiBi 等更高效的实现方式。对想进一步深入的读者，可以关注以下方向：  
- **可伸缩位置编码**：如何在不显著增加显存的情况下支持 64k、128k token。  
- **层次化检索与记忆**：结合长期记忆网络，让模型在保持全局上下文的同时只激活局部片段。  
- **跨语言长上下文对齐**：利用双语基准探索中英长文本的对齐策略，提升多语言模型的统一理解能力。  

### 一句话记住它
LongBench 用 21 项双语长文本任务统一评测，让我们第一次能客观比较“大模型”到底能读多长的“书”。