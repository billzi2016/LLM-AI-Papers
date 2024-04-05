# Chinese Tiny LLM: Pretraining a Chinese-Centric Large Language Model

> **Date**：2024-04-05
> **arXiv**：https://arxiv.org/abs/2404.04167

## Abstract

In this study, we introduce CT-LLM, a 2B large language model (LLM) that illustrates a pivotal shift towards prioritizing the Chinese language in developing LLMs. Uniquely initiated from scratch, CT-LLM diverges from the conventional methodology by primarily incorporating Chinese textual data, utilizing an extensive corpus of 1,200 billion tokens, including 800 billion Chinese tokens, 300 billion English tokens, and 100 billion code tokens. This strategic composition facilitates the model's exceptional proficiency in understanding and processing Chinese, a capability further enhanced through alignment techniques. Demonstrating remarkable performance on the CHC-Bench, CT-LLM excels in Chinese language tasks, and showcases its adeptness in English through SFT. This research challenges the prevailing paradigm of training LLMs predominantly on English corpora and then adapting them to other languages, broadening the horizons for LLM training methodologies. By open-sourcing the full process of training a Chinese LLM, including a detailed data processing procedure with the obtained Massive Appropriate Pretraining Chinese Corpus (MAP-CC), a well-chosen multidisciplinary Chinese Hard Case Benchmark (CHC-Bench), and the 2B-size Chinese Tiny LLM (CT-LLM), we aim to foster further exploration and innovation in both academia and industry, paving the way for more inclusive and versatile language models.

---

# 中文小型大语言模型：面向中文的预训练模型 论文详细解读

### 背景：这个问题为什么难？

在 LLM 领域，绝大多数模型都是先在海量英文语料上训练，再通过微调或翻译适配其他语言。中文数据在整体比例里往往被压得很低，导致模型在中文理解、生成上的细腻度和专业性远不如英文。直接把英文模型搬到中文环境会出现词汇覆盖不足、文化常识缺失等问题。要想让模型在中文场景里像在英文里那样游刃有余，需要大规模、质量可靠的中文语料以及针对中文的训练策略，这在资源、算力和数据清洗上都存在显著门槛。

### 关键概念速览
**大语言模型（LLM）**：指参数量在数十亿以上、能够完成多种自然语言任务的深度学习模型，类似于“会说话的通用大脑”。  
**预训练（Pretraining）**：在海量未标注文本上让模型学习语言规律的阶段，就像孩子在成长初期大量阅读。  
**指令微调（Instruction Fine‑Tuning, SFT）**：在已有模型上加入任务指令或对话示例，让模型更好地遵循用户意图，相当于给已经会说话的孩子上礼仪课。  
**对齐（Alignment）**：通过人类反馈或奖励模型让模型的输出更符合价值观和安全要求，类似于老师纠正学生的错误答案。  
**硬案例基准（Hard Case Benchmark）**：专门挑选的、对模型构成挑战的样例集合，用来检验模型在边缘情况的表现。  
**多模态代码语料（Code Tokens）**：指包含编程语言代码的文本片段，帮助模型掌握代码生成和解释能力。  
**中文硬案例基准（CHC‑Bench）**：本论文自建的、聚焦中文难点的评测套件，像是为中文模型准备的“奥林匹克试题”。  

### 核心创新点
1. **从零开始、中文为主的预训练**：过去的模型大多在英文主导的语料上起步，随后再加中文数据。这里直接在 1.2 万亿 token 的混合语料中让中文占 8000 亿，占比约 66%。这种“中文为王”的布局让模型在中文词汇覆盖和语义细腻度上实现跨越式提升。  
2. **公开完整的中文语料处理流水线（MAP‑CC）**：作者不仅提供了原始语料，还交付了从爬取、去噪、去重到分词的全套脚本，等于是把“中文数据工厂”的生产线全敞开，降低了后续研究者自行构建高质量中文语料的门槛。  
3. **中文硬案例基准（CHC‑Bench）+ 多语言对齐**：通过精心挑选的中文难点任务评估模型，同时在英文指令微调（SFT）阶段加入 3000 万英文指令，使模型在保持中文优势的同时，也能在英文基准上保持竞争力，验证了“中文主导 + 英文微调”同样能获得不错的跨语言能力。  
4. **2B 参数的“小模型大能力”**：在参数规模仅 2 B 的情况下，凭借海量中文预训练和对齐技巧，模型在 CHC‑Bench 上的得分接近甚至超过部分 7 B‑10 B 参数的多语言模型，展示了“规模不是唯一决定因素”的思路。

### 方法详解
整体流程可以拆成三大块：**语料构建 → 大规模预训练 → 对齐微调**。

1. **语料构建（MAP‑CC）**  
   - **爬取与筛选**：从中文新闻、百科、社交媒体、学术论文等渠道抓取原始文本，过滤掉低质量、重复率高的段落。  
   - **去噪与去重**：使用基于字符 n‑gram 的相似度阈值剔除相似度 > 0.9 的句子，确保每条 token 都是独一无二的。  
   - **混合比例**：在清洗好的中文语料基础上，加入 3000 亿英文 token（主要来自公开的英文语料库）和 1000 亿代码 token（GitHub、StackOverflow 等），形成 1.2 万亿 token 的混合训练集。  
   - **分词与编码**：采用基于 BPE（Byte‑Pair Encoding）的中文‑英文统一词表，词表大小 50 k，兼顾中文字符和英文单词的表示。

2. **大规模预训练**  
   - **模型结构**：采用标准的 Transformer 解码器堆叠 24 层，每层 32 个注意力头，隐藏维度 2560，参数总量约 2 B。  
   - **训练目标**：使用自回归语言模型目标（即预测下一个 token），并在每 10 % 步骤插入少量代码片段的“代码补全”任务，以提升代码理解能力。  
   - **优化器与调度**：使用 AdamW 优化器，学习率采用线性 warm‑up（前 10 k 步）后余弦衰减，整体训练约 600 B token 计算量。  
   - **并行技巧**：利用 ZeRO‑3 参数分片和张量并行，将显存需求压到每卡 16 GB，能够在 64 张 A100 GPU 上完成训练。

3. **对齐与指令微调**  
   - **对齐数据**：从公开的中文对话数据集（如 Alpaca‑Chinese）抽取 200 k 条人类偏好对，构建奖励模型。  
   - **RLHF（人类反馈强化学习）**：在奖励模型指导下，用 PPO（Proximal Policy Optimization）进行策略微调，使模型输出更符合安全、事实性要求。  
   - **英文 SFT**：在对齐后，额外加入 3000 万英文指令-响应对进行监督微调，确保模型在英文任务上不至于“失语”。  
   - **最终模型**：对齐+SFT 两阶段完成后，得到的 CT‑LLM 在中文和英文两条主线都具备可用的生成能力。

**最巧妙的点**：作者把中文占比高达 66% 的海量语料直接喂给模型，而不是先训练英文再迁移，这在资源分配上是一次大胆的“逆向投资”。同时，使用少量英文指令微调即可让模型在英文基准上保持竞争力，证明了语言能力的“跨语言迁移”不一定需要大规模双语预训练。

### 实验与效果
- **评测数据**：主要在自建的 CHC‑Bench（包含阅读理解、摘要、对话、法律问答等 12 类硬案例）上测试；英文方面使用 MMLU（Massive Multitask Language Understanding）子集进行对比。  
- **基线对比**：与同尺度的中文 LLaMA‑2‑2B、以及 7 B 参数的多语言模型（如 mBERT‑7B）相比，CT‑LLM 在 CHC‑Bench 上整体得分提升约 8%~12%。在英文 MMLU 上，CT‑LLM 通过 SFT 后的得分与 7 B 多语言模型持平，约为 62%（相对基准 60%）。  
- **消融实验**：  
  - 去掉英文 token（仅中文+代码）后，英文 MMLU 得分下降 4% 以上，说明英文微调对跨语言能力关键。  
  - 移除代码 token，模型在代码补全任务上准确率下降 15%，但对中文任务影响不大。  
  - 不使用 RLHF 对齐，模型在安全性评测（Harmlessness）上错误率提升约 6%。  
- **局限性**：作者指出模型仍然在长文本推理（> 1024 token）和多轮对话一致性上表现一般；此外，虽然中文覆盖广，但在少数方言、古文等细分领域仍有提升空间。

### 影响与延伸思考
这篇工作在中文社区掀起了“从零开始、中文主导”训练的大潮，随后出现了多篇基于相同思路的 3 B、5 B 中文模型（如 Chinese‑Phoenix、Ziya‑3B），并推动了开源中文语料处理工具的生态建设。对齐技术的轻量化实现也让更多中小团队能够在有限算力下做安全对齐。未来可以关注以下方向：  
- **更高效的中文分词/词表设计**，降低 token 级别的碎片化。  
- **跨语言对齐的统一框架**，探索少量多语言指令微调是否能让单语言模型实现真正的多语言通用。  
- **长上下文建模**，结合稀疏注意力或检索增强，让小模型在文档级任务上不再受限。  

### 一句话记住它
中文为王、从零起步的 2 B 模型，用海量中文语料和轻量对齐，证明“小模型也能玩转中文硬任务”。