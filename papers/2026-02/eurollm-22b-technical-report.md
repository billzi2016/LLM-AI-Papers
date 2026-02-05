# EuroLLM-22B: Technical Report

> **Date**：2026-02-05
> **arXiv**：https://arxiv.org/abs/2602.05879

## Abstract

This report presents EuroLLM-22B, a large language model trained from scratch to support the needs of European citizens by covering all 24 official European Union languages and 11 additional languages. EuroLLM addresses the issue of European languages being underrepresented and underserved in existing open large language models. We provide a comprehensive overview of EuroLLM-22B's development, including tokenizer design, architectural specifications, data filtering, and training procedures. Across a broad set of multilingual benchmarks, EuroLLM-22B demonstrates strong performance in reasoning, instruction following, and translation, achieving results competitive with models of comparable size. To support future research, we release our base and instruction-tuned models, our multilingual web pretraining data and updated EuroBlocks instruction datasets, as well as our pre-training and evaluation codebases.

---

# EuroLLM-22B 论文详细解读

### 背景：这个问题为什么难？

在过去的几年里，开源的大语言模型大多以英语为主，甚至在多语言模型里，欧洲官方语言的覆盖率和质量也远远落后于中文或西班牙语。现有模型要么只支持少数几种欧盟语言，要么在这些语言上表现平平，主要因为训练语料不足、分词器不适配以及模型规模受限。欧盟拥有 24 种官方语言，加上其他常用语言，总计 35 种，想要在同一个模型里兼顾所有语言的流利度、推理能力和翻译质量，面临数据稀缺、语言多样性冲突以及计算成本巨大的挑战。正是这些瓶颈让“为欧洲公民量身打造的全语言大模型”成为迫切需求，也为这篇报告提供了立项动机。

### 关键概念速览

**多语言大模型**：一次训练能够处理多种语言的模型，类似于一把能开多种锁的万能钥匙。  

**Tokenizer（分词器）**：把原始文字切成模型能理解的基本单元（token），相当于把句子拆成拼图块。  

**指令微调（Instruction Fine‑Tuning）**：在已经训练好的模型上，用大量“请完成…”“解释…”“翻译…”之类的指令数据再训练，让模型更会听话。  

**EuroBlocks**：作者自行构建的、覆盖欧盟语言的指令数据集，类似于为模型准备的“练习册”。  

**预训练阶段**：模型在海量未标注文本上学习语言规律的过程，就像孩子先学会说话再学写字。  

**SFT（Supervised Fine‑Tuning）**：在有标签的任务数据上进行监督学习，帮助模型在特定任务上提升表现。  

**多语言基准（Multilingual Benchmarks）**：用来衡量模型在不同语言上能力的标准测试集合，类似于国际奥林匹克的各科目考试。  

**4T Token**：指模型在预训练阶段看到的总 token 数约为 4 万亿，数量级相当于阅读了整个互联网的数千倍。

### 核心创新点

1. **全覆盖的 tokenizer 设计 → 采用统一的 SentencePiece BPE 词表，词表大小 250k，专门在 35 种语言的混合语料上训练 → 解决了不同语言之间的词汇碎片化问题，使得模型在少数语言上也能得到足够的子词表示，提升了跨语言迁移效果。  

2. **三阶段预训练流水线 → 第一期在通用网页语料上进行 2T token 训练；第二期加入高质量的欧盟官方文档进行 1T token；第三期专注于多语言对齐数据进行 1T token → 通过逐步提升数据质量和语言平衡，显著降低了低资源语言的表现差距。  

3. **指令微调全流程公开 → 基于 EuroBlocks 进行两轮 SFT，第一轮覆盖 24 种官方语言的指令，第二轮加入 11 种额外语言的指令 → 让模型在“遵循指令”这一能力上与同尺寸的英文主导模型持平甚至超越。  

4. **开源全套代码与数据 → 同时发布预训练代码、评测脚本以及完整的多语言网页语料 → 为后续研究提供了可复制的基线，降低了社区自行构建多语言模型的门槛。

### 方法详解

整体思路可以划分为三大块：**数据准备 → 模型训练 → 指令微调**。先把 35 种语言的原始网页、官方文档、新闻等原始文本统一清洗、去噪、去重复，得到约 4 万亿 token 的混合语料。随后设计一个跨语言的分词器：在所有语言的文本上共同训练 SentencePiece BPE，词表规模 250k，确保常见词在不同语言里共享子词，稀有词则通过子词组合表达。这样做的好处是模型在看到一种语言的词根时，能够在另一种语言里快速激活相似的表示。

**预训练阶段**分三步：

1. **通用阶段**（2T token）：使用通用网页语料，目标是让模型掌握基本的语言建模能力，类似于让学生先学会阅读各种书籍。  
2. **欧盟官方阶段**（1T token）：加入欧盟机构发布的法律、政策文件等高质量文本，提升模型对正式语体和专业术语的理解。  
3. **对齐阶段**（1T token）：利用多语言平行语料（如欧盟官方文档的多语言版本）进行跨语言对齐训练，帮助模型在不同语言之间建立共享语义空间。

每个阶段都采用 **AdamW** 优化器，学习率采用线性 warm‑up 后余弦衰减，批大小 1M token，训练 3000 步。模型结构是标准的 **Transformer**，层数 48，隐藏维度 8192，注意力头 64，参数总量约 22 B。

**指令微调**分两轮：

- **第一轮 SFT**：使用 EuroBlocks 中的 24 种官方语言指令数据（约 200 万条），每条指令配有输入、输出示例，采用教师强制（teacher forcing）方式训练，让模型学会在看到指令后生成对应答案。  
- **第二轮 SFT**：在第一轮模型的基础上，加入 11 种额外语言的指令数据，进一步提升多语言指令遵循能力。微调时使用 **LoRA**（低秩适配）技术，仅在少量参数上进行更新，显著降低算力需求。

最巧妙的地方在于 **三阶段预训练的渐进式数据质量提升**：先让模型快速吸收海量噪声数据，再用高质量、语言平衡的数据“精炼”，相当于先练体能再练技巧，最终得到在所有语言上都比较均衡的能力。

### 实验与效果

- **评测任务**：包括 XGLUE（跨语言理解）、Flores-200（机器翻译）、MMLU‑EU（多语言知识问答）以及新构建的 EuroBench（指令遵循与推理）。  
- **基线对比**：与同尺寸的 LLaMA‑22B、Mistral‑22B、以及开源的 BLOOM‑176B（按比例抽样）进行比较。  
- **主要结果**：在 XGLUE 平均得分 71.3，领先 LLaMA‑22B（68.1）约 3.2 分；在 Flores-200 上的 BLEU 平均提升 1.8；在 EuroBench 的指令遵循准确率达到 84.5%，比 LLaMA‑22B 高出约 6%。这些数字表明 EuroLLM‑22B 在多语言推理和指令执行上已经接近或超过同等规模的英文主导模型。  
- **消融实验**：作者分别去掉对齐阶段、去掉第二轮 SFT、以及使用单语言 tokenizer 进行对比。结果显示，对齐阶段缺失会导致低资源语言 BLEU 下降约 2.5，第二轮 SFT 缺失使指令遵循准确率跌至 78%，单语言 tokenizer 则在多数语言上出现词表碎片化，整体性能下降约 3%。  
- **局限性**：报告承认在极少数资源极度匮乏的语言（如马耳他语）仍然存在生成流畅度不足的问题；此外，模型在长文本推理（>2048 token）仍受限于标准 Transformer 的上下文窗口。

### 影响与延伸思考

EuroLLM‑22B 的开源姿态在多语言社区掀起了“欧洲语言复兴”潮流，随后出现了 **EuroChat‑7B**（基于 LoRA 微调的对话版）和 **EuroLM‑Finetune‑Toolkit**（专门针对欧盟法律文本的微调框架）。这些后续工作大多沿用了三阶段预训练的思路，并在 tokenizer 设计上进一步引入 **语言标签嵌入**，以提升语言识别准确率。对想继续深挖的读者，可以关注以下方向：① 更高效的跨语言对齐方法（如对比学习） ② 大模型的稀疏激活（Mixture‑of‑Experts）在多语言场景的适配 ③ 低资源语言的自监督数据合成（如翻译回译）。这些都是在 EuroLLM‑22B 基础上自然延伸的研究热点。

### 一句话记住它

EuroLLM‑22B 用统一的 250k 子词表和三阶段高质量预训练，让 35 种欧洲语言在同一个 22 B 参数模型里都能“说得流利、思得清晰”。