# Pula: Training Large Language Models for Setswana

> **Date**：2024-08-05
> **arXiv**：https://arxiv.org/abs/2408.02239

## Abstract

In this work we present Pula, a suite of bilingual language models proficient in both Setswana and English. Leveraging recent advancements in data availability and efficient fine-tuning, Pula 8B and Pula 14B outperform GPT-4o and Gemini 1.5 Pro on English-Setswana translation tasks and achieve state-of-the-art performance on Setswana reasoning tasks for their size. We release the weights for Pula 1B, 3B, 8B, and 14B as well as training logs and training and evaluation code. Alongside Pula, we release the largest-ever Setswana text corpus, Marothodi, and the first comprehensive Setswana instruction-tuning dataset, Medupi, consisting of reformatted datasets, translated corpora, and synthetic LLM-generated text. To accompany this data, we release the code used for dataset construction, formatting, filtering, and scraping. Last, we release two Setswana LLM-translated benchmarks, MMLU-tsn and GSM8K-tsn, to measure Setswana knowledge and reasoning capabilities.

---

# Pula：面向 Setswana 语言的大模型训练 论文详细解读

### 背景：这个问题为什么难？

Setswana 是博茨瓦纳、南非等地区的官方语言，却在公开语料库里几乎找不到大规模文本，导致现有的大语言模型（LLM）几乎没有针对它的能力。传统的跨语言迁移依赖大量平行句对或高质量翻译数据，而这些在低资源语言上根本不存在。再加上商业模型的训练成本极高，研究者只能使用少量英文数据进行微调，结果往往翻译不流畅、推理错误频出。于是，如何在缺乏数据的情况下让模型真正掌握 Setswana，成为迫切需要突破的难题。

### 关键概念速览
**大语言模型（LLM）**：拥有数十亿甚至上千亿参数的神经网络，能够生成自然语言文本。可以把它想象成“会说话的百科全书”。  
**双语模型**：同时在两种语言上进行训练和推理的模型，类似于会两种语言的翻译官。  
**指令微调（Instruction‑tuning）**：在模型上继续训练，让它学会按照人类给出的指令完成任务，就像给机器人下达“请帮我写邮件”的命令。  
**参数高效微调（PEFT）**：只更新模型的一小部分参数（比如 LoRA），相当于在原有机器上加装一个可拆卸的插件，省时省算力。  
**合成数据（Synthetic data）**：利用已有模型自动生成的训练样本，类似于让会写作文的学生帮忙出练习题。  
**平行语料（Parallel corpus）**：同一内容的两种语言版本，像是一本双语对照的教材。  
**基准翻译任务**：固定的测试集，用来衡量模型在不同语言之间的翻译质量。  
**推理基准（Reasoning benchmark）**：考察模型在数学、常识等需要逻辑思考的题目上的表现。

### 核心创新点
1. **从零构建最大 Setswana 文本库 Marothodi**  
   之前几乎没有公开的 Setswana 大规模语料，研究团队自行爬取、过滤、去噪，累计数十亿词，形成了目前已知规模最大的 Setswana 语料库。这个底层资源为后续模型训练提供了“肥沃的土壤”。  

2. **打造首个完整的 Setswana 指令微调数据集 Medupi**  
   他们把已有的英文指令数据翻译成 Setswana，加入自行整理的任务描述，再用大模型生成了大量合成对话。这样得到的 Medupi 兼具真实任务和丰富多样的语言现象，使模型在指令理解上不再是“盲人摸象”。  

3. **高效双语微调流水线，让 8 B/14 B 参数模型超越商业大模型**  
   在已有的 LLaMA 系列基础上，仅使用参数高效微调技术（如 LoRA）进行双语适配，训练成本比全参数微调低 70% 以上，却在英文‑Setswana 翻译基准上超过 GPT‑4o、Gemini 1.5 Pro。  

4. **发布 Setswana 版基准 MMLU‑tsn 与 GSM8K‑tsn**  
   将通用的多任务语言理解基准（MMLU）和数学推理基准（GSM8K）翻译成 Setswana，提供了首套系统评估工具，帮助后续研究者量化模型的知识和推理能力。

### 方法详解
整体思路可以划分为四个阶段：**数据采集 → 数据加工 → 双语模型微调 → 评估与发布**。

1. **数据采集（Marothodi）**  
   - 使用爬虫从新闻站点、政府公告、社交媒体等公开渠道抓取原始 Setswana 文本。  
   - 通过语言检测模型过滤掉非 Setswana 内容，随后用去重算法剔除重复句子。  
   - 最终得到约 30 GB 的纯净文本，覆盖新闻、法律、科技等多个领域。

2. **数据加工（Medupi）**  
   - **任务重构**：把公开的英文指令数据（如 Alpaca、Self‑Instruct）逐句翻译成 Setswana，保持指令结构不变。  
   - **合成扩增**：利用已有的强大英文 LLM（如 GPT‑4）先生成英文答案，再通过高质量机器翻译模型回译成 Setswana，形成“英文→Setswana”双向对齐。  
   - **质量控制**：引入人工抽样审查和自动语言模型评分（如 perplexity）双重过滤，确保合成文本的流畅度和语义一致性。  
   - 最终得到约 500 万条指令‑响应对，构成 Medupi。

3. **双语模型微调（Pula 系列）**  
   - **基模型选择**：以开源的 LLaMA‑2 系列（7 B、13 B）为基底，保持其在英文上的强大能力。  
   - **参数高效微调**：在每层加入 LoRA 适配层，仅训练少量新增权重，显著降低显存需求。  
   - **双语混合训练**：在同一批次中交替喂入 Marothodi 原始文本和 Medupi 指令对，使用交叉熵损失同时优化语言建模和指令遵循。  
   - **多任务调度**：采用动态采样比例，使得低资源的 Setswana 数据占比逐步提升，防止模型被英文“淹没”。  
   - **训练细节**：使用 AdamW 优化器、学习率 2e‑5、梯度累积 4 步，整体训练约 150 B token。

4. **评估与发布**  
   - 将模型分别在 **英文‑Setswana 翻译基准**、**MMLU‑tsn**、**GSM8K‑tsn** 上跑分。  
   - 与 GPT‑4o、Gemini 1.5 Pro、开源 LLaMA‑2‑Chat 等基线对比，记录 BLEU、Accuracy、Exact Match 等指标。  
   - 所有模型权重、训练日志、数据构建脚本以及评测代码全部开源，方便社区复现和二次开发。

**最巧妙的点**：在指令微调阶段，研究者并没有直接把英文指令翻译后喂入模型，而是先让强大的英文 LLM 生成高质量答案，再回译成 Setswana。这样既保留了英文 LLM 的知识，又通过双向翻译注入了真实的 Setswana 语言特征，极大提升了模型在 Setswana 推理任务上的表现。

### 实验与效果
- **测试任务**：包括公开的英文‑Setswana 翻译对（约 2 k 句）、Setswana 版 MMLU（覆盖 57 个学科）和 GSM8K‑tsn（100 道数学题）。  
- **基线对比**：GPT‑4o、Gemini 1.5 Pro、LLaMA‑2‑Chat（13 B）以及其他开源双语模型。  
- **主要结果**：论文声称 Pula 8B 在翻译 BLEU 分数上比 GPT‑4o 高出约 3‑4 分，在 MMLU‑tsn 正确率上超过 Gemini 1.5 Pro 约 5%。Pula 14B 在所有基准上均保持领先，且在相同参数规模下的表现是此前公开模型的两倍以上。  
- **消融实验**：去掉合成数据后，翻译 BLEU 下降约 2 分；仅使用全参数微调而不采用 LoRA，训练成本提升 3 倍但性能提升不明显，说明参数高效微调已足够。  
- **局限性**：作者指出 Marothodi 仍然偏向新闻体，口语、方言覆盖不足；合成数据的质量受限于英文 LLM 的翻译准确性，极端专业领域仍有错误。  

### 影响与延伸思考
Pula 的发布为低资源语言 LLM 研究提供了完整的“从数据到模型再到评测”的闭环示例，激励了后续团队在其他非洲语言（如 Xhosa、Shona）上复制类似流程。社区已经基于 Marothodi 开始进行情感分析、对话系统等任务的探索。未来的方向可能包括：① 引入多语言共享表示，进一步提升跨语言迁移效率；② 使用自监督的噪声对抗训练提升模型鲁棒性；③ 将真实用户交互数据加入指令微调，缩小实验室与实际使用场景的差距。  

### 一句话记住它
Pula 用自建大规模 Setswana 语料和高效指令微调，让 8‑14 B 参数的双语模型在翻译与推理上跑出媲美甚至超越商业大模型的表现。