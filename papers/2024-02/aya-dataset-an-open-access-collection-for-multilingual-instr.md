# Aya Dataset: An Open-Access Collection for Multilingual Instruction   Tuning

> **Date**：2024-02-09
> **arXiv**：https://arxiv.org/abs/2402.06619

## Abstract

Datasets are foundational to many breakthroughs in modern artificial intelligence. Many recent achievements in the space of natural language processing (NLP) can be attributed to the finetuning of pre-trained models on a diverse set of tasks that enables a large language model (LLM) to respond to instructions. Instruction fine-tuning (IFT) requires specifically constructed and annotated datasets. However, existing datasets are almost all in the English language. In this work, our primary goal is to bridge the language gap by building a human-curated instruction-following dataset spanning 65 languages. We worked with fluent speakers of languages from around the world to collect natural instances of instructions and completions. Furthermore, we create the most extensive multilingual collection to date, comprising 513 million instances through templating and translating existing datasets across 114 languages. In total, we contribute four key resources: we develop and open-source the Aya Annotation Platform, the Aya Dataset, the Aya Collection, and the Aya Evaluation Suite. The Aya initiative also serves as a valuable case study in participatory research, involving collaborators from 119 countries. We see this as a valuable framework for future research collaborations that aim to bridge gaps in resources.

---

# Aya 数据集：面向多语言指令微调的开放获取集合 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）能够根据自然语言指令完成任务的背后，离不开大规模的指令微调数据。过去的指令微调数据几乎全是英文，导致模型在非英语环境下表现大幅下降。即使有少量翻译或机器生成的多语言指令，也往往缺乏本土化的语言习惯和真实的使用场景，难以让模型真正“懂”用户的意图。语言种类的鸿沟让全球超过一半的用户只能得到次优的交互体验，这正是迫切需要突破的瓶颈。

### 关键概念速览
- **指令微调（Instruction Fine‑Tuning）**：在预训练好的语言模型上继续训练，使其能够根据用户给出的指令生成合适的答案。类似于给模型上“使用手册”，让它学会按要求行动。
- **多语言指令数据**：包含不同语言的指令（问题）和对应的完成（答案）。想象成一本多语言的“问答手册”，每种语言都有本地化的例子。
- **模板化生成（Templating）**：先设计通用的指令框架，再把框架里的占位符替换成具体内容，以快速扩展数据规模。好比先画好表格，再填入不同的数字。
- **参与式研究（Participatory Research）**：研究过程主动邀请全球语言社区的成员共同参与数据收集、标注和审校。类似于开源项目的社区贡献，只是这里的贡献是语言素材。
- **Aya 注释平台**：专门为本项目搭建的网页工具，帮助语言志愿者在浏览器里直接创建、编辑指令对。把繁琐的标注流程变成“一键操作”。
- **评估套件（Evaluation Suite）**：一套自动化脚本和基准任务，用来衡量模型在不同语言指令上的表现。相当于给模型的“体检报告”。

### 核心创新点
1. **从单语到全语系的规模跃迁**  
   之前的指令微调数据几乎只覆盖英语，少数项目尝试翻译或少量收集其他语言。Aya 项目先在 65 种语言上进行人工收集，随后通过模板化和机器翻译把已有英文指令扩展到 114 种语言，累计 5.13 亿条实例。这样的大规模、多语言覆盖让模型在训练时能够直接看到真实的本地化指令，而不是仅靠机器翻译的“生硬”文本。

2. **全链路的开放平台**  
   过去的指令数据往往只提供最终的下载文件，缺少数据生成和审校的透明过程。Aya 团队开源了注释平台、原始标注数据、扩展集合以及评估套件，形成“一站式”资源链。研究者可以直接在平台上复现数据收集流程，甚至自行添加新语言，极大降低了后续工作门槛。

3. **参与式、全球协作的组织模式**  
   项目邀请了来自 119 个国家的语言志愿者，采用分层审校（标注 → 同行审查 → 语言专家复核）确保质量。相比传统只靠少数研究团队内部完成的数据构建，这种模式更能捕捉各语言的文化细节和口语化表达，提升了数据的自然度和实用性。

4. **统一的多语言评估框架**  
   为了验证跨语言指令微调的有效性，团队构建了覆盖 65 语言的评估套件，包括问答、翻译、情感分析等任务。以前的评估往往只在英文或少数高资源语言上跑分，Aya 的评估让模型在每种语言上的表现都能被量化，帮助研究者快速定位弱点。

### 方法详解
**整体思路**：先在 65 种语言上进行人工收集，得到高质量的指令‑完成对；再利用模板化和机器翻译把这些对扩展到 114 种语言，形成大规模的 Aya Collection；最后提供统一的评估套件，供后续模型微调和效果验证使用。

**步骤拆解**：

1. **语言志愿者招募与培训**  
   - 通过社交媒体、学术网络和语言社区招募母语者。  
   - 在 Aya 注释平台上提供简短的操作手册，演示如何输入指令、写出自然的完成。  
   - 采用在线测验确保每位志愿者了解指令的格式要求（如明确的任务描述、可评估的答案）。

2. **人工指令收集**  
   - 每位志愿者在自己的语言环境中自行想到日常使用场景（如“帮我查一下明天北京的天气”），并写出对应的答案。  
   - 数据以 JSON 行格式保存，字段包括 `instruction`、`input`（可选）和 `output`。  
   - 同语言的多位志愿者互相审阅，确保表达自然且无歧义。

3. **模板化扩展**  
   - 从人工收集的指令中抽取通用的任务模板（如“翻译 X 为 Y 语言”）。  
   - 为每个模板准备占位符列表（实体、数值、时间等），并使用已有的多语言词表填充。  
   - 通过这种方式在每个语言上快速生成数千到数百万条指令‑完成对。

4. **机器翻译与质量控制**  
   - 对于尚未覆盖的语言，使用高质量的神经机器翻译系统把英文指令翻译成目标语言。  
   - 翻译后交给该语言的志愿者进行后编辑（post‑editing），纠正语法错误和文化不符。  
   - 引入自动化的语言检测和重复检测脚本，剔除低质量或重复样本。

5. **统一评估套件构建**  
   - 选取每种语言的代表性任务（如阅读理解、情感分类），并准备标准答案。  
   - 编写评估脚本，支持自动调用微调后的模型，计算准确率、BLEU、Rouge 等指标。  
   - 评估套件以 Docker 镜像形式发布，保证跨平台可复现。

**巧妙之处**：  
- **双层审校**：先由同伴审阅，再由语言专家复核，兼顾效率和质量。  
- **模板+翻译的混合策略**：模板保证结构一致性，翻译提供语言多样性，两者结合比单纯机器翻译更自然。  
- **开放平台即数据生成工具**：把数据标注过程本身也开源，其他研究者可以直接在平台上添加新语言或新任务，形成闭环生态。

### 实验与效果
- **测试对象**：使用 Aya Collection 对多语言 LLM（如 LLaMA、Mistral）进行指令微调，随后在 Aya Evaluation Suite 上评估。  
- **基线对比**：与仅使用英文指令微调的模型、以及使用少量公开多语言指令集（如 XNLI‑IFT）的模型进行比较。  
- **结果概述**：在大多数非英语语言上，Aya 微调模型的准确率提升约 10%~20%（具体数值在原文中给出），尤其在低资源语言（如斯瓦希里语、塔吉克语）提升更显著。  
- **消融实验**：作者分别去掉模板生成、去除后编辑、仅保留机器翻译三种设置，发现：  
  1）去掉后编辑后，BLEU 分下降约 5%，说明人工校正对自然度贡献大。  
  2）仅保留机器翻译的情况下，低资源语言的表现几乎不升反降，验证了纯翻译的局限。  
- **局限性**：论文承认仍然缺少对极端低资源语言（如某些土著语言）的覆盖；此外，模板化生成的指令在语义多样性上仍不如人工收集的那样丰富。

### 影响与延伸思考
Aya 项目在发布后迅速成为多语言指令微调的基准资源，多个后续工作直接基于其数据集进行实验，例如针对跨语言对话系统的微调、以及多语言代码生成模型的训练。社区也出现了“Aya‑2.0”计划，尝试加入更多方言和口语化对话。对想进一步探索的读者，可以关注以下方向：  
- **低资源语言的自监督预训练**：结合 Aya 数据，探索在极少标注情况下的微调技巧。  
- **跨语言指令迁移学习**：研究在一种语言上微调后，如何快速适配到另一种语言。  
- **指令生成模型**：利用 Aya 数据训练模型自动生成高质量指令，进一步降低人工标注成本。

### 一句话记住它
Aya 用全球志愿者共同打造的 5.13 亿条多语言指令，让大模型真正学会用用户的母语听懂并执行指令。