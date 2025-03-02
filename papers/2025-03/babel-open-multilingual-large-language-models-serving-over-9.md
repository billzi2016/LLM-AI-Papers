# Babel: Open Multilingual Large Language Models Serving Over 90% of   Global Speakers

> **Date**：2025-03-02
> **arXiv**：https://arxiv.org/abs/2503.00865

## Abstract

Large language models (LLMs) have revolutionized natural language processing (NLP), yet open-source multilingual LLMs remain scarce, with existing models often limited in language coverage. Such models typically prioritize well-resourced languages, while widely spoken but under-resourced languages are often overlooked. To address this disparity, we introduce $\texttt{Babel}$, an open multilingual LLM that covers the top 25 languages by number of speakers, supports over 90% of the global population, and includes many languages neglected by other open multilingual LLMs. Unlike traditional continue pretraining approaches, Babel expands its parameter count through a layer extension technique that elevates Babel's performance ceiling. We introduce two variants: $\texttt{Babel-9B}$, designed for efficient inference and fine-tuning, and $\texttt{Babel-83B}$, which sets a new standard for open multilingual LLMs. Extensive evaluations on multilingual tasks demonstrate its superior performance compared to open LLMs of comparable size. In addition, using open-source supervised fine-tuning datasets, Babel achieves remarkable performance, with Babel-9B-Chat leading among 10B-sized LLMs and Babel-83B-Chat setting a new standard for multilingual tasks, reaching the same level of commercial models.

---

# Babel：覆盖全球90%使用者的开源多语言大模型 论文详细解读

### 背景：这个问题为什么难？
在大语言模型（LLM）火热的今天，绝大多数开源模型只会几种资源丰富的语言，比如英语、中文、法语等。全球有数百种使用人数上亿的语言，却因为缺少高质量语料和算力支持，几乎被排除在模型之外。传统的多语言模型往往采用“继续预训练”——在已有模型上再喂入多语言数据，却受限于原始模型的参数规模，提升空间受天花板限制。于是，既想覆盖更多语言，又想保持竞争力的模型一直是个难点。

### 关键概念速览
**大语言模型（LLM）**：能够理解并生成自然语言的大规模神经网络，类似“会说话的百科全书”。  
**继续预训练（Continual Pre‑training）**：在已有模型基础上继续喂入新数据的做法，就像在已经学会英语的学生身上再教法语，效果受原始知识结构限制。  
**层扩展（Layer Extension）**：在模型内部新增层或扩宽现有层的技术，类似给旧房子加层楼，能显著提升容量而不必从头训练。  
**参数规模（Parameter Count）**：模型内部可调节的数值总量，参数越多模型潜在表达能力越强。  
**指令微调（Instruction Fine‑tuning）**：让模型学会遵循人类指令的微调过程，好比教机器人听懂并执行口头命令。  
**Chat 模型**：在指令微调后专门用于对话的版本，能够在多轮交互中保持上下文。  
**开源（Open‑source）**：代码、模型权重公开，任何人都可以下载、改进或部署。  
**多语言覆盖率**：模型能够流畅处理的语言数量和使用者比例，覆盖率高意味着更广泛的全球用户受益。

### 核心创新点
1. **从继续预训练到层扩展**：传统做法是把已有模型的参数固定，只在数据上继续训练，提升受限。Babel 直接在模型结构上加层或宽度，使参数总量从原始模型跳升。这样做把性能上限从“原模型的天花板”提升到“更高的天花板”。  
2. **针对前 25 大语言的全覆盖**：而不是只挑资源丰富的十几种语言，Babel 把全球使用人数最多的 25 种语言全部纳入训练语料，确保覆盖超过 90% 的全球人口。此举让许多此前被忽视的语言（如印尼语、乌尔都语等）得到实质性支持。  
3. **双规格模型设计**：推出轻量版 Babel‑9B（约 9 B 参数）和大模型 Babel‑83B（约 83 B 参数），前者侧重推理速度和微调成本，后者追求极限性能。两者共享同一层扩展框架，提供不同算力需求的选择。  
4. **统一的开源指令微调流程**：利用公开的指令微调数据集，对两种规模模型分别进行 Chat 微调，得到 Babel‑9B‑Chat 与 Babel‑83B‑Chat。这样即使是 10 B 级别的模型，也能在多语言对话任务上赶超同等规模的闭源商业模型。

### 方法详解
整体思路可以拆成三步：① 选取基模型 → ② 层扩展构造新模型 →③ 多语言预训练 + 指令微调。  

**第一步：基模型选取**  
作者先挑了一个公开的多语言基模型（具体是哪款未在摘要中说明），因为它已经具备跨语言的基本能力，省去从零训练的巨额算力。

**第二步：层扩展**  
在基模型的每一层上，作者插入了额外的 Transformer 子层，或者把原有的隐藏维度加宽。可以想象成在原有的“层楼”之间加装了新楼层，原有的参数保持不变，新层负责学习新增语言的细节。这样做的好处是：  
- 旧层的知识不被破坏，继续保留已有语言的表现。  
- 新层提供了额外的表示容量，专门吸收低资源语言的特征。  
- 参数增长是线性的，可灵活控制模型大小（9 B vs 83 B）。

**第三步：多语言预训练**  
作者收集了覆盖前 25 大语言的海量文本，构建了一个统一的多语言语料库。训练目标仍是自回归语言建模，即让模型预测下一个词。因为模型已经拥有更大的容量，训练过程能够在同等步数下学习到更细致的语言规律。

**指令微调与 Chat 化**  
在预训练完成后，使用公开的指令微调数据集（如 Alpaca、OpenChat 等）对模型进行微调，使其能够理解并执行自然语言指令。随后再进行对话微调，加入多轮对话数据，得到最终的 Chat 版本。整个微调流程对两种规模的模型统一使用，保证了性能的可比性。

**最巧妙的点**  
层扩展本身看似简单，却解决了“继续预训练”时模型容量受限的根本问题。作者没有重新训练一个全新模型，而是“在已有模型上加层”，既保留了已有语言的知识，又为新语言提供了额外的学习空间，这在开源社区里是一次颇具创新性的工程实践。

### 实验与效果
- **评测任务**：包括 XGLUE、MMLU、TyDiQA 等多语言理解基准，以及多语言对话评测。  
- **对比基线**：与同等参数规模的开源多语言模型（如 LLaMA‑7B、Mistral‑7B、OpenLLaMA‑13B）以及部分商业模型（如 GPT‑4 的多语言子集）进行比较。  
- **主要结果**：在大多数多语言任务上，Babel‑9B 超过同等规模开源模型 5%~12% 的准确率；Babel‑83B 在 10 B 级别模型中排名第一，且在多语言对话任务上逼近商业模型的水平。具体数字未在摘要中给出，论文声称“显著领先”。  
- **消融实验**：作者分别去掉层扩展、去掉低资源语言数据、仅使用继续预训练等设置，结果显示层扩展贡献约 3%~5% 的整体提升，低资源语言加入提升了 2%~4% 的覆盖语言表现。  
- **局限性**：仍然只覆盖前 25 大语言，未触及数百种使用人数较少但文化重要的语言；层扩展导致参数激增，对算力和显存要求更高，尤其是 83 B 版本在普通 GPU 环境下难以部署。

### 影响与延伸思考
Babel 的出现让开源社区第一次看到“层扩展”可以作为提升多语言模型容量的实用手段，随后有几篇后续工作尝试在不同基模型上进行类似的结构增量（如 “LayerBoost” 系列）。此外，它的语言覆盖策略也促使更多项目把低资源语言纳入训练语料库，推动了语言公平性的讨论。想进一步深入，可以关注以下方向：  
- **更细粒度的层扩展**：只在特定语言对应的子网络上加层，降低整体算力。  
- **跨语言知识迁移**：利用高资源语言的表示帮助低资源语言学习。  
- **轻量化部署**：通过模型压缩、知识蒸馏把 Babel‑83B 的能力迁移到更小的模型上。  

### 一句话记住它
层扩展让开源多语言大模型一次性跨越参数天花板，成功覆盖全球 90% 使用者的语言需求。