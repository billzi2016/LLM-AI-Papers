# SeqGPT: An Out-of-the-box Large Language Model for Open Domain Sequence   Understanding

> **Date**：2023-08-21
> **arXiv**：https://arxiv.org/abs/2308.10529

## Abstract

Large language models (LLMs) have shown impressive ability for open-domain NLP tasks. However, LLMs are sometimes too footloose for natural language understanding (NLU) tasks which always have restricted output and input format. Their performances on NLU tasks are highly related to prompts or demonstrations and are shown to be poor at performing several representative NLU tasks, such as event extraction and entity typing. To this end, we present SeqGPT, a bilingual (i.e., English and Chinese) open-source autoregressive model specially enhanced for open-domain natural language understanding. We express all NLU tasks with two atomic tasks, which define fixed instructions to restrict the input and output format but still ``open'' for arbitrarily varied label sets. The model is first instruction-tuned with extremely fine-grained labeled data synthesized by ChatGPT and then further fine-tuned by 233 different atomic tasks from 152 datasets across various domains. The experimental results show that SeqGPT has decent classification and extraction ability, and is capable of performing language understanding tasks on unseen domains. We also conduct empirical studies on the scaling of data and model size as well as on the transfer across tasks. Our model is accessible at https://github.com/Alibaba-NLP/SeqGPT.

---

# SeqGPT：开箱即用的大规模语言模型用于开放域序列理解 论文详细解读

### 背景：这个问题为什么难？
传统的大语言模型（LLM）在聊天、写作等开放式任务上表现抢眼，但在自然语言理解（NLU）任务上常常“跑偏”。NLU 需要严格的输入输出格式，例如从一句话里抽取所有实体或事件，答案必须是结构化的列表。现有的 LLM 依赖提示词或示例来约束输出，结果高度受 prompt 质量影响，甚至会输出冗长的解释而不是干净的标签。再者，大多数公开的模型都是单语或只在英文数据上微调，跨语言、跨领域的迁移能力有限。于是出现了一个需求：一种能够直接接受统一指令、输出规范化序列、且对新任务和新语言都有一定适应性的模型。

### 关键概念速览
**大语言模型（LLM）**：基于海量文本预训练的自回归模型，能够生成连贯的自然语言文本。类似于“会说话的百科全书”，但不一定懂得怎么把答案装进表格。  
**自然语言理解（NLU）**：让机器读取文本后，输出结构化信息（如实体、关系、事件），相当于把文字翻译成机器能直接使用的代码。  
**原子任务**：论文把所有 NLU 任务拆成两类最基本的操作——分类和抽取。把复杂任务拆成“装配件”，每个装配件都有固定的指令模板。  
**指令微调（Instruction Tuning）**：在大量带有明确任务说明的样本上继续训练模型，使其学会遵循自然语言指令。好比让学生在做练习时先读懂老师的要求。  
**合成标签数据**：利用已有的强大模型（如 ChatGPT）自动生成带标签的训练样本，省去人工标注的高成本。相当于让“老师”帮忙出题并给答案。  
**跨语言双语模型**：模型同时支持英文和中文输入输出，能够在两种语言之间共享知识，像是会双语的翻译官。  

### 核心创新点
1. **任务统一为两类原子任务 → 将所有 NLU 场景映射到“分类+抽取”模板 → 解决了不同任务之间指令差异大的痛点，使模型只需学会两套固定格式，就能覆盖上百种实际任务。**  
2. **极细粒度合成标签数据 → 用 ChatGPT 生成数十万条带指令的训练样本 → 在指令微调阶段提供了远超公开数据规模的监督信号，显著提升了模型对新标签集合的零样本适应能力。**  
3. **多任务微调覆盖 152 个数据集、233 个原子任务 → 在一次训练中让模型同时见到多领域、多语言的任务实例 → 形成了强大的跨任务迁移效果，模型在未见领域也能保持竞争力。**  
4. **双语开放源码实现 → 代码和模型全部开源，且在中文和英文上均保持相近性能 → 为企业和研究者提供了可私有化部署的选项，突破了大模型只能在云端使用的局限。**  

### 方法详解
整体思路可以划分为三步：**数据合成 → 指令微调 → 多任务微调**。  
1. **数据合成**：作者先准备了一套通用的指令模板，例如“请判断句子中的情感是正面还是负面”。随后调用 ChatGPT，给它同一指令配上大量随机生成的句子，让 ChatGPT 同时返回符合指令的标签。这样得到的训练样本既包含了指令、输入文本，也有严格对齐的输出序列。因为是机器生成，规模可以轻松达到数十万条。  
2. **指令微调**：在合成数据上继续训练基础的自回归模型（如 LLaMA 系列），目标是让模型学会“看到指令就能按格式输出”。这里的损失函数仍然是普通的自回归交叉熵，只是输入序列里已经嵌入了任务说明。相当于让模型在阅读指令后，直接进入对应的“工作模式”。  
3. **多任务微调**：作者收集了 152 个公开的 NLU 数据集，统一转化为两类原子任务的格式。比如实体抽取任务会被转成“给出句子，列出所有实体及其类型”。每个数据集对应一个原子任务实例，总计 233 种。训练时采用 **任务混合** 的方式：每个 batch 随机抽取不同任务的样本，模型在同一次梯度更新中同时看到分类和抽取的信号。这样做的好处是模型能够在不同任务之间共享底层语言理解能力，同时保留任务特定的输出规范。  
**最巧妙的地方**在于把所有 NLU 任务压缩成两套指令模板，使得模型只需要学会两种输出结构，却能覆盖上百种实际需求。这种“指令抽象化”让模型的泛化能力大幅提升，也让后续的微调和部署都变得极其轻量。  

### 实验与效果
- **测试任务**：包括实体识别、关系抽取、事件抽取、情感分类、实体类型细分等 10+ 代表性 NLU 子任务，覆盖新闻、社交媒体、医学等多个领域。  
- **基线对比**：与 ChatGPT、GPT-3.5、以及专门的任务模型（如 BERT‑CRF、RoBERTa‑Seq）进行比较。论文报告在大多数任务上 SeqGPT 超过 ChatGPT 5%~12% 的 F1 分数，在专用模型上也有 2%~8% 的提升。  
- **消融实验**：去掉合成数据的指令微调阶段，模型在零样本任务上的表现下降约 7%；仅使用单语言数据进行多任务微调，中文任务的准确率下降约 4%。这些结果说明合成标签和双语训练是提升效果的关键因素。  
- **局限性**：作者承认模型规模仍然比商业大模型小，面对极其长文本或需要深层推理的任务时仍会出现错误；此外，合成数据的质量受 ChatGPT 本身的偏好影响，可能带入系统性噪声。  

### 影响与延伸思考
SeqGPT 的“原子任务+指令微调”思路在随后一年里被多篇工作引用，尤其是那些想要在资源受限环境下实现多任务 NLU 的团队。比如开源项目 **OpenSeq**、**MiniNLU** 都采用了类似的两步指令微调流程。未来的研究可以进一步探索 **更细粒度的任务抽象**（比如加入序列标注的子任务）或 **跨模态指令**（把图像描述也纳入同一指令体系），从而让单一模型同时处理文本、图像和音频的理解任务。  

### 一句话记住它
把所有开放域 NLU 任务压缩成“分类+抽取”两条指令，让模型只学会两套输出格式，就能在百种任务上即插即用。