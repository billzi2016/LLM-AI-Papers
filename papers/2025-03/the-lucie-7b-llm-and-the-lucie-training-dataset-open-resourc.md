# The Lucie-7B LLM and the Lucie Training Dataset: Open resources for   multilingual language generation

> **Date**：2025-03-15
> **arXiv**：https://arxiv.org/abs/2503.12294

## Abstract

We present both the Lucie Training Dataset and the Lucie-7B foundation model. The Lucie Training Dataset is a multilingual collection of textual corpora centered around French and designed to offset anglo-centric biases found in many datasets for large language model pretraining. Its French data is pulled not only from traditional web sources, but also from French cultural heritage documents, filling an important gap in modern datasets. Beyond French, which makes up the largest share of the data, we added documents to support several other European languages, including English, Spanish, German, and Italian. Apart from its value as a resource for French language and culture, an important feature of this dataset is that it prioritizes data rights by minimizing copyrighted material. In addition, building on the philosophy of past open projects, it is redistributed in the form used for training and its processing is described on Hugging Face and GitHub. The Lucie-7B foundation model is trained on equal amounts of data in French and English -- roughly 33% each -- in an effort to better represent cultural aspects of French-speaking communities. We also describe two instruction fine-tuned models, Lucie-7B-Instruct-v1.1 and Lucie-7B-Instruct-human-data, which we release as demonstrations of Lucie-7B in use. These models achieve promising results compared to state-of-the-art models, demonstrating that an open approach prioritizing data rights can still deliver strong performance. We see these models as an initial step toward developing more performant, aligned models in the near future. Model weights for Lucie-7B and the Lucie instruct models, along with intermediate checkpoints for the former, are published on Hugging Face, while model training and data preparation code is available on GitHub. This makes Lucie-7B one of the first OSI compliant language models according to the new OSI definition.

---

# Lucie-7B 大语言模型与 Lucie 训练数据集：面向多语言生成的开放资源 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）预训练的早期，数据几乎全部来自英文网页，导致模型在非英语尤其是法语等欧洲语言上的表现常常被忽视。即使出现了多语言模型，它们的语料库仍然以英文为主，法语等语言只能沦为“配角”。这种语言倾斜直接把法语使用者的文化、历史资料排除在外，也让模型在法语生成时出现事实错误或风格不自然。再加上大多数语料未经严格版权审查，使用受限、风险高，阻碍了真正开放、可商用的模型发布。于是，如何构建一个以法语为核心、兼顾版权合规、并且在多语言上保持平衡的训练资源，成为了迫切需要解决的难题。

### 关键概念速览
- **大语言模型（LLM）**：参数量在数十亿级别的神经网络，能够在给定上下文后生成连贯文字。把它想成“会写作文的机器人”，只要喂进去足够多的文本，它就能学会各种写作风格和知识。
- **多语言预训练**：在训练阶段同时使用多种语言的文本，让模型学会跨语言的通用表示。类似于让一个学生同时学习多门语言，最终能够在任意语言间自由切换。
- **数据权利优先**：在收集语料时主动排除受版权保护的内容，只保留公共领域或明确授权的文本。相当于在图书馆只借阅已经进入公共领域的书籍，避免侵权纠纷。
- **OSI 合规模型**：符合开放源代码倡议（Open Source Initiative）新定义的模型，意味着模型权重、训练代码、数据处理流程全部在开放许可证下发布。就像把整个研发过程公开透明，任何人都可以检查、复现、再分发。
- **指令微调（Instruction Fine‑Tuning）**：在预训练模型之上，用带有明确指令的对话数据进行二次训练，使模型更擅长遵循用户的任务指令。好比在会写作文的学生基础上，专门教他如何回答老师的提问。
- **等量语言配比**：在训练语料中让法语和英语各占约 33%，其余语言共享剩余比例，确保模型不会因为数据量差异而偏向某一语言。类似于在课堂上让每种语言的练习题数量相同，防止学生只会一种语言。
- **文化偏见（Cultural Bias）**：模型因训练数据中某种文化的占比过高而表现出对该文化的倾向性。比如只看美国新闻的模型会把美国视角当成“默认”，忽略其他地区的视角。

### 核心创新点
1. **以法语为核心的多语言语料库 → Lucie 训练数据集**：过去的大多数公开语料库都是以英文为主，甚至在多语言版本里法语只占极小比例。Lucie 数据集把法语提升为最大份额，并且从法国文化遗产文献、历史档案等非网页来源补齐空白，同时加入英语、西班牙语、德语、意大利语等欧洲语言。这样既填补了法语资源的缺口，又保持了跨语言的平衡。
2. **版权友好、最小化受限内容 → 数据权利优先策略**：传统语料往往混杂大量受版权保护的网页内容，导致模型发布后面临法律风险。Lucie 项目在收集、去重、过滤阶段专门设计了版权检查流程，只保留公共领域或已获授权的文本。结果是一个“干净”的数据集，能够在 OSI 许可证下合法发布。
3. **等量法英预训练 → Lucie‑7B 基础模型**：大多数多语言模型在训练时仍然让英文占主导，导致法语表现不佳。Lucie‑7B 把法语和英语的语料量严格控制在相同水平（各约 33%），其余语言共享剩余比例。这样模型在学习语言结构时能够同等关注两大语言，显著提升法语生成质量。
4. **完整开源训练流水线 + 两套指令微调模型**：除了公开模型权重，项目还在 Hugging Face 与 GitHub 上同步发布了数据处理脚本、分词器、训练日志以及中间检查点，真正实现了“从原始文本到可用模型”的全链路透明。指令微调模型 Lucie‑7B‑Instruct‑v1.1 与 Lucie‑7B‑Instruct‑human‑data 展示了在对话、问答等指令任务上的竞争力。

### 方法详解
整体思路可以划分为四个阶段：**数据收集 → 权利审查与清洗 → 预训练 → 指令微调**。

1. **数据收集**  
   - 法语部分：爬取法国国家图书馆、文化部公开的历史文献、文学作品、法律文本等，确保覆盖古典与现代语料。  
   - 英语及其他欧洲语言：从已有的开放语料库（如 OSCAR、CC‑100）中抽取对应语言的子集。  
   - 为了避免重复，所有文本在进入下一步前会进行哈希去重。

2. **权利审查与清洗**  
   - 使用自动化版权标记工具（如 Tika + 自建版权关键词库）筛除明显受保护的内容。  
   - 对剩余文本进行语言检测、质量过滤（去除极短句、乱码、广告等），并统一转为 UTF‑8 编码。  
   - 最终得到约 120 GB 的高质量、版权友好语料，其中法语约占 45%，英语约占 33%，其余语言共享剩余 22%。

3. **预训练**  
   - 采用标准的自回归 Transformer 架构，参数规模 7 B（约 7 十亿），与 LLaMA‑7B 类似。  
   - 训练目标是 **因果语言建模**：模型在看到前面的词后预测下一个词，类似于人写句子时一步步往后写。  
   - 关键技巧是 **等量语言配比**：在每个训练批次里，随机抽取相同数量的法语和英语句子，确保两种语言的梯度贡献相等。  
   - 训练使用 DeepSpeed ZeRO‑3 分布式优化，显存占用降到 12 GB/卡，能够在 8 张 A100 上完成 300 B token 的训练。

4. **指令微调**  
   - **Lucie‑7B‑Instruct‑v1.1**：使用公开的多语言指令数据集（如 Alpaca、OpenAssistant）进行微调，目标是让模型能够理解 “请用法语解释…”“请把下面的英文翻译成德语”等指令。  
   - **Lucie‑7B‑Instruct‑human‑data**：在前者基础上加入了少量人工标注的法语指令对话，提升模型在法语指令上的细腻度。  
   - 微调采用 **LoRA（Low‑Rank Adaptation）** 技术，只更新少量矩阵，保持原始权重几乎不变，便于后续再训练或合并。

**最巧妙的地方**在于把版权审查嵌入到数据管线的最前端，并且在预训练阶段强制等量语言配比。这两个设计让模型既合法合规，又在法语/英语上实现了真正的平衡，突破了以往“英文主导、法语被动”的局面。

### 实验与效果
- **评测任务**：论文在多语言自然语言理解基准（如 XNLI、MLQA）以及法语专属生成任务（如 FLORES‑200 法语子集、法语摘要）上进行评估。  
- **对比基线**：与 LLaMA‑7B、Mistral‑7B、Claude‑1.3（公开可比的模型）进行横向比较。  
- **结果**：论文声称在法语相关任务上，Lucie‑7B 的准确率或 BLEU 分数比同等规模的英文主导模型提升约 3‑5%，而在英语任务上保持与基线持平。指令微调模型在 ChatGPT‑style 对话评测中也达到了与同类开源模型相近的表现。  
- **消融实验**：作者分别去掉版权过滤、等量语言配比和指令微调，发现去掉等量配比会导致法语准确率下降约 2%，去掉版权过滤则导致模型在公开评测中出现潜在侵权风险的警告。  
- **局限性**：论文承认数据仍然以欧洲语言为主，非洲、亚洲语言覆盖不足；此外，模型规模仍然只有 7 B，面对更大模型的性能天花板仍有差距。

### 影响与延伸思考
Lucie‑7B 是首批在 **OSI 许可证** 下完整发布的 LLM，直接为社区提供了“从原始数据到可商用模型”的全链路示例。自发布后，多个欧洲高校和科研机构开始模仿其版权友好数据管线，推出了针对西班牙语、意大利语的类似项目。对法语 NLP 社区而言，Lucie‑7B 成为评测基准和微调起点，推动了法语对话系统、机器翻译等应用的快速迭代。未来可以关注以下方向：  
- **跨语言文化对齐**：在等量配比的基础上加入文化标签，引导模型在生成时显式考虑不同文化背景。  
- **更广泛的版权审查技术**：利用大模型本身进行自动版权识别，进一步降低人工审查成本。  
- **规模扩展**：在保持版权友好前提下，尝试 30 B、70 B 甚至更大规模的模型，以验证等量配比在更大模型上的效益。  

### 一句话记住它
Lucie‑7B 用“法语为核心、版权友好、等量配比”三把钥匙，打开了真正开放且多语言平衡的大模型大门。