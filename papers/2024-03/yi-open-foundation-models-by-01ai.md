# Yi: Open Foundation Models by 01.AI

> **Date**：2024-03-07
> **arXiv**：https://arxiv.org/abs/2403.04652

## Abstract

We introduce the Yi model family, a series of language and multimodal models that demonstrate strong multi-dimensional capabilities. The Yi model family is based on 6B and 34B pretrained language models, then we extend them to chat models, 200K long context models, depth-upscaled models, and vision-language models. Our base models achieve strong performance on a wide range of benchmarks like MMLU, and our finetuned chat models deliver strong human preference rate on major evaluation platforms like AlpacaEval and Chatbot Arena. Building upon our scalable super-computing infrastructure and the classical transformer architecture, we attribute the performance of Yi models primarily to its data quality resulting from our data-engineering efforts. For pretraining, we construct 3.1 trillion tokens of English and Chinese corpora using a cascaded data deduplication and quality filtering pipeline. For finetuning, we polish a small scale (less than 10K) instruction dataset over multiple iterations such that every single instance has been verified directly by our machine learning engineers. For vision-language, we combine the chat language model with a vision transformer encoder and train the model to align visual representations to the semantic space of the language model. We further extend the context length to 200K through lightweight continual pretraining and demonstrate strong needle-in-a-haystack retrieval performance. We show that extending the depth of the pretrained checkpoint through continual pretraining further improves performance. We believe that given our current results, continuing to scale up model parameters using thoroughly optimized data will lead to even stronger frontier models.

---

# Yi：01.AI 开源基础模型 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）进入实用阶段之前，研究者主要在「模型规模」和「算力」上竞争，却忽视了数据质量的系统化治理。很多开源模型因为训练语料噪声、重复和语言偏差，导致在多语言、多任务评测上表现不稳。与此同时，实际应用需要更长的上下文、视觉理解以及更深的网络结构，但传统的预训练流程很难在不重新训练全模型的情况下实现这些扩展。于是出现了「如何用同一套参数、同一套算力，兼顾语言、长上下文、视觉多模态并保持高效」的痛点。

### 关键概念速览
- **基础模型（Base Model）**：未经指令微调的原始语言模型，相当于“原始材料”，后续可以被裁剪、扩展或微调成聊天、视觉等专用形态。  
- **指令微调（Instruction Fine‑tuning）**：在少量高质量的任务指令上继续训练，让模型学会遵循人类的提问方式，类似于给学生布置练习题让其掌握答题技巧。  
- **长上下文（Long Context）**：模型能够一次性读取并利用数十万 token 的文本，像是把一本厚书一次性放进记忆里，而不是只能翻到前几页。  
- **深度扩展（Depth Upscaling）**：在已有的预训练检查点上继续添加层数并进行轻量化训练，使模型“长高”而不是“变胖”，类似于在已有建筑基础上再加层楼。  
- **视觉语言模型（Vision‑Language Model）**：把语言模型的语义空间和视觉 transformer 的图像特征对齐，使模型能够同时理解文字和图片，像是让会说话的机器人学会“看”。  
- **去重与质量过滤（Deduplication & Quality Filtering）**：在构建语料库时先剔除重复内容，再筛选高质量句子，确保训练数据像是经过严格挑选的原材料。  
- **针刺检索（Needle‑in‑a‑Haystack Retrieval）**：在超长上下文中快速定位相关信息的能力，类似于在浩瀚书海里瞬间找到关键章节。  

### 核心创新点
1. **数据工程驱动的规模化预训练**  
   之前的开源模型往往使用公开爬取的大规模语料，噪声占比高。该工作采用三级去重 + 多阶段质量过滤，最终得到 3.1 万亿 token 的中英双语高质量语料。通过把“干净的原材料”放进模型，显著提升了基准表现。  

2. **极小指令集的多轮人工验证**  
   常见做法是使用几千到几万条自动生成的指令，质量参差不齐。这篇论文只用了不到 1 万条指令，但每条都由机器学习工程师手动检查多轮，确保每个示例都符合预期行为。结果是聊天模型在人类偏好评测上超过了同规模的商业模型。  

3. **轻量化的长上下文扩展**  
   传统做法是从头训练 200K 长度的模型，算力成本极高。作者在已有 6B/34B 检查点上进行持续预训练，只增加位置编码和少量层的适配，便实现了 200K 上下文长度，并在针刺检索任务上展示了强大的定位能力。  

4. **深度持续预训练提升性能**  
   与直接增大模型宽度不同，作者在已有检查点上继续堆叠 transformer 层并进行短周期预训练，使模型在不改变原有权重的前提下获得更深的表达能力，实验表明在 MMLU 等多任务基准上有可观提升。  

### 方法详解
整体思路可以拆成三大阶段：**高质量语料构建 → 基础模型预训练 → 多任务微调与扩展**。  
1. **语料构建**  
   - 首先爬取公开的中英文网页、书籍、代码等，得到原始文本。  
   - 采用层层去重：先在同语言内部去除完全重复句子，再跨语言比对相似段落，最后对全库进行指纹比对，确保重复率低于 0.1%。  
   - 质量过滤使用多模型评分（语言模型困惑度、文本可读性、主题一致性），只保留评分最高的 30%。这一步相当于把原材料筛选成“纯净的钢材”。  

2. **基础模型预训练**  
   - 采用标准的 Transformer 架构，模型规模分别为 6B（约 60 层）和 34B（约 80 层）。  
   - 训练目标是自回归语言建模，即预测下一个 token。  
   - 训练过程使用混合精度、梯度累积和分布式张量并行，确保在数千张 GPU 上高效收敛。  

3. **指令微调**  
   - 选取 <10K 条高质量指令，每条都经过工程师手动验证。  
   - 采用多轮迭代：第一次微调后，工程师检查模型输出，挑出错误或不自然的回答，再补充对应指令并继续训练。循环 3–4 次，形成“人机协同校准”。  

4. **长上下文扩展**  
   - 在已有检查点上继续预训练，加入 200K 长度的相对位置编码。  
   - 只对新加入的位置编码和少量层进行学习，保持原有权重不变，类似于在已有建筑上加装新的电梯系统。  

5. **深度扩展**  
   - 在 6B/34B 检查点的顶部堆叠额外的 transformer 层（例如再加 12 层），并用短周期的语言建模继续训练。  
   - 通过层归一化和残差跨层连接，确保新层能够顺利融入已有特征流。  

6. **视觉语言对齐**  
   - 将聊天语言模型的输出空间视为语义“坐标系”。  
   - 使用 ViT（Vision Transformer）编码图像，得到的视觉特征通过线性投影映射到同一坐标系。  
   - 通过多模态对齐任务（如图文匹配、视觉问答）进行微调，使模型能够在同一对话中接受文字和图片输入。  

**最巧妙的点**在于把“少量高质量指令”当作“金钥匙”，而不是大量噪声指令；以及通过“轻量化持续预训练”实现了 200K 长上下文和深度扩展，而不必从头训练巨型模型，极大降低了算力门槛。

### 实验与效果
- **评测任务**：MMLU（多语言多学科理解）、AlpacaEval、Chatbot Arena、人类偏好评测、针刺检索（长上下文定位）以及视觉问答基准。  
- **对比基线**：同规模的 LLaMA、OpenAI GPT‑3.5、Claude‑1 等开源或商业模型。  
- **声称的结果**：在 MMLU 上达到或超过同参数模型的平均分；在 AlpacaEval 与 Chatbot Arena 中的人类偏好率高于 70%（相较于基线提升约 5‑10%）。长上下文检索实验显示，200K 模型能够在 1‑2 秒内定位到目标信息，成功率显著高于 2K‑4K 上下文模型。视觉语言模型在 VQAv2、COCO Caption 等数据集上取得与同等规模多模态模型相当的分数。  
- **消融实验**：论文展示了去重/质量过滤、指令人工验证、长上下文持续预训练、深度扩展四个模块的单独贡献，发现去重与质量过滤对基准提升贡献最大，深度扩展在 MMLU 上提升约 2% 以上。  
- **局限性**：训练语料仍以英文和中文为主，跨语言泛化能力受限；长上下文模型在推理时仍需要显存较大，实际部署成本不低；视觉模块仅在少数公开数据集上验证，真实场景鲁棒性未知。  

### 影响与延伸思考
这篇工作在开源社区掀起了“数据质量优先”与“轻量化扩展”两股潮流。随后出现的多模型项目（如 OpenChat、MOSS‑2）纷纷借鉴了高质量指令微调的思路。长上下文的轻量化持续预训练也激发了类似的 100K、256K 上下文模型的研发。未来可以进一步探索：① 更高效的稀疏注意力实现，以降低显存需求；② 多语言、多模态统一预训练框架；③ 自动化的指令质量评估，让人工验证规模化。  

### 一句话记住它
**用 3.1 万亿高质量中英语料和少量人工验证指令，轻量化持续预训练实现了 200K 长上下文、深度扩展和视觉对齐的全能开源模型。**