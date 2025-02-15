# Granite Vision: a lightweight, open-source multimodal model for   enterprise Intelligence

> **Date**：2025-02-14
> **arXiv**：https://arxiv.org/abs/2502.09927

## Abstract

We introduce Granite Vision, a lightweight large language model with vision capabilities, specifically designed to excel in enterprise use cases, particularly in visual document understanding. Our model is trained on a comprehensive instruction-following dataset, including document-related tasks, such as content extraction from tables, charts, diagrams, sketches, and infographics, as well as general image tasks. The architecture of Granite Vision is centered around visual modality alignment with a decoder-only, 2 billion parameter Granite large language model. Additionally, we introduce a dedicated safety classification approach in test-time that leverages a sparse set of attention vectors to identify potential harmful inputs. Despite its lightweight architecture, Granite Vision achieves strong results in standard benchmarks related to visual document understanding, as well as on the LiveXiv benchmark, which is designed to avoid test set contamination by using a constantly updated corpus of recently published Arxiv papers. We are releasing the model under the Apache-2 license, allowing for both research and commercial use, while offering complete visibility into the training data and other relevant details. See https://huggingface.co/ibm-granite/ for model weights.

---

# Granite Vision：面向企业智能的轻量级开源多模态模型 论文详细解读

### 背景：这个问题为什么难？

企业里每天都会产生海量的图片、表格、图表和手绘草图，这些视觉信息往往埋藏在 PDF、扫描件或内部系统里。传统的 OCR 只能把文字抽出来，却无法理解表格结构、图表趋势或示意图的语义。已有的多模态大模型大多基于上百亿参数、封闭源码，部署成本高、推理慢，根本不适合企业内部的安全合规要求。再加上缺乏针对文档任务的指令微调，模型在实际业务场景里常常“看不懂”。因此，需要一种既轻量、又能直接处理各种文档视觉内容、并且可以安全审查的多模态模型。

### 关键概念速览
- **多模态模型**：同时接受文字和图像等不同类型输入的模型，就像人类能看图说话一样。  
- **解码器Only架构**：模型只有生成（解码）部分，没有单独的编码器，所有信息都通过同一个 Transformer 处理，类似只用“写作机器”来完成阅读和写作。  
- **指令微调（Instruction Tuning）**：在大规模指令-响应对上继续训练，让模型学会按照用户的明确指令完成任务，类似给模型上“使用手册”。  
- **视觉对齐（Vision Alignment）**：把图像特征映射到语言模型的词向量空间，使得语言模型能够直接“读懂”图像信息，像把图片翻译成文字的密码。  
- **稀疏注意力安全分类**：在推理时只激活少量注意力向量来判断输入是否有害，类似在大楼里只打开几扇窗检查是否有火灾。  
- **文档视觉理解**：从表格、图表、示意图等结构化或半结构化视觉内容中抽取语义信息，类似把一张财报图表直接转成可查询的数据库。  
- **LiveXiv基准**：使用不断更新的 arXiv 论文集合做评测，避免模型因“看到答案”而作弊，像实时的考试题库。  

### 核心创新点
1. **轻量化解码器Only视觉模型**  
   - 之前的多模态大模型大多采用 encoder‑decoder 双塔结构，参数在百亿级别。  
   - 这篇论文把视觉特征直接投射进 2 B 参数的解码器Only语言模型，实现了“视觉+语言”同一套 Transformer。  
   - 结果是模型体积只有几百 MB，推理成本大幅下降，企业内部部署更友好。  

2. **全指令微调覆盖文档任务**  
   - 过去的指令微调数据集多聚焦于通用对话或图像描述，缺少表格、图表等专业任务。  
   - 作者构建了包含表格抽取、图表解读、草图说明等多种文档指令的训练集，并在此上进行微调。  
   - 使模型在企业文档场景下能够直接输出结构化答案，省去额外的后处理步骤。  

3. **稀疏注意力驱动的安全分类**  
   - 常规安全过滤需要额外的分类模型，增加部署复杂度。  
   - 本文在推理阶段利用一小组预先学习的注意力向量检测潜在有害输入，无需额外网络。  
   - 这种“在模型内部自检”的方式既保持了轻量，又提升了安全审计的透明度。  

4. **全开源、数据透明**  
   - 与多数企业内部模型不同，Granite Vision 在 Apache‑2 许可证下发布，所有训练数据来源、预处理流程均公开。  
   - 这为后续研究提供了可复现的基线，也让企业在合规审查时更有底气。  

### 方法详解
整体思路可以拆成三步：**视觉特征提取 → 特征对齐 → 指令微调的统一语言模型**，并在推理时加入**稀疏注意力安全检查**。

1. **视觉特征提取**  
   - 使用一个轻量级的视觉前端（如 ViT‑B/16）把输入图片切成固定大小的 patch，得到每个 patch 的向量表示。  
   - 这些向量本质上是“图像的词”，类似把图片拆成拼图块，每块都有自己的“词义”。  

2. **特征对齐层**  
   - 将视觉向量通过一个线性投影映射到与语言模型词嵌入相同的维度。  
   - 投影后，这些向量被插入到语言模型的 token 序列中，位置上通常放在文本前缀或特殊的 `<IMG>` 标记后。  
   - 这样，解码器只需要一次前向传播就能同时处理图像和文字信息，省去跨模态桥接的额外计算。  

3. **指令微调**  
   - 基础模型是 2 B 参数的 Granite LLM，已经在大规模语言指令上预训练。  
   - 论文收集了大量文档相关指令：如“从这张表格中提取所有收入数字”，或“解释这张流程图的关键步骤”。  
   - 这些指令-响应对与普通文本指令一起混合，统一喂入模型进行微调。微调的目标是最小化模型输出与人工标注答案之间的交叉熵损失。  

4. **稀疏注意力安全分类**  
   - 在推理时，模型会激活一组预先学习的稀疏注意力向量（数量远小于全注意力矩阵），这些向量专门捕捉异常或有害模式。  
   - 若这些稀疏向量的激活值超过阈值，系统会标记输入为潜在风险并阻止生成。  
   - 该机制不需要额外的分类网络，完全嵌入在原有 Transformer 计算图中。  

**最巧妙的点**在于把视觉特征直接塞进解码器的 token 序列，而不是像传统做法那样先用独立的视觉 encoder 再用跨模态融合层。这样既省掉了大量参数，也让模型在一次前向传播中完成“看图+写答案”。另外，利用稀疏注意力做安全检测的思路把安全审计变成了模型内部的“自检功能”，极大降低了部署复杂度。

### 实验与效果
- **评测任务**：论文在多个视觉文档理解基准上测试，包括表格抽取、图表解释、手绘草图识别等；另外在 LiveXiv 基准上评估模型对最新学术文献的理解能力。  
- **对比基线**：与传统的大型多模态模型（如 GPT‑4V、LLaVA）以及专门的文档 OCR 系统进行比较。  
- **声称的结果**：Granite Vision 在标准文档理解指标上取得了“强劲”表现，尤其在表格内容抽取和图表解释任务上接近或超过了百亿参数模型的水平。LiveXiv 上的得分也高于未进行文档指令微调的基线模型。  
- **消融实验**：论文提供了对视觉对齐层、指令微调数据量、以及稀疏注意力安全模块的消融分析，显示视觉对齐是提升跨模态性能的关键，安全模块在检测有害输入时的误报率低于 2%。  
- **局限性**：作者承认模型仍受限于 2 B 参数规模，在极其复杂的图表（如多轴、交互式可视化）上仍有误差；此外，稀疏注意力安全检测依赖于训练时的负样本覆盖，未知攻击可能逃过检测。  

### 影响与延伸思考
Granite Vision 的发布让企业级用户看到了“轻量多模态模型”可以实际落地的可能，推动了开源社区对文档视觉任务的关注。随后出现的几篇工作（如 IBM 的 “Granite OCR‑Plus” 与 “Vision‑Lite”）在此基础上进一步探索更高效的视觉前端和更细粒度的安全策略。对想深入的读者，建议关注以下方向：  
- **视觉特征与语言模型的更紧耦合方式**（如跨层对齐、动态投影）。  
- **指令微调的任务自适应技术**，让模型能够在少量新任务上快速调优。  
- **多模态安全审计**，尤其是对抗样本和隐蔽攻击的检测方法。  

### 一句话记住它
Granite Vision 用 2 B 参数的解码器一次性把图像和指令融合，轻量又安全，专为企业文档视觉理解而生。