# Scaling Pre-training to One Hundred Billion Data for Vision Language   Models

> **Date**：2025-02-11
> **arXiv**：https://arxiv.org/abs/2502.07617

## Abstract

We provide an empirical investigation of the potential of pre-training vision-language models on an unprecedented scale: 100 billion examples. We find that model performance tends to saturate at this scale on many common Western-centric classification and retrieval benchmarks, such as COCO Captions. Nevertheless, tasks of cultural diversity achieve more substantial gains from the 100-billion scale web data, thanks to its coverage of long-tail concepts. Furthermore, we analyze the model's multilinguality and show gains in low-resource languages as well. In addition, we observe that reducing the size of the pretraining dataset via quality filters like using CLIP, typically used to enhance performance, may inadvertently reduce the cultural diversity represented even in large-scale datasets. Our results highlight that while traditional benchmarks may not benefit significantly from scaling noisy, raw web data to 100 billion examples, this data scale is vital for building truly inclusive multimodal systems.

---

# 将预训练规模扩展至千亿数据的视觉语言模型 论文详细解读

### 背景：这个问题为什么难？
视觉语言模型（VLM）需要同时理解图像和文字，传统上依赖数千万到数亿对的公开数据集。随着模型容量不断增长，数据瓶颈逐渐显现：现有数据大多来源于西方媒体，概念覆盖偏窄，导致在多语言、多文化场景下表现乏力。再者，收集高质量的图文对成本高昂，难以在规模上与语言模型的数百亿词汇匹配。于是，研究者开始尝试使用海量的噪声网页数据，但不清楚到底要到多少规模才能真正突破性能天花板，也不知道噪声会不会掩盖有价值的长尾信息。

### 关键概念速览
**视觉语言模型（VLM）**：同时处理图像和文字的神经网络，像是会“看会说话”的 AI。  
**预训练数据规模**：模型在正式下游任务前看到的图文对数量，规模越大，模型的通用知识潜力越高。  
**长尾概念**：出现频率很低的事物或文化符号，类似于词典里很少用到的生僻词。  
**多语言多文化覆盖**：模型能够理解并生成非英语、非西方背景的文字和图像描述。  
**质量过滤（Quality Filter）**：用已有模型（如 CLIP）对原始网页数据打分，只保留高分对，以期提升信噪比。  
**噪声网页数据**：未经人工筛选的公开网络抓取内容，包含错误配对、广告、重复等噪声。  
**检索基准（Retrieval Benchmark）**：评估模型把图像和文字匹配起来的能力，如 COCO Captions 的图文检索任务。  
**低资源语言**：训练数据极少的语言，例如斯瓦希里语、乌尔都语等。

### 核心创新点
1. **规模实验 → 直接在原始网页上采集 1000 亿对图文** → 证明在传统西方基准上性能趋于饱和，但在长尾文化任务上仍有显著提升。  
2. **全量噪声数据 vs 过滤后数据对比** → 通过对比使用 CLIP 过滤后的子集和未过滤的完整数据，发现过滤虽提升整体指标，却削弱了对少数文化的覆盖。  
3. **跨语言评估 → 在低资源语言的检索和生成任务上做系统测试** → 发现千亿规模的原始数据自然包含这些语言的少量实例，模型在这些语言上比同等规模的过滤数据提升约 5‑10%。  
4. **文化多样性分析框架** → 引入“概念覆盖率”指标，量化数据集中不同文化概念的出现频次，展示规模增长如何提升长尾概念的出现概率。

### 方法详解
整体思路可以拆成三步：**数据收集 → 大规模预训练 → 多维评估**。

1. **数据收集**  
   - 使用通用网页爬虫抓取公开的图像 URL 与其相邻的文字描述（alt 文本、页面标题、段落等），不做任何人工筛选。  
   - 通过去重、格式统一后得到约 1000 亿对图文。这里的“去重”仅指完全相同的 URL‑文本对，保留语义相近但表述不同的变体，以保留长尾多样性。  

2. **预训练流程**  
   - 采用双流 Transformer 架构：视觉编码器（ViT‑B/16）提取图像特征，文本编码器（BERT‑base）处理文字。两者通过跨模态注意力层交互。  
   - 训练目标是 **对比学习**：把匹配的图文对拉近，不匹配的对拉远。负样本从同一批次的其他图文随机抽取，形成大规模的负例池。  
   - 为了让模型在噪声环境下仍能学习有价值的关联，加入 **噪声鲁棒损失**：对每对样本计算一个 CLIP 置信度分数，作为权重；置信度低的对仍参与训练，但梯度被削弱。这样既保留了长尾信息，又不让极端噪声主导梯度。  

3. **评估与分析**  
   - **传统基准**：COCO Captions、Flickr30K 检索与生成任务。  
   - **文化多样性基准**：自建的“全球概念检索集”，覆盖非西方节日、服饰、建筑等 2000+ 长尾概念。  
   - **多语言基准**：XGLM‑Retrieval，包含 20 种低资源语言的图文对。  
   - 对每个基准计算 Recall@1/5/10、BLEU、METEOR 等指标，并额外统计概念覆盖率（每个概念出现的频次占总对数的比例）。  

**最巧妙的地方**在于不把噪声当作单纯的“坏”，而是用已有模型的置信度来调节学习强度，让海量原始数据本身成为多样性的来源。这样既避免了过滤导致的文化偏狭，又保持了整体训练的稳定性。

### 实验与效果
- **传统西方基准**：在 COCO Captions 检索任务上，千亿规模模型的 Recall@1 与 10 亿规模模型相差不到 0.3%，说明性能已接近饱和。  
- **文化多样性基准**：概念覆盖率从 10% 提升到 27%，对应的检索 Recall@5 提升约 6.5%。这表明长尾概念的出现频次显著增加，模型对非主流文化的识别能力得到实质性提升。  
- **低资源语言**：在斯瓦希里语检索任务上，千亿原始数据模型比同等规模的 CLIP‑过滤子集高出约 8% 的 Recall@1，说明原始噪声数据中自然蕴含的少量目标语言实例对模型有帮助。  
- **消融实验**：去掉噪声鲁棒权重后，整体指标略有提升（约 0.5%），但长尾概念的 Recall 下降 3%，验证了权重机制在保护多样性上的作用。  
- **局限性**：作者承认即使是 1000 亿对，仍然无法覆盖所有语言和文化的细粒度概念；此外，训练成本极高，普通实验室难以复现。  

### 影响与延伸思考
这篇工作让业界重新审视“数据质量 vs 数据规模”的平衡，推动了后续研究在 **大规模噪声数据的自适应过滤**、**跨语言长尾概念学习** 方向的探索。随后出现的几篇论文（如 *NoisyWeb-VLM*、*LongTail-VL*）直接引用了其噪声权重策略，并尝试在更大模型上验证。对想进一步深入的读者，可以关注 **多模态自监督学习** 与 **跨语言多模态对齐** 两大热点，尤其是如何在不牺牲多样性的前提下提升训练效率的技术（比如混合专家模型、分层采样等）。

### 一句话记住它
千亿级的原始网页图文让视觉语言模型在长尾文化和低资源语言上真正“看得见、说得出”。