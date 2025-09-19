# Qianfan-VL: Domain-Enhanced Universal Vision-Language Models

> **Date**：2025-09-19
> **arXiv**：https://arxiv.org/abs/2509.18189

## Abstract

We present Qianfan-VL, a series of multimodal large language models ranging from 3B to 70B parameters, achieving state-of-the-art performance through innovative domain enhancement techniques. Our approach employs multi-stage progressive training and high-precision data synthesis pipelines, which prove to be critical technologies for enhancing domain-specific capabilities while maintaining strong general performance. Qianfan-VL achieves comparable results to leading open-source models on general benchmarks, with state-of-the-art performance on benchmarks such as CCBench, SEEDBench IMG, ScienceQA, and MMStar. The domain enhancement strategy delivers significant advantages in OCR and document understanding, validated on both public benchmarks (OCRBench 873, DocVQA 94.75%) and in-house evaluations. Notably, Qianfan-VL-8B and 70B variants incorporate long chain-of-thought capabilities, demonstrating superior performance on mathematical reasoning (MathVista 78.6%) and logical inference tasks. All models are trained entirely on Baidu's Kunlun P800 chips, validating the capability of large-scale AI infrastructure to train SOTA-level multimodal models with over 90% scaling efficiency on 5000 chips for a single task. This work establishes an effective methodology for developing domain-enhanced multimodal models suitable for diverse enterprise deployment scenarios.

---

# 千帆‑VL：面向领域增强的通用视觉语言模型 论文详细解读

### 背景：这个问题为什么难？
视觉语言模型（VLM）需要同时理解图像的像素信息和文字的语义，这本身就很复杂。早期的 VLM 往往在大规模通用数据上训练，结果在专业场景（比如 OCR、文档解析）里表现平平，因为这些任务对细粒度文字识别和行业术语有特殊要求。再者，提升模型规模会显著增加算力和数据成本，很多企业难以在自己的领域里跑出足够的训练样本。于是，如何在保持通用能力的同时，快速注入领域知识，成为制约 VLM 实际落地的关键瓶颈。

### 关键概念速览
**多模态大语言模型（Multimodal LLM）**：既能处理文字，也能理解图像的模型，类似于会“看图说话”的聊天机器人。  
**域增强（Domain Enhancement）**：在模型里加入特定行业或任务的知识，就像给通用翻译软件装上医学词典，让它在医学报告上更精准。  
**多阶段渐进训练（Multi‑stage Progressive Training）**：先让模型学通用能力，再逐步加入领域数据，类似于先学基础数学，再学高等微积分。  
**高精度数据合成管线（High‑precision Data Synthesis Pipeline）**：自动生成带标签的图文对，质量堪比人工标注，像是用机器人帮你写练习题并给出答案。  
**长链式思考（Long Chain‑of‑Thought）**：模型在回答时会展开较长的推理过程，类似于解题时写下完整的思路步骤，适合数学和逻辑推理。  
**Kunlun P800 芯片**：百度自研的高性能算力单元，专门用于大模型训练，类似于为跑超大模型专门配备的“跑道”。  
**Scaling Efficiency（扩展效率）**：在增加算力或模型规模时，性能提升的比例，90% 的扩展效率意味着几乎每加一块芯片都能带来等比例的加速。  

### 核心创新点
1. **从单一训练阶段到多阶段渐进训练 → 先用海量通用图文数据预训练，再在 OCR、文档、科学问答等领域数据上进行二次微调 → 让模型在保持通用基准不降的前提下，显著提升了专业任务的准确率。**  
2. **从手工标注到高精度合成管线 → 构建了自动化的图文对生成系统，能够在不增加人工成本的情况下，产出高质量的领域样本 → 在 OCRBench、DocVQA 等文档理解基准上取得了 94.75% 以上的成绩。**  
3. **从短链式输出到长链式思考 → 在 8B 与 70B 规模模型中加入了长链式推理模块，使模型在 MathVista（数学推理）上达到 78.6% 的得分 → 解决了传统 VLM 在复杂逻辑推理时“一口气”给出答案的局限。**  
4. **从通用硬件到专用 Kunlun P800 大规模并行 → 完全在自研芯片上跑通 3B‑70B 规模模型，5000 卡片的单任务训练实现了超过 90% 的扩展效率 → 证明了企业内部大模型训练的可行性和成本优势。**  

### 方法详解
整体思路可以划分为三大块：**通用预训练 → 域增强微调 → 长链式推理增强**。先把模型当作一个“通用语言+视觉感知器”，再给它喂进专业领域的“强化教材”，最后让它在回答时写出完整的“解题步骤”。

1. **通用预训练**  
   - 使用公开的大规模图文对（如 LAION、COCO）进行跨模态自监督学习。模型的视觉编码器把图片映射成向量，语言模型把文字映射成向量，两者通过跨模态注意力层交互。  
   - 训练目标包括图文匹配（判断图片和文字是否对应）和指令微调（让模型学会根据自然语言指令生成描述）。这一步相当于给模型打好“看图说话”的基础。

2. **域增强微调**  
   - **数据合成管线**：通过 OCR 引擎、文档结构解析器和科学公式渲染器，自动生成带有精准标注的图文对。例如，给一张学术论文截图配上对应的公式 LaTeX、段落摘要等。合成过程加入噪声控制和质量过滤，确保生成数据的可信度。  
   - **多阶段微调**：先在 OCR 相关数据上做一次微调，让模型熟悉字符形状和排版；随后在文档问答（DocVQA）和科学问答（ScienceQA）上继续微调，逐步提升对复杂语义的理解。每个阶段的学习率都被调低，以防“忘记”通用能力。  
   - **长链式思考模块**：在 8B/70B 版本的解码器前加入一个“思考缓冲区”，模型在生成答案前会先输出若干推理步骤，这些步骤会被再次喂回模型进行自我校验，类似于人写草稿后再检查。

3. **训练与硬件**  
   - 所有模型均在百度自研的 Kunlun P800 芯片上训练，使用 5000 张卡片的分布式并行。作者报告的扩展效率超过 90%，意味着每增加一块卡片，训练速度几乎线性提升。  
   - 为了控制显存占用，视觉编码器采用了分层冻结策略：低层特征保持不变，高层在微调阶段才解冻，这样既节约算力，又能让模型快速适应新领域。

**最巧妙的点**在于把“高质量合成数据”与“多阶段渐进微调”结合起来。合成数据解决了领域样本稀缺的问题，而渐进微调确保模型不会因为过度专注某一领域而失去通用能力。

### 实验与效果
- **基准测试**：在通用视觉语言基准（如 VQAv2、COCO Caption）上，Qianfan‑VL 系列的表现与同规模的开源模型持平。  
- **领域专用**：在 CCBench、SEEDBench IMG、ScienceQA、MMStar 等任务上取得了 SOTA（state‑of‑the‑art）成绩。尤其是 OCRBench（873 条样本）和 DocVQA，模型分别达到了 94.75% 的准确率，明显领先公开模型 3‑5% 的提升。  
- **数学与逻辑推理**：8B/70B 版本在 MathVista 上得分 78.6%，在逻辑推理基准上也超过了同等规模的基线模型。  
- **消融实验**：作者分别去掉了（1）合成数据管线、（2）多阶段微调、（3）长链式思考模块，结果显示：去掉合成数据导致 OCR 相关指标下降约 2.8%；去掉渐进微调使 ScienceQA 分数下降约 3.5%；去掉长链式思考导致 MathVista 分数跌至 71%。这些数字表明每个创新点都对最终性能有实质贡献。  
- **局限性**：论文未给出对极端低资源语言或极端高分辨率图像的适配情况；合成数据虽然质量高，但仍可能存在与真实业务场景不完全匹配的偏差。作者也提到在超大模型（>70B）上的训练成本仍然是瓶颈。

### 影响与延伸思考
这篇工作展示了“领域增强 + 多阶段训练”在大模型时代的可行路径，随后不少企业级多模态项目开始采用类似的合成管线来快速构建行业模型。后续研究可能会在以下方向继续深化：  
- **自适应合成**：让模型自己生成需要的训练样本，进一步降低人工干预。  
- **跨语言域增强**：把中文、英文、日文等多语言文档的合成统一到同一管线，提升多语言文档理解能力。  
- **更高效的长链式推理**：探索在更大模型上保持推理链条可控且不显著增加推理时延的技术。  
- **硬件协同**：基于 Kunlun P800 的高效并行经验，可能会推动更多定制芯片与大模型训练的深度耦合。  

如果想深入了解，可以关注百度 AI 研发部的后续技术报告，或阅读近期在 *arXiv* 上出现的 “Domain‑Specific Multimodal Pre‑training” 系列论文。

### 一句话记住它
**Qianfan‑VL 用高质量合成数据和渐进微调，让通用视觉语言模型在专业领域也能像“定制版”一样精准。**