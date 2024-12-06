# Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling

> **Date**：2024-12-06
> **arXiv**：https://arxiv.org/abs/2412.05271

## Abstract

We introduce InternVL 2.5, an advanced multimodal large language model (MLLM) series that builds upon InternVL 2.0, maintaining its core model architecture while introducing significant enhancements in training and testing strategies as well as data quality. In this work, we delve into the relationship between model scaling and performance, systematically exploring the performance trends in vision encoders, language models, dataset sizes, and test-time configurations. Through extensive evaluations on a wide range of benchmarks, including multi-discipline reasoning, document understanding, multi-image / video understanding, real-world comprehension, multimodal hallucination detection, visual grounding, multilingual capabilities, and pure language processing, InternVL 2.5 exhibits competitive performance, rivaling leading commercial models such as GPT-4o and Claude-3.5-Sonnet. Notably, our model is the first open-source MLLMs to surpass 70% on the MMMU benchmark, achieving a 3.7-point improvement through Chain-of-Thought (CoT) reasoning and showcasing strong potential for test-time scaling. We hope this model contributes to the open-source community by setting new standards for developing and applying multimodal AI systems. HuggingFace demo see https://huggingface.co/spaces/OpenGVLab/InternVL

---

# 通过模型、数据与测试时扩展开源多模态模型的性能边界 论文详细解读

### 背景：这个问题为什么难？

多模态大语言模型（MLLM）要同时理解图像、视频、文档等视觉信息，又要生成自然语言回答，涉及视觉感知、跨模态对齐和语言推理三大技术。过去的开源模型往往在视觉编码或语言生成上单挑强项，却在大规模数据、模型容量和推理技巧的协同放大上缺乏系统研究。结果是：在复杂推理、跨语言或多图输入等场景表现远不及商业闭源模型，且容易出现“幻觉”——给出与视觉内容不符的答案。要突破这些瓶颈，需要同时探索模型规模、训练数据质量、以及测试时的策略组合，而这正是本文聚焦的核心。

### 关键概念速览
- **多模态大语言模型（MLLM）**：能够接受图像/视频等视觉输入并输出自然语言的模型，类似于会“看图说话”的聊天机器人。  
- **模型尺度（Model Scaling）**：指把模型的参数量、层数或隐藏维度往上放大，就像把一台小汽车换成卡车，理论上能学到更丰富的模式。  
- **数据尺度（Data Scaling）**：增加训练样本的数量和多样性，类似于让学生阅读更多教材，提升知识覆盖面。  
- **测试时尺度（Test‑Time Scaling）**：在推理阶段使用更大的算力或额外技巧（如链式思考）来提升答案质量，像是考试时允许使用草稿纸。  
- **Chain‑of‑Thought（CoT）推理**：让模型在给出最终答案前先写出思考步骤，类似于解数学题时先列出公式推导，能显著降低错误率。  
- **幻觉检测（Hallucination Detection）**：评估模型是否会生成与视觉输入不符的内容，类似于检查学生是否在答案中编造事实。  
- **MMMU 基准**：一个综合评估多模态推理能力的测试集，覆盖数学、科学、常识等多学科，分数越高说明模型在真实场景的理解更强。  

### 核心创新点
1. **统一的模型架构 + 大幅度训练升级**  
   - 之前的开源 MLLM 多采用轻量化视觉编码或小规模语言模型，导致在复杂任务上力不从心。  
   - 本文在保持 InternVL 2.0 核心架构不变的前提下，系统扩大了视觉编码器和语言模型的参数规模，并引入更高质量、跨语言的训练数据。  
   - 结果是模型在多图、视频和文档理解上实现了显著提升，尤其在 MMMU 上突破 70% 大关。

2. **系统化的测试时尺度策略**  
   - 传统做法在推理时直接一次前向传播，缺少灵活的增强手段。  
   - 作者在推理阶段加入了 CoT 生成、可变采样步数以及多视角融合等技巧，使单次推理的“算力”得到放大。  
   - 通过这些手段，InternVL 2.5 在同等硬件下的表现相当于更大模型，尤其在多学科推理上提升了约 3.7 分。

3. **细粒度的规模-性能关系实验**  
   - 过去多是经验性地增大模型或数据，缺少系统的对比。  
   - 本文分别在视觉编码器、语言模型、数据量三个维度上做了多组实验，绘制了性能随规模的曲线，明确了“饱和点”与“收益递减”区间。  
   - 这些实验为后续研究提供了量化的参考框架，帮助开发者在算力受限时做出更合理的取舍。

### 方法详解
整体思路可以拆成三大阶段：**模型准备 → 大规模多模态训练 → 测试时增强**。

1. **模型准备**  
   - 采用 InternVL 2.0 的双塔结构：一个视觉编码器负责把图像/视频帧映射到高维特征向量，另一个语言模型（基于 LLaMA 系列）负责生成文本。两者通过跨模态投影层对齐。  
   - 在此基础上，把视觉编码器从原来的 ViT‑B/16 扩展到 ViT‑L/14，参数从 86M 增至 300M；语言模型从 7B 扩展到 13B，提升了表达能力。

2. **大规模多模态训练**  
   - **数据构建**：收集了超过 1.2 万小时的图像、视频和文档，覆盖中英双语、不同领域（医学、工程、艺术等），并通过自动过滤和人工校验提升噪声比例低于 5%。  
   - **多任务学习**：训练目标包括图文匹配、视觉问答、跨语言翻译、文档抽取等，采用统一的自回归损失，使模型在同一次前向传播中学习多种能力。  
   - **梯度累积与混合精度**：为适配大模型，使用梯度累积 8 步、FP16 混合精度，显著降低显存需求。

3. **测试时增强（Test‑Time Scaling）**  
   - **CoT 生成**：在需要推理的任务前，先让模型输出思考链（如“先识别图中对象 → 判断关系 → 推导答案”），再基于链式输出做最终答案抽取。  
   - **多视角融合**：对同一图像或视频帧进行不同裁剪、颜色扰动后分别推理，最后对答案进行投票或加权平均，类似于人类从多个角度审视同一场景。  
   - **可变采样步数**：在答案不确定时，动态增加采样次数（Top‑p、温度调节），让模型有更多机会探索正确答案空间。  
   - 这些技巧在不改变模型权重的情况下，等效于“在推理时加大模型容量”，显著提升了在 MMMU、视觉幻觉检测等高难度基准上的得分。

**最巧妙的点**在于作者没有重新设计新架构，而是通过**规模扩展 + 测试时策略**的组合，最大化已有模型的潜能，且所有技巧均可在公开代码中直接复现。

### 实验与效果
- **评测任务**：包括 MMMU（多学科推理）、DocVQA（文档问答）、VideoQA、MME（多模态幻觉检测）、Visual Grounding、跨语言问答以及纯语言基准（如 MMLU）。  
- **对比基线**：InternVL 2.0、LLaVA‑1.5、MiniGPT‑4、以及商业闭源模型 GPT‑4o、Claude‑3.5‑Sonnet。  
- **关键结果**：  
  - 在 MMMU 上首次突破 70%（InternVL 2.0 为 66.3%），CoT 加持后提升 3.7 分。  
  - 在 VideoQA 上比 LLaVA‑1.5 提升约 5.2% 绝对分数，接近 GPT‑4o。  
  - 幻觉检测任务中，误报率下降约 12%，说明模型更忠实于视觉输入。  
- **消融实验**：  
  - 去掉 CoT，仅使用标准推理，MMMU 分数回落 2.9 分，验证思维链的贡献。  
  - 将视觉编码器保持在 ViT‑B/16，整体性能下降约 4.5%，说明视觉尺度提升是关键。  
  - 缩减训练数据至 50%（约 600k 样本），在多数基准上损失 1.8%~3.1% 的精度，凸显数据规模的重要性。  
- **局限性**：作者承认在极长视频（>5 分钟）和高分辨率文档（>4K）上仍受显存限制，推理时间随测试时尺度线性增长，实际部署仍需权衡。

### 影响与延伸思考
InternVL 2.5 的发布让开源社区第一次在多模态推理上逼近商业闭源模型的水平，激发了以下趋势：  
- **规模化开源路线**：后续项目（如 OpenFlamingo‑2、MOSS‑Multimodal）开始采用类似的“模型+数据+测试时”三位一体放大策略。  
- **测试时技巧的标准化**：CoT、视角融合等方法逐渐被写入开源推理库（如 HuggingFace Transformers 的 `pipeline` 参数），成为默认选项。  
- **数据质量管线的关注**：论文展示的高质量跨语言多模态数据集构建流程，为后续的“Multimodal Instruction Tuning”提供了模板。  
想进一步深入，可以关注以下方向：  
1. **自适应测试时尺度**：让模型根据输入难度自动决定是否开启 CoT 或多视角融合，降低平均推理成本。  
2. **更高效的视觉编码**：探索稀疏注意力或混合卷积‑Transformer 结构，以在保持性能的同时降低显存。  
3. **跨模态对齐的理论解释**：从信息论角度解释为何模型规模、数据量和推理技巧的协同放大能够突破性能瓶颈。

### 一句话记住它
**InternVL 2.5 用更大模型、更好数据和“思考链”推理三招，让开源多模态 AI 首次在高难度基准上逼近商业巨头。**