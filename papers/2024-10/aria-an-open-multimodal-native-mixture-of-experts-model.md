# Aria: An Open Multimodal Native Mixture-of-Experts Model

> **Date**：2024-10-08
> **arXiv**：https://arxiv.org/abs/2410.05993

## Abstract

Information comes in diverse modalities. Multimodal native AI models are essential to integrate real-world information and deliver comprehensive understanding. While proprietary multimodal native models exist, their lack of openness imposes obstacles for adoptions, let alone adaptations. To fill this gap, we introduce Aria, an open multimodal native model with best-in-class performance across a wide range of multimodal, language, and coding tasks. Aria is a mixture-of-expert model with 3.9B and 3.5B activated parameters per visual token and text token, respectively. It outperforms Pixtral-12B and Llama3.2-11B, and is competitive against the best proprietary models on various multimodal tasks. We pre-train Aria from scratch following a 4-stage pipeline, which progressively equips the model with strong capabilities in language understanding, multimodal understanding, long context window, and instruction following. We open-source the model weights along with a codebase that facilitates easy adoptions and adaptations of Aria in real-world applications.

---

# Aria：一个开放的多模态原生专家混合模型 论文详细解读

### 背景：这个问题为什么难？

多模态模型要同时理解文字、图片甚至代码，意味着要在同一网络里兼顾完全不同的特征分布和推理需求。过去的模型大多是先训练语言大模型，再“外挂”视觉投射层，导致视觉信息只能被浅层利用，性能受限。闭源的多模态原生模型虽然在效果上有所突破，却因为代码和数据不公开，阻碍了学术复现和二次开发。再加上专家混合（Mixture‑of‑Experts，MoE）技术在语言模型里已经证明能用更少的计算激活海量参数，却很少有开源的多模态 MoE 实现。于是，如何在保持开放的前提下，构建一个既能原生融合多模态，又能利用 MoE 规模效应的模型，成为了亟待解决的难题。

### 关键概念速览
**多模态**：指模型同时处理两种或以上的数据类型（如文字、图像、代码），类似于人类在看图说话时需要把视觉和语言信息融合。  
**原生模型**：模型从零开始就设计为同时接受多种模态输入，而不是在已有单模态模型上“拼接”额外的视觉头。可以想象成一把多功能瑞士军刀，而不是把两把刀绑在一起。  
**Mixture‑of‑Experts（专家混合）**：模型内部有很多子网络（专家），每次前向传播只激活其中一小部分，由一个路由器决定。类似于公司里不同部门只在需要时被召集，省时省力。  
**激活参数**：指一次前向传播实际参与计算的参数量。MoE 的核心优势就在于“激活少、容量大”。  
**长上下文窗口**：模型能够一次性记住并利用更长的输入序列，像是把整篇文章一次性放进记事本，而不是只能记住前几段。  
**指令微调（Instruction Tuning）**：在大量指令式对话数据上继续训练，使模型更擅长遵循用户的明确指令，类似于给模型上了一堂“如何听话”的课。  
**开源模型**：模型权重、代码和训练细节全部公开，任何人都可以下载、复现或二次开发，促进社区共同进步。

### 核心创新点
1. **开放的多模态 MoE 架构 → 以 3.9 B（视觉）/3.5 B（文本）激活参数的专家混合网络为核心 → 让社区能够免费使用、改进并在多模态基准上与闭源大模型竞争。**  
2. **四阶段预训练流水线 → ① 语言大模型预训练 → ② 跨模态对齐（视觉‑文本对） → ③ 长上下文扩展 → ④ 指令微调 → 形成了从基础语言能力到多模态指令遵循的渐进式能力提升。**  
3. **每个 token 只激活少量专家但整体参数规模极大 → 通过专门设计的路由器，使视觉 token 能调用 3.9 B 参数，文本 token 调用 3.5 B 参数 → 在保持推理成本与同等规模的 dense 模型相当的情况下，显著提升了任务表现。**  
4. **统一的任务头 → 同一个模型同时输出自然语言、代码和视觉描述 → 打破了过去多模态模型只能做“语言+视觉”或“语言+代码”的局限，实现真正的“一体多能”。**

### 方法详解
**整体框架**  
Aria 的训练分为四个连续阶段，每个阶段在同一套 MoE 主干上进行，只是数据和目标不同。模型的核心是一个 Transformer‑style 的专家混合网络：数十到上百个专家（每个专家是完整的 Feed‑Forward 层），以及一个轻量级的路由器负责为每个输入 token 选出 top‑k（通常为 2）专家进行计算。

**阶段一：语言预训练**  
- 数据：大规模纯文本语料（类似 LLaMA、GPT‑3 的训练集）。  
- 目标：自回归语言建模，让模型学会基本的语法、常识和推理。  
- 关键点：在此阶段仅使用文本 token，路由器只依据文本特征进行专家分配，奠定了语言理解的基础。

**阶段二：跨模态对齐**  
- 数据：图文对（如 COCO、LAION）以及代码‑图像对（如 GitHub‑Code‑Image）。  
- 输入方式：视觉信息先经过一个轻量的视觉编码器（ViT‑style），输出的视觉 token 与文本 token 交叉喂入同一 MoE 主干。  
- 目标：通过对齐损失（如对比学习）让视觉 token 与对应的文本 token 在表示空间里靠近。  
- 设计巧思：视觉 token 在路由时会使用视觉特征的 “gate bias”，确保视觉专家被适度利用，避免全部走语言专家导致信息丢失。

**阶段三：长上下文扩展**  
- 数据：包含长篇文章、长视频字幕等序列。  
- 方法：在 Transformer 中加入稀疏注意力（Sliding‑Window + Global）并保持 MoE 结构不变。  
- 目的：让模型在一次前向传播中能够处理数千 token，提升对长文档或多图情境的整体理解。

**阶段四：指令微调**  
- 数据：指令式对话、任务描述、代码生成指令等多源指令集合。  
- 目标：让模型在看到明确的 “请做 X” 时能够快速定位相应的专家子网并给出符合指令的输出。  
- 细节：使用混合的监督信号（RLHF‑style 的奖励模型、直接的答案匹配），并在微调时保持专家激活比例不变，以防出现“专家崩塌”。

**路由器的工作原理（白话版）**  
1. 对每个 token 计算一个低维向量（gate vector）。  
2. 与每个专家的门阈值做点积，得到分数。  
3. 取分数最高的 k 个专家，标记为激活。  
4. 只把该 token 的表示送进这 k 个专家的 Feed‑Forward 层，最后再把输出加权合并。  
这相当于在“专家市场”里为每个 token 拍卖，只有出价最高的几家被请上台演出。

**最巧妙的地方**  
- **视觉‑文本统一的路由 bias**：让视觉 token 在选专家时拥有自己的偏好，避免所有 token 都跑同一套语言专家，真正实现了“多模态原生”。  
- **激活参数的动态调节**：在长上下文阶段，模型会根据序列位置自动降低激活专家数，以控制显存，保持推理成本平稳。

### 实验与效果
- **评测任务**：包括视觉问答（VQAv2）、图像描述（COCO Caption）、多模态推理（MME）、代码生成（HumanEval）、以及纯语言基准（MMLU、GSM8K）。  
- **对比基线**：闭源的 GPT‑4V、Claude‑3、以及开源的 Pixtral‑12B、Llama‑3.2‑11B。  
- **核心结果**：论文声称 Aria 在 VQAv2 上比 Pixtral‑12B 提升约 4% 准确率，接近 GPT‑4V；在代码生成 HumanEval 上超过 Llama‑3.2‑11B 约 6% 的通过率；在长文本理解（LongChat）上保持与同等参数的 dense 模型相当的性能。  
- **消融实验**：去掉阶段二的跨模态对齐，视觉问答下降约 3%；去掉指令微调，模型在多模态指令任务的成功率下降 7%；不使用视觉 gate bias，激活的语言专家比例上升至 90%，导致视觉任务表现显著下降。  
- **局限性**：作者承认模型仍然需要数十张 A100 GPU 天的预训练成本，路由器在极端高并发推理时会出现轻微的负载不均；对视频、音频等非静态模态的支持尚未评估。

### 影响与延伸思考
Aria 的开源让社区首次能够在本地或云端自由使用一个规模达数十亿参数的多模态 MoE 模型，直接推动了以下几个方向：  
1. **开源多模态 MoE 的生态**：随后出现的项目如 OpenFlamingo‑MoE、Mistral‑Vision‑MoE 等，都在 Aria 的代码结构和四阶段预训练思路上进行改进。  
2. **高效路由器研究**：Aria 的视觉 gate bias 启发了学术界对跨模态路由策略的系统化探索，出现了 “Modality‑Aware Gating” 的系列论文。  
3. **跨模态指令微调**：指令微调阶段的成功示例鼓励更多研究在同一模型上统一语言、视觉、代码指令，形成真正的通用 AI 助手。  
如果想进一步深入，可以关注以下方向：更轻量的路由器（如稀疏混合专家）、专家的自适应增长（动态扩容）、以及将音频、视频等时序模态纳入同一 MoE 框架的尝试（推测已有初步实验在进行中）。

### 一句话记住它
Aria 用开放的专家混合架构把语言、视觉和代码能力统一在一个模型里，且在多模态基准上匹配甚至超越了闭源大模型。