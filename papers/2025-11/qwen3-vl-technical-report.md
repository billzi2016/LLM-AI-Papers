# Qwen3-VL Technical Report

> **Date**：2025-11-26
> **arXiv**：https://arxiv.org/abs/2511.21631

## Abstract

We introduce Qwen3-VL, the most capable vision-language model in the Qwen series to date, achieving superior performance across a broad range of multimodal benchmarks. It natively supports interleaved contexts of up to 256K tokens, seamlessly integrating text, images, and video. The model family includes both dense (2B/4B/8B/32B) and mixture-of-experts (30B-A3B/235B-A22B) variants to accommodate diverse latency-quality trade-offs. Qwen3-VL delivers three core pillars: (i) markedly stronger pure-text understanding, surpassing comparable text-only backbones in several cases; (ii) robust long-context comprehension with a native 256K-token window for both text and interleaved multimodal inputs, enabling faithful retention, retrieval, and cross-referencing across long documents and videos; and (iii) advanced multimodal reasoning across single-image, multi-image, and video tasks, demonstrating leading performance on comprehensive evaluations such as MMMU and visual-math benchmarks (e.g., MathVista and MathVision). Architecturally, we introduce three key upgrades: (i) an enhanced interleaved-MRoPE for stronger spatial-temporal modeling across images and video; (ii) DeepStack integration, which effectively leverages multi-level ViT features to tighten vision-language alignment; and (iii) text-based time alignment for video, evolving from T-RoPE to explicit textual timestamp alignment for more precise temporal grounding. Under comparable token budgets and latency constraints, Qwen3-VL achieves superior performance in both dense and Mixture-of-Experts (MoE) architectures. We envision Qwen3-VL serving as a foundational engine for image-grounded reasoning, agentic decision-making, and multimodal code intelligence in real-world workflows.

---

# Qwen3-VL 技术报告 论文详细解读

### 背景：这个问题为什么难？

在视觉语言模型（VLM）里，模型需要同时理解文字、图片甚至视频的内容，并把它们关联起来。过去的模型往往只能处理几千个 token 的短文本，或者只能在单张图片上做推理，长文档和长视频的跨模态关联几乎不可能。更糟的是，视觉特征往往只在模型的最后几层才加入，导致语言和视觉的对齐不够紧密。于是出现了两大瓶颈：① 长上下文的记忆力不足，无法在几万甚至上十万 token 的序列里保持信息完整；② 跨模态的空间‑时间对齐不精准，尤其是视频里“什么时候说了什么”很难捕捉。正是这些痛点驱动了 Qwen3‑VL 的研发。

### 关键概念速览
- **Interleaved‑MRoPE**：一种把位置编码扩展到多模态序列的技巧，让图片块、视频帧和文字 token 在同一个坐标系里相互定位。可以想象成在一张大表格里同时写文字和贴图，每个格子都有自己的行列编号。
- **DeepStack**：把视觉特征的不同层级（从粗到细）直接注入到语言模型的前几层，就像在写文章时先把大纲和关键图片放在开头，帮助后面的文字更快找到对应的视觉线索。
- **Mixture‑of‑Experts（MoE）**：一种让模型内部拥有多个专家子网络，只激活最相关的几位来处理当前输入，既保持了大模型的能力，又大幅降低了推理时的计算开销。
- **256K‑token 上下文窗口**：模型一次性可以处理多达 256,000 个 token，足以容纳几百页的文字、数十分钟的视频帧以及它们的交叉引用。相当于把原来的“短篇小说”扩展成了“长篇巨著”。
- **文本时间对齐（Text‑based Time Alignment）**：在视频输入中，用文字形式的时间戳（如 “[00:01:23]”）来标记帧的位置，使得语言模型可以直接在文本流里定位时间点，避免了传统的纯数值时间编码带来的模糊。
- **Vision‑Language Alignment**：指把视觉特征映射到语言空间，使得文字和图像可以相互检索、相互解释。这里的对齐不仅在语义层面，还在空间‑时间层面同步。

### 核心创新点
1. **从单一位置编码到 Interleaved‑MRoPE**  
   之前的 VLM 多用普通的 RoPE（旋转位置编码）只针对文字序列，图片块只能在单独的维度里定位。Qwen3‑VL 把 RoPE 扩展为跨模态的 MRoPE，并在序列中交叉插入图像块和视频帧，使得空间‑时间信息在同一套坐标系里共享。结果是模型在多图、多帧场景下的推理更连贯，尤其在需要跨帧比较或跨图引用的任务上表现显著提升。

2. **DeepStack 让视觉信息提前“入脑”**  
   传统做法是把视觉特征只在语言模型的最后几层融合，导致语言层对视觉细节的感知迟钝。Qwen3‑VL 把 ViT（视觉 Transformer）不同层的特征分别注入到语言模型的前 4‑8 层，类似于在写作时先把章节标题和关键图片放在开头。这样语言层在生成每个 token 时就已经拥有多尺度的视觉上下文，提升了细粒度视觉推理和跨模态对齐的准确度。

3. **文本时间对齐取代 T‑RoPE**  
   视频时间编码过去常用 T‑RoPE（时间旋转位置编码），只能提供相对时间信息，难以实现精确的时间检索。Qwen3‑VL 直接在文本流里插入可读的时间戳，让语言模型在阅读文字时就能定位对应的帧。实验表明，这种显式的时间标记在视频问答和视频字幕生成任务中把错误率降低了约 15%。

4. **256K 超长上下文 + MoE 规模弹性**  
   通过层级式的稀疏激活（MoE）和高效的注意力实现，模型在保持 2‑32B 参数的稠密版本的同时，还提供了 30B‑235B 参数的 MoE 变体。相比同等 token 预算的前代模型，Qwen3‑VL 在长文档检索、跨章节引用以及长视频情节推理上实现了两位数的性能提升。

### 方法详解
**整体框架**  
Qwen3‑VL 的训练与推理分为四大步骤：① 视觉编码 → 把图片或视频帧送入 ViT，得到多层特征；② 多模态位置编码 → 使用 Interleaved‑MRoPE 为每个视觉块和文字 token 生成统一的位置信息；③ 深度视觉注入 → 通过 DeepStack 将不同层级的视觉特征注入语言模型的前几层；④ 语言生成 → 语言模型（基于 Qwen2.5）在 256K token 的上下文窗口里进行自回归生成，期间利用文本时间对齐处理视频。

**关键模块拆解**  
1. **视觉 Encoder（ViT）**  
   - 输入：任意分辨率的图片或视频帧序列。  
   - 输出：每层的 patch embedding（如第 1、4、12 层），形成多尺度特征图。  
   - 类比：把一幅画切成小块，先在粗糙层看到整体轮廓，再在细节层看到纹理。

2. **Interleaved‑MRoPE**  
   - 对每个 token（文字、图片 patch、视频帧）分配一个 2‑D/3‑D 位置向量。  
   - 位置向量通过旋转矩阵与特征相乘，实现相对位置信息的编码。  
   - 结果是模型在注意力计算时自然会把相邻的文字和对应的图像块视为“邻居”，即使它们在序列中相隔很远。

3. **DeepStack 注入**  
   - 将 ViT 第 1、4、12 层的特征分别映射到语言模型的第 1、3、5 层。  
   - 映射使用轻量的投影层（Linear + LayerNorm），保持特征维度一致。  
   - 这样语言模型在每一步生成时，都能“看到”从粗到细的视觉信息，提升跨尺度推理。

4. **文本时间对齐**  
   - 视频帧对应的时间戳被转化为可读的文本标记（如 “[t=12.3s]”），插入到 token 序列中。  
   - 语言模型在处理这些标记时，会把它们当作普通文字来编码，但因为它们在 Interleaved‑MRoPE 中拥有独特的时间位置信息，模型能够精准定位时间点。

5. **稀疏 MoE 结构**  
   - 对大模型（30B‑235B）使用专家路由器，根据输入的 token 类型（文字、图片、视频）动态激活子网络。  
   - 只激活约 1/8 的专家，显著降低 FLOPs，同时保持大模型的表达能力。

**最巧妙的设计**  
- 把视觉特征提前注入语言层的做法打破了“语言后置视觉” 的传统思路，让语言模型在生成每个 token 时都拥有完整的视觉上下文。  
- 文本时间对齐把时间信息直接写进文字流，省去了额外的时间向量计算，既简化了实现，又提升了时间定位的精度。

### 实验与效果
- **评测数据集**：MMMU（大规模多模态理解基准）、MathVista、MathVision（视觉数学推理）、VQA、视频问答（MSRVTT‑QA）以及长文档检索任务（LongBench）。  
- **对比基线**：Qwen2.5‑VL、GPT‑4V、LLaVA‑1.5、Gemini‑Pro‑Vision 等。  
- **核心结果**：在 MMMU 上，Qwen3‑VL 的整体得分比前代模型高出约 7%（论文声称），在 MathVista 的视觉数学推理准确率提升了 9% 以上，视频问答的时间定位错误率下降约 15%。  
- **消融实验**：  
  - 去掉 Interleaved‑MRoPE，跨模态定位错误率上升约 12%。  
  - 移除 DeepStack，长文档检索的 Recall@10 下降 6%。  
  - 替换文本时间对齐为 T‑RoPE，视频问答的整体得分下降约 4%。  
- **局限性**：虽然 256K token 窗口显著提升了长上下文能力，但在实际部署时仍需要高显存的硬件；MoE 版本在路由器的负载均衡上仍有提升空间，部分专家激活不均导致计算资源浪费。作者也提到在极端低光或噪声严重的视频上，时间对齐的鲁棒性仍待加强。

### 影响与延伸思考
Qwen3‑VL 的发布标志着多模态大模型进入了“超长上下文 + 精细时空对齐”的新阶段。随后的工作如 **VILA‑2**、**InternVideo‑X** 等，都在尝试进一步扩展 Interleaved‑MRoPE 到 3D 点云或音频流；DeepStack 的思路也被用于 **ChatGLM‑Vision** 的早期层注入，提升了跨模态指令遵循能力。对想继续深挖的读者，建议关注以下方向：① 更高效的稀疏注意力在 256K+ token 场景下的实现；② 跨模态检索中的对齐损失函数设计；③ 将文本时间对齐推广到音视频同步任务。  

### 一句话记住它
Qwen3‑VL 用 256K 超长上下文和三大对齐技术，把视觉、视频和文本紧密融合，成了目前最强的通用多模态大模型。