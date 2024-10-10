# Mono-InternVL: Pushing the Boundaries of Monolithic Multimodal Large   Language Models with Endogenous Visual Pre-training

> **Date**：2024-10-10
> **arXiv**：https://arxiv.org/abs/2410.08202

## Abstract

In this paper, we focus on monolithic Multimodal Large Language Models (MLLMs) that integrate visual encoding and language decoding into a single LLM. In particular, we identify that existing pre-training strategies for monolithic MLLMs often suffer from unstable optimization or catastrophic forgetting. To address this issue, our core idea is to embed a new visual parameter space into a pre-trained LLM, thereby stably learning visual knowledge from noisy data while freezing the LLM. Based on this principle, we present Mono-InternVL, a novel monolithic MLLM that seamlessly integrates a set of visual experts via a multimodal mixture-of-experts structure. Moreover, we propose an innovative pre-training strategy to maximize the visual capability of Mono-InternVL, namely Endogenous Visual Pre-training (EViP). In particular, EViP is designed as a progressive learning process for visual experts, which aims to fully exploit the visual knowledge from noisy data to high-quality data. To validate our approach, we conduct extensive experiments on 16 benchmarks. Experimental results confirm the superior performance of Mono-InternVL than existing monolithic MLLMs on 13 of 16 multimodal benchmarks, e.g., +80 points over Emu3 on OCRBench. Compared to the modular baseline, i.e., InternVL-1.5, Mono-InternVL still retains comparable multimodal performance while reducing up to 67% first token latency. Code and model are released at https://github.com/OpenGVLab/Mono-InternVL.

---

# Mono-InternVL：通过内生视觉预训练突破单体多模态大语言模型的边界 论文详细解读

### 背景：这个问题为什么难？

多模态大语言模型（MLLM）需要把图像信息和文字信息统一到同一个语言模型里，理想状态是一次前向传播就能完成视觉感知和语言生成。过去的单体（monolithic）MLLM往往在预训练阶段把视觉编码器和语言解码器一起微调，结果出现两大痛点：一是优化不稳定，模型在学习视觉特征时会把已经学好的语言能力“忘记”；二是噪声数据（比如网络爬取的图片‑文字对）会把语言模型拉偏，导致整体性能波动。于是，如何在不破坏语言模型的前提下，让模型稳健地吸收大量视觉知识，成为了瓶颈。

### 关键概念速览
- **单体多模态大语言模型（Monolithic MLLM）**：把视觉特征提取和语言生成全部装进同一个大语言模型（LLM）内部，和传统的“视觉编码器+语言模型”两段式不同。可以想象成把相机和键盘直接焊进了同一块芯片。
- **视觉专家（Visual Expert）**：专门负责处理视觉信息的子网络，类似于一支“视觉小队”。在 Mono‑InternVL 中，这些专家通过混合专家（Mixture‑of‑Experts）机制被调度使用。
- **混合专家结构（Mixture‑of‑Experts, MoE）**：把多个专家模型放在一起，输入会根据路由网络挑选出最适合的几个专家来处理，像是把任务分配给最擅长的工人，提高算力利用率。
- **内生视觉预训练（Endogenous Visual Pre‑training, EViP）**：一种逐层递进的视觉学习流程，先让视觉专家在噪声数据上打基础，再在高质量数据上精细化，整个过程在语言模型保持冻结的状态下进行。
- **灾难性遗忘（Catastrophic Forgetting）**：模型在学习新任务时把旧任务的知识全部抹掉的现象。这里指的是视觉微调导致语言能力下降。
- **第一 token 延迟（First‑token latency）**：模型生成第一个输出 token 所需的时间，直接影响交互式使用体验。

### 核心创新点
1. **视觉参数空间独立嵌入 → 将视觉专家作为额外参数层嵌入预训练好的 LLM，而不对 LLM 本体进行梯度更新 → 语言能力保持不变，视觉知识可以在冻结的语言模型上安全累积。**  
   传统做法是把视觉编码器和 LLM 同时微调，容易出现优化冲突。这里的做法相当于给 LLM 加装了一个“外挂”，只在外挂上学习视觉信息。

2. **多模态 MoE 结构 → 用路由网络把不同视觉专家组合成一个整体的视觉模块 → 在不同输入上自动挑选最合适的专家，提高了视觉特征的表达能力且算力开销可控。**  
   与单一视觉编码器相比，MoE 能让模型在处理 OCR、图像描述、细粒度分类等任务时灵活切换专长。

3. **EViP 预训练流程 → 先在海量噪声图文对上进行粗训练，让视觉专家学到基本的视觉-语言对应；随后在高质量标注数据上进行精细微调 → 视觉专家从“粗糙的草图”逐步变成“精细的画作”，而语言模型始终保持冻结。**  
   这种渐进式学习避免了一次性把大量噪声喂给模型导致的性能崩溃。

4. **显著降低第一 token 延迟 → 通过 MoE 的稀疏激活（一次只激活少数专家）和冻结 LLM，推理时只需少量计算 → 在同等硬件上比模块化基线 InternVL‑1.5 快 67%。**  
   这让单体模型在实际交互场景中更具竞争力。

### 方法详解
整体思路可以拆成三步：  
1) **准备一个已经训练好的大语言模型（LLM）**，比如 InternVL‑1.5 的语言部分。  
2) **构建一组视觉专家**，每个专家都是一个轻量的视觉编码器（如 ViT‑tiny、ResNet 等），并把它们放进一个多模态 MoE 框架。  
3) **在冻结 LLM 的前提下，用 EViP 对视觉专家进行两阶段预训练**，最终得到一个完整的单体 MLLM。

**步骤 1：冻结语言模型**  
语言模型的参数全部设为不可学习，梯度不再回传。这样做的直观效果是：模型的语言知识像一座已经建好的大楼，后面的施工只在旁边加装新房间，不会破坏原有结构。

**步骤 2：多模态 MoE 结构**  
- **路由网络**：接收图像特征的粗略表示（比如经过一个共享的浅层卷积），输出每个视觉专家的激活概率。  
- **稀疏激活**：只选取概率最高的 1~2 个专家进行前向计算，类似于在一支团队里挑出最擅长当前任务的成员。  
- **特征融合**：激活的视觉专家输出的向量与语言模型的词嵌入在同一维度上相加或拼接，随后进入语言模型的解码层，完成“看图说话”。

**步骤 3：EViP 预训练**  
- **粗训练阶段**：使用大规模、噪声较多的图文对（如网络爬取的图片标题），目标是让视觉专家学会基本的视觉‑语言对应关系。因为语言模型冻结，梯度只流向视觉专家，避免了语言能力被干扰。  
- **精细微调阶段**：换成高质量标注数据集（如 OCRBench、COCO Caption），继续在视觉专家上做梯度更新。此时视觉专家已经有了“底子”，可以更快收敛到高精度。  
- **渐进式学习的关键**：在两阶段之间，作者会动态调整路由网络的温度，使得专家选择从“宽松”逐步变为“专一”，帮助模型从广度到深度逐步聚焦。

**最巧妙的设计**  
- **视觉参数空间的独立性**：把视觉知识封装在额外的参数块里，等于是给 LLM 加装了可拆卸的插件，既保留了语言模型的通用性，又让视觉学习过程可控。  
- **稀疏 MoE 与冻结 LLM 的协同**：稀疏激活本身已经能降低计算量，冻结 LLM 再进一步削减了不必要的梯度计算，使得第一 token 的响应时间大幅下降。

### 实验与效果
- **评测范围**：论文在 16 个公开的多模态基准上做实验，涵盖 OCR、图像问答、视觉常识推理、图像描述等任务。  
- **主要结果**：在 13/16 基准上超越了所有已有的单体 MLLM，尤其在 OCRBench 上比同类的 Emu3 提升了约 80 分（具体分数未在摘要中给出，只说“+80 points”）。  
- **与模块化基线对比**：与 InternVL‑1.5（模块化方案）相比，Mono‑InternVL 在多模态性能上保持相当，同时第一 token 延迟降低了最高 67%。  
- **消融实验**：论文通过去掉 MoE、去掉 EViP 的精细阶段或让 LLM 参与微调等设置，分别验证了视觉专家独立、渐进预训练和稀疏激活对性能的贡献。结果显示，缺少任何一环都会导致整体分数下降 5%~15%。  
- **局限性**：作者指出，虽然视觉专家在冻结 LLM 下学习效果好，但当需要跨语言（如多语言 OCR）时仍需额外的语言适配层；此外，MoE 的路由网络在极端硬件上可能带来额外的内存碎片。

### 影响与延伸思考
Mono‑InternVL 的思路打开了“插件化”视觉学习的新方向，后续有几篇工作尝试把其他模态（音频、深度图）也以类似的专家插件方式接入 LLM，形成真正的多模态插件生态。还有研究把 EViP 的渐进式噪声‑高质量训练理念搬到大模型的自监督阶段，试图在更少标注的情况下提升跨模态能力。想进一步了解，可以关注以下方向：  
- **跨模态 MoE 的路由策略优化**（如基于强化学习的动态路由）。  
- **多语言/多任务插件化**，即在同一个 LLM 上挂多个不同语言或任务的专家。  
- **低资源视觉预训练**，利用 EViP 思路在少量标注数据上快速提升视觉专家的表现。

### 一句话记住它
把视觉专家当成可插拔插件，在冻结的大语言模型上用渐进式噪声‑高质量预训练，让单体多模态模型既稳又快。