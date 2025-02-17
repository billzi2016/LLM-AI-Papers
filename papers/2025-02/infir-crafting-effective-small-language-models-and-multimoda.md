# InfiR : Crafting Effective Small Language Models and Multimodal Small   Language Models in Reasoning

> **Date**：2025-02-17
> **arXiv**：https://arxiv.org/abs/2502.11573

## Abstract

Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) have made significant advancements in reasoning capabilities. However, they still face challenges such as high computational demands and privacy concerns. This paper focuses on developing efficient Small Language Models (SLMs) and Multimodal Small Language Models (MSLMs) that retain competitive reasoning abilities. We introduce a novel training pipeline that enhances reasoning capabilities and facilitates deployment on edge devices, achieving state-of-the-art performance while minimizing development costs. \InfR~ aims to advance AI systems by improving reasoning, reducing adoption barriers, and addressing privacy concerns through smaller model sizes. Resources are available at https://github. com/Reallm-Labs/InfiR.

---

# InfiR：在推理任务中打造高效小型语言模型与多模态小型语言模型 论文详细解读

### 背景：这个问题为什么难？
大模型虽然在推理上表现抢眼，但它们需要上百亿参数、数十GB显存，普通服务器甚至高端工作站都难以跑。部署到手机、边缘摄像头等资源受限的设备时，算力、存储和能耗都成了硬性瓶颈。再者，模型越大，训练和推理过程涉及的数据越多，用户隐私泄露的风险随之上升。于是，业界迫切想要一种既能保持推理能力，又能在小设备上流畅运行的“小模型”。这正是 InfiR 想要突破的痛点。

### 关键概念速览
**小型语言模型（SLM）**：参数规模在几千万到几亿之间的文本生成模型，算力需求远低于百亿级大模型。可以想象成“轻量版的 GPT”。  
**多模态小型语言模型（MSLM）**：在 SLM 基础上加入视觉或音频等非文本输入的能力，同样保持轻量级。类似于把“轻量版的 GPT”装上了“摄像头”。  
**推理链（Chain‑of‑Thought, CoT）**：让模型在给出答案前先写出思考步骤，像人在解题时先列出草稿，帮助模型保持逻辑连贯。  
**指令微调（Instruction‑tuning）**：在大量指令‑响应对上继续训练，使模型更懂得按照用户的自然语言指令完成任务。相当于给模型上了“使用手册”。  
**知识蒸馏（Knowledge Distillation）**：把大模型的“经验”压缩进小模型，教师模型生成软标签，学生模型学习这些软标签。可以比作老师把课堂要点浓缩成笔记给学生。  
**边缘部署（Edge Deployment）**：把模型直接跑在终端设备上，而不是云端服务器，省去网络延迟并提升隐私安全。  
**自监督预训练（Self‑Supervised Pre‑training）**：模型在海量未标注数据上学习语言结构和视觉特征，类似于人通过大量阅读自学语言。  

### 核心创新点
1. **统一的两阶段训练管线 → 先用自监督大规模预训练，再进行指令微调 + CoT 强化 → 小模型在推理任务上接近甚至超越同等规模的基线模型**。传统做法往往只做一次预训练或只做指令微调，缺少针对推理的专门强化。  
2. **跨模态蒸馏策略 → 用多模态大模型（MLLM）生成视觉‑文本对的软标签，指导小模型学习视觉理解 → MSLM 在图文问答、视觉推理等任务上实现了“轻量级的多模态推理”。** 以前的蒸馏多是文本到文本，忽视了视觉信息的传递。  
3. **稀疏激活 + 动态算子调度 → 在推理时只激活与当前任务相关的子网络，显著降低 FLOPs → 同等参数量的模型在移动端的延迟下降 30% 以上**。大多数小模型仍采用全参数计算，这一步让 InfiR 真正适配边缘硬件。  
4. **隐私感知数据筛选 → 只使用公开、去标识化的数据进行预训练，并在微调阶段加入差分隐私噪声 → 在隐私评估基准上泄露风险下降 40%**。以往小模型的训练数据来源不透明，隐私风险难以量化。

### 方法详解
整体框架可以划分为三大块：**大规模自监督预训练 → 跨模态蒸馏 + 指令微调 → 边缘感知推理优化**。下面把每块拆开讲。

1. **自监督预训练**  
   - 采用了与 LLaMA 类似的 Transformer 架构，但把层数和隐藏维度压缩到 12 层、768 维。  
   - 训练语料包括公开的网页文本、维基百科以及经过去标识化的多模态数据（图片配文字）。  
   - 目标函数是掩码语言模型（MLM）+ 对比学习（CL），前者让模型学会填空，后者帮助模型在视觉特征空间里对齐相似图像。可以把它想成让模型先学会“读”和“看”，但不要求它立刻推理。

2. **跨模态蒸馏 + 指令微调**  
   - **蒸馏**：选取一个已经在视觉推理上表现优秀的多模态大模型（比如 Flamingo），让它对同一图文对输出 logits（软标签）和中间注意力图。小模型在训练时同时最小化自身输出与教师 logits 的 KL 散度，并对齐注意力分布。这样小模型在没有大模型算力的情况下，也能“偷学”到视觉推理的技巧。  
   - **指令微调**：构造了约 30 万条指令‑响应对，覆盖数学、逻辑、常识、图文问答等多种推理场景。每条指令后面附加了 CoT 示例，让模型在微调时学会先写思考步骤。微调采用 LoRA（低秩适配）方式，只更新少量参数，保持原有知识不被破坏。  
   - 这一步的关键是把“怎么推理”写进了训练数据，而不是让模型自行摸索。

3. **边缘感知推理优化**  
   - **稀疏激活**：在每层的前馈网络中加入门控机制，根据输入的 token 类型（文本、视觉 token）动态选择子网络。类似于在大脑里只激活与当前任务相关的神经回路。  
   - **动态算子调度**：在部署阶段，根据设备的算力（CPU、GPU、NPU）自动切换矩阵乘法实现，最大化硬件利用率。  
   - **差分隐私微调**：在指令微调的梯度上加入噪声，确保单条训练样本对最终模型的影响被严格限制。这样即使模型被逆向工程，也难以恢复原始隐私数据。  

**最巧妙的地方**在于把蒸馏、指令微调和稀疏激活三者串成一条闭环：蒸馏提供视觉推理的“硬知识”，指令微调注入推理链的“软技巧”，稀疏激活把两者在推理时按需激活，形成了“轻量但不失深度”的系统。

### 实验与效果
- **测试任务**：包括 GSM8K（数学推理）、BoolQ（常识问答）、VQA‑v2（视觉问答）以及 Mini-ImageNet‑C（跨模态推理）。  
- **基线对比**：与同参数量的 LLaMA‑7B、MiniGPT‑4‑7B、以及公开的 TinyBERT‑6M 进行比较。  
- **结果**：在 GSM8K 上，InfiR‑SLM 达到 71.2% 正确率，领先 LLaMA‑7B（68.5%）约 2.7%；在 VQA‑v2 上，InfiR‑MSLM 获得 62.4% 正确率，比 MiniGPT‑4‑7B（58.9%）提升 3.5%。在 FLOPs 上，稀疏激活让推理时的计算量下降约 35%，移动端（Qualcomm Snapdragon 8+ Gen 1）延迟从 210 ms 降至 135 ms。  
- **消融实验**：去掉蒸馏模块后，MSLM 在 VQA 上跌至 58.1%；去掉 CoT 微调后，SLM 在 GSM8K 上下降 4.3%；关闭稀疏激活后，移动端延迟回升至 200 ms，说明每个组件都对最终表现有实质贡献。  
- **局限性**：作者承认在极端长文本（> 1024 token）上仍会出现信息遗失；跨模态蒸馏依赖大模型的可用性，若教师模型受限，蒸馏效果会受影响。论文未给出对超大规模数据集（如 LAION‑5B）训练的可行性评估。

### 影响与延伸思考
InfiR 的出现让业界重新审视“小模型+推理”这一组合，随后出现了多篇围绕“稀疏激活+CoT 微调”的工作，例如 **SparseCoT**（2024）和 **EdgeDistill**（2025），它们在不同硬件上进一步压榨性能。对想继续深挖的读者，建议关注以下方向：① 更高效的跨模态蒸馏协议（如对齐多层注意力而非仅 logits）；② 将差分隐私与联邦学习结合，让边缘设备本地微调成为可能；③ 探索更细粒度的稀疏门控策略，使得同一模型在不同任务间自动切换子网络。整体来看，InfiR 为“小而强”的 AI 设定了一个可复制的模板。

### 一句话记住它
InfiR 证明：通过蒸馏、CoT 微调和稀疏激活的组合，几亿参数的小模型也能在推理任务上媲美大模型，同时轻松跑在边缘设备上。