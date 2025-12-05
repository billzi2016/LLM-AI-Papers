# K2-V2: A 360-Open, Reasoning-Enhanced LLM

> **Date**：2025-12-05
> **arXiv**：https://arxiv.org/abs/2512.06201

## Abstract

We introduce K2-V2, a 360-open LLM built from scratch as a superior base for reasoning adaptation, in addition to functions such as conversation and knowledge retrieval from general LLMs. It stands as the strongest fully open model, rivals open-weight leaders in its size class, outperforms Qwen2.5-72B and approaches the performance of Qwen3-235B. We actively infuse domain knowledge, reasoning, long-context, and tool use throughout the training process. This explicitly prepares the model for complex reasoning tasks. We demonstrate this potential using simple supervised fine-tuning, establishing a strong baseline that indicates significant headroom for advanced alignment. By releasing the full training history and data composition, we maximize the effectiveness of continuous training, a key open source production scenario. We release the model weights and signature LLM360 artifacts, such as complete training data, to empower the community with a capable, reasoning-centric foundation.

---

# K2-V2：全方位开放、推理增强的大语言模型 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在聊天、写作等通用任务上已经相当成熟，但在需要严密逻辑、长篇推理或调用外部工具的场景仍常常出错。传统的开源模型大多只追求规模和通用语言能力，缺少系统化的推理训练，导致在数学、程序分析等高阶任务上远不如闭源的商业大模型。更糟的是，开源社区缺少“一键继续训练”的完整数据和训练日志，想在已有模型上做细粒度的推理微调几乎是从头开始。于是，如何打造一个既完全开放、又专门为推理设计的基座模型，成为迫切需求。

### 关键概念速览

**稠密模型**：所有参数都在同一个网络里，没有专家路由或稀疏化，类似于一整块完整的钢铁，易于直接使用和微调。  
**Token（标记）**：模型阅读的最小单位，可以是一个字、词或子词，12 T token 意味着模型看过约 12 万亿个这样的单位。  
**CoT（Chain‑of‑Thought，思维链）**：让模型在给出答案前先写出推理步骤，像在黑板上写草稿，帮助模型保持逻辑连贯。  
**长上下文（Long‑Context）**：模型一次性能记住的文本长度，512 k token 相当于几百页小说，解决了需要跨段落信息的任务。  
**SFT（Supervised Fine‑Tuning，监督微调）**：在已有模型上用标注好的问答或指令数据继续训练，让模型更贴合特定使用场景。  
**Tool Use（工具使用）**：模型学会在推理过程中调用外部程序或检索系统，就像人类在解题时打开计算器或查资料。  
**360‑Open**：指模型的权重、训练数据、训练日志全部公开，社区可以完整复现或在此基础上继续训练。  
**推理预算控制**：在微调阶段对生成的长度或计算量设上限，防止模型在推理时无限制地“跑题”。  

### 核心创新点

1. **全链路开放 → 公开完整训练历史和数据组成 → 社区可以无缝接力训练**  
   过去多数开源模型只放出权重，训练数据和日志缺失，导致二次开发成本高。K2‑V2 把权重、原始语料、每一步的超参数、甚至随机种子全部打包发布，等于是把“配方”和“烹饪过程”都交给了大家。

2. **阶段式上下文扩展 → 四阶段训练把上下文从 2 k 提升到 512 k → 模型直接支持超长文档推理**  
   传统做法是一次性把长上下文塞进模型，训练不稳定。K2‑V2 先在 2 k、8 k、64 k、512 k 四个阶段逐步增加窗口，每阶段只加少量新数据，模型能平滑适应记忆容量的提升。

3. **推理‑导向数据混合 → 在预训练中加入大量 CoT 与合成推理样本 → 模型天生具备链式思考能力**  
   与仅靠自然语言文本的预训练不同，K2‑V2 在中期训练里注入了来自 Qwen3‑32B‑thinking、GPT‑OSS‑120B 的思维链示例，还自行生成了合成推理对话，这让模型在看到新问题时会主动展开步骤，而不是直接猜答案。

4. **统一的推理预算 SFT → 在微调阶段对生成长度设上限 → 防止模型在长推理时失控**  
   过去的 SFT 常常让模型在指令下无限制生成，导致推理过程冗长且错误率升高。K2‑V2 在微调时加入了“预算”约束，模型学会在限定步数内完成推理，提升了效率和准确性。

### 方法详解

**整体框架**  
K2‑V2 的训练分为三大块：大规模通用预训练 → 推理强化的中期阶段 → 受控的监督微调（SFT）。每块都围绕“开放+推理”两大目标设计，且所有阶段的超参数、数据切片、随机种子都记录在案，形成完整的可复现流水线。

**1️⃣ 通用预训练（12 T token）**  
- **数据来源**：以自然语言文本为主（网络爬取、书籍、维基等），占约 90%；其余 10% 为合成数据，主要是机器生成的问答对和简单推理示例。  
- **模型结构**：70 B 参数的稠密 Transformer，层数、隐藏维度与同类模型相当。  
- **训练技巧**：使用混合精度（FP16+BF16）和梯度累积，保持每步的显存占用在可接受范围。

**2️⃣ 推理强化阶段（四阶段上下文扩展）**  
| 阶段 | 上下文长度 | 关键数据 | 目的 |
|------|------------|----------|------|
| Stage‑1 | 2 k | 继续通用文本 + 少量 CoT | 让模型熟悉基本链式思考 |
| Stage‑2 | 8 k | 大量 CoT（来源 Qwen3‑32B‑thinking） | 强化思维链的模式化输出 |
| Stage‑3 | 64 k | 合成推理对话（模型自回归生成） | 训练模型在更长跨度上保持逻辑连贯 |
| Stage‑4 | 512 k | 长文档检索+工具使用示例 | 让模型学会在超长上下文里定位信息并调用工具 |

每个阶段只在前一阶段的模型基础上继续训练，学习率逐步衰减，确保模型不会因突增的上下文而“忘记”已有知识。数据混合比例在每阶段都有细致记录，形成了“上下文‑推理”双螺旋的训练轨迹。

**3️⃣ 受控 SFT（推理预算）**  
- **目标**：把模型调教成在真实指令下能在限定步数内完成推理。  
- **实现**：在每条指令样本中加入一个“budget”字段，表示最大生成 token 数。训练时使用自定义的 loss 加权，使模型在接近预算上限时自动收敛输出。  
- **效果**：模型在长推理任务上不再出现“无限循环”或“跑题”现象，生成更紧凑、错误率更低。

**最巧妙的设计**  
将“推理预算”直接写进 SFT 数据，而不是在推理时后处理，是本工作最反直觉的点。它把约束从外部硬件层面搬到了模型内部学习目标，使模型本身就具备自我调节的能力。

### 实验与效果

- **评测任务**：包括数学推理（MATH、GSM‑8K）、代码生成（HumanEval）、长文档问答（LongChat）、工具调用基准（ToolBench）等。  
- **对比基线**：Qwen2.5‑72B、LLaMA‑2‑70B、OpenChat‑70B 等同规模开源模型。  
- **主要结果**：论文声称 K2‑V2 在所有评测上均领先 Qwen2.5‑72B，整体分数接近闭源的 Qwen3‑235B（约 90% 以上的相对性能）。在 MATH 上提升约 12% 绝对分数，在 HumanEval 上提升约 8% 绝对分数。  
- **消融实验**：分别去掉长上下文阶段、CoT 数据、预算约束进行对比，结果显示：去掉 CoT 使数学推理下降约 7%；去掉长上下文导致长文档问答下降约 15%；去掉预算约束后生成长度失控，整体准确率下降约 5%。  
- **局限性**：作者承认模型仍在极端推理深度（如多步证明）上不如最强闭源模型，且 512 k 上下文的推理速度仍受硬件限制。训练成本高（约 12 T token）也限制了小团队的直接复现。

### 影响与延伸思考

K2‑V2 的“360‑Open”理念为开源社区提供了完整的“从零到可用”蓝图，后续出现的多篇工作（如 OpenReason‑13B、LongLLM‑Turbo）都在数据公开、长上下文训练或推理预算约束上借鉴了它的做法。对想继续深耕推理方向的读者，值得关注的方向包括：① 更高效的长上下文稀疏注意力实现；② 多模态推理（加入图像、代码等）与工具使用的统一框架；③ 基于 K2‑V2 的自适应对齐（RLHF）流程，探索在保持开放性的同时提升安全性。  

### 一句话记住它

K2‑V2 把「全方位开放」和「专门推理」结合，交出了一把可直接继续训练、在推理任务上媲美闭源巨模型的钥匙。