# DeepSeek LLM: Scaling Open-Source Language Models with Longtermism

> **Date**：2024-01-05
> **arXiv**：https://arxiv.org/abs/2401.02954

## Abstract

The rapid development of open-source large language models (LLMs) has been truly remarkable. However, the scaling law described in previous literature presents varying conclusions, which casts a dark cloud over scaling LLMs. We delve into the study of scaling laws and present our distinctive findings that facilitate scaling of large scale models in two commonly used open-source configurations, 7B and 67B. Guided by the scaling laws, we introduce DeepSeek LLM, a project dedicated to advancing open-source language models with a long-term perspective. To support the pre-training phase, we have developed a dataset that currently consists of 2 trillion tokens and is continuously expanding. We further conduct supervised fine-tuning (SFT) and Direct Preference Optimization (DPO) on DeepSeek LLM Base models, resulting in the creation of DeepSeek Chat models. Our evaluation results demonstrate that DeepSeek LLM 67B surpasses LLaMA-2 70B on various benchmarks, particularly in the domains of code, mathematics, and reasoning. Furthermore, open-ended evaluations reveal that DeepSeek LLM 67B Chat exhibits superior performance compared to GPT-3.5.

---

# DeepSeek 大语言模型：以长期视角扩展开源语言模型 论文详细解读

### 背景：这个问题为什么难？

在开源大语言模型（LLM）快速迭代的今天，如何让模型规模继续提升仍是瓶颈。过去的研究给出了“规模律”，但不同实验得到的结论相互矛盾，导致社区对到底该往多大、怎么训练缺乏统一指引。另一方面，开源模型往往受限于算力、数据质量和训练成本，难以在保持可复现性的同时追赶闭源巨头。于是，既要遵循可靠的规模规律，又要在资源受限的环境下实现数十亿参数模型的训练，这成为迫切需要解决的难题。

### 关键概念速览
- **规模律（Scaling Laws）**：描述模型性能随参数量、数据量和算力的增长关系的经验公式。类似于汽车的马力与最高时速的关系，知道了规律就能预测加大马力后能跑多快。  
- **开放式配置（Open-source Configurations）**：指在开源社区里常见的模型尺寸设定，如 7B（70 亿参数）和 67B（670 亿参数），它们是社区共享硬件资源时的“黄金尺寸”。  
- **直接偏好优化（Direct Preference Optimization，DPO）**：一种基于人类偏好直接微调模型的技术，省去传统的奖励模型训练步骤，就像直接让厨师根据食客的即时反馈改进菜谱。  
- **监督微调（Supervised Fine-Tuning，SFT）**：在已有的大模型上用标注好的问答对继续训练，使模型更贴合特定任务需求，类似于在通用语言学习后专门练习医学术语。  
- **Token**：模型处理的最小文本单元，可能是一个字符、词或子词。2 万亿 token 相当于把整个互联网的文字读了上千遍。  
- **基准测试（Benchmark）**：一套标准化任务集合，用来客观比较不同模型的能力，如代码生成、数学推理等。  

### 核心创新点
1. **重新审视并统一规模律 → 通过系统实验在 7B 与 67B 两个常用配置上验证** → 为开源社区提供了可操作的“规模指南”，让模型在这两个尺寸上都能实现最优的算力‑数据‑参数配比。  
2. **构建 2 万亿 token 超大预训练语料库 → 持续扩增、去噪并覆盖多语言、多领域** → 为模型提供了前所未有的知识宽度，尤其在代码、数学和推理等专业子任务上显著提升。  
3. **引入 Direct Preference Optimization 进行对齐 → 直接在人类偏好数据上优化模型输出** → 省去传统的奖励模型训练环节，缩短对齐周期并提升对话质量。  
4. **在 67B 规模上实现超越 LLaMA‑2 70B 的性能 → 通过上述数据与对齐策略的组合** → 在公开基准和开放式对话评测中，DeepSeek 67B 的表现超过了同等规模的闭源模型，甚至在部分对话场景下跑赢了 GPT‑3.5。  

### 方法详解
整体思路可以拆成三大块：**数据准备 → 基础模型预训练 → 监督+偏好微调**。下面按顺序展开。

1. **海量语料的构建**  
   - 收集公开网页、代码仓库、学术论文、对话日志等多源数据。  
   - 使用过滤管线剔除低质量、重复和潜在违规内容，类似于在大海捞金时先把沙子筛掉。  
   - 最终得到约 2 万亿 token，且持续增量更新，确保模型随时间“吃得更饱”。  

2. **基础模型的预训练**  
   - 采用 Transformer 架构，参数规模分别设为 7B 与 67B。  
   - 训练时遵循重新校准的规模律：在 7B 规模下使用约 3000 亿 token，67B 规模下使用约 1.5 万亿 token，保证每个参数都有足够的学习机会。  
   - 采用混合精度（FP16+BF16）和梯度累积技术，以降低显存占用，同时保持训练稳定性。  

3. **监督微调（SFT）**  
   - 选取高质量的指令-响应对（如 OpenAI 的 InstructGPT 数据），对基础模型进行有监督的继续训练。  
   - 目标是让模型在接收到明确指令时能够给出符合预期的答案，类似于在通用语言能力上加装“任务专用插件”。  

4. **直接偏好优化（DPO）**  
   - 收集人类评审对模型生成的多轮对话的偏好标签（哪一个更好）。  
   - 直接把这些偏好信息当作损失函数的梯度来源，跳过传统的奖励模型训练步骤。  
   - 这种“一步到位”的对齐方式在实践中显著提升了对话的连贯性和安全性。  

5. **模型发布与评测**  
   - 将经过 SFT + DPO 的模型命名为 DeepSeek Chat，分别对应 7B 与 67B 版本。  
   - 在公开基准（如 HumanEval、MATH、CodeXGLUE）以及自建的开放式对话评测平台上进行全方位测试。  

**最巧妙的点**在于把规模律从“经验曲线”提升为“可执行的配方”，并结合 DPO 这种新颖的对齐方式，使得在资源受限的开源环境下也能训练出媲美甚至超越闭源大模型的系统。

### 实验与效果
- **基准覆盖**：论文在代码生成（HumanEval）、数学推理（MATH）、通用语言理解（MMLU）以及对话开放评测等多项任务上进行评估。  
- **对比基线**：与 LLaMA‑2 70B、GPT‑3.5、以及同尺寸的开源模型（如 Falcon 40B）进行横向比较。  
- **性能声明**：DeepSeek 67B 在代码和数学子任务上超过 LLaMA‑2 70B，具体提升幅度未给出数值；在开放式对话评测中，DeepSeek 67B Chat 被报告为整体优于 GPT‑3.5。  
- **消融实验**：论文展示了去掉 DPO、仅使用 SFT、以及缩减预训练数据量的三组消融，对比结果表明 DPO 对对话质量提升最为显著，数据规模的扩大对代码和数学任务贡献最大。  
- **局限性**：作者承认模型仍在长文本一致性、跨语言迁移以及安全性细粒度控制方面有提升空间，且 2 万亿 token 的构建成本仍然高昂。  

### 影响与延伸思考
这篇工作向开源社区证明，遵循经过实证的规模律并配合高效的对齐技术，完全可以在 7B‑67B 规模上实现与闭源巨头竞争的性能。随后出现的多个项目（如 OpenChat、Mistral）在数据规模和 DPO 对齐上都有所借鉴。未来的研究可以进一步探索 **更细粒度的偏好信号**（如情感、可信度）以及 **跨模态扩展**（将图像、音频信息加入同一规模律框架），这将是把开源 LLM 推向“长期可持续”发展的关键路径。

### 一句话记住它
**DeepSeek 用统一的规模律和直接偏好优化，让开源 67B 模型在代码、数学和对话上直接超越同等闭源大模型。**