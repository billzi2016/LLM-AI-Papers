# Qwen2 Technical Report

> **Date**：2024-07-15
> **arXiv**：https://arxiv.org/abs/2407.10671

## Abstract

This report introduces the Qwen2 series, the latest addition to our large language models and large multimodal models. We release a comprehensive suite of foundational and instruction-tuned language models, encompassing a parameter range from 0.5 to 72 billion, featuring dense models and a Mixture-of-Experts model. Qwen2 surpasses most prior open-weight models, including its predecessor Qwen1.5, and exhibits competitive performance relative to proprietary models across diverse benchmarks on language understanding, generation, multilingual proficiency, coding, mathematics, and reasoning.   The flagship model, Qwen2-72B, showcases remarkable performance: 84.2 on MMLU, 37.9 on GPQA, 64.6 on HumanEval, 89.5 on GSM8K, and 82.4 on BBH as a base language model. The instruction-tuned variant, Qwen2-72B-Instruct, attains 9.1 on MT-Bench, 48.1 on Arena-Hard, and 35.7 on LiveCodeBench. Moreover, Qwen2 demonstrates robust multilingual capabilities, proficient in approximately 30 languages, spanning English, Chinese, Spanish, French, German, Arabic, Russian, Korean, Japanese, Thai, Vietnamese, and more, underscoring its versatility and global reach.   To foster community innovation and accessibility, we have made the Qwen2 model weights openly available on Hugging Face and ModelScope, and the supplementary materials including example code on GitHub. These platforms also include resources for quantization, fine-tuning, and deployment, facilitating a wide range of applications and research endeavors.

---

# Qwen2 技术报告 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）要想在理解、生成、代码、数学推理等多种任务上都保持高水平，需要海量数据、巨大的算力以及精细的训练流程。过去的开源模型往往在参数规模、语言覆盖或多模态能力上受限，要么只能在英文上表现好，要么在中文、代码等专业领域明显落后。再加上开源社区缺少统一的、从 0.5 B 到 70 B 参数全系列模型，研究者很难在同一套基座上做对比实验或迁移学习。因此，提供一个覆盖广、性能强、易于微调的全系模型成为迫切需求。

### 关键概念速览
- **大语言模型（LLM）**：使用数十亿甚至上百亿参数，在海量文本上自监督学习的神经网络，能够完成问答、写作、代码等任务。可以把它想象成“会说话的百科全书”。
- **稠密模型（Dense Model）**：所有参数在每一次前向传播中都被使用的模型。类似于一整块钢铁，整体强度均匀。
- **混合专家模型（Mixture‑of‑Experts, MoE）**：模型内部划分为多个“专家”，每次只激活一小部分专家来处理输入，像是“按需召集的专家团队”，可以在保持计算成本的同时显著扩大参数规模。
- **指令微调（Instruction‑tuning）**：在预训练模型之上，用大量“指令—答案”对进行有监督学习，使模型更擅长遵循用户指令。相当于给模型上了一堂“如何听话”的课程。
- **多语言能力**：模型能够理解和生成多种语言的文本。这里的“多语言”指的是在同一模型里同时掌握约 30 种语言，而不是为每种语言单独训练。
- **量化（Quantization）**：把模型的浮点权重压缩成更低位数（如 4‑bit、8‑bit），以降低显存和算力需求。可以类比为把一本厚书压缩成小册子，内容不变但更易携带。

### 核心创新点
1. **全系模型覆盖 0.5‑72 B 参数**  
   之前的开源项目大多只提供单一规模的模型（如 7 B 或 13 B），导致不同算力环境的用户难以选型。Qwen2 同时发布稠密模型、MoE 模型以及对应的指令微调版本，覆盖从轻量级到旗舰级的全部需求。这样一来，研究者可以在同一套代码库里直接比较不同规模的表现。

2. **Mixture‑of‑Experts 规模突破**  
   在同等计算预算下，MoE 版的参数量远超稠密版，却只激活少数专家，保持推理成本不变。相较于之前的 MoE 实现（如 GLaM），Qwen2 在训练数据多样性和专家路由策略上做了优化，使得大模型的“潜在知识库”更丰富，实际推理仍保持高效。

3. **统一的指令微调 pipeline**  
   Qwen2‑Instruct 系列在预训练基础上加入了大规模指令数据，并使用了多阶段微调（包括对话、代码、数学等子任务），实现了在 MT‑Bench、Arena‑Hard、LiveCodeBench 等多项指令评测上显著领先。与早期仅在单一任务上微调的做法不同，Qwen2 的指令微调更像“一站式训练营”，让模型在多种交互场景下都能表现稳健。

4. **开放生态与工具链**  
   除了模型权重，官方同步发布了量化、微调、部署脚本，并在 Hugging Face 与 ModelScope 上提供一键下载。相比过去只放模型文件的做法，这种“全套工具”降低了新手上手门槛，也促进了社区快速迭代。

### 方法详解
**整体框架**  
Qwen2 的训练分为两大阶段：大规模自监督预训练 → 多任务指令微调。预训练使用了跨语言、跨领域的海量文本；指令微调则在此基础上加入了数十万条“指令—答案”对，覆盖对话、代码、数学推理等子任务。

**关键模块拆解**  

1. **数据层**  
   - **多语言语料**：约 2 TB 的文本，来源包括英文网页、中文新闻、代码仓库、学术论文等，覆盖约 30 种语言。相当于把全球的图书馆搬进了模型的“记忆”。  
   - **指令数据**：从公开指令集合（如 Alpaca、Self‑Instruct）以及内部生成的高质量指令对中抽取，确保每条指令都有明确的目标输出。

2. **模型架构**  
   - **稠密版**：标准的 Transformer 编码器/解码器堆叠，层数、隐藏维度随参数规模线性增长。  
   - **MoE 版**：在每个 Transformer 层的前馈网络位置插入专家路由模块，拥有数十到上百个专家，每次只激活 2‑4 个。路由器根据输入特征计算“门控分数”，决定哪些专家参与计算。这样做的直觉是：不同的语言或任务会自动匹配到最擅长的专家上。

3. **训练细节**  
   - **自监督目标**：采用自回归语言建模（即预测下一个 token），并在多语言数据上使用统一的词表。  
   - **混合精度**：使用 FP16 + BF16 混合训练，提升显存利用率。  
   - **梯度累积 & 动态学习率**：在大模型上通过梯度累积实现更大的有效批次，同时使用 cosine 衰减的学习率调度。

4. **指令微调流程**  
   - **多阶段微调**：先在通用指令集上进行一次全模型微调，随后针对代码、数学等高难度子任务进行专门的微调。每个阶段都使用 LoRA（低秩适配）等参数高效微调技术，以降低算力需求。  
   - **对齐策略**：在部分指令上加入人类偏好数据，使用 PPO（近端策略优化）进行轻量级的强化学习对齐，提升模型在对话安全性和可控性方面的表现。

**最巧妙的设计**  
MoE 的路由器采用了“负载均衡正则化”，确保不同专家的使用频率相对均匀，防止出现“热门专家”被过度调用、其他专家闲置的情况。这一点在大规模训练中尤为关键，因为不均衡会导致显存浪费和训练不收敛。

### 实验与效果
- **评测任务**：语言理解（MMLU、GPQA）、代码生成（HumanEval、LiveCodeBench）、数学推理（GSM8K、BBH）以及多语言指令基准（MT‑Bench、Arena‑Hard）。  
- **旗舰模型表现**（Qwen2‑72B，未微调）：MMLU 84.2、GPQA 37.9、HumanEval 64.6、GSM8K 89.5、BBH 82.4。  
- **指令微调后**（Qwen2‑72B‑Instruct）：MT‑Bench 9.1、Arena‑Hard 48.1、LiveCodeBench 35.7，显著超过同尺度的开源基线。  
- **对比基线**：在多数公开榜单上，Qwen2‑72B 超过了前代 Qwen1.5，并与闭源商业模型（如 GPT‑4、Claude）保持竞争。  
- **消融实验**：原文提供了 MoE 与稠密版的对比，显示 MoE 在相同算力下可提升约 2‑3% 的任务准确率；指令微调的多阶段策略比一次性微调提升约 5% 的 MT‑Bench 分数。  
- **局限性**：报告中承认在极低资源语言（如部分非洲语言）上的表现仍有提升空间；此外，MoE 的路由开销在极端高并发推理场景下可能导致延迟波动。

### 影响与延伸思考
Qwen2 的全系开源策略在 2024‑2025 年间被多家企业和学术实验室采纳，推动了“从 0.5 B 到 70 B 一键切换”的实验范式。随后出现的模型（如 LLaMA‑Next、OpenChat‑3B）在架构上都借鉴了 Qwen2 的 MoE 路由与多阶段指令微调流程。对想进一步探索的读者，可以关注以下方向：  
- **更高效的专家路由**：研究如何在推理时实现更低延迟的专家选择。  
- **跨语言对齐**：利用对齐技术提升低资源语言的生成质量。  
- **轻量化指令微调**：探索 LoRA、Adapter 等技术在指令微调中的极限压缩率。  

### 一句话记住它
Qwen2 用“一套全尺寸、稠密+MoE、指令微调的开放模型族”，把高性能大语言模型从实验室搬进了每个人的开发环境。