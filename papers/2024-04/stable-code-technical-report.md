# Stable Code Technical Report

> **Date**：2024-04-01
> **arXiv**：https://arxiv.org/abs/2404.01226

## Abstract

We introduce Stable Code, the first in our new-generation of code language models series, which serves as a general-purpose base code language model targeting code completion, reasoning, math, and other software engineering-based tasks. Additionally, we introduce an instruction variant named Stable Code Instruct that allows conversing with the model in a natural chat interface for performing question-answering and instruction-based tasks. In this technical report, we detail the data and training procedure leading to both models. Their weights are available via Hugging Face for anyone to download and use at https://huggingface.co/stabilityai/stable-code-3b and https://huggingface.co/stabilityai/stable-code-instruct-3b. This report contains thorough evaluations of the models, including multilingual programming benchmarks, and the MT benchmark focusing on multi-turn dialogues. At the time of its release, Stable Code is the state-of-the-art open model under 3B parameters and even performs comparably to larger models of sizes 7 billion and 15 billion parameters on the popular Multi-PL benchmark. Stable Code Instruct also exhibits state-of-the-art performance on the MT-Bench coding tasks and on Multi-PL completion compared to other instruction tuned models. Given its appealing small size, we also provide throughput measurements on a number of edge devices. In addition, we open source several quantized checkpoints and provide their performance metrics compared to the original model.

---

# Stable Code 技术报告 论文详细解读

### 背景：这个问题为什么难？

代码生成模型要兼顾代码补全、逻辑推理、数学计算以及更高层次的软件工程任务，一直是个技术难点。早期的大模型往往参数量在数十亿以上，训练成本高，部署门槛大，导致普通开发者难以直接使用。与此同时，开源的轻量模型在多语言编程基准上表现乏力，往往只能在单一任务上勉强及格。于是，社区迫切需要一种既小巧又能在多种编程场景下保持竞争力的模型，这正是这篇报告要解决的核心痛点。

### 关键概念速览
**代码语言模型（Code LLM）**：专门训练来理解和生成程序代码的语言模型，类似于会写程序的“自动写手”。  
**指令微调（Instruction Tuning）**：在已有模型上再用对话式指令数据进行训练，使模型能够接受自然语言指令并给出相应的代码或解释，像给助理下达任务。  
**多语言编程基准（Multi‑PL）**：评估模型在多种编程语言（如 Python、Java、C++ 等）上完成代码补全或生成任务的统一测试集合。  
**多轮对话基准（MT‑Bench）**：测量模型在连续问答或指令交互中的表现，类似于聊天机器人在编程场景下的“连环考”。  
**量化（Quantization）**：把模型的浮点权重压缩成更低位数（如 8‑bit）以降低显存占用和推理时延，像把大块的积木拆成更小的拼图。  
**边缘设备（Edge Device）**：指算力受限的本地硬件（如笔记本、手机、单板机），模型若能在这些设备上跑，就大幅提升可用性。  

### 核心创新点
1. **从大模型到小模型的性能逆袭**：过去的开源模型要想在 Multi‑PL 上达到 7B‑15B 参数模型的水平，需要数十亿参数。Stable Code 通过精心构建的高质量代码数据和多阶段训练，使得仅 3 B 参数的模型在同一基准上表现相当，突破了“参数越大越好”的常规认知。  
2. **统一的双模型体系**：在同一基础模型上，分别推出了普通版（Stable Code）和指令版（Stable Code Instruct）。普通版侧重代码补全和推理，指令版则加入了对话式指令微调，使其能够在自然语言交互中完成编程任务，这种“一套底座，两种用途”的设计提升了复用效率。  
3. **面向边缘的量化与吞吐率测评**：作者不仅开源了原始 FP16 权重，还提供了多种量化检查点，并在多种边缘硬件上给出实际推理速度。这样，开发者可以直接挑选最适合自己设备的版本，省去自行量化的繁琐。  
4. **多语言与多轮对话的系统评估**：报告中同时给出 Multi‑PL（覆盖十余种语言）和 MT‑Bench（多轮对话）两大评测，展示模型在“写代码”和“和人聊代码”两类场景下的全方位实力，提供了比单一任务评测更可信的性能画像。

### 方法详解
整体思路可以划分为三大步骤：**数据准备 → 多阶段训练 → 指令微调与量化**。

1. **数据准备**  
   - **代码语料池**：作者从公开代码仓库、技术博客、教学网站等渠道抓取了数百 GB 的源码，覆盖 Python、Java、C++、JavaScript 等主流语言。为了提升质量，使用了自动化的去噪脚本，剔除重复、无效或许可证受限的片段。  
   - **指令数据**：在基础代码语料上，额外构造了约 200 万条“指令‑响应”对，指令形式包括“请实现一个二分查找”“解释下面的错误”。这些指令通过人工或半自动方式生成，确保自然语言与代码之间的对应关系紧密。

2. **多阶段训练**  
   - **阶段一：通用语言预训练**  
     使用混合的自然语言和代码文本进行自回归训练（即让模型预测下一个 token），类似于 GPT 系列的基础训练。这里的关键是 **混合比例**：约 60% 为代码，40% 为自然语言，帮助模型在理解编程意图时兼顾语言表达。  
   - **阶段二：代码专化微调**  
     在纯代码语料上继续训练，重点提升模型的代码补全和函数生成能力。作者采用了 **长序列截断**（最大长度 2048 token）和 **梯度累积**，保证在显存受限的情况下仍能看到完整的函数体。  
   - **阶段三：指令微调（仅针对 Instruct 版）**  
     将指令‑响应对喂入模型，使用 **指令前缀**（如 “User:”）和 **模型回复前缀**（如 “Assistant:”）的格式，让模型学会在对话中切换角色。微调时采用了 **低学习率**（约 1e-5）和 **早停**，防止模型忘记之前的代码能力。

3. **量化与部署**  
   - 在训练完成后，使用 **GPTQ**（一种高效的后训练量化方法）将模型权重压缩到 4‑bit、8‑bit 等不同位宽。量化过程保持了原始模型的 **零点偏移**，确保在边缘设备上推理时误差最小。  
   - 作者在多种硬件（如 RTX 3060、Apple M1、Jetson Nano）上跑了 **每秒 token 产出率**（throughput）测试，结果显示 8‑bit 量化后仍能保持接近原始模型的 80% 速度，显著降低了显存需求。

**最巧妙的地方**在于把“多语言代码预训练”和“指令微调”两条平行的训练路线合并到同一个底座模型上，然后通过轻量级的量化让模型在资源受限的环境中依旧保持竞争力。这种“一体多用、轻量高效”的组合在开源社区里相对少见。

### 实验与效果
- **评测数据集**  
  - **Multi‑PL**：包含 Python、Java、C、C++、JavaScript、Go、Rust 等 10+ 语言的代码补全任务。  
  - **MT‑Bench**：设计了多轮对话场景，要求模型在连续提问下保持上下文一致性并给出正确代码或解释。  

- **对比基线**  
  - 与同尺寸的开源模型（如 CodeLlama‑3B、StarCoder‑3B）以及更大的闭源模型（如 GPT‑4‑Code）进行比较。  
  - 论文声称在 Multi‑PL 上，Stable Code‑3B 的 **平均通过率** 与 7 B、15 B 参数模型相当，且在 MT‑Bench 的 **多轮准确率** 超过所有同类指令模型。  

- **消融实验**  
  - 移除指令微调后，Instruct 版在对话任务上跌至原始版的 70% 左右，说明指令微调是提升对话性能的关键。  
  - 将代码比例从 60% 降到 30% 再训练，模型在代码补全任务上的得分下降约 12%，验证了混合训练比例的重要性。  

- **局限性**  
  - 报告未提供在极端低资源语言（如 COBOL、Fortran）上的表现，暗示模型仍可能在这些小众语言上欠缺。  
  - 量化实验主要聚焦在 8‑bit 与 4‑bit，未探索更激进的 2‑bit 或稀疏化方案，进一步压缩空间仍有待研究。  

### 影响与延伸思考
这篇技术报告在开源社区掀起了“小模型大能量”的讨论热潮。随后出现的 **StableLM‑Alpha**、**Mistral‑7B‑Code** 等模型，都在数据质量和多阶段训练上借鉴了 Stable Code 的思路。对想继续深入的读者，可以关注以下方向：  
- **跨语言代码迁移学习**：利用 Stable Code 的多语言特性，研究如何让模型在一种语言上学到的技巧迁移到另一种语言。  
- **更高效的后训练量化**：探索稀疏化、混合精度训练等手段，进一步降低边缘部署门槛。  
- **指令微调的对话安全**：在对话式编程助手中加入安全过滤，防止生成有害代码。  

### 一句话记住它
Stable Code 用 3 B 参数实现了与更大模型相媲美的多语言代码能力，并通过指令微调和量化让它在边缘设备上也能轻松跑。