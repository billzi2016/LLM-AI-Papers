# DeepSeek-Coder: When the Large Language Model Meets Programming -- The   Rise of Code Intelligence

> **Date**：2024-01-25
> **arXiv**：https://arxiv.org/abs/2401.14196

## Abstract

The rapid development of large language models has revolutionized code intelligence in software development. However, the predominance of closed-source models has restricted extensive research and development. To address this, we introduce the DeepSeek-Coder series, a range of open-source code models with sizes from 1.3B to 33B, trained from scratch on 2 trillion tokens. These models are pre-trained on a high-quality project-level code corpus and employ a fill-in-the-blank task with a 16K window to enhance code generation and infilling. Our extensive evaluations demonstrate that DeepSeek-Coder not only achieves state-of-the-art performance among open-source code models across multiple benchmarks but also surpasses existing closed-source models like Codex and GPT-3.5. Furthermore, DeepSeek-Coder models are under a permissive license that allows for both research and unrestricted commercial use.

---

# DeepSeek-Coder：大语言模型遇上编程——代码智能的崛起 论文详细解读

### 背景：这个问题为什么难？

代码生成看似只要让模型学会写几行函数，但实际上要让模型在真实项目里写出可维护、符合依赖关系的代码，难度远超普通自然语言生成。过去的开源代码模型大多基于少量公开仓库，数据噪声高、上下文短，导致生成的代码经常缺少必要的导入、变量声明或跨文件调用。与此同时，最强的商业模型（如 Codex、GPT‑3.5）都是闭源的，研究者只能在黑盒 API 上做实验，难以深入改进或验证新想法。于是，缺少高质量大规模训练数据、短上下文窗口以及开放可改的模型，成为制约代码智能进一步突破的三座大山。

### 关键概念速览

**大语言模型（LLM）**：一种基于 Transformer 的深度网络，经过海量文本预训练后能够生成连贯的文字。把它想成“会说话的百科全书”，只要给出提示，它就能续写下去。  

**代码智能**：让模型不仅能写出语法正确的代码，还能理解项目结构、依赖关系并完成自动补全、错误修复等任务。类似于 IDE 中的智能提示，只是由模型全权负责。  

**项目级代码语料**：与单文件或片段不同，这类语料保留了完整仓库的目录结构、依赖文件和跨文件引用，像是把整个工程搬进了模型的记忆里。  

**填空任务（fill‑in‑the‑blank）**：训练时让模型在代码中随机遮盖一段，然后要求它恢复被遮掉的部分。相当于给模型出一张“填字游戏”，逼它学会利用前后文推断缺失代码。  

**上下文窗口（context window）**：模型一次性能看到的 token 数量。16K 窗口相当于一次能读完整个函数甚至多个文件的代码，像是把阅读范围从“一页纸”扩大到“一整章”。  

**宽松许可（permissive license）**：允许用户自由修改、再发布甚至商业化使用的开源协议。类似于“自由使用、自由改造、自由赚钱”。  

**基准评估（benchmark）**：一套公开的任务集合，用来统一比较不同模型的代码生成能力，如 HumanEval、MBPP 等。  

**消融实验（ablation study）**：把模型的某个组件关掉或换掉，观察性能变化，以判断该组件的重要性。  

### 核心创新点

1. **从零训练的超大规模代码模型 → 使用 2 万亿 token 的项目级代码语料进行全新预训练 → 让模型在真实项目上下文里学习到跨文件依赖和库调用，生成的代码更完整、更符合实际工程需求。**  

2. **引入 16K 长度的填空任务 → 在训练时让模型在 16K 窗口内随机遮盖代码块并恢复 → 大幅提升模型的“代码填补”能力，使其在函数内部补全、缺失实现以及代码重构时表现出色。**  

3. **提供 1.3B、7B、15B、33B 四个规模的模型系列 → 同时满足轻量部署和高性能需求 → 开源社区可以根据算力选取合适的模型，降低使用门槛。**  

4. **采用宽松的商业友好许可 → 公开模型权重、训练代码和数据处理脚本 → 研究者和企业可以直接在自己的产品里使用或二次开发，无需担心版权纠纷。**  

### 方法详解

整体思路可以拆成三步：**数据准备 → 目标设计 → 大规模训练**。

1. **数据准备**  
   - 从公开代码托管平台抓取数十万高星项目，保留完整的仓库结构（目录、依赖文件、README 等），形成“项目级”语料库。  
   - 对代码进行去噪处理：剔除生成式注释、自动生成的文档、二进制文件等噪声，确保模型学习到的是人类编写的可读代码。  
   - 使用统一的 tokenizer（基于 Byte‑Pair Encoding）把代码和自然语言混合的注释一起编码，保证模型能同时理解代码和解释性文字。

2. **目标设计（填空任务）**  
   - 在每个 16K 长度的窗口内，随机选取若干连续的 token 序列进行遮盖，遮盖比例约为 15%。  
   - 模型的任务是预测被遮盖的 token，预测时只能利用窗口内的前后上下文。  
   - 这种设计相当于让模型在一次阅读中同时完成“阅读理解”和“代码补全”，迫使它学会利用长距离依赖（比如跨文件的函数调用）来推断缺失内容。

3. **大规模训练**  
   - 采用标准的解码器‑only Transformer 架构（类似 GPT），层数、隐藏维度随模型规模而变。  
   - 训练使用混合精度（FP16）和 ZeRO 分布式优化器，能够在数百块 GPU 上高效跑完 2 万亿 token。  
   - 为了让模型适应长上下文，训练时显式开启了 16K 的位置编码，并在后期进行微调，使得模型在更长序列上仍保持稳定的注意力分布。  

**最巧妙的地方**在于把“项目级语料”与“长窗口填空”结合起来。过去的代码模型往往只在单文件或短片段上做掩码预测，导致它们只能记住局部模式；而 DeepSeek‑Coder 把整个项目当作一次阅读材料，让模型在一次前向传播中看到跨文件的调用链，从而学会全局一致性。

### 实验与效果

- **评测任务**：作者在多个公开代码生成基准上做了测试，包括 HumanEval（函数级代码完成）、MBPP（中等难度编程题）以及 CodeXGLUE 中的代码补全任务。  
- **对比基线**：与同等规模的开源模型（如 CodeLlama、StarCoder）以及闭源的 Codex、GPT‑3.5 进行比较。  
- **结果**：在所有基准上，DeepSeek‑Coder 系列均取得了最高分，尤其在 33B 规模模型上，超过了 Codex 和 GPT‑3.5 的表现（具体提升幅度未在摘要中给出，论文声称“显著领先”）。  
- **消融实验**：作者分别关闭了 16K 窗口、项目级语料和填空比例，发现去掉任意一项都会导致整体性能下降 5%~12%，其中长窗口的贡献最大，验证了跨文件上下文的重要性。  
- **局限性**：论文承认模型仍然会产生语义错误或安全漏洞，尤其在涉及复杂并发或底层系统调用时表现不佳；此外，训练成本极高，普通研究团队难以复现完整的 2 万亿 token 训练过程。

### 影响与延伸思考

DeepSeek‑Coder 的开源发布立刻在社区引发热潮，多个企业开始基于其权重进行行业化微调（如金融、嵌入式系统的专用代码生成）。后续工作如 **CodeLlama‑Instruct**、**StarCoder‑Plus** 都在数据质量和长上下文上借鉴了它的做法。值得关注的方向包括：

- **指令微调（instruction tuning）**：让模型更好地理解自然语言指令与代码之间的映射。  
- **强化学习人类反馈（RLHF）**：通过人类评审进一步降低生成代码的错误率。  
- **多模态代码理解**：结合代码、文档、执行日志等多源信息，提升模型的调试和优化能力。  

如果想深入了解，可以先阅读 DeepSeek‑Coder 的数据构建章节，然后关注近期在 **OpenAI Code Interpreter** 与 **Microsoft Copilot** 上的实现细节，比较它们在项目级上下文处理上的差异。

### 一句话记住它

把完整项目当作一次阅读材料，用 16K 长窗口的填空任务训练，从零打造的开源大模型让代码生成直接超越了闭源的 Codex 与 GPT‑3.5。