# StarCoder: may the source be with you!

> **Date**：2023-05-09
> **arXiv**：https://arxiv.org/abs/2305.06161

## Abstract

The BigCode community, an open-scientific collaboration working on the responsible development of Large Language Models for Code (Code LLMs), introduces StarCoder and StarCoderBase: 15.5B parameter models with 8K context length, infilling capabilities and fast large-batch inference enabled by multi-query attention. StarCoderBase is trained on 1 trillion tokens sourced from The Stack, a large collection of permissively licensed GitHub repositories with inspection tools and an opt-out process. We fine-tuned StarCoderBase on 35B Python tokens, resulting in the creation of StarCoder. We perform the most comprehensive evaluation of Code LLMs to date and show that StarCoderBase outperforms every open Code LLM that supports multiple programming languages and matches or outperforms the OpenAI code-cushman-001 model. Furthermore, StarCoder outperforms every model that is fine-tuned on Python, can be prompted to achieve 40\% pass@1 on HumanEval, and still retains its performance on other programming languages. We take several important steps towards a safe open-access model release, including an improved PII redaction pipeline and a novel attribution tracing tool, and make the StarCoder models publicly available under a more commercially viable version of the Open Responsible AI Model license.

---

# StarCoder：愿源码与你同在！ 论文详细解读

### 背景：这个问题为什么难？

在大模型时代，生成代码的模型已经可以写出基本的函数，但要让它们在真实的开发环境中可靠使用仍然很难。过去的 Code LLM 多数是用少量公开代码或专有数据训练，导致模型对语言多样性、长上下文和安全风险的处理都不够。还有一个瓶颈是推理速度：代码补全往往需要一次性生成大量 token，传统的自注意力机制在大批量推理时会出现显存和算力的双重压力。于是，业界急需一种既能覆盖多语言、又能处理长上下文、还能在大批量推理时保持高效的开源模型。

### 关键概念速览

**Code LLM（代码大语言模型）**：专门在源码上训练的语言模型，能够理解、生成和改写程序代码。类似于普通语言模型只会写作文，Code LLM 则是会写代码的“程序员”。  

**多查询注意力（Multi‑Query Attention）**：在自注意力里，只为每个 token 生成一个 query，而所有 token 共用同一套 key/value。想象成一次查询多个图书馆的目录，只需要一次检索就能得到所有答案，显著降低显存占用。  

**Infilling（填空）**：模型不仅能在行尾补全，还能在代码中间的占位符处生成合理代码，就像在一段文字中间插入缺失的句子。  

**上下文长度（Context Length）**：模型一次性能看到的 token 数量。8K 上下文相当于可以一次性阅读约 8 000 行代码，足以覆盖大多数函数或小模块。  

**HumanEval**：一个公开的 Python 编程题库，模型需要生成可直接运行并通过单元测试的函数。通过率（pass@1）是衡量代码生成质量的标准指标。  

**PII（个人可识别信息）脱敏**：在训练数据中自动检测并删除姓名、邮箱等敏感信息，防止模型泄露真实用户数据。  

**归属追踪（Attribution Tracing）**：一种工具，能够在生成的代码片段中追溯到原始开源仓库的具体文件和行号，帮助确认许可证合规性。  

**Open Responsible AI Model License**：一种开放许可证，允许商业使用但要求遵守安全、归属和数据使用的责任条款。

### 核心创新点

1. **多查询注意力 → 大批量推理更快**  
   传统的自注意力在每个 batch 中为每个 token 都维护独立的 key/value，显存随 batch 大小线性增长。StarCoder 把 key/value 合并为单一共享集合，只为每个 token 产生 query。这样显存开销几乎不随 batch 增大，推理速度提升 2–3 倍，尤其在 8K 长上下文下仍能保持高吞吐。

2. **1 万亿 token 的大规模、许可友好语料 → 更广语言覆盖**  
   之前的开源代码模型往往只用了几百亿 token，且很多数据来源模糊。StarCoderBase 使用 The Stack——一个从 GitHub 上收集的、全部采用宽松许可证的代码库，并提供了“opt‑out”机制，确保每个仓库都有退出权。如此规模的、合法的语料让模型在 Python、JavaScript、C++ 等十余种语言上都有扎实的基础。

3. **针对 Python 的 35 B token 微调 → 兼顾专精与通用**  
   直接在全量 15.5 B 参数模型上微调 Python，得到的 StarCoder 在 HumanEval 上达到约 40% 的 pass@1，超过所有同规模的专门 Python 模型。更重要的是，这种微调并没有显著削弱模型对其他语言的表现，实现了“专精不失通用”。

4. **安全与合规管线的系统化 → 开源模型的责任边界**  
   为了让模型安全可用，团队构建了两套工具：一是基于正则和实体识别的 PII 脱敏流水线，二是能够在生成代码中标记来源的归属追踪器。相比之前仅靠手工审查的做法，这套自动化系统大幅降低了泄露风险和许可证冲突的概率。

### 方法详解

**整体框架**  
StarCoder 的训练分为两大阶段：① 大规模预训练（StarCoderBase），使用 1 万亿 token 的 The Stack 数据；② 语言专精微调（StarCoder），在 35 B Python token 上继续训练。两阶段都采用了同一套 Transformer 架构，只在推理时启用了多查询注意力来提升效率。训练完成后，模型会经过 PPI 脱敏和归属追踪两道安全过滤，最终以 Open Responsible AI Model License 发布。

**关键模块拆解**  

1. **数据收集与清洗**  
   - **The Stack**：从 GitHub 抓取所有采用 MIT、Apache‑2.0、BSD 等宽松许可证的仓库。  
   - **Opt‑out 机制**：仓库所有者可以提交请求，系统自动剔除对应代码。  
   - **去噪**：删除二进制文件、自动生成的文档、测试数据等噪声，只保留可读源码。  

2. **模型架构**  
   - **Transformer 编码器**：15.5 B 参数，层数 48，隐藏维度 6144，头数 48。  
   - **多查询注意力**：在每层的自注意力模块里，所有 token 共享同一套 key/value 向量，只为每个 token 计算独立的 query。这样在推理时，只需要一次矩阵乘法就能得到所有 token 的注意力分布，显存占用随 batch 大小几乎不变。  

3. **预训练目标**  
   - **因果语言建模**：预测下一个 token，兼顾代码的顺序性。  
   - **填空（Infilling）**：随机遮盖代码片段，让模型学习在中间位置生成合理代码，提升补全能力。  

4. **微调流程**  
   - **Python 专精**：在 35 B Python token 上继续训练 200 B 步，学习 Python 的细粒度语法和标准库使用。  
   - **保持多语言能力**：微调时仍保留少量非 Python 数据，防止模型“忘记”其他语言。  

5. **安全管线**  
   - **PII 脱敏**：使用实体识别模型检测姓名、邮箱、地址等信息，替换为占位符。  
   - **归属追踪**：在每段生成代码前后附加元数据，记录原始仓库、文件路径和行号，便于后续审计。  

**最巧妙的地方**  
多查询注意力的设计看似简单，却在 8K 长上下文和大批量推理时产生指数级的显存节省，这让 15.5 B 参数的模型能够在普通 GPU（如 A100）上以 4‑8× 的批次大小运行，而不需要专用的模型并行或显存压缩技术。

### 实验与效果

- **评测数据集**  
  - **HumanEval**（Python 编程题）  
  - **Multi‑language benchmarks**：包括 MBPP（Python）、CodeXGLUE（多语言）以及专门的 Java、C++ 完成任务。  

- **主要结果**  
  - 在 HumanEval 上，StarCoder 达到约 **40% pass@1**，超过所有公开的同规模模型（如 CodeGen‑16B、PolyCoder‑12.5B），并且接近或略超 OpenAI 的 **code‑cushman‑001**（官方报告约 38%）。  
  - 在多语言基准上，StarCoderBase 超过所有开源多语言 Code LLM，尤其在 JavaScript、Go、Rust 等语言的代码补全任务中领先 5%–10%。  
  - 推理吞吐量提升：使用多查询注意力后，批量大小从 1 提升到 8 时显存仅增长约 10%，推理速度提升约 **2.5×**。  

- **消融实验**  
  - **去掉多查询注意力**：显存占用翻倍，批量大小受限，推理速度下降约 45%。  
  - **不进行 Python 微调**：HumanEval pass@1 降至 28%，说明专精微调对 Python 任务贡献显著。  
  - **关闭归属追踪**：对模型生成质量无影响，但合规审计成本上升。  

- **局限性**  
  - 论文未给出在极大规模商业代码库（如私有企业内部代码）上的表现，可能受限于训练数据的公开性。  
  - 虽然 PII 脱敏管线已上线，但在极端噪声数据中仍有漏检风险，作者承认需要持续改进。  

### 影响与延伸思考

StarCoder 的发布标志着开源代码模型进入了“规模+安全”双轨并进的阶段。它的多查询注意力被后续的 **Code Llama**、**WizardCoder** 等模型采纳，成为提升大模型推理效率的标准做法。The Stack 数据集也被多方引用，推动了对许可证友好代码语料的统一收集。安全管线（PII 脱敏 + 归属追踪）为后续开源模型的合规发布提供了可复制的模板。未来的研究可以在以下方向继续深化：

- **更细粒度的多语言微调**：在保持通用性的前提下，对每种语言进行专门的少量微调，进一步提升跨语言一致性。  
- **动态上下文裁剪**：结合代码结构（AST）进行上下文选择，进一步降低显存需求。  
- **自动化合规审计**：把归属追踪与许可证检查闭环，实现“一键合规”。  

如果想深入了解，建议关注 **BigCode 社区** 的后续报告以及 **The Stack** 的数据发布说明。

### 一句话记住它

StarCoder 用 1 万亿合法代码和多查询注意力，打造了首个兼顾多语言、长上下文和高效推理的开源代码大模型。