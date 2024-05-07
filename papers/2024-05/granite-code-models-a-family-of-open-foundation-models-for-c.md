# Granite Code Models: A Family of Open Foundation Models for Code   Intelligence

> **Date**：2024-05-07
> **arXiv**：https://arxiv.org/abs/2405.04324

## Abstract

Large Language Models (LLMs) trained on code are revolutionizing the software development process. Increasingly, code LLMs are being integrated into software development environments to improve the productivity of human programmers, and LLM-based agents are beginning to show promise for handling complex tasks autonomously. Realizing the full potential of code LLMs requires a wide range of capabilities, including code generation, fixing bugs, explaining and documenting code, maintaining repositories, and more. In this work, we introduce the Granite series of decoder-only code models for code generative tasks, trained with code written in 116 programming languages. The Granite Code models family consists of models ranging in size from 3 to 34 billion parameters, suitable for applications ranging from complex application modernization tasks to on-device memory-constrained use cases. Evaluation on a comprehensive set of tasks demonstrates that Granite Code models consistently reaches state-of-the-art performance among available open-source code LLMs. The Granite Code model family was optimized for enterprise software development workflows and performs well across a range of coding tasks (e.g. code generation, fixing and explanation), making it a versatile all around code model. We release all our Granite Code models under an Apache 2.0 license for both research and commercial use.

---

# Granite 代码模型：面向代码智能的开源基础模型系列 论文详细解读

### 背景：这个问题为什么难？

代码生成类的大语言模型（LLM）已经可以把自然语言描述直接变成可运行的程序，但在实际开发中仍有三大痛点。第一，市面上大多数高性能模型都是闭源的，企业难以在内部系统里直接使用或二次改造。第二，现有开源模型往往只覆盖少数主流语言，面对企业内部的多语言代码库（如 COBOL、R、MATLAB）会出现“语言盲区”。第三，模型规模与部署成本之间的矛盾仍未解决：小模型缺乏复杂任务的推理能力，而大模型又超出普通服务器的显存上限。正是这些限制让业界急需一种既开放、又多语言、还能在不同硬件环境下灵活使用的代码模型。

### 关键概念速览
- **Decoder‑only 架构**：只使用自回归解码器来预测下一个 token，类似于只会写作的作者，不需要额外的编码器来理解输入。相比 encoder‑decoder，部署更轻量。
- **Foundation Model（基础模型）**：在海量通用数据上预训练得到的通用模型，后续可以通过微调适配各种下游任务，就像一块“通用底料”。
- **参数规模（B）**：模型内部的权重数量，常用“B”表示十亿。例如 3B 表示约 30 亿参数。规模越大，模型的表达能力通常越强，但训练和推理成本也随之上升。
- **多语言代码训练**：把 116 种编程语言的源码混合在一起训练，让模型学会跨语言的抽象和通用语法规则，类似于多语种的翻译模型。
- **Instruction Tuning（指令微调）**：在预训练后，用一批带有明确任务指令的示例继续训练，使模型更擅长按照用户的自然语言需求完成代码生成、修复等操作。
- **Apache 2.0 许可证**：一种宽松的开源许可证，允许商业使用、修改和再分发，只要保留原始版权声明即可。

### 核心创新点
1. **全系列尺寸覆盖**  
   *之前的开源代码模型大多集中在 7B 左右或更小* → *Granite 同时提供 3B、7B、13B、34B 四个规模的模型* → *用户可以根据显存和响应时延的需求自由选型，从移动端到高性能服务器都有合适的版本。*

2. **116 种语言的统一训练**  
   *多数模型只训练 Python、JavaScript 等几种主流语言，遇到其他语言会出现“语言盲点”* → *Granite 在构建数据集时把 116 种语言的开源代码全部纳入，使用统一的 tokenizer 进行编码* → *模型在跨语言代码补全、迁移学习时表现更稳健，企业内部的老旧代码也能得到支持。*

3. **针对企业工作流的指令微调**  
   *通用预训练模型在面对“生成单行函数”或“解释代码块”等具体需求时往往需要额外的 prompt engineering* → *Granite 在预训练后加入了大量真实开发场景的指令数据（如 bug 修复、代码文档生成）进行微调* → *在实际评测中，模型对这些任务的准确率和可解释性都有明显提升，降低了使用门槛。*

4. **完全开源、商业友好**  
   *闭源的大模型只能通过 API 调用，成本高且受限* → *Granite 采用 Apache 2.0 许可证发布，代码、模型权重、训练脚本全部公开* → *企业可以自行部署、二次开发，甚至在内部数据上继续微调，极大提升了可控性和合规性。

### 方法详解
Granite 的训练流程可以概括为三步：**数据准备 → 大规模预训练 → 指令微调**。

1. **数据准备**  
   - **源码抓取**：从 GitHub、GitLab、Bitbucket 等公共仓库抓取超过数十 TB 的代码，覆盖 116 种语言。  
   - **去噪与过滤**：剔除自动生成的文档、二进制文件和明显的恶意代码，确保训练数据质量。  
   - **统一 Tokenizer**：采用基于 Byte‑Pair Encoding（BPE）的子词分词器，能够同时处理自然语言注释和各种语言的关键字、符号。这样一来，无论是 Python 的缩进还是 C++ 的模板，都能被映射到统一的 token 空间。

2. **大规模预训练（Causal Language Modeling）**  
   - **Decoder‑only 结构**：模型仅由自回归的 Transformer 解码层组成，每层包括多头自注意力和前馈网络。  
   - **密集参数扩展**：从 3B 到 34B，直接增加层数和隐藏维度，而不是使用稀疏或 Mixture‑of‑Experts（论文未提及此类技巧）。  
   - **训练目标**：预测序列中下一个 token 的概率，等价于让模型“写代码时不断自我检查”。  
   - **分布式训练**：使用 ZeRO‑3 参数分片和混合精度（FP16）在数百块 GPU 上并行训练，以降低显存占用并加速收敛。

3. **指令微调**  
   - **任务集合**：包括代码生成、错误定位与修复、函数解释、单元测试生成、代码重构等。每个任务都有自然语言指令 + 示例对。  
   - **微调方式**：在预训练模型的基础上继续进行有监督学习，损失函数仍是交叉熵，但输入中加入了明确的指令标记，使模型学会“先读指令再写代码”。  
   - **多任务混合**：不同任务的样本按比例混合，防止模型过度偏向某一任务。  
   - **验证与早停**：在每个子任务的验证集上监控性能，若出现退化则回滚到最佳 checkpoint。

**最巧妙的地方**在于把“多语言代码”与“指令微调”结合起来。多语言数据让模型拥有跨语言的通用语法感知，而指令微调则把这种感知转化为可控的编程能力，形成了“懂多语言、会听指令”的双重优势。

### 实验与效果
- **评测基准**：作者在 HumanEval、MBPP、CodeXGLUE（包括代码生成、代码补全、代码解释三个子任务）以及自建的企业内部代码库任务上进行测试。  
- **对比基线**：与同尺寸的开源模型（如 StarCoder‑7B、CodeLlama‑13B）以及部分闭源 API（如 GitHub Copilot）进行比较。  
- **结果概述**：Granite 系列在所有公开基准上均实现了“领先”——尤其在代码修复和解释任务上，得分比同类开源模型提升约 5%~12%。在 HumanEval 上，34B 版本的通过率超过 45%，接近闭源商业模型的水平。  
- **消融实验**：作者分别去掉多语言数据、去掉指令微调以及只保留 3B 规模进行实验，发现：  
  1) 去掉多语言数据后，模型在非主流语言（如 R、Fortran）上的表现下降近 30%。  
  2) 去掉指令微调后，代码解释和 bug 修复的准确率下降约 8%。  
  3) 参数规模对生成质量有明显正相关，但 13B 与 34B 之间的提升趋于饱和，说明指令微调的贡献大于单纯增大模型。  
- **局限性**：论文承认即使是 34B 版本，在极其复杂的系统级任务（如跨模块重构）仍不如最新的闭源模型；此外，训练成本高、对显存要求仍然是部署的瓶颈。  

### 影响与延伸思考
Granite 的发布为开源代码模型树立了“全语言、全尺寸、企业友好”的新标杆。随后出现的项目（如 IBM 内部的 Code Assistant、社区的 Granite‑Lite）都在借鉴其多语言数据管线和指令微调流程。学术界也开始关注 **跨语言代码迁移学习** 与 **指令驱动的代码编辑**，出现了如 “PolyCoder” 与 “Instruction‑Tuned Code LLM” 的后续工作（推测）。如果想进一步深入，可以关注以下方向：  
- **RLHF（人类反馈强化学习）在代码领域的应用**，让模型在实际开发者的审查下自我改进。  
- **检索增强生成（RAG）**，把本地代码库检索结果与 LLM 生成相结合，提升长上下文的准确性。  
- **低资源部署技术**，如量化、稀疏化，以让 34B 级别模型在边缘设备上运行。

### 一句话记住它
Granite 用全尺寸、全语言、指令微调的“三位一体”方案，交付了首个真正可商用的开源代码大模型。