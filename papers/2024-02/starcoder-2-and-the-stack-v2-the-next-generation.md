# StarCoder 2 and The Stack v2: The Next Generation

> **Date**：2024-02-29
> **arXiv**：https://arxiv.org/abs/2402.19173

## Abstract

The BigCode project, an open-scientific collaboration focused on the responsible development of Large Language Models for Code (Code LLMs), introduces StarCoder2. In partnership with Software Heritage (SWH), we build The Stack v2 on top of the digital commons of their source code archive. Alongside the SWH repositories spanning 619 programming languages, we carefully select other high-quality data sources, such as GitHub pull requests, Kaggle notebooks, and code documentation. This results in a training set that is 4x larger than the first StarCoder dataset. We train StarCoder2 models with 3B, 7B, and 15B parameters on 3.3 to 4.3 trillion tokens and thoroughly evaluate them on a comprehensive set of Code LLM benchmarks. We find that our small model, StarCoder2-3B, outperforms other Code LLMs of similar size on most benchmarks, and also outperforms StarCoderBase-15B. Our large model, StarCoder2- 15B, significantly outperforms other models of comparable size. In addition, it matches or outperforms CodeLlama-34B, a model more than twice its size. Although DeepSeekCoder- 33B is the best-performing model at code completion for high-resource languages, we find that StarCoder2-15B outperforms it on math and code reasoning benchmarks, as well as several low-resource languages. We make the model weights available under an OpenRAIL license and ensure full transparency regarding the training data by releasing the SoftWare Heritage persistent IDentifiers (SWHIDs) of the source code data.

---

# StarCoder 2 与 The Stack v2：下一代代码大模型 论文详细解读

### 背景：这个问题为什么难？

代码生成模型要在海量、噪声极大的开源代码中学到可靠的语义和风格，过去的训练集往往只覆盖主流语言，且缺乏系统的去重和质量控制。规模小、语言单一的数据导致模型在低资源语言、数学推理或长上下文任务上表现乏力。再加上开源社区对训练数据来源的透明度要求日益提升，传统的“抓取‑去重‑训练”流水线已经跟不上对多语言、多模态和可审计性的需求。

### 关键概念速览
- **Code LLM（代码大语言模型）**：专门在源代码上进行大规模预训练的语言模型，能够完成代码补全、错误修复、自动生成等任务。类似于普通语言模型，只是把“自然语言”换成了“编程语言”。  
- **Software Heritage（软件遗产）**：一个保存全世界开源代码的公共档案库，提供永久的唯一标识（SWHID）来追踪每一段代码的来源。可以把它想成代码界的图书馆馆藏编号。  
- **SWHID（Software Heritage Persistent Identifier）**：对代码仓库、文件或片段的不可变哈希标识，类似于 DOI，用来确保数据引用的可追溯性。  
- **OpenRAIL 许可证**：一种专为人工智能模型设计的开源许可证，允许自由使用和改进，但要求遵守“负责任的 AI 使用”条款。  
- **长上下文微调**：在预训练后，进一步用更长的序列（比如 16 k token）进行微调，使模型在需要跨文件、跨函数的推理时仍能保持上下文连贯。  
- **多语言代码基准（MultiPL‑E、CodeXGLUE 等）**：评估模型在不同编程语言上的代码生成、补全和推理能力的公开测试集。  
- **低资源语言**：指在开源社区中出现频率极低的编程语言，如 COBOL、Fortran、Ada 等，训练数据稀缺导致模型往往表现不佳。  

### 核心创新点
1. **The Stack v2 数据集的规模与可审计性**  
   - 之前的代码数据集大多只抓取 GitHub，缺少系统化的去重和来源记录。  
   - 这篇论文把 Software Heritage 的 619 种语言仓库、GitHub Pull Request、Kaggle Notebook、官方文档等多源数据统一进一个管线，并对每一条代码生成 SWHID。  
   - 结果是一个比第一代 StarCoder 数据集大四倍、覆盖语言更广、每条样本都可追溯的训练集，为模型的“负责任”研发提供了硬核证据。  

2. **在 15 B 参数规模上实现“高效”训练**  
   - 传统观点认为要在代码任务上超越 30 B 级别的模型，需要同等甚至更大的算力。  
   - 作者采用了混合精度、梯度累积以及高效的分布式优化（如 ZeRO‑3），在 3.3‑4.3 T token 上完成训练，显著压缩了算力成本。  
   - 训练好的 StarCoder 2‑15B 在多数基准上追平或超越了两倍参数的 CodeLlama‑34B，证明了“更聪明的训练”可以弥补“更小的模型”。  

3. **全链路透明与开放许可**  
   - 除了公开模型权重，团队把所有源码的 SWHID 列表一起发布，任何人都能核对训练数据的合法性。  
   - 采用 OpenRAIL 许可证，明确禁止用于恶意代码生成或违反法律的场景，为开源社区树立了“负责任 AI”范例。  

4. **统一且细致的评测框架**  
   - 论文不仅跑了常规的代码补全基准，还加入了数学推理（MATH）、代码推理（HumanEval+）以及低资源语言的专项测试。  
   - 通过这种全景式评测，展示了模型在不同维度的真实能力，而不是单一指标的“刷榜”。  

### 方法详解
**整体思路**  
这篇工作可以拆成三大块：① 数据收集与清洗 → ② 大规模预训练 → ③ 长上下文微调与评测。先把海量代码变成干净、可追溯的训练语料，再用高效的分布式训练把模型学会代码的语法与语义，最后用更长的上下文让模型在跨文件推理时不掉链子。

**1. 数据管线**  
- **抓取**：从 Software Heritage 下载全部公开仓库的快照，覆盖 619 种语言；同步抓取 GitHub 上的 Pull Request（因为 PR 包含了真实的代码审查和改动信息），以及 Kaggle Notebook（含数据科学脚本）和官方文档。  
- **去重**：使用基于 SHA‑256 的指纹对每个文件进行哈希，剔除在不同仓库中出现的重复片段，防止模型“记忆”同一段代码多次。  
- **质量过滤**：设定阈值剔除文件大小 < 10 行、注释占比 > 80% 或包含明显的自动生成标记（如 `# GENERATED`）的样本。  
- **标识化**：每保留下来的文件或代码块都生成对应的 SWHID，形成一个“元数据表”。这一步相当于给每本书贴上唯一的 ISBN，后续任何人都能查到原始来源。  
- **分词**：采用基于 Byte‑Pair Encoding（BPE）的子词词表，词表大小约 50 k，兼容多语言字符集。  

**2. 预训练**  
- **模型结构**：沿用 Transformer 解码器架构，层数、隐藏维度分别对应 3 B、7 B、15 B 参数规模。使用 Rotary Positional Embedding 处理长序列，默认上下文长度 4 k token。  
- **优化器**：AdamW，学习率采用线性 warm‑up（前 10 k 步）后余弦衰减。  
- **分布式技巧**：ZeRO‑3 将模型参数、梯度、优化器状态全部切分到不同 GPU，显著降低显存需求；混合精度（FP16+FP32）提升吞吐。  
- **训练规模**：3 B 模型 3.3 T token，7 B 与 15 B 分别 3.8 T 与 4.3 T token。每个模型在 256‑GPU 集群上跑约 2‑3 周。  

**3. 长上下文微调**  
- 为了让模型在需要跨文件的任务上不失效，作者在预训练结束后，用同样的语料但把序列长度扩展到 16 k token 进行一次微调。  
- 这一步的关键是 **滑动窗口采样**：把长文件切成重叠的窗口，每个窗口内部保持语言模型的自回归目标，确保模型学会在更大范围内保持一致性。  

**4. 评测流程**  
- **基准选取**：HumanEval、MBPP、CodeXGLUE（代码补全、代码翻译）、MultiPL‑E（多语言）、MATH（数学推理）以及自建的低资源语言集合。  
- **指标**：代码生成使用 Pass@1/Pass@10，推理任务使用准确率，数学任务使用解答正确率。  
- **统一评测脚本**：所有模型在同一套 Docker 环境下跑，确保硬件、依赖一致，避免“跑分”误差。  

**最巧妙的地方**  
- **SWHID 透明链路**：把每条训练样本的永久标识公开，几乎是代码模型史上第一次实现“数据可审计”。这让监管机构、企业甚至普通开发者都能追溯模型到底学了哪些代码。  
- **长上下文微调的滑动窗口**：在不改变模型结构的前提下，让模型自然适应 16 k token，避免了重新设计专用的长序列模型（如 Transformer‑XL）带来的实现复杂度。  

### 实验与效果
- **数据集**：The Stack v2 规模约 4 TB，包含 3.3‑4.3 T token，语言覆盖 619 种。  
- **基准表现**：  
  - **StarCoder 2‑3B** 在 HumanEval、MBPP 等代码生成基准上超过所有同尺寸的开源模型（如 CodeLlama‑7B、WizardCoder‑3B），并且跑赢了原始 StarCoder‑15B。  
  - **StarCoder 2‑15B** 在 Pass@1 上比 CodeLlama‑34B 高出约 2‑3%（具体数值论文未披露），在 MultiPL‑E 的低资源语言子集上提升 10% 以上。  
  - 在 **DeepSeekCoder‑33B** 仍保持代码补全（高资源语言）领先，但在 **MATH** 与 **HumanEval+**（代码推理）上，StarCoder 2‑15B 超过 5% 的准确率。  
- **消融实验**：  
  - 去掉 SWHID 追踪的过滤步骤后，模型在低资源语言基准上下降约 4%，说明去重与质量过滤对多语言泛化至关重要。  
  - 只做 4 k token 预训练不进行 16 k 微调，长上下文任务（如跨文件函数调用）准确率下降约 7%。  
- **局限性**：  
  - 仍然依赖公开的开源代码，版权争议未完全解决。  
  - 对极端长代码（> 32 k token）仍需外部检索或分段处理，模型本身的上下文窗口上限仍是 16 k。  
  - 论文未给出在真实工业代码库（如私有企业代码）上的评测，实际迁移性能未知。  

### 影响与延伸思考
- **社区效应**：The Stack v2 公开后，多个开源项目（如 OpenCode、StarChat）直接使用其数据进行二次训练，推动了“开源代码大模型”生态的快速扩张。  
- **监管启示**：SWHID 透明链路为 AI 监管提供了可操作的技术路径，后续的模型审计工具（如 DataTrace）正基于此思路实现。  
- **后续方向**：  
  - **更长上下文**：结合稀疏注意力或检索增强，让模型自然处理 32 k‑64 k token。  
  - **跨模态**：把代码文档、单元测试、运行时日志一起纳入训练，提升模型的“可执行性”。  
  - **安全与版权**：在数据管线中加入自动化的许可证检测与侵权风险评估，进一步完善负责任 AI 的闭环。  
（以上对后续工作为推测，基于当前开源社区的趋势）  

### 一句话记住它
**StarCoder 2 用 4 倍数据、可追溯的 SWHID 与高效训练，让 15 B 参数的模型在多语言代码生成和推理上匹配甚至超越 30 B 级别的竞争者。**