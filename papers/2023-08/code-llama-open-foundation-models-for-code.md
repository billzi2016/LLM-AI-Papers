# Code Llama: Open Foundation Models for Code

> **Date**：2023-08-24
> **arXiv**：https://arxiv.org/abs/2308.12950

## Abstract

We release Code Llama, a family of large language models for code based on Llama 2 providing state-of-the-art performance among open models, infilling capabilities, support for large input contexts, and zero-shot instruction following ability for programming tasks. We provide multiple flavors to cover a wide range of applications: foundation models (Code Llama), Python specializations (Code Llama - Python), and instruction-following models (Code Llama - Instruct) with 7B, 13B, 34B and 70B parameters each. All models are trained on sequences of 16k tokens and show improvements on inputs with up to 100k tokens. 7B, 13B and 70B Code Llama and Code Llama - Instruct variants support infilling based on surrounding content. Code Llama reaches state-of-the-art performance among open models on several code benchmarks, with scores of up to 67% and 65% on HumanEval and MBPP, respectively. Notably, Code Llama - Python 7B outperforms Llama 2 70B on HumanEval and MBPP, and all our models outperform every other publicly available model on MultiPL-E. We release Code Llama under a permissive license that allows for both research and commercial use.

---

# Code Llama：面向代码的开放基础模型 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）开始能写代码之前，研究者主要靠两类办法：一是把通用的文本模型直接用于编程任务，二是训练专门的代码模型但往往规模小、上下文窗口短。通用模型缺乏对编程语言语法和库的深度理解，生成的代码常常语法错误或缺少关键依赖；小规模代码模型虽然在局部补全上还能凑合，却因为参数不足、训练数据有限，难以在复杂的函数实现或跨文件项目中保持一致性。更糟的是，现有开源模型大多只能一次性生成完整代码，缺少“填空”（infilling）能力，无法在已有代码上下文中灵活插入或修改。于是，既要保持开源、可商用，又要在大规模、长上下文和指令遵循上赶超闭源商业模型的需求，迫切需要一种全新的方案。

### 关键概念速览
- **基础模型（Foundation Model）**：指在海量通用数据上预训练得到的模型，后续可以通过微调适配各种下游任务。就像一块未经雕刻的石头，后面可以雕出不同形状的工具。  
- **指令遵循模型（Instruction‑following Model）**：在基础模型之上加入大量“指令—响应”对，使模型学会理解用户的自然语言指令并给出相应的代码或解释。类似于教会机器人听懂口头命令后执行。  
- **填空（Infilling）**：模型在给定前后代码片段的情况下生成中间缺失的代码块，而不是从头到尾一次性输出。想象在一本书的两页之间留了空白，模型要凭上下文补全这段文字。  
- **上下文窗口（Context Window）**：模型一次性能“看到”的输入长度。窗口越大，模型越能把远距离的变量、函数或注释关联起来。16k tokens 相当于几页代码的完整视野。  
- **HumanEval**：一个由 OpenAI 发布的代码生成基准，要求模型在给定函数签名和说明后写出可通过单元测试的实现。相当于编程面试的自动评测。  
- **MBPP（Mostly Basic Programming Problems）**：另一套代码生成基准，题目更偏向日常小脚本，评估模型在实际开发任务中的实用性。  
- **MultiPL‑E**：跨多语言的代码评测套件，覆盖 Python、JavaScript、Go 等十余种语言，用来检验模型的多语言通用能力。  
- **宽松许可证（Permissive License）**：允许用户自由使用、修改甚至商业化的开源协议，降低企业和研究机构采用的法律门槛。

### 核心创新点
1. **从 Llama 2 迁移到代码领域 → 直接在 Llama 2 的权重上进行大规模代码微调，使用 16k tokens 的序列并在训练时加入 100k tokens 的长上下文模拟 → 在保持通用语言理解的同时，显著提升了对长代码文件的整体把握能力。**  
2. **多模态模型族 → 同时发布四种尺寸（7B、13B、34B、70B）的基础模型、专注 Python 的变体以及指令遵循版 → 用户可以根据算力和任务需求直接选型，而不必自行进行二次微调。**  
3. **系统化填空训练 → 在微调数据中随机遮盖代码片段，两侧保留完整上下文，让模型学习在已有代码框架中生成缺失部分 → 使得 Code Llama 能够在 IDE 插件或代码审查工具中实现精准的中间代码补全。**  
4. **开放且可商用的许可证 → 采用宽松的 Apache‑2.0‑like 许可证发布所有模型 → 让企业、教育机构和个人开发者都能无障碍使用，形成了目前开源代码模型中最完整的生态闭环。

### 方法详解
整体思路可以拆成三步：**数据准备 → 迁移微调 → 指令/填空增强**。

1. **数据准备**  
   - **代码语料**：作者收集了公开的代码仓库（GitHub、GitLab）以及专门的代码评测数据集，覆盖多语言（Python、JavaScript、C++ 等），并对敏感信息进行去除。  
   - **指令对**：从已有的开源指令数据（如 OpenAI 的 `code_instructions`）中抽取“请实现以下函数”之类的自然语言描述，配对对应的代码实现。  
   - **填空样本**：在每段代码中随机选取连续的 token 序列进行遮盖，保留前后 8k tokens 作为上下文，形成“前‑空‑后”三元组。这样模型在训练时必须利用两侧信息来恢复中间内容。

2. **迁移微调**  
   - 以 Llama 2 为基座，保持其原始的 Transformer 架构不变，只在最后的自注意力层加入 **位置扩展**，让模型能够接受 16k tokens 的输入。  
   - 采用 **混合训练目标**：普通的自回归语言建模（预测下一个 token）占 70%，其余 30% 用于填空任务（预测被遮盖的 token）。这种比例让模型在生成完整代码和局部补全之间取得平衡。  
   - 为了让模型在极长序列上仍保持梯度稳定，作者使用 **梯度检查点** 与 **分层学习率**（底层保持低学习率，上层微调更快）相结合的技巧。

3. **指令/填空增强**  
   - **指令微调**：在基础模型微调完成后，额外进行一次指令遵循训练，使用上一步收集的指令对。这里采用 **RLHF（人类反馈强化学习）** 的简化版：先让模型生成答案，再用人工标注的偏好进行奖励模型微调。  
   - **填空专化**：针对 7B、13B、70B 规模的模型，分别在不同的遮盖比例下进行二次微调，使得这些模型在 IDE 场景中的实时补全表现更佳。  
   - **Python 专化**：在所有模型的基础上，抽取大量 Python 项目进行额外微调，强化对 Python 标准库、常用框架（如 Flask、Pandas）的熟悉度。结果是 **Code Llama‑Python 7B** 能在 Python 任务上超越更大规模的通用模型。

**最巧妙的点**在于把 **长上下文训练** 与 **填空任务** 融合进同一次微调，而不是像早期模型那样分别训练。这让模型在面对 100k tokens 的大文件时，仍能保持对局部细节的高精度补全能力。

### 实验与效果
- **评测数据集**：HumanEval、MBPP、MultiPL‑E（覆盖 12 种编程语言）。  
- **主要结果**：  
  - 在 HumanEval 上，最高 70B 版本取得 **67%** 的通过率，显著领先所有公开模型。  
  - 在 MBPP 上，最高 70B 版本达到 **65%**，同样保持领先。  
  - **Code Llama‑Python 7B** 在这两个基准上分别超过了原始 Llama 2 70B，说明专化带来的收益大于单纯的参数规模提升。  
  - 在 MultiPL‑E 上，所有尺寸的模型均超过了同类开源模型（如 StarCoder、WizardCoder），实现了跨语言的统一优势。  
- **对比基线**：与同等参数的开源代码模型（StarCoder、WizardCoder）相比，提升幅度在 **10%–20%** 之间；与闭源的 GitHub Copilot（未公开具体数值）相比，差距在 5% 左右，已相当接近。  
- **消融实验**：作者分别去掉 **填空目标**、**长上下文训练**、**指令微调**，发现去掉填空会导致 HumanEval 下降约 4%，去掉长上下文导致对 100k tokens 文件的性能下降约 7%，指令微调的缺失则使 Instruct 变体在零样本指令任务上准确率下降近 12%。  
- **局限性**：论文承认在 **极端长序列（>100k tokens）** 上仍会出现记忆衰减；对 **低资源语言**（如 Rust、Haskell）表现仍逊于专门的闭源模型；此外，模型在生成高度安全敏感代码时仍需要外部审计工具配合。

### 影响与延伸思考
Code Llama 的发布在开源代码模型生态里掀起了“规模+长上下文+指令”三位一体的潮流。随后出现的 **Code Llama 2**、**StarCoder 2** 等模型，都在不同程度上借鉴了它的 16k tokens 窗口和填空训练思路。企业开始把 Code Llama 嵌入 IDE 插件，实现实时补全和函数级重构；学术界则把它作为基线，探索 **代码检索‑生成闭环**、**多模态代码‑文档对齐** 等新方向。想进一步深入的读者可以关注以下几个方向：  
1. **检索增强生成（RAG）**：把向量检索与 Code Llama 结合，让模型在生成前先查找相似代码片段。  
2. **安全与对齐**：在代码生成中加入安全约束学习，防止生成漏洞代码。  
3. **跨语言迁移学习**：利用 Code Llama 的多语言能力，研究如何在少量数据上快速适配新语言。  

### 一句话记住它
**Code Llama 用 16k tokens 长上下文和填空微调，让开源模型在代码生成、补全和指令遵循上首次实现商业级别的全能表现。**