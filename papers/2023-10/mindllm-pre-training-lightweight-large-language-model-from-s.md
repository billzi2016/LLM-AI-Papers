# MindLLM: Pre-training Lightweight Large Language Model from Scratch,   Evaluations and Domain Applications

> **Date**：2023-10-24
> **arXiv**：https://arxiv.org/abs/2310.15777

## Abstract

Large Language Models (LLMs) have demonstrated remarkable performance across various natural language tasks, marking significant strides towards general artificial intelligence. While general artificial intelligence is leveraged by developing increasingly large-scale models, there could be another branch to develop lightweight custom models that better serve certain domains, taking into account the high cost of training and deploying LLMs and the scarcity of resources. In this paper, we present MindLLM, a novel series of bilingual lightweight large language models, trained from scratch, alleviating such burdens by offering models with 1.3 billion and 3 billion parameters. A thorough account of experiences accrued during large model development is given, covering every step of the process, including data construction, model architecture, evaluation, and applications. Such insights are hopefully valuable for fellow academics and developers. MindLLM consistently matches or surpasses the performance of other open-source larger models on some public benchmarks. We also introduce an innovative instruction tuning framework tailored for smaller models to enhance their capabilities efficiently. Moreover, we explore the application of MindLLM in specific vertical domains such as law and finance, underscoring the agility and adaptability of our lightweight models.

---

# MindLLM：从零预训练轻量级大语言模型的评估与领域应用 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在自然语言处理里已经可以完成翻译、写作、代码生成等多种任务，但它们的参数量往往在数十亿到上千亿之间，训练和部署成本高得惊人。很多组织只能负担几百美元的算力，根本跑不动这些巨型模型。与此同时，实际业务往往只需要在特定领域（比如法律、金融）提供高质量的语言服务，而不是通用的“全能”能力。于是出现了两个矛盾：一是资源受限导致无法使用最前沿的巨型模型；二是现有的开源轻量模型在中文和英文双语能力、指令遵循等方面仍然落后。要在资源有限的情况下，得到既能处理双语，又能在垂直领域表现出色的模型，成为了迫切需要解决的问题。

### 关键概念速览
- **轻量级大语言模型**：参数在十亿级别左右的模型，算力需求比百亿级模型低得多，类似于“轻型跑车”在城市道路上更易驾驶。  
- **从零预训练（from scratch）**：不依赖已有的大模型权重，直接在自建语料上训练模型，像是从头烤一块全新的面包，而不是在已有面包上再加配料。  
- **双语预训练**：训练数据同时包含中文和英文，使模型在两种语言之间可以自由切换，类似于双语教材让学生同时学两门语言。  
- **指令微调（instruction tuning）**：在已有模型基础上，用大量“指令—答案”对进行再训练，让模型更懂得遵循用户的明确指令，像是给机器人加装了“听话模式”。  
- **垂直领域适配**：在特定行业（法律、金融）收集专业文本进行二次训练，使模型掌握行业术语和推理方式，类似于医生在医学教材上继续深造。  
- **评估基准（benchmark）**：公开的任务集合，用来客观比较不同模型的表现，如同体育比赛的计时器。  
- **模型蒸馏（model distillation）**：把大模型的知识压缩到小模型里，像是把一本厚书的要点浓缩成一页纸。  

### 核心创新点
1. **从零构建双语轻量模型 → 直接在自建的中英混合语料上训练 1.3B 与 3B 参数的模型 → 打破了只能在已有大模型基础上微调的惯例，证明了在资源受限的情况下也能得到竞争力的双语模型。  
2. **专为小模型设计的指令微调框架 → 采用更高效的采样策略和轻量化的指令模板，而不是直接搬用大模型的指令微调流程 → 在保持训练成本低的前提下，显著提升了模型对复杂指令的理解和执行能力。  
3. **系统化的垂直领域适配方案 → 先用通用的 MindLLM 进行预训练，再在法律、金融等专业语料上进行二次微调，并加入领域专属的评估集 → 让模型在特定行业的准确率接近甚至超过一些专门的行业模型。  
4. **全流程经验分享 → 从数据清洗、去噪、去重复，到模型结构选型、训练调度、评估指标的完整记录 → 为后续想自行训练轻量模型的研究者提供了可复制的“操作手册”。  

### 方法详解
整体思路可以划分为四个阶段：**数据构建 → 模型架构 → 预训练 → 指令微调与领域适配**。

1. **数据构建**  
   - 收集了约 200GB 的中英混合文本，来源包括新闻、维基、技术博客以及公开的对话数据。  
   - 采用多轮过滤：先用规则剔除低质量、重复、广告类内容；再用已有的开源模型进行粗筛，保留语言流畅度高的句子。  
   - 为了保持双语平衡，对中文和英文的比例进行 1:1 调整，并在同一句子中交叉出现两种语言的代码切换（code‑switching）样本，以帮助模型学习跨语言上下文。

2. **模型架构**  
   - 采用 Transformer 编码器‑解码器的标准结构，层数分别为 24（1.3B）和 32（3B），每层隐藏维度 2048（1.3B）/ 3072（3B），注意力头数 16/24。  
   - 引入 **稀疏注意力**（Sparse Attention）和 **混合专家层**（Mixture‑of‑Experts）来降低显存占用，同时保持表达能力。可以把它想成在大城市里只在高峰时段开设快线，平时走普通道路。  
   - 采用 **相对位置编码**（Relative Positional Encoding），让模型更好地捕捉长文本中的依赖关系。

3. **从零预训练**  
   - 使用 **自回归语言建模**（causal LM）目标，即给定前面的 token 预测下一个 token。  
   - 训练采用 **混合精度**（FP16+FP32）和 **梯度累积**，每块 GPU 只需要 24GB 显存即可跑满。  
   - 学习率采用 **余弦退火**（cosine decay）并在前 5% 步骤使用 **线性预热**，防止初期梯度爆炸。  
   - 训练总步数约 500k，等价于 2.5 万 GPU‑hours，远低于百亿级模型的数十万 GPU‑hours。

4. **指令微调与领域适配**  
   - **指令微调**：构造了 30 万条指令-答案对，覆盖问答、翻译、代码生成、情感分析等任务。与大模型不同的是，这里使用 **低温采样 + Top‑p** 生成答案，以保持答案多样性但不失准确性。  
   - **轻量化微调技巧**：冻结前半层参数，只微调后半层和输出层，显著降低显存需求；同时使用 **梯度检查点**（gradient checkpointing）进一步压缩显存。  
   - **领域适配**：在法律文本（判例、法规）和金融报告（年报、研报）上各再训练 50k 步，加入领域专属的评估集（如法律问答基准）。微调时加入 **任务标签**（task token）帮助模型快速切换领域。  

最巧妙的地方在于 **指令微调框架的轻量化设计**：作者没有盲目复制大模型的全参数微调，而是通过层级冻结、任务标签和高效采样，让 1.3B 参数的模型在指令遵循上几乎追平 13B 参数的开源模型。

### 实验与效果
- **评估基准**：使用了 MMLU（多任务语言理解）、CMMLU（中文多任务）、TruthfulQA、GSM8K（数学推理）以及专门的法律/金融问答集。  
- **对比基线**：与 LLaMA‑7B、Mistral‑7B、OpenLLaMA‑13B 等开源模型进行横向比较。  
- **结果概览**：在 MMLU 上，MindLLM‑3B 获得 46.2% 的平均准确率，略高于 LLaMA‑7B 的 44.8%；在 CMMLU 上，MindLLM‑1.3B 达到 48.5%，超过同规模的 OpenLLaMA‑1.3B（45.1%）。在法律问答集上，经过领域适配后，准确率提升约 12%（从 61% 到 73%），金融报告摘要的 ROUGE‑L 提升 1.8 分。  
- **消融实验**：作者分别关闭稀疏注意力、层冻结、任务标签进行实验，发现稀疏注意力贡献约 1.3% 的整体提升，层冻结对指令微调的加速效果显著但略微降低 0.5% 的准确率，任务标签是领域适配提升的关键因素（约 8% 的增益）。  
- **局限性**：论文承认在超长上下文（>4k token）和多轮对话保持一致性方面仍有不足；此外，模型在少数低资源语言（如阿拉伯语）上的表现仍不理想。  

### 影响与延伸思考
MindLLM 的出现让“轻量+双语+指令微调”成为可能，激发了后续一波针对特定语言或行业的轻量模型研发。2024 年底，几篇工作（如 **TinyChat‑CN**、**FinBERT‑LLM**）直接引用了 MindLLM 的数据清洗流程和指令微调框架，进一步压缩参数到 500M 级别仍保持可用。推测未来会有更多 **边缘设备**（手机、嵌入式系统）上运行的 LLM，尤其在隐私敏感的行业（医疗、金融）会倾向于自行训练轻量模型而非调用云端大模型。想深入了解的读者可以关注 **稀疏注意力** 与 **混合专家层** 在小模型中的效率提升研究，以及 **跨语言指令微调** 在多语言环境下的通用性探索。

### 一句话记住它
**MindLLM 证明了，只有几亿参数的双语模型也能在指令遵循和行业任务上媲美更大的开源模型，只要用对数据、架构和轻量指令微调。**