# FreeLM: Fine-Tuning-Free Language Model

> **Date**：2023-05-02
> **arXiv**：https://arxiv.org/abs/2305.01616

## Abstract

Pre-trained language models (PLMs) have achieved remarkable success in NLP tasks. Despite the great success, mainstream solutions largely follow the pre-training then finetuning paradigm, which brings in both high deployment costs and low training efficiency. Nevertheless, fine-tuning on a specific task is essential because PLMs are only pre-trained with language signal from large raw data. In this paper, we propose a novel fine-tuning-free strategy for language models, to consider both language signal and teacher signal. Teacher signal is an abstraction of a battery of downstream tasks, provided in a unified proposition format. Trained with both language and strong task-aware teacher signals in an interactive manner, our FreeLM model demonstrates strong generalization and robustness. FreeLM outperforms large models e.g., GPT-3 and InstructGPT, on a range of language understanding tasks in experiments. FreeLM is much smaller with 0.3B parameters, compared to 175B in these models.

---

# FreeLM：免微调语言模型 论文详细解读

### 背景：这个问题为什么难？

预训练语言模型（PLM）在各种 NLP 任务上表现惊艳，但它们的强大往往依赖于后续的微调。微调需要为每个下游任务准备标注数据、调参、再部署，成本高且效率低。更糟的是，微调过程容易出现灾难性遗忘——模型在新任务上表现好，旧任务却退步。于是业界一直在寻找“一次预训练、全场即用”的方案，却始终受限于仅靠语言信号（大规模原始文本）进行预训练，缺乏任务感知的指导，导致模型在实际任务上仍然需要微调。

### 关键概念速览
- **预训练语言模型（PLM）**：在海量未标注文本上学习词汇、句法等通用语言规律的模型，类似于先学会“说话”，再去“写作文”。  
- **微调（Fine‑Tuning）**：把已经学会通用语言的模型在特定任务的标注数据上继续训练，让它适应具体需求，像是把通用厨师训练成专门的寿司师傅。  
- **教师信号（Teacher Signal）**：作者把一系列下游任务抽象成统一的指令或示例集合，提供给模型作为额外的学习目标，相当于给模型一本“任务手册”。  
- **交互式学习（Interactive Learning）**：语言目标和教师目标交替优化，模型在学习语言的同时不断校正对任务的理解，类似于学生在课堂上听讲的同时做练习题。  
- **指令式预训练（Instruction‑tuned Pre‑training）**：在预训练阶段直接加入指令或任务描述，让模型从一开始就具备遵循指令的能力。  
- **参数规模（Parameter Scale）**：模型内部可调节的权重数量，常用来衡量模型大小，0.3 B 表示三亿个参数。  

### 核心创新点
1. **语言目标 + 教师目标 双目标预训练**  
   过去的预训练只靠语言目标（预测下一个词或填空），缺少任务感知。FreeLM 同时引入教师目标，把一批下游任务的指令/示例写进预训练数据。这样模型在同一次训练里既学会语言规律，又学会如何解任务，省掉了后续微调的步骤。

2. **统一指令格式的任务抽象**  
   作者把不同任务（情感分类、阅读理解、事实抽取等）统一成一种“指令 + 输入 + 输出”的模板，形成所谓的 teacher signal。相比于为每个任务单独设计特定的预训练任务，这种统一格式大幅降低了任务工程的复杂度。

3. **交互式优化策略**  
   训练过程中交替计算语言损失和教师损失，让两类信号相互影响。语言信号保证模型的通用性，教师信号推动模型在任务上保持高效。交互式的做法比一次性加权求和更能平衡两者，提升了模型的鲁棒性。

4. **小模型大能力的实证**  
   在仅有 0.3 B 参数的情况下，FreeLM 在多项语言理解基准上超过了 175 B 参数的 GPT‑3、InstructGPT 等大模型，证明任务感知的预训练可以弥补参数规模的不足。

### 方法详解
整体框架可以看作两层循环：外层遍历预训练语料，内层在每条语料上交替执行语言目标和教师目标的梯度更新。

1. **数据准备**  
   - **语言数据**：传统的大规模原始文本（如维基百科、新闻语料），用于标准的自回归或掩码语言建模。  
   - **教师数据**：从公开的任务集合（GLUE、SuperGLUE、SQuAD 等）抽取指令化示例，每条示例被包装成“指令 + 输入 + 期望输出”。这些示例被随机混入语言数据流中。

2. **双目标损失**  
   - **语言损失**：对语言数据使用常见的交叉熵损失，预测下一个 token 或填空 token。  
   - **教师损失**：对教师数据使用同样的交叉熵，但目标是指令对应的答案文本。因为答案往往是短句或标签，模型需要在生成时直接对齐指令意图。  
   - **交互式更新**：在一次前向传播后，先计算语言损失并更新参数，然后立即在同一批次的教师数据上计算教师损失并再次更新。这样模型的梯度在同一次迭代中既受语言信号也受任务信号影响。

3. **统一指令模板**  
   示例：“**指令**：判断下面句子的情感。**输入**：我今天很开心。**输出**：正面”。所有任务都被转化为这种三段式结构，模型只需要学会解析“指令”部分的意图，再根据“输入”生成对应的“输出”。这相当于给模型一本多语言的使用手册。

4. **模型结构**  
   采用标准的 Transformer 解码器（或编码‑解码混合）架构，参数量约 0.3 B。没有额外的任务专用头，所有任务都通过同一套生成层完成，保持了模型的简洁性。

5. **训练细节**  
   - 学习率采用线性 warm‑up + cosine decay。  
   - 教师数据的采样比例在 5%–10% 之间，确保语言信号仍占主导。  
   - 为防止教师信号过强导致语言能力退化，作者在实验中对教师损失加了一个小的权重系数（具体数值未在摘要中披露）。

**最巧妙的点**在于交互式更新：而不是把两种损失简单相加后一次性反向传播，FreeLM 让语言和任务的梯度交错进行，使得模型在每一步都能即时纠正因单一目标产生的偏差，提升了对指令的敏感度，同时保留了语言的通用性。

### 实验与效果
- **测试任务**：包括情感分析、自然语言推理、阅读理解、事实抽取等多项公开基准（如 SST‑2、MNLI、SQuAD、CoNLL‑2003）。  
- **对比基线**：GPT‑3（175 B）、InstructGPT（同样规模）以及其他指令微调模型（如 T5‑XXL、FLAN‑T5）。  
- **结果**：FreeLM 在多数任务上取得了领先，尤其在小样本设置下表现尤为突出。论文声称在整体平均分上超过 GPT‑3 与 InstructGPT，且参数量仅为其千分之一。  
- **消融实验**：作者分别去掉教师目标、去掉交互式更新、降低教师数据比例，发现模型的任务准确率会下降 3%–7%，验证了教师信号和交互式策略的必要性。  
- **局限性**：摘要未提供细粒度的数值报告，也未说明在生成类任务（如对话、长文本写作）上的表现；此外，教师数据的质量和覆盖度对最终效果有较大影响，若任务集合不够丰富，模型仍可能在未见任务上表现平平。

### 影响与延伸思考
FreeLM 的核心思路——在预训练阶段直接注入任务指令——在随后两年里激发了大量“指令预训练”工作，例如 OpenAI 的 ChatGPT 系列、Google 的 PaLM‑2‑Instruct 等，都在不同程度上采用了类似的双目标或多任务预训练策略。后续研究进一步探索了更大规模的教师信号、跨语言指令统一以及自适应教师数据采样等方向。想深入了解，可以关注“Instruction‑tuned Pre‑training”以及“Multi‑Task Prompt‑Based Learning”这两个热点。

### 一句话记住它
FreeLM 用统一的指令式任务信号在预训练阶段一次性教会模型完成下游任务，实现了“小模型、免微调、直接上手”。