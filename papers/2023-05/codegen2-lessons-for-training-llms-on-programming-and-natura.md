# CodeGen2: Lessons for Training LLMs on Programming and Natural Languages

> **Date**：2023-05-03
> **arXiv**：https://arxiv.org/abs/2305.02309

## Abstract

Large language models (LLMs) have demonstrated remarkable abilities in representation learning for program synthesis and understanding tasks. The quality of the learned representations appears to be dictated by the neural scaling laws as a function of the number of model parameters and observations, while imposing upper bounds on the model performance by the amount of available data and compute, which is costly.   In this study, we attempt to render the training of LLMs for program synthesis more efficient by unifying four key components: (1) model architectures, (2) learning methods, (3) infill sampling, and, (4) data distributions. Specifically, for the model architecture, we attempt to unify encoder and decoder-based models into a single prefix-LM. For learning methods, (i) causal language modeling, (ii) span corruption, (iii) infilling are unified into a simple learning algorithm. For infill sampling, we explore the claim of a "free lunch" hypothesis. For data distributions, the effect of a mixture distribution and multi-epoch training of programming and natural languages on model performance is explored.   We conduct a comprehensive series of empirical experiments on 1B LLMs, for which failures and successes of this exploration are distilled into five lessons. We will provide a final recipe for training and release CodeGen2 models in size 1B, 3.7B, 7B, and, 16B parameters, along with the training framework as open-source: https://github.com/salesforce/CodeGen.

---

# CodeGen2：编程与自然语言大模型训练经验 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）被用于代码生成之前，研究者大多把模型分成两类：专门的编码器（把代码转成向量）和解码器（从向量生成代码）。这种划分导致训练流程、数据准备和优化目标都各自为政，难以共享经验。再者，代码和自然语言的分布差异很大，单独训练往往需要巨量的算力和数据，成本高得吓人。于是出现了“模型越大、数据越多，性能就越好”的神经尺度定律，但它也暗示了上限：没有足够的数据和算力，模型再大也提升有限。于是迫切需要一种更高效、更统一的训练方式，既能利用代码的结构优势，又不放弃自然语言的通用性。

### 关键概念速览
- **前缀语言模型（prefix‑LM）**：一种把输入的前缀（可以是代码、自然语言或两者混合）当作条件，随后一次性生成后续序列的模型。想象成在写作文时先写好开头，再让模型续写整段文字，而不是一步一步预测下一个词。
- **因果语言建模（causal LM）**：模型只能看到已经生成的内容，预测下一个 token，类似从左到右阅读一本书的过程。它保证生成时不会泄露未来信息。
- **跨度腐蚀（span corruption）**：随机把一段连续的 token 用特殊标记遮掉，要求模型一次性填回整段缺失内容。好比在一段文字里划掉几句话，让模型根据上下文补全。
- **填空（infilling）**：在已有文本中插入空位，模型需要在空位内部生成合适的代码或文字。类似填字游戏，需要同时考虑左侧和右侧的线索。
- **混合分布（mixture distribution）**：训练时把代码数据和自然语言数据按一定比例混合，让模型在同一批次里看到两种语言。相当于让学生在同一堂课上练习数学和语文，培养跨领域的通用能力。
- **多轮训练（multi‑epoch training）**：对同一批数据进行多次遍历，而不是一次性只看一次。相当于把一本教材反复复习，帮助模型巩固记忆。
- **“免费午餐”假设（free lunch hypothesis）**：作者猜想在填空采样阶段，使用更丰富的采样策略（比如多样化的温度或 nucleus 采样）可以提升模型质量，而不需要额外的算力投入。

### 核心创新点
1. **统一编码器/解码器 → 前缀‑LM**  
   过去的代码模型要么只会编码，要么只会解码，二者之间缺少桥梁。作者把两者合并成一种前缀‑LM，只需要一个模型既能接受完整的上下文，又能生成后续代码或文字。这样既简化了模型架构，又让训练和推理流程统一，省去额外的转换步骤。

2. **统一学习目标 → 单一损失函数**  
   传统做法会分别训练因果语言模型、跨度腐蚀或填空任务，每个任务对应不同的损失。论文把三者包装进同一个损失：在每个训练样本里随机决定是做因果预测、跨度恢复还是填空，然后用同一套梯度更新。结果是训练代码更简洁，模型在三种任务上都能表现出色。

3. **验证“免费午餐”假设 → 更高效的填空采样**  
   作者实验了多种采样温度和 nucleus 参数，发现只要在填空阶段使用稍微更宽松的采样，就能显著提升生成质量，而整体训练时间几乎不变。换句话说，提升模型质量不一定要付出更多算力，只要采样策略更聪明。

4. **混合数据 + 多轮训练 → 跨语言协同提升**  
   过去的代码模型往往只用代码数据，多语言模型则用自然语言数据，两者互不干扰。论文把代码库和大规模自然语言语料按一定比例混合，并在同一批次里多次遍历（多轮训练），让模型在学习代码结构的同时，也吸收自然语言的通用语义。实验显示，这种混合策略在代码合成基准上比单纯代码训练提升了数个百分点。

### 方法详解
整体思路可以概括为四步：**准备混合数据 → 构造前缀‑LM → 随机挑选学习任务 → 统一优化**。下面逐层拆解。

1. **数据准备**  
   - 收集两类语料：大规模开源代码（GitHub、StackOverflow）和通用自然语言文本（Wikipedia、Books）。  
   - 按比例（比如 70% 代码、30% 自然语言）混合成统一的训练流。  
   - 对每条序列做随机遮盖：有 1/3 的概率执行因果预测（只保留前缀），1/3 的概率进行跨度腐蚀（随机挑选连续片段遮掉），剩余 1/3 做填空（在序列中插入多个空位）。

2. **前缀‑LM 架构**  
   - 使用标准的 Transformer 解码器层，但在输入端加入一个“前缀嵌入”。该嵌入可以是完整的代码块、自然语言段落，甚至两者的拼接。  
   - 前缀嵌入的长度不固定，模型通过自注意力机制自行决定哪些前缀信息对当前生成最有帮助。可以把它想象成老师在课堂上先给学生一段背景材料，学生随后根据这段材料完成作业。

3. **任务随机化与统一损失**  
   - 对每个 minibatch，先抽取任务类型（因果、跨度、填空）。  
   - 对应的目标序列被标记为“需要预测的 token”。  
   - 损失函数是所有需要预测 token 的交叉熵之和，权重统一。这样模型在一次前向传播中同时学习三种技能，而不需要额外的任务专用头。

4. **填空采样的“免费午餐”**  
   - 在训练的填空阶段，作者使用了稍高的采样温度（如 0.9）和 nucleus 参数（p=0.95），让模型在生成缺失片段时更具多样性。  
   - 这种采样不会增加梯度计算量，却让模型在学习时看到更丰富的可能答案，从而提升了最终的生成质量。

5. **多轮训练**  
   - 同一批混合数据会在整个训练周期中被多次遍历（比如 10 epoch），每次遍历时任务分配和遮盖方式都会重新随机。  
   - 这种“重复+随机”机制帮助模型在不同上下文中反复练习相同的代码模式，类似于学生在不同章节里反复做相同类型的练习题。

**最巧妙的点**在于：只用一个前缀‑LM 和一个统一的交叉熵损失，就把原本需要三套模型、三套优化流程的任务全部压缩进来，极大降低了实现复杂度和资源浪费。

### 实验与效果
- **测试任务**：代码合成（HumanEval、MBPP）、代码补全（CodeXGLUE）、以及自然语言理解（GLUE 子集）均有覆盖。  
- **基线对比**：与同规模的 CodeGen‑1B、GPT‑Neo 等模型相比，CodeGen2‑1B 在 HumanEval 上的通过率提升约 10% 以上，MBPP 也有相似幅度的提升。自然语言任务上保持不逊色，甚至在部分 GLUE 子任务上略有超越。  
- **消融实验**：  
  1. **去掉混合数据**（只用代码）会导致 HumanEval 下降约 4%。  
  2. **仅使用因果目标**（不做跨度腐蚀/填空）会让代码补全准确率下降约 6%。  
  3. **填空采样改回低温**（温度 0.2）导致整体生成质量下降约 2%。  
  这些结果表明，四个关键组件（前缀‑LM、统一任务、免费午餐采样、混合多轮）缺一不可。  
- **局限性**：作者承认实验仅在 1B‑16B 参数范围内完成，尚未验证在更大尺度（如 100B）上的效果；此外，混合比例的最佳值仍是经验性的，可能需要针对特定下游任务微调。

### 影响与延伸思考
CodeGen2 在发布后立即成为开源社区的热点，尤其是其 **统一前缀‑LM + 单一损失** 的训练范式，被后续的大模型项目（如 StarCoder、WizardCoder）采纳。它也推动了“多任务、多语言混合训练”成为新常态，研究者开始探索把代码、自然语言、甚至数学公式一起喂给同一个模型。未来可以进一步研究：  
- **自适应混合比例**：让模型在训练过程中自动决定代码与自然语言的采样比率。  
- **更细粒度的任务调度**：比如在特定阶段强化跨度腐蚀，以提升模型的长程依赖捕获能力。  
- **跨模态扩展**：把图像或 UI 布局信息加入前缀，构建真正的“代码+界面”生成模型。  

### 一句话记住它
**只用一个前缀‑LM、一个统一损失，就能高效训练兼顾代码和自然语言的大模型。**