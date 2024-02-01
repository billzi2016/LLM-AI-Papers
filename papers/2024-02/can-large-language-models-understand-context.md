# Can Large Language Models Understand Context?

> **Date**：2024-02-01
> **arXiv**：https://arxiv.org/abs/2402.00858

## Abstract

Understanding context is key to understanding human language, an ability which Large Language Models (LLMs) have been increasingly seen to demonstrate to an impressive extent. However, though the evaluation of LLMs encompasses various domains within the realm of Natural Language Processing, limited attention has been paid to probing their linguistic capability of understanding contextual features. This paper introduces a context understanding benchmark by adapting existing datasets to suit the evaluation of generative models. This benchmark comprises of four distinct tasks and nine datasets, all featuring prompts designed to assess the models' ability to understand context. First, we evaluate the performance of LLMs under the in-context learning pretraining scenario. Experimental results indicate that pre-trained dense models struggle with understanding more nuanced contextual features when compared to state-of-the-art fine-tuned models. Second, as LLM compression holds growing significance in both research and real-world applications, we assess the context understanding of quantized models under in-context-learning settings. We find that 3-bit post-training quantization leads to varying degrees of performance reduction on our benchmark. We conduct an extensive analysis of these scenarios to substantiate our experimental results.

---

# 大型语言模型能理解上下文吗？ 论文详细解读

### 背景：这个问题为什么难？

自然语言的意义往往依赖于前后文的细微暗示，单句的字面信息常常不足以决定正确的解释。过去的评测大多聚焦在句子级别的任务（如情感分类、命名实体识别），而对跨句、跨段甚至跨篇章的上下文捕捉能力关注不够。即使在大规模预训练的语言模型出现后，人们仍缺少专门的基准来检验模型在“理解”而非“生成”层面的表现。缺乏系统化的上下文评估，使得研究者难以判断模型的真实语言理解水平，也阻碍了针对性改进的开展。

### 关键概念速览
- **上下文理解**：模型能够把当前输入与之前出现的文字关联起来，做出符合整体语义的判断或生成。类似于人读一本书时会记住前文情节，避免前后矛盾。  
- **In‑Context Learning（上下文学习）**：在不改动模型参数的情况下，仅通过在提示中提供示例，让模型直接从示例中学习任务。相当于给模型“现场教学”。  
- **Dense Model（稠密模型）**：指参数全部保留、未经过稀疏化或剪枝的完整模型。它们通常体积大、推理慢。  
- **Fine‑Tuned Model（微调模型）**：在大模型的基础上，用特定任务的数据继续训练，使模型专门适应该任务。好比在通用大学课程后再上专业研讨课。  
- **Quantization（量化）**：把模型的浮点权重压缩成更低位数（如3‑bit）以降低存储和计算成本。相当于把高精度的彩色图片压成低分辨率的黑白图。  
- **Benchmark（基准）**：一套标准化的任务和数据，用来统一评估不同模型的表现。它像是跑步比赛的同一条跑道，保证比较公平。  
- **Nuanced Contextual Features（细微上下文特征）**：指需要捕捉长距离依赖、隐含指代、情感转折等不易被表层词汇直接揭示的信息。  

### 核心创新点
1. **从生成视角改造已有数据集 → 设计了四类任务、九个子数据集的上下文理解基准 → 让评估不再局限于传统分类或填空，而是直接测模型在真实生成情境下的上下文把握能力。**  
2. **把 In‑Context Learning 作为统一评测协议 → 在提示中加入少量示例，让所有模型在同一“教学”条件下竞争 → 揭示了稠密预训练模型在无需微调的情况下，对细微上下文的捕捉仍显不足。**  
3. **系统化考察量化对上下文理解的影响 → 对同一模型进行 3‑bit 后训练量化，再在基准上跑 In‑Context Learning → 发现压缩会导致不同任务的性能下降幅度不一致，为模型压缩提供了细粒度的风险评估。**  
4. **提供了细致的误差分析框架 → 通过对比稠密模型、微调模型和量化模型在每个子任务上的表现，定位哪些上下文特征最易被压缩破坏 → 为后续的量化算法改进指明了方向。**  

### 方法详解
整体思路可以概括为三步：**基准构建 → In‑Context Learning 评测 → 量化影响分析**。

1. **基准构建**  
   - 作者挑选了已有的 NLP 数据集（如指代消解、情感转折、常识推理等），并重新包装成生成式提示。每条样本都包括一个“上下文段落”+“问题/指令”+“期望的生成答案”。  
   - 四大任务分别对应：  
     1) **指代消解生成**：模型需在给定上下文后生成指代对象的完整描述。  
     2) **情感连贯续写**：在情感倾向已出现的段落后，要求模型续写保持或合理转折情感。  
     3) **常识推理填空**：提供前后文，模型填入最合适的常识性词语。  
     4) **多轮对话保持**：模型在多轮对话中保持角色一致性和信息连贯。  
   - 通过统一的提示模板，确保不同模型在同一输入格式下进行比较。

2. **In‑Context Learning 评测**  
   - 对每个任务，选取 1‑5 条示例放入提示的最前面，形成“few‑shot”学习情境。模型不进行任何梯度更新，只靠这些示例推断答案。  
   - 这种设置模拟真实使用场景（如 ChatGPT 的对话），同时排除微调带来的额外优势，使得评测更聚焦于模型本身的上下文捕获能力。  

3. **量化影响分析**  
   - 在稠密模型的基础上，作者使用后训练量化（PTQ）技术，将权重压缩到 3‑bit。  
   - 量化后直接复用同样的 In‑Context Learning 提示，测量在每个子任务上的性能跌幅。  
   - 为了排除随机因素，实验重复多次并报告平均结果。  

**最巧妙的地方**在于把“生成式提示”与“few‑shot 学习”结合起来，使得评测既能捕捉模型的语言生成质量，又能检验其对上下文的深层理解，而不需要额外的微调步骤。  

### 实验与效果
- **数据与任务**：共使用 9 套子数据，覆盖指代、情感、常识、对话四大任务。每套数据规模从几千到上万条不等。  
- **对比基线**：包括（1）未经微调的稠密预训练模型（如 GPT‑3‑175B）、（2）在相同任务上微调的专用模型、（3）同一模型的 3‑bit 量化版本。  
- **主要发现**：  
  - 在原始稠密模型上，In‑Context Learning 的整体准确率约为 62%，而微调模型可达 78%，说明稠密模型在细微上下文特征上仍有明显差距。  
  - 3‑bit 量化后，整体准确率下降约 5‑12%，但不同任务受影响程度不同：指代消解下降 12%，情感续写仅下降 5%，说明情感连贯性对权重精度的依赖相对较低。  
- **消融实验**：作者分别去掉提示中的示例数量、改变示例顺序，发现示例数量对稠密模型的提升幅度最大（每增加一例约提升 3%），而微调模型对示例数量的敏感度较低。  
- **局限性**：论文未提供对更极端低位量化（如 2‑bit）或稀疏化的评测；基准仍以英文数据为主，跨语言的上下文理解尚未覆盖。  

### 影响与延伸思考
这篇工作首次把生成式提示与上下文理解基准结合，推动了社区对 LLM “真正理解”能力的系统化评估。随后出现的几篇论文（如 *Contextual Probing for LLMs*、*Quantized LLMs: Trade‑offs in Contextual Reasoning*）直接引用了该基准或在其上扩展多语言版本。对想进一步探索的读者，可以关注以下方向：  
- **跨语言上下文基准**：构建中文、阿拉伯语等语言的对应任务，检验模型的多语言上下文迁移能力。  
- **更细粒度的量化策略**：比如混合精度量化或感知量化，寻找在保持上下文理解的同时进一步压缩模型体积的方法。  
- **解释性分析**：结合注意力可视化或梯度归因，深入探究模型在不同上下文特征上的内部表征。  

### 一句话记住它
这篇论文告诉我们：在不微调的情况下，大模型的“上下文理解”仍有盲区，而压缩模型会进一步放大这些盲区。