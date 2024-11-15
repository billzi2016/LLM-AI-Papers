# LLaVA-CoT: Let Vision Language Models Reason Step-by-Step

> **Date**：2024-11-15
> **arXiv**：https://arxiv.org/abs/2411.10440

## Abstract

Large language models have demonstrated substantial advancements in reasoning capabilities. However, current Vision-Language Models (VLMs) often struggle to perform systematic and structured reasoning, especially when handling complex visual question-answering tasks. In this work, we introduce LLaVA-CoT, a large VLM designed to conduct autonomous multistage reasoning. Unlike chain-of-thought prompting, LLaVA-CoT independently engages in sequential stages of summarization, visual interpretation, logical reasoning, and conclusion generation. This structured approach enables LLaVA-CoT to achieve marked improvements on reasoning-intensive tasks. To accomplish this, we construct the LLaVA-CoT-100k dataset, integrating samples from various visual question answering sources and providing structured reasoning annotations. Besides, we propose a test-time stage-wise retracing search method (SWIRES), which enables effective and efficient test-time scaling. Remarkably, with only 100k training samples and test-time scaling, LLaVA-CoT not only outperforms its base model by 9.4% on a wide range of multimodal reasoning benchmarks, but also surpasses the performance of larger and even closed-source models, such as Gemini-1.5-pro, GPT-4o-mini, and Llama-3.2-90B-Vision-Instruct. The code, dataset, and pre-trained weights are publicly available at https://github.com/PKU-YuanGroup/LLaVA-CoT.

---

# LLaVA‑CoT：让视觉语言模型逐步推理 论文详细解读

### 背景：这个问题为什么难？

视觉语言模型（VLM）需要把图像信息和文字指令融合后给出答案，但大多数模型在面对需要多步推理的视觉问答时会直接给出结论，缺乏中间的思考过程。传统的 VLM 往往只在视觉特征上做一次注意力加权，然后交给语言模型生成答案，这种“一次性”方式在处理涉及多对象关系、计数、比较或隐含逻辑的题目时容易出错。换句话说，模型缺少把复杂视觉信息拆解成若干可管理子任务的能力，这也是近年来大语言模型（LLM）通过“思维链”（CoT）提升推理的关键所在，却没有被完整搬到多模态场景中。

### 关键概念速览
- **视觉语言模型（VLM）**：能够同时接受图像和文字输入，并输出文字答案的模型，类似于把看图和聊天功能合在一起的机器人。  
- **思维链（CoT，Chain‑of‑Thought）**：让模型在给出最终答案前先写出推理步骤，就像解数学题时先列出算式再算出结果。  
- **多阶段推理**：把一次完整的推理过程拆成若干明确的子阶段，每个阶段只负责一种特定的认知任务，例如先把图像要点写下来，再做逻辑推导。  
- **LLaVA‑CoT‑100k 数据集**：作者手工标注的 10 万条视觉问答样本，每条样本都配有四段结构化的推理注释（摘要、视觉解释、逻辑推理、结论），相当于给模型提供了“思考模板”。  
- **阶段式回溯搜索（SWIRES）**：在测试时动态决定每一步是否需要回头检查或重新生成，类似于人类在解题时发现前一步不对就回头改正的过程。  
- **自监督微调**：在已有的大语言模型基础上，用视觉-语言对进行进一步训练，使模型学会在视觉上下文中使用语言的推理能力。  

### 核心创新点
1. **从单轮输出到四阶段自驱动**  
   - 之前的 VLM 多采用一次性生成答案的方式，或者依赖外部 CoT 提示。  
   - LLaVA‑CoT 让模型自行进入“摘要 → 视觉解释 → 逻辑推理 → 结论”四个阶段，每个阶段的输出都作为下一个阶段的输入。  
   - 这种自我组织的流程让模型在复杂视觉任务上表现出更系统的思考，显著提升了答案的正确率。

2. **结构化推理标注的训练数据**  
   - 过去缺少大规模、细粒度的多模态 CoT 数据，导致模型难以学习阶段化的思考方式。  
   - 作者构建了 LLaVA‑CoT‑100k，给每条问答配上完整的四段推理文本，等于是给模型提供了“思考脚本”。  
   - 只用了 10 万条样本，却让模型在多模态推理基准上提升了约 9.4%。

3. **测试时的阶段式回溯搜索（SWIRES）**  
   - 传统测试只跑一次前向传播，若某一步出错就直接得到错误答案。  
   - SWIRES 在每个阶段结束后评估生成的可信度，若低于阈值则重新生成或回溯到前一阶段，类似于人类的“检查-改正”循环。  
   - 该机制在不显著增加计算成本的前提下，进一步提升了模型的鲁棒性。

4. **跨模型对标，超越更大闭源系统**  
   - 通过上述两大技术（多阶段推理 + SWIRES），LLaVA‑CoT 在公开的多模态推理基准上超过了 Gemini‑1.5‑pro、GPT‑4o‑mini 以及 Llama‑3.2‑90B‑Vision‑Instruct 等更大、更昂贵的模型，展示了“少量数据+结构化推理”可以弥补规模劣势。

### 方法详解
**整体框架**  
LLaVA‑CoT 把一次视觉问答拆成四个顺序模块：  
1) **摘要（Summarization）**：把问题和图像的整体信息压缩成一句话或几句话的概览。  
2) **视觉解释（Visual Interpretation）**：基于图像特征，描述关键对象、属性和空间关系。  
3) **逻辑推理（Logical Reasoning）**：把摘要和视觉解释当作前提，进行一步或多步的演绎、比较或计数。  
4) **结论生成（Conclusion Generation）**：把推理结果转化为最终答案。  

每个模块都是同一个大语言模型（LLaVA 的语言骨干）在不同的提示下运行，模型内部共享视觉编码器的输出，只是提示词（prompt）指明当前阶段的任务。

**关键模块拆解**  
- **视觉编码**：使用预训练的 CLIP‑ViT 或类似的视觉 Transformer 将图像映射为一组向量。  
- **阶段提示**：例如，摘要阶段的提示可能是 “请用一句话概括以下图片和问题的核心”。视觉解释阶段的提示是 “列出图片中出现的主要对象及其属性”。逻辑推理阶段的提示是 “基于上述信息，逐步推导出答案”。结论阶段的提示是 “给出最终答案”。这些提示在训练时与标注的四段文本一起喂入模型，使模型学会对应的输出格式。  
- **信息流**：摘要的输出直接拼接到视觉解释的输入，视觉解释的输出再拼接到逻辑推理的输入，依此类推。这样模型在每一步都能看到前一步的完整文字描述，形成“文字记忆”。  
- **SWIRES（Stage‑Wise Retracing Search）**：在推理时，每完成一个阶段模型会计算一个置信度分数（比如基于生成的 log‑prob）。如果分数低于预设阈值，系统会回到上一阶段重新生成，或者在当前阶段进行多次采样取最佳。整个过程在推理图上形成一个小的搜索树，但因为只有四层，搜索空间极小，实际开销接近一次前向传播。

**最巧妙的设计**  
- **自驱动阶段切换**：模型不依赖外部 CoT 提示，而是内部通过提示词和前一步输出自行决定进入下一个阶段，这让系统在真实使用时更像人类的思考流程。  
- **结构化标注的“教学”效应**：通过在训练中强制模型输出四段固定格式，模型学会了在不同抽象层次上组织信息，这种“教学式”训练比单纯的答案对齐更能提升推理能力。

### 实验与效果
- **测试数据**：作者在多个公开的多模态推理基准上评估，包括 VQA‑Reason、ScienceQA‑Vis、NLVR2‑Logic 等，这些任务都要求模型进行计数、比较或多步逻辑。  
- **整体提升**：在所有基准的平均得分上，LLaVA‑CoT 比其未使用多阶段推理的基线模型高出约 9.4%。  
- **对标更大模型**：在同样的评测环境下，LLaVA‑CoT 超过了 Gemini‑1.5‑pro、GPT‑4o‑mini 以及 Llama‑3.2‑90B‑Vision‑Instruct，说明即使模型规模相对较小，结构化推理也能带来显著优势。  
- **消融实验**：论文中分别去掉（1）四阶段结构，仅保留一次性生成；（2）SWIRES 回溯机制；（3）结构化标注的训练数据。结果显示，去掉任何一项都会导致整体性能下降 3%~6%，其中阶段化最为关键。  
- **局限性**：作者承认当前的四阶段设计仍然是固定的，面对需要更多或更少步骤的任务时可能不够灵活；此外，SWIRES 的阈值需要手动调节，自动化仍是未来工作。

### 影响与延伸思考
LLaVA‑CoT 把“思维链”从纯文本扩展到视觉语言，多模态社区随后出现了不少模仿其阶段化设计的工作，例如将视觉解释细化为“属性抽取 + 关系图构建”。还有研究尝试让阶段数自适应，或者把阶段输出直接转化为可执行的程序代码（Program‑of‑Thought）。如果想进一步深入，可以关注以下方向：  
- **自适应阶段生成**：让模型根据问题难度动态决定需要几步推理。  
- **跨模态图结构学习**：把视觉解释阶段的文字描述转化为显式的图结构，再用于逻辑推理。  
- **更高效的回溯搜索**：结合强化学习或贝叶斯优化，让模型在测试时自动学习最佳回溯策略。  

### 一句话记住它
让视觉语言模型像人一样分四步思考，并在每步不确定时回头检查，少量结构化数据就能把推理能力甩开更大模型。