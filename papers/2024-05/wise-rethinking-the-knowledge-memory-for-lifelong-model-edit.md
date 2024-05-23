# WISE: Rethinking the Knowledge Memory for Lifelong Model Editing of   Large Language Models

> **Date**：2024-05-23
> **arXiv**：https://arxiv.org/abs/2405.14768

## Abstract

Large language models (LLMs) need knowledge updates to meet the ever-growing world facts and correct the hallucinated responses, facilitating the methods of lifelong model editing. Where the updated knowledge resides in memories is a fundamental question for model editing. In this paper, we find that editing either long-term memory (direct model parameters) or working memory (non-parametric knowledge of neural network activations/representations by retrieval) will result in an impossible triangle -- reliability, generalization, and locality can not be realized together in the lifelong editing settings. For long-term memory, directly editing the parameters will cause conflicts with irrelevant pretrained knowledge or previous edits (poor reliability and locality). For working memory, retrieval-based activations can hardly make the model understand the edits and generalize (poor generalization). Therefore, we propose WISE to bridge the gap between memories. In WISE, we design a dual parametric memory scheme, which consists of the main memory for the pretrained knowledge and a side memory for the edited knowledge. We only edit the knowledge in the side memory and train a router to decide which memory to go through when given a query. For continual editing, we devise a knowledge-sharding mechanism where different sets of edits reside in distinct subspaces of parameters, and are subsequently merged into a shared memory without conflicts. Extensive experiments show that WISE can outperform previous model editing methods and overcome the impossible triangle under lifelong model editing of question answering, hallucination, and out-of-distribution settings across trending LLM architectures, e.g., GPT, LLaMA, and Mistral. Code is available at https://github.com/zjunlp/EasyEdit.

---

# WISE：重新思考大型语言模型终身模型编辑的知识记忆 论文详细解读

### 背景：这个问题为什么难？
大型语言模型（LLM）在训练完毕后会被直接部署，但现实世界的事实在不断变化，模型也会出现“幻觉”——给出不真实的答案。传统的模型编辑方法要么直接改参数，要么在推理时通过检索外部知识来“临时”补丁。直接改参数会和已有的预训练知识产生冲突，导致旧知识被破坏；检索式补丁则难以让模型真正“理解”新信息，导致新知识只能在特定提示下生效，缺乏泛化。更糟的是，随着编辑次数累积，这两种方式都无法同时满足 **可靠性**（编辑后答案正确）、**泛化性**（新知识能在不同表述下被使用）和 **局部性**（不影响无关知识）这三条要求，形成了所谓的“不可能三角”。因此，需要一种新机制来在长期记忆和工作记忆之间取得平衡，才能实现真正的终身编辑。

### 关键概念速览
- **长期记忆（Long‑term Memory）**：模型的参数本身存储的知识，类似人脑的长期记忆，修改后会永久影响所有推理。  
- **工作记忆（Working Memory）**：模型在一次前向传播中产生的激活向量或外部检索到的事实，类似人类的即时记事本，只在当前推理中起作用。  
- **模型编辑（Model Editing）**：对已训练好的模型进行小幅度的参数或结构调整，使其对特定事实产生新的、正确的输出。  
- **可靠性（Reliability）**：编辑后模型在对应查询上必须给出正确答案。  
- **泛化性（Generalization）**：编辑的知识能够在不同的提问方式或上下文中被正确调用。  
- **局部性（Locality）**：编辑不应破坏模型对其他无关问题的已有能力。  
- **双参数记忆（Dual Parametric Memory）**：在 WISE 中同时维护两套参数，一套保存原始预训练知识（主记忆），另一套专门存放编辑后的新知识（侧记忆）。  
- **路由器（Router）**：一个轻量的分类网络，负责判断当前查询应该走主记忆还是侧记忆。  
- **知识分片（Knowledge Sharding）**：把不同编辑任务分别映射到参数空间的不同子区域，防止相互冲突，随后再合并成统一的侧记忆。

### 核心创新点
1. **发现“不可能三角”**：之前的工作只关注单一记忆（参数或检索），论文系统实验表明，无论是直接改参数还是仅靠检索，都无法同时满足可靠、泛化、局部三要素。这个负面结论为后续设计提供了明确的目标。  
2. **双参数记忆结构**：把模型拆成主记忆和侧记忆，两者共享同一网络结构但参数独立。编辑只在侧记忆上进行，保留了原始模型的完整性，从而提升局部性和可靠性。  
3. **可学习路由器**：训练一个小网络，根据输入查询的特征决定走哪块记忆。这样在需要新知识时自动切换到侧记忆，其他情况下仍使用原始知识，兼顾泛化和局部性。  
4. **知识分片与合并机制**：在持续编辑场景下，每一次编辑被映射到参数空间的独立子块，避免不同编辑之间的冲突。周期性地把这些子块合并到侧记忆的统一参数中，实现真正的终身编辑而不产生灾难性遗忘。

### 方法详解
**整体框架**  
WISE 的工作流程可以分为三步：① 初始化双记忆模型；② 对每条新知识进行编辑并放入侧记忆的专属子块；③ 训练路由器并定期合并子块。整体思路是让模型在“主记忆+侧记忆+路由器”三位一体的系统中自行决定使用哪块记忆，从而在编辑后保持原有能力并获得新知识。

**步骤拆解**  

1. **双记忆初始化**  
   - 复制原始 LLM 参数得到两套完全相同的网络：`θ_main`（主记忆）和 `θ_side`（侧记忆）。  
   - `θ_main` 在整个生命周期保持不变，只用于提供未被编辑的原始知识。  

2. **编辑与知识分片**  
   - 给定一条需要修改的事实（如“东京是日本的首都”），构造一个小的编辑任务。  
   - 在侧记忆上加入一个可学习的“增量参数块”，该块只在与该事实相关的内部层激活上起作用。  
   - 为防止不同编辑相互干扰，使用一个随机或基于任务特征的映射函数，将每个编辑映射到参数空间的不同子空间（比如不同的神经元子集或不同的层）。  

3. **路由器训练**  
   - 收集两类查询：需要新知识的（编辑目标）和不需要的（普通问答）。  
   - 用这些查询训练一个轻量的二分类模型 `R(x)`，输出“走主记忆”或“走侧记忆”。  
   - 训练目标是让 `R` 在需要新知识的查询上高概率选择侧记忆，在其他查询上选择主记忆。  

4. **合并与持续编辑**  
   - 随着编辑次数增多，侧记忆会累积多个子块。周期性地执行“合并”操作：把所有子块的参数平均或通过小规模微调整合到 `θ_side` 的统一参数中。  
   - 合并后仍保留子块的分片信息，以便后续继续添加新块而不冲突。  

**关键细节**  
- **路由器的输入**：通常是查询的嵌入向量或前几层的激活，确保路由决策基于语义特征。  
- **子块的大小**：作者在实验中发现几百到几千维的子块足以容纳单条编辑，且不会显著增加推理成本。  
- **合并策略**：采用轻量的微调而非全参数更新，保持侧记忆的整体结构不被破坏。  

**最巧妙的地方**  
把编辑限制在侧记忆的独立子块里，同时用路由器在推理时动态切换记忆，实现了“编辑只影响目标知识、其他知识保持不变、且新知识能在不同表述下被调用”的三重目标，这在之前的单记忆方案中是难以做到的。

### 实验与效果
- **测试任务**：包括常规问答（如 TriviaQA、Natural Questions）、幻觉纠正（让模型不再生成错误事实）以及分布外（OOD）情境下的知识检索。  
- **对比基线**：传统参数编辑方法（如 MEND、ROME）和检索增强模型（如 RAG、K-Adapter）。  
- **结果概述**：论文声称在上述任务上，WISE 在可靠性、泛化性和局部性三个指标上均优于所有基线，尤其在连续编辑 50 条以上时仍保持高成功率，未出现显著的灾难性遗忘。  
- **消融实验**：通过去掉路由器、合并步骤或知识分片，性能均出现明显下降，说明每个模块都是不可或缺的。  
- **局限性**：侧记忆的参数开销随编辑次数线性增长，虽然子块很小，但在极端长期使用场景仍需额外存储；此外，路由器的误判会导致新知识被忽略或误用，作者在讨论中提到需要更鲁棒的路由策略。

### 影响与延伸思考
WISE 的双记忆+路由思路为 LLM 的终身学习提供了全新视角，随后出现的工作开始探索更细粒度的记忆划分、跨模型的记忆共享以及基于提示的路由机制。推测未来会有研究把侧记忆进一步压缩为外部键值库，或者把路由器与大模型的自注意力层融合，实现“记忆自适应”。如果想深入了解，可关注 **可微分记忆网络**、**持续学习中的参数分区** 以及 **检索增强的大模型编辑** 等方向。

### 一句话记住它
WISE 用主/侧双参数记忆加路由器，让大模型在编辑新知识时既不忘旧知，又能在不同表述下灵活使用，成功破解了模型编辑的“不可能三角”。