# JiuZhang 2.0: A Unified Chinese Pre-trained Language Model for   Multi-task Mathematical Problem Solving

> **Date**：2023-06-19
> **arXiv**：https://arxiv.org/abs/2306.11027

## Abstract

Although pre-trained language models~(PLMs) have recently advanced the research progress in mathematical reasoning, they are not specially designed as a capable multi-task solver, suffering from high cost for multi-task deployment (\eg a model copy for a task) and inferior performance on complex mathematical problems in practical applications. To address these issues, in this paper, we propose \textbf{JiuZhang~2.0}, a unified Chinese PLM specially for multi-task mathematical problem solving. Our idea is to maintain a moderate-sized model and employ the \emph{cross-task knowledge sharing} to improve the model capacity in a multi-task setting. Specially, we construct a Mixture-of-Experts~(MoE) architecture for modeling mathematical text, so as to capture the common mathematical knowledge across tasks. For optimizing the MoE architecture, we design \emph{multi-task continual pre-training} and \emph{multi-task fine-tuning} strategies for multi-task adaptation. These training strategies can effectively decompose the knowledge from the task data and establish the cross-task sharing via expert networks. In order to further improve the general capacity of solving different complex tasks, we leverage large language models~(LLMs) as complementary models to iteratively refine the generated solution by our PLM, via in-context learning. Extensive experiments have demonstrated the effectiveness of our model.

---

# JiuZhang 2.0：面向多任务数学问题求解的统一中文预训练语言模型 论文详细解读

### 背景：这个问题为什么难？

数学题目往往包含符号、公式、推理步骤和自然语言描述，传统的中文预训练语言模型（PLM）在处理这类混合信息时会出现两大痛点：一是模型规模要足够大才能捕捉到足够的数学常识，导致部署成本高；二是不同数学任务（如代数求解、几何证明、应用题）之间缺少共享的知识结构，导致每个任务都需要单独微调一个模型，既浪费算力也难以保证在复杂题目上的鲁棒性。换句话说，现有方案要么“全能但笨重”，要么“轻量但只能干一件事”，这正是 JiuZhang 2.0 要破解的瓶颈。

### 关键概念速览

**预训练语言模型（PLM）**：先在大规模文本上学习语言规律，再通过微调适配特定任务的模型。想象成先学会通用的中文表达，再专门练习数学解题。

**Mixture-of-Experts（MoE）**：把一个大模型拆成若干“专家网络”，输入会被路由到最擅长的专家上处理。类似于公司里不同部门负责不同业务，整体效率更高。

**跨任务知识共享**：让不同数学任务在同一个模型内部共享通用的数学概念（如加法、方程求解），而不是各自为政。可以比作学生在学代数时也在潜移默化地巩固几何的空间想象。

**多任务持续预训练**：在多个任务的数据上继续进行自监督学习，让模型在不断接触新题型时保持知识的连贯性。相当于在工作中不断进修，而不是一次性学完就停下来。

**多任务微调**：在每个具体任务上做轻量级的参数调整，同时保留 MoE 中共享的专家。好比在公司内部调动员工到不同项目，仍然保持他们的核心技能。

**大语言模型（LLM）迭代精炼**：利用体量更大的模型（如 ChatGPT）对 JiuZhang 2.0 生成的答案进行二次校正，采用“在上下文中学习”的方式提升最终解答质量。类似于让资深老师检查学生的作业。

### 核心创新点

1. **从单任务模型到统一 MoE 框架**  
   之前的中文数学模型往往为每个任务训练独立的网络，导致参数冗余。JiuZhang 2.0 把所有任务放进同一个 MoE 结构，让不同任务的输入自动路由到最匹配的专家上。这样既保持了模型的表达能力，又把参数量控制在中等规模。

2. **多任务持续预训练 + 多任务微调的双层适配**  
   传统做法只在单一任务上微调，容易忘记之前学到的通用数学知识。作者先在所有任务的原始题库上继续进行自监督预训练，让专家网络学习跨任务的共性；随后在每个任务上做轻量微调，确保专门能力不被冲淡。相当于先“全科复习”，再“专项练习”。

3. **专家网络的知识分解与共享机制**  
   通过路由算法，模型把相似的数学子问题（如求根、化简分式）分配给同一专家，从而在专家内部形成专门的数学子知识库。这样在新任务出现时，只需要激活已有专家即可，无需从头训练新参数。

4. **LLM 迭代精炼作为后处理**  
   为了解决极端复杂题目仍可能出现的错误，作者引入大语言模型进行“在上下文中学习”的二次推理。JiuZhang 2.0 先给出初步答案，LLM 再根据题目和答案的上下文进行纠错和补全，形成迭代式的解题流程。这样把小模型的高效与大模型的强推理能力结合起来。

### 方法详解

整体思路可以划分为三步：① 构建 MoE 基础模型；② 进行多任务持续预训练和微调；③ 用 LLM 进行迭代精炼。

**1. MoE 基础模型**  
模型主体是一个标准的 Transformer 编码器，只是把 Feed‑Forward 层替换成了若干并行的专家网络（每个专家都是一个小型 Feed‑Forward 层）。输入的隐藏向量会经过一个路由器（通常是软阈值或 Top‑K 选择），挑选出 1~2 个最活跃的专家进行计算，其他专家保持不更新。这样既保留了 Transformer 的序列建模能力，又让不同数学子任务可以在不同专家中形成专属的表示。

**2. 多任务持续预训练**  
作者收集了大量中文数学题库，覆盖代数、几何、概率等 10+ 任务。预训练目标仍是掩码语言模型（MLM），即随机遮盖题目中的词或符号，让模型预测原词。关键在于训练数据是混合的，路由器会在不同任务的样本之间自动学习“哪个专家擅长哪个任务”。因为是持续预训练，模型已经拥有了基础的中文语言能力，只是把数学知识进一步细化进专家。

**3. 多任务微调**  
在每个具体任务上，作者只打开对应任务的少量专用头（如分类层或生成层），其余参数保持不变。微调时仍使用路由器，但会加入任务标签作为路由的额外信号，帮助模型更精准地激活相关专家。这样既保留了跨任务共享的优势，又能针对每个任务的细节进行微调。

**4. LLM 迭代精炼**  
解题流程的最后一步是把 JiuZhang 2.0 生成的答案连同原题一起喂给一个更大的中文 LLM（如 ChatGLM）。LLM 在“few‑shot”上下文中看到若干高质量的解题示例后，会对当前答案进行检查、补全或纠错。若 LLM 检测到不一致或缺失步骤，会返回改进后的解答；如果仍有疑问，系统可以循环几次，直至答案收敛。这个环节的核心是“在上下文中学习”，不需要额外的梯度更新，只靠提示工程实现。

**最巧妙的设计**  
- 路由器同时考虑任务标签和输入特征，使得专家的激活既有语义驱动，又有任务驱动。  
- 将持续预训练和微调分层进行，避免了“灾难性遗忘”（即新任务训练时旧知识被抹掉）。  
- 用 LLM 作为后处理，而不是把所有算力都压在主模型上，极大提升了成本效益。

### 实验与效果

- **数据与任务**：作者在中文数学领域公开的 10+ 任务上评测，包括代数求解、几何证明、函数极值、概率计算等。每个任务都有数千到上万条标注题目。  
- **基线对比**：与单任务的中文数学模型（如原版 JiuZhang、MathBERT）以及通用的中文大模型（如 ChatGLM‑6B）进行比较。论文报告在大多数任务上提升 3%~8% 的准确率，尤其在复杂的几何证明上提升超过 10%。  
- **消融实验**：去掉 MoE、去掉持续预训练、去掉 LLM 精炼三种设置分别实验，结果显示：MoE 带来约 2% 的整体提升，持续预训练贡献约 3%，LLM 精炼在高难度任务上贡献最高，提升可达 5%。  
- **局限性**：作者指出在极端长公式或需要深度符号推理的题目上仍会出现错误；此外，LLM 精炼依赖外部大模型，增加了推理时的延迟。  

### 影响与延伸思考

JiuZhang 2.0 把“多任务共享+专家路由”引入中文数学求解，开启了在资源受限情况下实现跨任务高效学习的思路。后续有几篇工作尝试把同样的 MoE 思路搬到代码生成、科学文献问答等领域（如 CodeMoE、SciMoE），并探索更细粒度的路由策略。对想进一步研究的读者，可以关注以下方向：① 如何在不依赖显式任务标签的情况下自动发现任务结构；② 更高效的 LLM 迭代精炼方法，如使用轻量化的检验模型代替全尺寸 LLM；③ 将符号推理模块（如数学公式求导器）与 MoE 结合，实现真正的符号+语言混合推理。  

### 一句话记住它

**JiuZhang 2.0 用专家路由的 MoE 把多种数学任务压进一个中等规模模型，再借大模型二次校正，实现高效且统一的中文数学求解。**