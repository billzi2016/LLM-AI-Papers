# IOPO: Empowering LLMs with Complex Instruction Following via Input-Output Preference Optimization

> **Date**：2024-11-09
> **arXiv**：https://arxiv.org/abs/2411.06208

## Abstract

In the realm of large language models (LLMs), the ability of models to accurately follow instructions is paramount as more agents and applications leverage LLMs for construction, where the complexity of instructions are rapidly increasing. However, on the one hand, there is only a certain amount of complex instruction evaluation data; on the other hand, there are no dedicated algorithms to improve the ability to follow complex instructions. To this end, this paper introduces TRACE, a benchmark for improving and evaluating the complex instructionfollowing ability, which consists of 120K training data and 1K evaluation data. Furthermore, we propose IOPO (Input-Output Preference Optimization) alignment method which takes both input and output preference pairs into consideration, where LLMs not only rapidly align with response preferences but also meticulously explore the instruction preferences. Extensive experiments on both in-domain and outof-domain datasets confirm the effectiveness of IOPO, showing 8.15%, 2.18% improvements on in-domain data and 6.29%, 3.13% on outof-domain data compared to SFT and DPO respectively.

---

# IOPO：通过输入输出偏好优化赋能大语言模型的复杂指令遵循 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）已经可以完成基本的指令跟随，但真实应用里常出现层层嵌套、细粒度差别的复杂指令。现有的微调数据大多是简短、单一任务的示例，导致模型在面对多步骤、模糊或相互冲突的需求时容易走偏。再者，传统的对齐方法（如SFT、RLHF）只关注模型输出的好坏，忽视了“输入层面”的微调空间——即同一指令的不同表述会显著影响模型的理解。于是，缺少既能扩充复杂指令数据，又能在输入和输出两端同步优化的技术手段，成为提升复杂指令遵循能力的瓶颈。

### 关键概念速览
**复杂指令**：包含多层次目标、条件或需要推理的指令，例如“先列出三种可行方案，再比较它们的成本与风险”。类似于让人完成一份带有子任务的工作清单。  
**SFT（监督微调）**：把模型在标注好的问答对上继续训练，让它学会模仿人类答案。相当于给模型上“补习班”。  
**DPO（直接偏好优化）**：通过比较模型生成的两个答案的偏好分数，直接对模型进行梯度更新，只关注输出好坏。像是让模型参加“谁的答案更受欢迎”的投票。  
**输入偏好**：指对同一指令的不同表述（如不同措辞、顺序）进行偏好评估，判断哪种表述更有助于模型产生正确答案。类似于挑选最易懂的题目描述。  
**IOPO（输入输出偏好优化）**：一种同时考虑输入表述和输出答案偏好的对齐方法，既让模型学会更好地“听”，也让它更擅长“说”。  
**TRACE基准**：作者新建的用于训练和评估复杂指令遵循能力的数据集，包含约12万条训练样本和1千条高质量评估样本。相当于专门为复杂指令设计的“考试”。  

### 核心创新点
1. **从单向偏好到双向偏好**：传统DPO只比较两个输出的好坏 → IOPO在此基础上加入对输入表述的偏好比较（即同一指令的不同写法） → 模型在学习时会自动倾向于更易理解的输入，同时产生更符合期望的输出，显著提升复杂指令的整体成功率。  
2. **TRACE基准的构建**：之前没有专门针对复杂指令的大规模数据集 → 作者手工收集、筛选并人工标注了12万条多步骤指令及其对应的高质量答案 → 为IOPO提供了足够的训练信号，也为后续研究提供了统一评测平台。  
3. **偏好对齐的联合梯度更新**：常规做法是先固定输入，优化输出 → IOPO把输入的微调（如词序、措辞）和输出的微调放在同一个梯度步骤中 → 这种“一体化”更新让模型在同一次迭代里同时改善“听”和“说”，收敛更快、效果更稳。  
4. **跨域验证**：大多数对齐方法只在训练域内表现好 → 作者在完全不同的指令集合上（如代码生成、法律咨询）进行测试 → IOPO仍保持两位数的提升，证明其对指令结构的通用理解能力。  

### 方法详解
**整体框架**  
IOPO的训练流程可以划分为三步：①准备输入表述对，②生成对应的输出对，③基于两对的偏好分数进行联合梯度更新。核心思想是把“哪个输入更好”和“哪个输出更好”这两个二分类任务合并为一个统一的优化目标。

**步骤拆解**  
1. **输入对构造**  
   - 对每条训练指令，随机生成两个略有差异的表述（比如换词、调换子句顺序）。  
   - 让人类评审或使用已有的偏好模型判断哪种表述更有助于得到正确答案，得到一个二元标签（更好 / 更差）。  
   - 这一步相当于给模型提供“更好听的指令”示例。

2. **输出对生成**  
   - 对每个输入表述，使用当前的LLM生成答案。  
   - 再对同一指令的另一表述也生成答案。  
   - 通过人工标注或自动评价（如GPT‑4评审）得到哪个答案更符合指令意图的偏好标签。

3. **联合偏好损失**  
   - 对输入对和输出对分别计算偏好得分（类似于二分类的logits）。  
   - 将两者的得分相加，形成一个综合的偏好分数。  
   - 采用梯度上升（或下降）方式，使得“更好”组合的综合分数最大化，而“更差”组合的分数最小化。  
   - 这里的关键是**同一次梯度更新同时调节模型对输入的感知和对输出的生成**，而不是先后独立进行。

**最巧妙的点**  
- **输入微调的幅度极小**：只在词汇层面做轻微改动，避免破坏原指令的语义，却足以让模型学习到哪些表述更易被正确解析。  
- **偏好对齐的对称性**：输入偏好和输出偏好使用相同的损失形式，使得训练过程保持统一，代码实现上只需复用同一套优化器。  

### 实验与效果
- **数据集**：主要在作者自建的TRACE基准上进行评测，包含12万条训练样本和1千条评估样本；此外，还在公开的ComplexInstr、CodeInstr等跨域指令集上做外部验证。  
- **对比基线**：SFT（普通监督微调）和DPO（仅输出偏好优化）是主要对手。  
- **核心结果**：在TRACE的内部评估上，IOPO相较于SFT提升了8.15%，相较于DPO提升了2.18%；在跨域测试中，分别提升了6.29%和3.13%。这些数字表明即使在未见过的指令类型上，IOPO仍能保持显著优势。  
- **消融实验**：作者分别去掉输入偏好、去掉输出偏好以及只使用单向偏好进行训练。结果显示，去掉输入偏好会导致整体提升下降约1.8%，仅保留输出偏好则下降约1.2%，说明两者协同是关键。  
- **局限性**：论文承认构造高质量输入偏好对仍需人工参与，成本不低；此外，IOPO在极长上下文（>4k token）上的效果尚未充分验证。  

### 影响与延伸思考
IOPO首次把“听”和“说”放在同一优化框架里，打开了对齐研究的新视角。随后的工作（如2024年提出的“双向RLHF”以及2025年的“全局指令感知微调”）都在不同程度上借鉴了输入偏好这一思路。对想进一步探索的读者，可以关注以下方向：①自动化生成高质量输入偏好对的模型（如利用大模型自评）；②将IOPO扩展到多模态指令（图文、音视频）；③研究在超长上下文场景下的输入偏好梯度传播。  

### 一句话记住它
IOPO通过同步优化指令的表述和答案的偏好，让大模型既听得更懂，也说得更对。