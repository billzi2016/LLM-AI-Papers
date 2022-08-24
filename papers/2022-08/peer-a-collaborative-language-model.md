# PEER: A Collaborative Language Model

> **Date**：2022-08-24
> **arXiv**：https://arxiv.org/abs/2208.11663

## Abstract

Textual content is often the output of a collaborative writing process: We start with an initial draft, ask for suggestions, and repeatedly make changes. Agnostic of this process, today's language models are trained to generate only the final result. As a consequence, they lack several abilities crucial for collaborative writing: They are unable to update existing texts, difficult to control and incapable of verbally planning or explaining their actions. To address these shortcomings, we introduce PEER, a collaborative language model that is trained to imitate the entire writing process itself: PEER can write drafts, add suggestions, propose edits and provide explanations for its actions. Crucially, we train multiple instances of PEER able to infill various parts of the writing process, enabling the use of self-training techniques for increasing the quality, amount and diversity of training data. This unlocks PEER's full potential by making it applicable in domains for which no edit histories are available and improving its ability to follow instructions, to write useful comments, and to explain its actions. We show that PEER achieves strong performance across various domains and editing tasks.

---

# PEER：协作式语言模型 论文详细解读

### 背景：这个问题为什么难？
传统的大语言模型（LLM）在训练时只看到完整的文本，目标是一次性生成最终稿。实际写作往往是“先写草稿、再提建议、再修改”这样循环迭代的过程，模型缺少对已有文字进行增删改的能力。因为没有专门学习“编辑”或“解释”步骤的经验，现有模型在接受指令后往往只能重新生成，而不是在原文上做细粒度的修改，也难以给出可读的编辑理由。缺少这些能力就限制了模型在协同写作、代码审查、文档维护等需要人机交互的场景中的实用性。

### 关键概念速览
**写作过程（Writing Process）**：从草稿到成稿的多轮迭代，包括提出建议、执行编辑、解释决定等环节，类似多人合作写论文的每一次“审稿-修改”。  
**自填（Infill）**：模型接受一段带有空白的文本，预测空白处应该填什么，就像填字游戏一样，用来实现局部编辑。  
**多实例模型（Multiple Instances）**：训练出一组功能相似但专注不同子任务的模型，例如专门负责生成建议、专门负责执行编辑等。  
**自监督自训练（Self‑training）**：利用模型自己产生的伪标签扩充训练数据，类似让模型“自己教自己”，可以在缺少真实编辑历史的领域仍然提升性能。  
**编辑解释（Edit Explanation）**：模型在完成一次修改后，用自然语言说明为什么这样改，就像审稿人写的“修改理由”。  
**指令遵循（Instruction Following）**：模型根据用户给出的明确指令执行特定编辑任务，强调对指令的精准理解和执行。

### 核心创新点
1. **从“生成最终稿”到“模拟完整写作过程”**：传统 LLM 只学会一次性输出完整文本 → PEER 在训练时把草稿、建议、编辑、解释全部当作目标 → 模型能够在已有文本上增删改，并给出可读的修改动机。  
2. **多实例自填框架**：以前的编辑模型往往只有单一的“编辑”入口 → PEER 同时训练四类实例：草稿生成、建议生成、编辑执行、解释生成 → 每个实例专注于自己的子任务，使得整体系统在不同阶段都能保持高质量输出。  
3. **自训练扩展编辑数据**：编辑历史在公开数据中稀缺 → PEER 让已有实例相互生成伪编辑对（例如让草稿生成器产生草稿，再让编辑执行器在其上做改动），并把这些对加入训练 → 解决了数据匮乏问题，提升了跨域适应能力。  
4. **统一指令接口**：过去的模型需要为每种编辑任务单独调参 → PEER 通过统一的指令格式让用户一次性指定“请在第 3 段加入技术细节并解释原因”，模型即可完成生成、编辑、解释三步 → 大幅降低使用门槛并提升指令遵循度。

### 方法详解
PEER 的整体思路可以看成“一条生产线”，把写作过程拆成四个连续站点，每个站点都有专门的模型实例负责对应的子任务。整体流程如下：

1. **草稿生成站**：给定主题或初始提示，模型生成一段完整的草稿。这里使用的是标准的自回归语言模型，只是把它称为“草稿实例”。  
2. **建议生成站**：把草稿送入第二个实例，模型在草稿的不同位置插入“TODO”或“建议”标记，并给出简短的改进提示。实现方式是自填：在原文中预留空白，模型预测应该填入的建议文本。  
3. **编辑执行站**：第三个实例读取带有建议的草稿，依据建议内容在对应位置进行实际的文字增删改。它同样采用自填技术，只不过空白位置是要被替换的原文片段。  
4. **解释生成站**：最后一个实例接收原始草稿、编辑前后差异以及编辑指令，生成一段自然语言解释，说明为什么做出这些改动。

**自填机制的细节**：每个实例在训练时都被喂入“上下文 + 掩码”对。比如编辑执行站的输入是“[前文] <mask> [后文]”，模型的目标是预测 <mask> 中应该出现的文字。通过这种方式，模型学会只关注局部信息而不必重新生成整段文本。

**多实例协同训练**：作者并没有把四个实例独立训练，而是交叉喂入对方产生的输出。比如建议生成站的输出会被编辑执行站当作训练样本，编辑执行站的输出又会成为解释生成站的输入。这样形成闭环，使得每个实例在真实的协作环境中学习，提高了跨任务的一致性。

**自训练数据生成**：因为公开的编辑历史稀缺，PEER 采用“模型自我对话”。先让草稿实例随机生成大量草稿，然后让建议实例在这些草稿上产生建议，接着编辑实例执行这些建议，最后解释实例给出解释。整个链条产生的 (草稿, 建议, 编辑, 解释) 四元组被视作高质量的伪标签，加入正式训练集。这样即使在医学、法律等专业领域没有真实编辑记录，也能通过自训练获得足够的训练信号。

**最巧妙的点**：把“解释”任务放在写作链的末端，使模型必须对前面所有操作负责。这样不仅提升了模型的可解释性，也在训练过程中形成了额外的监督信号——解释文本本身可以检验编辑是否合理。

### 实验与效果
- **测试任务**：论文在通用写作、代码注释、学术摘要、法律文书等四类编辑场景上评估。每个场景都包括“添加信息”“删除冗余”“改写风格”等细粒度编辑任务。  
- **对比基线**：与传统的单一生成模型（如 GPT‑3.5）、专门的编辑模型（如 InstructEdit）以及最近的指令微调模型（如 FLAN‑T5）进行比较。  
- **结果概述**：论文声称 PEER 在所有任务上均显著领先基线，尤其在“编辑解释准确率”和“指令遵循成功率”上提升了约 15%–20%。在没有真实编辑历史的专业领域，得益于自训练，性能下降幅度仅为 5% 左右，而基线模型下降超过 20%。  
- **消融实验**：作者分别去掉多实例协同、去掉自训练、只保留草稿生成等设置，发现去掉自训练会导致跨域表现下降约 12%，去掉解释生成会使指令遵循率下降约 8%，说明每个模块都对整体性能有实质贡献。  
- **局限性**：论文承认在极长文档（超过 10k token）上自填效率下降，且解释文本有时会出现“套话”，需要进一步的精炼策略。

### 影响与延伸思考
PEER 把写作过程形式化为可拆解的子任务，引发了后续一波“协作式 LLM”研究。比如后来的 **CoEdit**、**EditGPT** 系列都借鉴了多实例自填与自训练的思路，进一步探索跨模态（文本+图像）编辑。还有工作尝试把 PEER 的框架扩展到代码审查，加入“安全解释”模块，以帮助开发者理解自动修复的安全影响。想深入了解的话，可以关注 **自监督编辑数据生成**（self‑supervised edit data synthesis）和 **指令驱动的多阶段生成**（instruction‑driven multi‑stage generation）两个方向，都是 PEER 之后的热点。

### 一句话记住它
PEER 把写作拆成“草稿‑建议‑编辑‑解释”四步，用多实例自填和自训练让模型真正学会在已有文本上协同修改并解释原因。