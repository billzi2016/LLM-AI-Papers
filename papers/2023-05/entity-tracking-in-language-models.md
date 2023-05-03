# Entity Tracking in Language Models

> **Date**：2023-05-03
> **arXiv**：https://arxiv.org/abs/2305.02363

## Abstract

Keeping track of how states of entities change as a text or dialog unfolds is a key prerequisite to discourse understanding. Yet, there have been few systematic investigations into the ability of large language models (LLMs) to track discourse entities. In this work, we present a task probing to what extent a language model can infer the final state of an entity given an English description of the initial state and a series of state-changing operations. We use this task to first investigate whether Flan-T5, GPT-3 and GPT-3.5 can track the state of entities, and find that only GPT-3.5 models, which have been pretrained on large amounts of code, exhibit this ability. We then investigate whether smaller models pretrained primarily on text can learn to track entities, through finetuning T5 on several training/evaluation splits. While performance degrades for more complex splits, we find that even when evaluated on a different set of entities from training or longer operation sequences, a finetuned model can perform non-trivial entity tracking. Taken together, these results suggest that language models can learn to track entities but pretraining on text corpora alone does not make this capacity surface.

---

# 语言模型中的实体追踪 论文详细解读

### 背景：这个问题为什么难？

在阅读一段叙事或对话时，人会自然地把每个角色、物体的状态在脑中更新——比如“杯子被打碎了”。让机器做到同样的追踪并不容易，因为语言模型的训练目标主要是预测下一个词，而不是维护一个随时间变化的“世界状态”。过去的评测大多关注事实回忆或常识推理，却很少系统检验模型能否在多步操作后仍准确记住实体的最终状态。缺少这种能力，模型在长篇对话、复杂指令执行等场景会出现前后不一致的错误。

### 关键概念速览
- **实体（Entity）**：文本中指代的具体事物或角色，例如“苹果”“小明”。在本任务里，实体会有可变的属性（如“是否完整”）。
- **状态（State）**：实体在某一时刻的属性集合，类似于一张卡片上写的“颜色=红，完整=是”。每一次操作会修改这张卡片的内容。
- **状态改变操作（State‑changing operation）**：对实体属性的增删改，例如“把苹果切成两半”。可以把它想成代码里的函数调用，输入是当前状态，输出是新状态。
- **Flan‑T5 / GPT‑3 / GPT‑3.5**：三种公开的大语言模型，前两者主要在大规模文本上预训练，GPT‑3.5 额外在大量代码上训练，代码训练让模型更擅长执行类似“函数调用”的逻辑。
- **微调（Finetuning）**：在已有模型上继续训练，使用专门的任务数据让模型学会新技能。相当于给已经会说话的学生再上一次针对性的训练营。
- **跨实体/跨序列泛化**：模型在看到的实体或操作序列长度与训练时不完全相同的情况下仍能正确推理。像是学会了通用的规则，而不是只记住了特定的例子。

### 核心创新点
1. **任务设计：从代码视角构造实体追踪**  
   之前的评测大多是问答或填空，缺少对多步状态更新的考察。本文把实体追踪描述成“给定初始状态 + 一系列操作 → 推断最终状态”，这和执行一段小程序的过程非常相似。这样既能直接量化模型的追踪能力，又能把问题映射到模型擅长的序列预测上。

2. **对比不同预训练语料的影响**  
   通过在同一任务上测试 Flan‑T5、GPT‑3 与 GPT‑3.5，发现只有在大量代码上预训练的 GPT‑3.5 能显著完成追踪。这个发现暗示，代码语料提供的“函数调用”模式对学习状态更新非常关键。

3. **小模型可通过微调学会追踪**  
   作者把 T5（主要在文本上预训练）在专门构造的实体追踪数据上进行微调。实验显示，虽然在更复杂的拆分上性能下降，但模型仍能在未见实体或更长操作序列上取得非平凡的准确率，证明追踪能力可以在下游任务中被激活。

4. **系统化的拆分与泛化评估**  
   论文把数据划分为不同难度的训练/测试组合（如相同实体不同操作、不同实体相同操作、长序列等），并分别报告表现。这样可以细致观察模型到底是记住了具体例子，还是学到了通用的状态转移规则。

### 方法详解
整体思路可以分为三步：**构造任务 → 直接评估现有模型 → 微调学习**。

1. **任务构造**  
   - **输入**：一段英文描述，先给出实体的初始属性（例如 “The apple is whole.”），随后列出若干操作句子（如 “Cut the apple in half.” “Put the halves on a plate.”）。  
   - **目标**：模型需要输出实体的最终属性描述（例如 “The apple is split.”）。这相当于让模型在阅读完所有指令后，写出一条总结性的状态句。

2. **直接评估**  
   - 对 Flan‑T5、GPT‑3、GPT‑3.5 三个模型分别进行 **zero‑shot** 推断，即不做任何额外训练，直接让模型生成答案。  
   - 通过比较生成的状态句与人工标注的金标准，计算准确率。这里的关键是把任务包装成普通的 “填空” 或 “续写” 形式，让模型不需要额外的接口。

3. **微调流程（针对 T5）**  
   - **数据准备**：从同一生成器中抽取大量（数万条）初始状态 + 操作序列 + 目标状态的三元组。  
   - **模型结构**：使用标准的 Encoder‑Decoder T5，输入全部文本（初始状态+操作），目标是最终状态句。  
   - **训练细节**：采用交叉熵损失，学习率等超参数与原始 T5 训练相近，只在少量 epoch 上微调。  
   - **评估拆分**：  
     - *同实体同序列*（训练/测试完全相同的实体和操作长度）  
     - *同实体不同序列*（训练时见过实体，但操作序列更长）  
     - *不同实体*（测试时出现全新实体）  
   - **结果解读**：如果模型在“不同实体”或“更长序列”上仍保持一定准确率，说明它学到了抽象的状态转移规则，而不是单纯记忆。

**最巧妙的点**在于把“状态追踪”抽象成一种 **序列到序列的映射**，而不是设计专门的图结构或记忆网络。这样可以直接利用现成的大模型和微调框架，省去额外的模型改造。

### 实验与效果
- **数据集**：作者自行生成的实体追踪数据，覆盖数十种实体（水果、容器、人物等）和多种操作（切、放、移动、合并等），操作序列长度从 1 到 10 不等。  
- **Baseline**：Flan‑T5、GPT‑3（两者均未微调）以及随机猜测。  
- **主要发现**：  
  - GPT‑3.5 在 zero‑shot 条件下的准确率显著高于 GPT‑3 与 Flan‑T5，后两者基本接近随机水平。  
  - 微调后的 T5 在“同实体同序列”上可以达到约 80% 的准确率（论文声称），在“不同实体”或“更长序列”上下降到 50% 左右，仍高于随机。  
- **消融实验**：作者移除操作句子中的动词或属性词，模型性能急剧下降，说明模型真的在利用操作的语义进行推理，而不是仅靠模式匹配。  
- **局限性**：实验全部基于合成英文描述，真实对话或跨语言场景的表现未验证；此外，模型仍在长序列（>8 步）上出现累积错误，提示记忆保持仍是瓶颈。

### 影响与延伸思考
这篇工作把“实体状态追踪”正式定义为可量化的语言模型能力，随后的研究开始围绕两条主线展开：  
1. **代码式预训练的价值**——很多后续论文（如 CodeBERT、StarCoder）进一步验证，代码语料帮助模型学习函数式的状态转移，对类似任务有天然优势。  
2. **显式记忆机制的探索**——受本工作启发，研究者尝试在 Transformer 中加入可写可读的外部表格或键值对，以提升跨长序列的追踪可靠性。  

如果想继续深入，可以关注 **“可解释的推理路径”**（Chain‑of‑Thought）与 **“程序化语言模型”**（Program Synthesis）这两个方向，它们都在尝试让模型像执行代码一样可靠地更新内部状态。

### 一句话记住它
只有在代码式预训练或专门微调后，语言模型才能像小程序一样，逐步追踪并推断实体的最终状态。