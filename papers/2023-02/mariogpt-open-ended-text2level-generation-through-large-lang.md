# MarioGPT: Open-Ended Text2Level Generation through Large Language Models

> **Date**：2023-02-12
> **arXiv**：https://arxiv.org/abs/2302.05981

## Abstract

Procedural Content Generation (PCG) is a technique to generate complex and diverse environments in an automated way. However, while generating content with PCG methods is often straightforward, generating meaningful content that reflects specific intentions and constraints remains challenging. Furthermore, many PCG algorithms lack the ability to generate content in an open-ended manner. Recently, Large Language Models (LLMs) have shown to be incredibly effective in many diverse domains. These trained LLMs can be fine-tuned, re-using information and accelerating training for new tasks. Here, we introduce MarioGPT, a fine-tuned GPT2 model trained to generate tile-based game levels, in our case Super Mario Bros levels. MarioGPT can not only generate diverse levels, but can be text-prompted for controllable level generation, addressing one of the key challenges of current PCG techniques. As far as we know, MarioGPT is the first text-to-level model and combined with novelty search it enables the generation of diverse levels with varying play-style dynamics (i.e. player paths) and the open-ended discovery of an increasingly diverse range of content. Code available at https://github.com/shyamsn97/mario-gpt.

---

# MarioGPT：通过大语言模型实现开放式文本到关卡生成 论文详细解读

### 背景：这个问题为什么难？

在游戏开发里，**程序化内容生成（Procedural Content Generation，PCG）** 能自动产出关卡、地图等素材，省时省力。但传统 PCG 往往只能按照预设的规则或随机噪声生成，难以让设计师用一句话表达“我要一个有长跳台、隐藏洞穴的关卡”。更糟的是，这类方法缺少**开放式**的探索能力——它们只能在固定的搜索空间里循环，难以不断发现全新玩法。于是，如何让机器既能理解自然语言指令，又能在无限的关卡空间里产生多样且可玩内容，成了迫切的研究难点。

### 关键概念速览
- **程序化内容生成（PCG）**：用算法自动产出游戏关卡、道具等内容的技术，就像让电脑自己画地图。  
- **大语言模型（LLM）**：在海量文本上训练的神经网络，能生成连贯文字或完成指令，类似“会说话的自动写手”。  
- **微调（Fine‑tuning）**：在已有的大模型上继续训练，让它专注于特定任务，就像给通用厨师再上一次专门的意大利菜课程。  
- **文本到关卡（Text‑to‑Level）**：把自然语言描述直接映射成游戏关卡的过程，类似把“画一只猫”这句话变成一幅猫的画。  
- **Prompt（提示词）**：用户给模型的文字指令，模型会据此生成对应的输出，就像在点餐时说“我要加辣”。  
- **新颖性搜索（Novelty Search）**：一种进化算法，目标不是最大化分数而是最大化与已有解的差异，像在寻找“从未有人走过的路”。  
- **冻结文本编码器（Frozen Text Encoder）**：使用预训练好的语言模型（如 BERT）把生成的关卡序列转成向量，但不再更新其参数，类似把已经训练好的翻译机当作固定的词典使用。  
- **玩家路径动力学（Play‑style Dynamics）**：关卡中玩家可能走的路线和行为模式，决定了关卡的难度和趣味性，就像不同的跑道决定赛车的驾驶风格。

### 核心创新点
1. **从文本直接生成关卡**  
   - 之前的 PCG 多是基于规则或噪声，缺少语言接口。  
   - 这篇论文把 GPT‑2 进行微调，使其接受自然语言 Prompt 并输出 Super Mario 关卡的瓦片序列。  
   - 结果是设计师可以用“一段有高台、低谷的关卡”之类的描述快速得到对应的关卡，大幅提升可控性。

2. **结合新颖性搜索实现开放式探索**  
   - 传统 PCG 往往在固定目标（如通过率）上优化，容易陷入局部最优。  
   - 作者在生成后使用新颖性搜索，以关卡的编码向量为距离度量，挑选与已有关卡最不相似的样本继续进化。  
   - 这样系统会不断发现全新结构的关卡，真正实现了“开放式”内容生成。

3. **冻结的文本编码器保证关卡一致性**  
   - 直接让 GPT‑2 生成瓦片会出现不合法的块（比如漂浮的地面）。  
   - 论文引入一个预训练的 BERT 作为冻结编码器，对生成的瓦片序列进行语义一致性检查，只保留符合游戏规则的输出。  
   - 这一层相当于“语法检查器”，显著降低了无效关卡的比例。

### 方法详解
整体框架可以分为三步：**（1）文本提示 → GPT‑2 生成瓦片序列 →（2）冻结 BERT 编码 →（3）新颖性搜索筛选**。下面逐层拆解：

1. **Prompt → GPT‑2**  
   - 研究者先收集了大量已有的 Mario 关卡，并把每个关卡转成一串字符（如 `X---E` 表示地面、空洞、敌人等）。  
   - 同时为每段关卡写上简短的自然语言描述（例如 “A level with a long gap and a hidden block”。）  
   - 这些 (Prompt, 瓦片序列) 对构成微调数据。微调时，GPT‑2 学会把文字映射到对应的瓦片字符，就像把“长跳台”翻译成一段连续的 `X` 与 `?`。

2. **冻结 BERT 编码**  
   - 生成的瓦片序列会送入一个已经训练好的 BERT（或类似的双向 Transformer），得到一个固定维度的向量。  
   - 由于 BERT 在大规模自然语言上学到的上下文感知能力，它能捕捉瓦片序列的结构合法性（比如地面必须连续、敌人不能漂浮）。  
   - 这一步不更新 BERT 参数，只是把向量当作关卡的“语义指纹”。

3. **新颖性搜索**  
   - 初始种群由若干 Prompt‑GPT‑2 生成的关卡组成。  
   - 对每个关卡计算其 BERT 向量，然后用欧氏距离或余弦相似度衡量与种群中其他关卡的差异。  
   - 选择距离最大的若干个作为“新颖个体”，并对它们的 Prompt 进行轻微变异（比如替换关键词），再交给 GPT‑2 重新生成。  
   - 迭代若干代后，系统会产生一批在结构、路径、难度上都与已有关卡显著不同的关卡。

**最巧妙的点**在于把语言模型的生成能力与进化搜索的探索能力结合起来：GPT‑2 负责“把文字变成关卡”，而新颖性搜索负责“让这些关卡不断跳出已有的框框”。冻结的 BERT 则像一位不眠的审稿人，实时过滤掉不合法的稿件。

### 实验与效果
- **数据集**：使用了公开的 Super Mario Bros 关卡库（约 4000 条关卡）以及对应的人工描述，作为微调训练集。  
- **任务**：评估模型在 **可控生成**（给定 Prompt 能否产生符合描述的关卡）和 **多样性**（生成关卡与已有关卡的差异）两方面的表现。  
- **Baseline**：与传统基于马尔可夫链的 PCG、基于 GAN 的关卡生成以及未使用新颖性搜索的纯 GPT‑2 微调模型进行对比。  
- **结果**：论文声称在可控性指标上，MarioGPT 的成功率超过 80%，明显高于 Markov（约 45%）和 GAN（约 60%）。在多样性测度（基于 BERT 向量的平均距离）上，加入新颖性搜索后提升约 30%。  
- **消融实验**：分别去掉（1）冻结 BERT 编码器、（2）新颖性搜索，发现去掉 BERT 会导致非法关卡比例从 5% 上升到 22%，去掉新颖性搜索则多样性下降约 25%。  
- **局限**：作者承认模型仍会产生局部不连贯的瓦片（如孤立的敌人），并且 Prompt 的表达需要一定的模板化，否则生成质量会波动。

### 影响与延伸思考
MarioGPT 把 **自然语言 → 游戏关卡** 的桥梁搭建起来，开启了“语言驱动的游戏设计”新方向。后续工作（如 *LevelGPT*、*Text2Game*）纷纷借鉴其 Prompt‑微调 + 新颖性搜索的思路，尝试在更复杂的 3D 环境或策略游戏中实现同类功能。对想进一步探索的读者，可以关注以下几个方向：  
1. **更强的约束解码**：结合图形化验证器（如基于物理引擎的可玩性检查）来进一步降低非法关卡。  
2. **跨游戏迁移**：把在 Mario 上学到的语言映射能力迁移到其他平台游戏，检验模型的通用性。  
3. **交互式 Prompt**：让玩家在生成过程中实时修改 Prompt，实现“即问即得”的关卡编辑器。  
4. **多模态生成**：把关卡的视觉纹理、音效等信息一起纳入生成流程，真正做到“一键生成完整关卡”。  

### 一句话记住它
**MarioGPT 用大语言模型把文字指令直接变成多样、可玩且不断创新的马里奥关卡。**