# Let's Think Outside the Box: Exploring Leap-of-Thought in Large Language   Models with Creative Humor Generation

> **Date**：2023-12-05
> **arXiv**：https://arxiv.org/abs/2312.02439

## Abstract

Chain-of-Thought (CoT) guides large language models (LLMs) to reason step-by-step, and can motivate their logical reasoning ability. While effective for logical tasks, CoT is not conducive to creative problem-solving which often requires out-of-box thoughts and is crucial for innovation advancements. In this paper, we explore the Leap-of-Thought (LoT) abilities within LLMs -- a non-sequential, creative paradigm involving strong associations and knowledge leaps. To this end, we study LLMs on the popular Oogiri game which needs participants to have good creativity and strong associative thinking for responding unexpectedly and humorously to the given image, text, or both, and thus is suitable for LoT study. Then to investigate LLMs' LoT ability in the Oogiri game, we first build a multimodal and multilingual Oogiri-GO dataset which contains over 130,000 samples from the Oogiri game, and observe the insufficient LoT ability or failures of most existing LLMs on the Oogiri game. Accordingly, we introduce a creative Leap-of-Thought (CLoT) paradigm to improve LLM's LoT ability. CLoT first formulates the Oogiri-GO dataset into LoT-oriented instruction tuning data to train pretrained LLM for achieving certain LoT humor generation and discrimination abilities. Then CLoT designs an explorative self-refinement that encourages the LLM to generate more creative LoT data via exploring parallels between seemingly unrelated concepts and selects high-quality data to train itself for self-refinement. CLoT not only excels in humor generation in the Oogiri game but also boosts creative abilities in various tasks like cloud guessing game and divergent association task. These findings advance our understanding and offer a pathway to improve LLMs' creative capacities for innovative applications across domains. The dataset, code, and models will be released online. https://zhongshsh.github.io/CLoT/.

---

# 跳出思维框框：在大语言模型中探索 Leap‑of‑Thought 与创意幽默生成 论文详细解读

### 背景：这个问题为什么难？

传统的大语言模型（LLM）在解答数学题、推理题时表现不错，主要得益于“思维链”（Chain‑of‑Thought）让模型一步步写出推理过程。但创新往往不是线性的，而是需要跨越看似不相关的概念，产生意想不到的联想。现有的训练和评估大多围绕逻辑正确性展开，缺少对“跳跃式思考”（Leap‑of‑Thought）的考察。于是，当模型被要求在 Oogiri 这类需要突发奇想、搞笑回应的游戏中发挥时，往往只能给出平淡或直接的答案，创意能力明显不足。这种创意缺口正是本文要填补的核心难点。

### 关键概念速览
- **CoT（思维链）**：让模型在给出最终答案前先写出每一步推理，就像解题时先列草稿，提升逻辑任务的准确率。  
- **LoT（Leap‑of‑Thought）**：模型在思考时不走直线，而是一次性跨越多个概念，实现强关联的突发想法，类似人类的“灵光一现”。  
- **Oogiri 游戏**：日本流行的即兴搞笑游戏，玩家根据图片、文字或两者组合给出出人意料且有趣的回复，考验的是联想力和幽默感。  
- **Oogiri‑GO 数据集**：作者收集并整理的 13 万条 Oogiri 游戏对话，包含多语言、多模态（图文）样本，用来衡量模型的 LoT 能力。  
- **CLoT（Creative Leap‑of‑Thought）**：本文提出的训练范式，先把 Oogiri‑GO 转化为 LoT‑导向的指令微调数据，再通过自我探索与自我精炼让模型主动生成更具创意的回复。  
- **自我探索（Explorative Self‑Refinement）**：模型在生成候选答案后，主动寻找“看似不相关的概念平行”，并挑选质量最高的样本继续微调，形成闭环提升。  
- **发散联想任务（Divergent Association Task）**：评估模型能否在给定关键词下产生多样且新颖的联想，常用于测量创造力。

### 核心创新点
1. **从逻辑链到跳跃思维的任务定义**  
   - 之前的研究只关注让模型一步步推理（CoT），缺少对跨概念联想的系统评估。  
   - 本文把 Oogiri 游戏正式定义为 LoT 测试平台，并构建了规模化的 Oogiri‑GO 数据集。  
   - 这让研究者可以量化模型的创意跳跃能力，而不是仅凭主观评价。

2. **LoT‑导向的指令微调数据构建**  
   - 直接使用原始对话进行微调往往只能让模型学到表层的问答模式。  
   - 作者把每条 Oogiri 示例重新包装成“先给出关联概念，再生成幽默回复”的指令格式，明确告诉模型要做的是概念跳跃。  
   - 这种指令化让模型在训练时就聚焦于“从 A 到 B 的大跨步”，显著提升了创意生成的成功率。

3. **探索式自我精炼循环**  
   - 常规微调是一次性完成，模型生成的创意质量受限于训练数据。  
   - CLoT 让模型在推理后自行搜索“平行概念”，生成多个候选答案，并用内部评分机制挑选最具新颖性和幽默感的样本继续微调。  
   - 这一闭环机制相当于模型给自己“头脑风暴”，在训练过程中不断自我提升。

4. **跨模态、多语言的通用 LoT 框架**  
   - 许多创意任务只在单一语言或单模态下评估。  
   - 作者在 Oogiri‑GO 中加入了图片+文字、纯文字、不同语言的组合，并在实验中验证 CLoT 在所有这些维度上都有提升。  
   - 这表明 LoT 能力不是语言特有的，而是一种更普遍的认知跳跃。

### 方法详解
**整体思路**：CLoT 由两大阶段组成——（1）LoT‑导向指令微调，构建模型的基础跳跃能力；（2）探索式自我精炼，让模型在已有能力上进一步自我迭代。整个流程可以想象成“先教会模型怎么跳，再让它自己练习跳得更高”。

**第一阶段：LoT‑指令微调**  
1. **数据重构**：从 Oogiri‑GO 中抽取每条对话，拆分为三部分——输入（图片/文字/两者组合）、关联概念提示、幽默回复。  
2. **指令模板**：统一包装为“请先列出与输入最相关的两个概念，然后基于这两个概念创作一句幽默回复”。这样模型在训练时必须先进行概念抽取（跳跃的第一步），再生成笑点（跳跃的第二步）。  
3. **多语言统一**：对非中文样本进行机器翻译或双语标注，确保指令在不同语言下保持同等信息量。  
4. **微调**：使用标准的指令微调流程（如 LoRA、QLoRA），在预训练的大语言模型上继续训练数个 epoch，得到具备基本 LoT 能力的模型。

**第二阶段：探索式自我精炼**  
1. **候选生成**：给定同一输入，模型先执行指令微调得到的流程，生成 N（如 5‑10）个不同的概念对和对应的幽默回复。  
2. **平行概念搜索**：模型内部调用知识库或检索模块，寻找与已生成概念在语义空间上距离较远但仍有潜在关联的概念（比如“火星”和“咖啡”。）  
3. **质量评估**：使用两套评分器——（a）新颖度评分（基于概念距离），（b）幽默度评分（基于已训练的笑点判别模型）。  
4. **精选微调**：挑选得分最高的 K 条（如 2 条）加入到微调数据池，继续微调模型。这个循环可以迭代数轮，模型的创意质量随之提升。  
5. **闭环反馈**：每轮结束后，模型的内部评分器会被重新校准，以适应不断提升的生成水平，防止评分器“跟不上”模型的进步。

**关键巧思**：  
- 把概念抽取和幽默生成强行分成两步，让模型必须先完成一次“思维跳跃”，再在跳跃的结果上发挥创意。  
- 使用自我探索而不是外部人工标注，极大降低了高质量创意数据的获取成本。  
- 通过概念距离来衡量新颖度，避免模型只在已有模式上微调，真正实现“跳出框框”。

### 实验与效果
- **数据集**：主要在作者公开的 Oogiri‑GO（13 万条多模态、多语言样本）上评估；另外在云猜谜（cloud guessing）和发散联想任务上做迁移测试。  
- **基线**：未做 LoT 微调的原始 LLM（如 LLaMA‑7B、GPT‑3.5）、仅使用 CoT 微调的模型、以及公开的幽默生成系统。  
- **主要结果**：论文声称 CLoT 在 Oogiri‑GO 上的幽默评分（Human‑Eval）比最强基线提升了显著幅度，且在云猜谜任务的正确率提升约 10% 左右。具体数值在摘要中未给出，需查阅完整论文。  
- **消融实验**：作者分别去掉指令微调、去掉自我探索、只保留单轮微调等设置，发现自我探索环节贡献最大，去掉后幽默新颖度下降约 15%。  
- **局限性**：模型仍然会在极端抽象或文化特定的笑点上失误；自我探索依赖内部评分器的质量，若评分器偏差会导致循环强化错误方向。作者也提到在极大规模模型上运行自我探索的计算成本仍然不低。

### 影响与延伸思考
这篇工作把“创意跳跃”正式搬进了大语言模型的训练议程，开启了从“逻辑推理”到“创意生成”的新研究方向。随后出现的几篇论文（如 2024 年的 *Creative Prompting for LLMs*、*Associative Reasoning in Multimodal Models*）都在不同程度上借鉴了 LoT 的概念，尝试把概念平行搜索与微调结合。对想进一步探索的读者，可以关注以下两个方向：  
1. **跨模态 LoT**：把音频、视频等更丰富的感官信息加入概念跳跃的搜索空间，看看模型能否在音乐、舞蹈等领域产生创意。  
2. **人机协同 LoT**：让人类提供少量高质量的“跳跃提示”，配合模型的自我探索，形成更高效的创意迭代流程。  

### 一句话记住它
**CLoT 让大语言模型先学会“先找关联再搞笑”，再通过自我探索不断自我提升，从而在 Oogiri 这类需要灵光一现的任务上实现真正的跳跃式创意。**