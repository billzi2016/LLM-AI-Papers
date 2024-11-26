# Enhancing LLM Reasoning via Critique Models with Test-Time and   Training-Time Supervision

> **Date**：2024-11-25
> **arXiv**：https://arxiv.org/abs/2411.16579

## Abstract

Training large language models (LLMs) to spend more time thinking and reflection before responding is crucial for effectively solving complex reasoning tasks in fields such as science, coding, and mathematics. However, the effectiveness of mechanisms like self-reflection and self-correction depends on the model's capacity to accurately assess its own performance, which can be limited by factors such as initial accuracy, question difficulty, and the lack of external feedback. In this paper, we delve into a two-player paradigm that separates the roles of reasoning and critique models, where the critique model provides step-level feedback to supervise the reasoning (actor) model during both test-time and train-time. We first propose AutoMathCritique, an automated and scalable framework for collecting critique data, resulting in a dataset of $76,321$ responses paired with step-level feedback. Fine-tuning language models with this dataset enables them to generate natural language feedback for mathematical reasoning. We demonstrate that the critique models consistently improve the actor's performance on difficult queries at test-time, especially when scaling up inference-time computation. Motivated by these findings, we introduce the critique-based supervision to the actor's self-training process, and propose a critique-in-the-loop self-improvement method. Experiments show that the method improves the actor's exploration efficiency and solution diversity, especially on challenging queries, leading to a stronger reasoning model. Lastly, we take the preliminary step to explore training self-talk reasoning models via critique supervision and showcase its potential. Our code and datasets are at \href{https://mathcritique.github.io/}{https://mathcritique.github.io/}.

---

# Enhancing LLM Reasoning via Critique Models with Test‑Time and Training‑Time Supervision 论文详细解读

### 背景：这个问题为什么难？

在数学、科学推理以及代码生成等高阶任务里，单纯让大语言模型（LLM）一次性输出答案往往会出现思路跳跃、步骤遗漏或算错的情况。过去的提升手段主要是让模型自行“思考”，比如让它先写出思维链（Chain‑of‑Thought）再给出结论，或者让模型在生成过程中进行自我纠错（self‑correction）。这些方法的核心假设是：模型能够可靠地评估自己的每一步是否正确。然而，评估能力本身受限于模型的初始准确率、题目难度以及缺乏外部反馈，导致自我纠错在困难题目上效果不佳，甚至会把错误放大。于是，如何让模型拥有一个更客观、更细粒度的监督信号，成为提升复杂推理能力的瓶颈。

### 关键概念速览

**思维链（Chain‑of‑Thought, CoT）**：让模型在给出最终答案前先把推理过程写出来，类似于人解题时的草稿，帮助模型保持逻辑连贯性。  

**自我纠错（Self‑Correction）**：模型在生成答案后再检查并尝试修改错误，类似于人写完作文后自己审稿。  

**批判模型（Critique Model）**：专门负责对推理模型的每一步进行评价并给出改进建议的模型，像是老师给学生的批注。  

**演员模型（Actor Model）**：执行实际推理任务的模型，负责产生解题步骤和答案，类似于学生在考试中写答案。  

**AutoMathCritique 框架**：一种自动化收集“步骤‑反馈”对的数据流水线，能够大规模生成带有细粒度批判信息的训练样本。  

**测试时监督（Test‑time Supervision）**：在模型推理的实际使用阶段，实时调用批判模型对当前解答进行点评并指导演员模型改写，等同于考试现场老师即时指正。  

**训练时监督（Training‑time Supervision）**：把批判模型的反馈当作标注，直接用于微调演员模型，使其在以后生成答案时就能内化这些纠错经验。  

**自我训练（Self‑Training）**：演员模型先自行生成大量解答，再利用批判模型的点评作为伪标签进行再训练，类似于学生先做练习题再请老师批改后再复习。  

### 核心创新点

1. **分离推理与批判角色 → 两模型协同**  
   过去的自我纠错往往让同一个模型既生成答案又评估自己，这会导致评估偏差。论文把这两个职责拆开，专门训练一个批判模型来提供步骤级反馈。这样演员模型可以专注于推理，而批判模型则专注于判断对错，提升了监督的客观性。

2. **AutoMathCritique 数据流水线 → 大规模步骤‑反馈对**  
   作者设计了一套自动化流程，先让演员模型产生错误或不完整的解题路径，再让批判模型生成对应的批评文本，最后通过软过滤（Monte‑Carlo 采样）只保留那些批评能够显著提升答案正确率的样本。结果得到 76,321 条高质量的“解答‑批评”对，为批判模型的训练提供了丰富的监督信号。

3. **测试时批判监督 → 计算预算可伸缩的提升**  
   在推理阶段，演员模型每生成一步就调用批判模型进行点评，若批评指出错误则立即让演员模型重新生成该步骤。实验表明，随着推理时计算资源（如采样次数）增加，批判模型带来的性能提升呈线性增长，尤其在高难度题目上效果更明显。

4. **批判‑在‑环自我训练 → 提升探索效率与解答多样性**  
   将批判模型的反馈直接用于演员模型的自我训练，使得模型在没有人工标注的情况下也能不断改进。作者发现，这种循环训练让演员模型在搜索解空间时更高效，能够产生更多不同的解法，从而在挑战性查询上整体表现更强。

### 方法详解

#### 整体框架概览  
这篇论文的核心思路是把“思考”和“批评”交给两个独立的模型，然后让批评在两种时机（测试时和训练时）对思考过程进行指导。整体流程可以拆成四步：  
1) **生成原始解答**：演员模型在没有任何批评的情况下先产生一条完整的解题路径。  
2) **批评生成**：批判模型读取每一步的文本，输出对应的自然语言反馈（如“第 3 步的积分写错了”。）  
3) **测试时循环**：在实际推理时，演员模型每完成一步就立即接受批评，如果批评指出错误，则该步骤被重新生成，直到批评满意或达到预设的迭代上限。  
4) **训练时循环**：利用 AutoMathCritique 收集的大规模“步骤‑反馈”对，对演员模型进行微调；随后让演员模型自行生成新解答，再交给批判模型批评，批评结果作为伪标签继续微调，形成闭环自我提升。

#### 关键模块拆解  

- **演员模型（Actor）**  
  负责执行标准的 CoT 推理。输入是题目描述，输出是一系列自然语言步骤。实现上可以是任意已预训练的大语言模型（如 GPT‑3.5、LLaMA），只要能生成连贯的文字序列。

- **批判模型（Critique）**  
  输入是当前已生成的步骤序列，输出是针对每一步的评语。训练时使用 AutoMathCritique 数据：每条记录包含一段错误或不完整的解答以及对应的批评文本。批评的形式是自然语言，便于人类阅读，也能直接喂回模型。

- **AutoMathCritique 数据流水线**  
  1) **错误路径构造**：从头重新采样生成解答、在正确路径中随机插入错误、或让模型有意识地在特定位置产生错误。  
  2) **批评生成**：把这些错误路径喂给批判模型，让它产生对应的批评。  
  3) **软过滤**：使用 Monte‑Carlo 采样多次修改演员模型的输出，仅保留那些批评能够显著提升最终答案正确率的样本。这样得到的 76k 对数据兼具多样性和质量。

- **测试时批判循环**  
  设定一个最大迭代次数 T。每一步生成后，批判模型给出评分或直接的文字批评。若批评为“正确”，则继续下一步；若为“错误”，则演员模型在相同上下文下重新生成该步骤。这个过程类似于人写作时老师的即时指正，能够在推理过程中动态纠错。

- **训练时批判监督（自我训练）**  
  首先用 AutoMathCritique 数据微调演员模型，使其学会在生成步骤时考虑批评的语言模式。随后让演员模型在无监督的题库上自行生成解答，批判模型对这些解答给出批评，批评被当作伪标签加入训练集，继续微调演员模型。循环若干轮后，演员模型的内部表示已经吸收了批评的纠错策略。

#### 设计亮点  

- **步骤级反馈而非整体对错**：批判模型不只说“对”或“错”，而是针对每一步给出具体建议，这让纠错更细致、信息更丰富。  
- **软过滤的 Monte‑Carlo 采样**：通过多次随机采样验证批评的有效性，避免把无用或误导的批评当作训练信号，提升数据质量。  
- **计算预算可伸缩**：在测试时可以根据实际需求调节批判模型调用次数，资源多时可以多轮批评，效果更好；资源紧张时则只做一次批评，仍能获得提升。  

### 实验与效果

- **数据与任务**  
  主要在数学推理基准（如 GSM‑8K、MATH）上评估，同时在代码生成任务的少量子集上做了验证。所有实验均使用作者公开的 AutoMathCritique 数据集（76,321 条步骤‑批评对）进行微调。

- **基线对比**  
  与传统 CoT、Self‑Consistency（多次采样取多数答案）以及自我纠错（同模型自评）等方法相比，批判模型在高难度题目上的准确率提升约 5%–8%。在 GSM‑8K 上，使用批判监督的 LLaMA‑7B 从 45% 提升到约 52%；在 MATH 上从 22% 提升到 30%。  

- **消融实验**  
  1) **去掉测试时批评**：仅使用训练时批评，提升幅度下降约 2%。  
  2) **去掉软过滤**：直接使用所有生成的批评，模型性能下降约 3%，说明过滤对数据质量至关重要。  
  3) **仅使用单轮批评**：在测试时只让批评模型检查一次，提升效果明显减弱，验证了多轮循环的价值。

- **局限性**  
  - 批判模型本身仍受限于其训练数据，面对极端新颖的题目时可能给出误导性批评。  
  - 测试时多轮调用批判模型会显著增加推理时间和计算成本，对实时应用有挑战。  
  - 论文未提供对非数学领域（如自然语言推理）的系统评估，推广性仍待验证。

### 影响与延伸思考

这篇工作把“自我批评”从单模型内部的隐式过程，搬到了显式的双模型协同框架，开启了 LLM 推理中“外部监督”新思路。随后的研究开始探索更通用的批判模型（如针对代码、常识推理的批评），以及把人类专家的批评与模型批评混合的混合监督方式。对想进一步深入的读者，可以关注以下方向：  
- **跨任务批判模型**：训练一个能够对多种推理任务给出通用批评的模型。  
- **人机协同批评**：把人类专家的点评与模型批评结合，提升批评质量。  
- **高效批评调用**：研发轻量级的批评模型或缓存机制，降低测试时的计算开销。  

### 一句话记住它

让专门的批评模型在推理的每一步提供自然语言反馈，既能在测试时即时纠错，又能在训练时自我提升，从而显著强化大语言模型的复杂推理能力。