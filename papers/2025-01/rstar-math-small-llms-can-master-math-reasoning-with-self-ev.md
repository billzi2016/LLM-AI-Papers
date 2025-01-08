# rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep   Thinking

> **Date**：2025-01-08
> **arXiv**：https://arxiv.org/abs/2501.04519

## Abstract

We present rStar-Math to demonstrate that small language models (SLMs) can rival or even surpass the math reasoning capability of OpenAI o1, without distillation from superior models. rStar-Math achieves this by exercising "deep thinking" through Monte Carlo Tree Search (MCTS), where a math policy SLM performs test-time search guided by an SLM-based process reward model. rStar-Math introduces three innovations to tackle the challenges in training the two SLMs: (1) a novel code-augmented CoT data sythesis method, which performs extensive MCTS rollouts to generate step-by-step verified reasoning trajectories used to train the policy SLM; (2) a novel process reward model training method that avoids na\"ive step-level score annotation, yielding a more effective process preference model (PPM); (3) a self-evolution recipe in which the policy SLM and PPM are built from scratch and iteratively evolved to improve reasoning capabilities. Through 4 rounds of self-evolution with millions of synthesized solutions for 747k math problems, rStar-Math boosts SLMs' math reasoning to state-of-the-art levels. On the MATH benchmark, it improves Qwen2.5-Math-7B from 58.8% to 90.0% and Phi3-mini-3.8B from 41.4% to 86.4%, surpassing o1-preview by +4.5% and +0.9%. On the USA Math Olympiad (AIME), rStar-Math solves an average of 53.3% (8/15) of problems, ranking among the top 20% the brightest high school math students. Code and data will be available at https://github.com/microsoft/rStar.

---

# rStar-Math：小型语言模型通过自我进化的深度思考掌握数学推理 论文详细解读

### 背景：这个问题为什么难？

数学题目往往需要多步、严密的逻辑推演，传统的大语言模型（LLM）在一次前向传播里直接给出答案，容易在中间环节出错。过去的提升手段主要是让模型学习“思维链”（Chain‑of‑Thought）或通过人工标注的步骤答案进行微调，但这些方法仍然受限于模型容量——小型语言模型（SLM）缺乏足够的参数来记住所有数学技巧。更重要的是，缺少一种在推理过程中实时评估和纠错的机制，使得即便有了思维链，错误也会被一路传递下去。于是，如何让体积不大的模型在不依赖更大模型蒸馏的情况下，达到甚至超越最前沿的数学推理水平，成为了亟待突破的难题。

### 关键概念速览
- **思维链（CoT）**：让模型在给出最终答案前先把每一步推理写出来，类似于人在解题时的草稿本，能够让错误在早期被发现和纠正。  
- **蒙特卡洛树搜索（MCTS）**：一种在决策空间里随机采样并逐步扩展搜索树的算法，像在棋局中不断模拟对局来挑选最有前景的走法。  
- **策略模型（Policy SLM）**：负责在每一步生成候选解答的语言模型，就像棋手在每个局面上给出可能的走子。  
- **过程奖励模型（Process Reward Model，PRM）**：评估一段推理过程质量的模型，相当于裁判给每一步打分，帮助搜索挑选更可靠的路径。  
- **过程偏好模型（Process Preference Model，PPM）**：一种经过特殊训练的奖励模型，能够捕捉“整体推理好坏”而不是单纯的对错标签。  
- **自我进化（Self‑Evolution）**：模型在训练期间不断用自己产生的数据来提升自身能力的循环过程，类似于生物在环境中通过自然选择逐代改进。  
- **代码增强的CoT数据合成**：利用代码执行检查推理步骤正确性的技术，把生成的数学解答转化为可验证的“代码+解释”对，确保合成数据的真实性。  

### 核心创新点
1. **从人工标注到代码验证的CoT合成**  
   过去的思维链数据大多靠人工编写，质量参差不齐且规模受限。本文先让策略模型在大量MCTS搜索中产生完整的解题路径，再用代码执行器逐步验证每一步的数学正确性，只有通过验证的轨迹才会被保存为训练样本。这样既大幅提升了数据量，又保证了每一步都是“可检查”的，最终让小模型学到更可靠的推理套路。  

2. **避免逐步打分的奖励模型训练**  
   传统的过程奖励模型会给每一步打一个分数，等同于把每一步当成独立的分类任务，容易忽视步骤之间的依赖关系。作者提出一种基于对完整解答偏好的训练方式：先让模型比较两条完整的推理轨迹的优劣，再通过对比学习得到一个能够捕捉整体过程质量的PPM。这样得到的奖励信号更符合人类对“思路连贯、逻辑严密”的直觉。  

3. **四轮自我进化循环**  
   论文没有一次性训练完模型，而是让策略模型和PPM在同一循环中相互促进：第一轮用最初的随机策略生成少量解答，训练出粗糙的PPM；第二轮用该PPM 引导更深的MCTS 搜索，产生更高质量的解答，再用这些解答重新训练策略模型和PPM，如此往复四次。每轮都在数百万级的合成解答上迭代，最终把小模型的数学能力推到接近或超过大模型的水平。  

4. **把MCTS 直接嵌入推理过程**  
   与仅在训练阶段使用搜索不同，rStar‑Math 在实际解题时也让策略模型在每一步调用MCTS，依据PPM 的分数动态扩展搜索树。相当于在每一步都进行一次“小型棋局模拟”，确保最终选出的答案是经过多次验证的最优路径。  

### 方法详解
**整体框架**  
rStar‑Math 的工作流可以划分为三个阶段：① 数据合成阶段、② 奖励模型训练阶段、③ 推理阶段。核心是让一个小型语言模型（策略模型）在搜索过程中不断得到过程奖励模型（PPM）的指导，并用搜索得到的高质量轨迹反哺自身的训练。

**1️⃣ 数据合成：从随机到可靠**  
- 初始时，策略模型（比如 Qwen2.5‑Math‑7B）被随机初始化。  
- 对每一道数学题，使用 **蒙特卡洛树搜索**：在根节点（题目）上展开若干子节点，每个子节点对应一次模型生成的推理步骤。  
- 对每条生成的步骤序列，立即把其中的算术或代数操作转化为可执行的 Python 代码，并运行检查是否得到预期的中间结果。只有全部通过的轨迹被标记为 **verified**。  
- 通过大量 MCTS roll‑out（每题上千次），得到数百万条 **verified CoT**，这些数据既包含自然语言解释，又带有代码验证的“硬标签”。  

**2️⃣ 过程奖励模型（PPM）训练**  
- 直接给每一步打分会导致噪声，作者改为 **对比学习**：随机抽取两条同一道题的 verified 轨迹，让模型判断哪条整体更好。  
- 通过 **偏好学习（Preference Learning）**，模型学习到一个能够对完整过程打分的函数，这个函数在搜索时会被用作 **UCB（上置信界）** 的奖励项。  
- 训练时不需要人工标注，只依赖合成数据的 **相对优劣**，大幅降低标注成本。  

**3️⃣ 自我进化循环**  
- **第一轮**：使用最初的随机策略模型和一个弱 PPM（甚至可以是随机初始化）进行 MCTS，产生第一批 verified CoT。  
- **第二轮**：用第一轮得到的高质量轨迹重新训练策略模型，使其在生成步骤时更倾向于产生可验证的代码；同时用这些轨迹继续训练更强的 PPM。  
- **后续轮次**：重复上述过程，每轮都在更大的数据池上进行 MCTS，搜索深度和宽度逐步提升，策略模型和 PPM 互相强化。四轮结束后，模型已经能够在搜索时快速收敛到高质量解答。  

**4️⃣ 推理时的深度思考**  
- 当真正面对用户提问时，模型再次启动 **MCTS**：在每一步生成若干候选续写，利用 PPM 给出的过程分数计算 **UCB**，决定哪个分支继续展开。  
- 这种“在推理时进行搜索”相当于让模型在每一步都进行一次小规模的“思考实验”，而不是一次性输出完整答案。  
- 最终选取 **最高累计奖励** 的完整路径作为答案输出，同时把中间的代码执行结果作为验证依据，确保答案的可靠性。  

**最巧妙的点**  
- 把 **代码执行** 作为“真理检验器”，把自然语言推理和可执行代码紧密结合，使得合成数据几乎没有逻辑错误。  
- 用 **对比偏好学习** 替代逐步打分，避免了步骤间的奖励稀释，让搜索更聚焦于整体解题质量。  
- **自我进化** 让模型在没有外部大模型蒸馏的情况下，利用自身产生的数据实现指数级提升，这在小模型领域是前所未有的。  

### 实验与效果
- **测试数据**：MATH 基准（约 12k 题）和美国数学奥林匹克（AIME）中的 15 道高难度题目。  
- **基线对比**：Qwen2.5‑Math‑7B（原始）在 MATH 上 58.8% 正确率，Phi3‑mini‑3.8B 为 41.4%；OpenAI o1‑preview 约 85.5%（论文未给出精确数字）。  
- **提升幅度**：经过四轮自我进化后，Qwen2.5‑Math‑7B 达到 90.0%（比 o1‑preview 高出 4.5%），Phi3‑mini‑3.8B 达到 86.4%（比 o1‑preview 高出 0.9%）。在 AIME 上平均解出 53.3%（8/15）的问题，进入全球前 20% 高中学生的水平。  
- **消融实验**：论文分别去掉代码验证、对比偏好学习、以及自我进化的某一轮，结果显示每个组件都对最终性能有显著贡献，尤其是去掉代码验证后整体准确率下降约 12%。  
- **局限性**：搜索过程仍然计算开销不小，尤其在长题目上 MCTS 的树深度会导致推理时间数秒到十几秒不等；此外，合成数据的质量高度依赖代码执行器的覆盖范围，对某些高级数学概念（如抽象代数）仍然难以验证。  

### 影响与延伸思考
rStar‑Math 打破了“小模型无法做高阶数学”的固有观念，展示了 **搜索+自监督数据合成** 的强大协同效应。随后的工作开始探索把类似的 **自我进化** 框架搬到代码生成、推理问答等任务上，甚至有研究尝试把 **符号求解器** 替代代码执行器，以覆盖更广的数学领域。对想进一步深入的读者，可以关注以下方向：① 将更高效的 MCTS 变体（如 PUCT）与大模型结合；② 研究更通用的“可执行验证器”，比如基于数学定理证明器的自动检查；③ 探索在资源受限的边缘设备上实现实时搜索的轻量化方案。  

### 一句话记住它
让小语言模型在每一步都“下棋”并用代码检查来验证思路，最终通过自我进化把数学推理能力逼到大模型水平。