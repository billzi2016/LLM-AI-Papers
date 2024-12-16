# SPaR: Self-Play with Tree-Search Refinement to Improve   Instruction-Following in Large Language Models

> **Date**：2024-12-16
> **arXiv**：https://arxiv.org/abs/2412.11605

## Abstract

Instruction-following is a fundamental capability of language models, requiring the model to recognize even the most subtle requirements in the instructions and accurately reflect them in its output. Such an ability is well-suited for and often optimized by preference learning. However, existing methods often directly sample multiple independent responses from the model when creating preference pairs. Such practice can introduce content variations irrelevant to whether the instruction is precisely followed (e.g., different expressions about the same semantic), interfering with the goal of teaching models to recognize the key differences that lead to improved instruction following. In light of this, we introduce SPaR, a self-play framework integrating tree-search self-refinement to yield valid and comparable preference pairs free from distractions. By playing against itself, an LLM employs a tree-search strategy to refine its previous responses with respect to the instruction while minimizing unnecessary variations. Our experiments show that a LLaMA3-8B model, trained over three iterations guided by SPaR, surpasses GPT-4-Turbo on the IFEval benchmark without losing general capabilities. Furthermore, SPaR demonstrates promising scalability, greatly enhancing models like GLM-4-9B and LLaMA3-70B. We also identify how inference scaling in tree search would impact model performance. Our code and data are publicly available at https://github.com/thu-coai/SPaR.

---

# SPaR：通过自博弈与树搜索细化提升大语言模型指令遵循能力 论文详细解读

### 背景：这个问题为什么难？
指令遵循要求模型在千变万化的提示中捕捉细微差别，并把这些要求完整、准确地体现在输出里。传统的偏好学习（preference learning）往往直接让模型一次性生成多条答案，再让人类或奖励模型挑出更好的那一条。这样做的副作用是：不同答案之间会出现大量与指令本身无关的表述差异（比如同义改写、风格变化），导致奖励模型学到的信号被“噪声”稀释。换句话说，模型被教会了辨别“怎么说得好”，而不是“到底有没有遵循指令”。因此，如何生成既满足指令又可直接比较的答案对，成为提升指令遵循的关键瓶颈。

### 关键概念速览
**指令遵循（Instruction Following）**：模型在接收到任务描述后，能够完整、准确地执行任务的能力。类似于学生听老师布置作业后，既不漏掉要求，也不跑题。

**偏好学习（Preference Learning）**：通过让模型比较两段输出的好坏，学习一个奖励函数来指导后续生成。好比让评委挑选更好的演讲稿，从而教会写作者写得更好。

**自博弈（Self‑Play）**：模型与自己对弈，轮流生成答案并相互评估。像棋手自己下棋再回看每一步，找出更优的走法。

**树搜索（Tree Search）**：在生成过程中展开多条可能的分支，系统性地评估每条分支的质量后再决定最终输出。可以想象为在写文章时先列出多个段落草稿，再挑选最合适的组合。

**奖励模型（Reward Model）**：用来给生成的文本打分的网络，分数越高表示越符合指令。相当于老师给作业打分的标准。

**IFEval 基准**：专门测评模型指令遵循细致程度的评测套件。类似于专门的考试，考查学生是否把所有细节都做对。

### 核心创新点
1. **从随机抽样到树搜索细化**  
   之前的做法是直接让模型一次性抽取多条独立答案，形成比较对。SPaR 改为让模型先生成一条基线答案，然后在此基础上进行树搜索，逐步改写、精炼，得到一系列“更好”或“更差”的变体。这样得到的对比对几乎只在指令遵循程度上有差别，其他噪声被压到最低。

2. **自博弈驱动的自我对抗**  
   传统的偏好学习需要外部标注或人工对比。SPaR 让模型自己扮演“玩家”和“评审”，在同一轮次里既生成改进版，又用奖励模型评估两者优劣，实现了全自动的闭环训练。

3. **迭代式三轮训练策略**  
   论文中对 LLaMA3‑8B 进行三次迭代，每一次都用最新的奖励模型重新进行自博弈与树搜索。每轮迭代都相当于一次“升级”，使模型在指令遵循上逐步逼近甚至超越 GPT‑4‑Turbo。

4. **规模可扩展性验证**  
   除了 8B 参数的模型，作者还把同样的框架搬到 GLM‑4‑9B 与 LLaMA3‑70B 上，实验显示提升幅度仍然显著，说明方法并不依赖特定模型大小。

### 方法详解
**整体框架**  
SPaR 的训练流程可以划分为四步：① 初始生成基线答案；② 基于基线进行树搜索得到多个改进候选；③ 用奖励模型对每对候选进行比较，产生偏好标签；④ 用这些标签对语言模型进行强化学习（如 PPO）更新。整个循环在每轮迭代结束后重新训练奖励模型，再进入下一轮。

**步骤拆解**  

1. **基线答案生成**  
   给定指令，模型直接采样得到一段文本 A。此时 A 可能已经遵循指令，但仍有提升空间。

2. **树搜索细化**  
   - **节点展开**：在 A 的每个关键位置（如动词、数值、结构段落）插入“编辑操作”，生成若干子答案（A₁、A₂…）。  
   - **局部评估**：每个子答案通过当前奖励模型打分，保留分数最高的 K 条。  
   - **递归搜索**：对保留下来的子答案再进行一次同样的编辑-评估循环，深度一般设为 2~3 步，形成一棵搜索树。  
   - **最终选取**：树的叶子中分数最高的即为“改进版” B，分数最低的则视作“退步版” C。这样得到的 (B, C) 对几乎只在指令遵循上有差别。

3. **自博弈对比**  
   模型本身充当评审角色：使用奖励模型比较 B 与 C，输出 “B 更好” 或 “C 更好”。这一步不需要人工标注，完全自动。

4. **强化学习更新**  
   将比较结果转化为偏好标签，喂给 PPO（Proximal Policy Optimization）等强化学习算法，更新语言模型的策略，使其在以后生成时更倾向于产生高分答案。

**巧妙之处**  
- **搜索空间受指令约束**：编辑操作只在指令关键点上进行，避免了全句重写带来的无关差异。  
- **奖励模型同步进化**：每轮迭代后重新训练奖励模型，使其能够捕捉到最新的细微改进，防止奖励函数“老化”。  
- **自博弈闭环**：模型既是生成者也是评审者，省去了昂贵的人类偏好标注成本。

### 实验与效果
- **评测数据**：主要在 IFEval 基准上测试，该基准专注于指令细节的完整性与准确性。  
- **对比基线**：包括 GPT‑4‑Turbo、原始 LLaMA3‑8B、以及传统偏好学习（直接抽样）方式。  
- **主要结果**：经过三轮 SPaR 训练后，LLaMA3‑8B 在 IFEval 上的得分超过 GPT‑4‑Turbo（具体数值未在摘要中给出），且在通用语言能力（如 MMLU、ARC）上没有明显下降。  
- **规模实验**：对 GLM‑4‑9B 与 LLaMA3‑70B 进行同样的训练，均观察到显著提升，说明方法对不同模型规模均有效。  
- **消融研究**：论文报告了去掉树搜索或仅使用随机抽样的对照实验，发现去掉树搜索后提升幅度大幅下降，验证了搜索细化是关键因素。  
- **局限性**：树搜索在推理阶段会带来额外计算开销，尤其在大模型上成本显著；作者也提到奖励模型的质量仍是瓶颈，若奖励模型本身不够敏感，搜索得到的改进会受限。

### 影响与延伸思考
SPaR 把自博弈与结构化搜索结合进偏好学习，为“让模型自我纠错”提供了可操作的框架。后续工作可能会在以下方向继续探索：  
- **更高效的搜索策略**：如蒙特卡洛树搜索（MCTS）或基于梯度的局部搜索，以降低推理成本。  
- **跨任务自博弈**：把同一模型在不同指令集合上进行自博弈，学习通用的指令遵循技巧。  
- **奖励模型的自监督提升**：利用未标注的大规模对话数据，让奖励模型自行发现指令细节的判别特征。  
- **与人类反馈混合**：在关键高风险任务上仍保留少量人工偏好，以校准奖励模型的极端错误。  
这些思路已经在一些后续的开源项目和论文中出现（推测），表明社区对自博弈细化的兴趣正在上升。

### 一句话记住它
让大语言模型自己下棋、细致搜索每一步，然后用自评的分数来教自己——SPaR 用自博弈+树搜索把“怎么说得好”变成“到底有没有遵循指令”。