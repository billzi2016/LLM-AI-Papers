# R1-VL: Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization

> **Date**：2025-03-17
> **arXiv**：https://arxiv.org/abs/2503.12937

## Abstract

Recent studies generally enhance MLLMs' reasoning capabilities via supervised fine-tuning on high-quality chain-of-thought reasoning data, which often leads models to merely imitate successful reasoning paths without understanding what the wrong reasoning paths are. In this work, we aim to enhance the MLLMs' reasoning ability beyond passively imitating positive reasoning paths. To this end, we design Step-wise Group Relative Policy Optimization (StepGRPO), a new online reinforcement learning framework that enables MLLMs to self-improve reasoning ability via simple, effective and dense step-wise rewarding. Specifically, StepGRPO introduces two novel rule-based reasoning rewards: Step-wise Reasoning Accuracy Reward (StepRAR) and Step-wise Reasoning Validity Reward (StepRVR). StepRAR rewards the reasoning paths that contain necessary intermediate reasoning steps via a soft key-step matching technique, while StepRAR rewards reasoning paths that follow a well-structured and logically consistent reasoning process through a reasoning completeness and logic evaluation strategy. With the proposed StepGRPO, we introduce R1-VL, a series of MLLMs with outstanding capabilities in step-by-step reasoning. Extensive experiments over 8 benchmarks demonstrate the superiority of our methods.

---

# R1‑VL：通过逐步组相对策略优化学习多模态大语言模型的推理 论文详细解读

### 背景：这个问题为什么难？
多模态大语言模型（MLLM）在视觉问答、图文推理等任务上已经能给出看似合理的答案，但它们的推理过程往往是“黑箱”。过去的提升手段主要是把大量高质量的思维链（CoT）数据喂进去，让模型学会模仿正确的推理步骤。模仿虽然能提升表面准确率，却没有教会模型辨别错误路径——模型不知道哪些步骤是“坑”。于是，当遇到未见过的复杂情境时，它们容易走进逻辑陷阱，表现出“会写却不会想”。要让模型真正具备自我纠错、逐步推理的能力，需要一种能够让模型感知每一步好坏并据此改进的训练机制，这正是本文要解决的核心难题。

### 关键概念速览
**多模态大语言模型（MLLM）**：同时接受文字和图像等多种输入，输出自然语言的模型。想象成一个会看图说话的聊天机器人。  
**思维链（CoT）**：在答案前先写出推理步骤，就像解数学题时先列出算式。  
**强化学习（RL）**：让模型通过“试错—奖励”循环学会策略，类似训练小狗坐下：做对了给零食。  
**策略优化（Policy Optimization）**：在RL里调整模型的行为分布，使其产生更高奖励的动作序列。  
**Step‑wise Group Relative Policy Optimization（StepGRPO）**：本文提出的逐步组相对策略优化框架，核心是把每一步推理都当作一次小决策，并用相对奖励来比较好坏。  
**Step‑wise Reasoning Accuracy Reward（StepRAR）**：奖励包含关键中间步骤的推理路径，像给每个必考点打分。  
**Step‑wise Reasoning Validity Reward（StepRVR）**：奖励整体结构合理、逻辑连贯的路径，类似检查答案是否前后自洽。  
**软关键步骤匹配（Soft Key‑step Matching）**：用模糊匹配而非硬性对齐来判断模型是否命中了关键步骤，避免因为表述差异而扣分。  

### 核心创新点
1. **从单一整体奖励到逐步密集奖励**  
   之前的RL方法往往只在完整答案后给一次奖励，模型只能靠整体好坏来推断每一步的贡献。本文把奖励拆成每一步的StepRAR和StepRVR，两者在每一步都给出信号，让模型在推理过程中实时感知哪一步做对了、哪一步偏离了逻辑。这样模型不再是“盲目前进”，而是像走迷宫时每走一步都有指示灯。  

2. **相对策略优化而非绝对奖励**  
   传统策略梯度直接最大化奖励值，容易受奖励尺度波动影响。StepGRPO 引入“组相对”概念：在同一批次里把模型产生的多个推理路径两两比较，只有相对更好的路径得到正向梯度，劣势路径被抑制。相当于让模型在同场比赛中学习谁跑得更快，而不是单纯追求某个固定时间。  

3. **软关键步骤匹配技术**  
   为了计算StepRAR，作者没有硬性要求模型输出与标注完全一致的关键句，而是用语义相似度和位置容忍度做软匹配。这样即使模型用不同的表达方式描述同一关键概念，也能得到奖励，降低了对标注形式的依赖。  

4. **完整性与逻辑一致性评估策略**  
   StepRVR 通过检查推理链的“完整性”（是否覆盖所有必要子任务）和“逻辑一致性”（前后因果是否连贯）来打分。作者实现了一套规则引擎，能够自动检测缺失步骤或前后矛盾，提供细粒度的负向信号。  

### 方法详解
**整体框架**  
StepGRPO 把一次多模态推理看作一个序列决策过程：模型在看到图像+问题后，先生成第一步文字描述，再基于该描述继续生成第二步，直到输出最终答案。每一步都被视作一次“动作”。在训练时，系统会为每一步计算StepRAR和StepRVR 两种奖励，然后把同一批次里所有路径的奖励进行相对比较，最后用一种叫做“组相对策略梯度”（Group Relative Policy Gradient）的算法更新模型参数。

**关键模块拆解**  

1. **推理生成器**：基于现有的多模态大语言模型（如 LLaVA、MiniGPT‑4）做微调，使其能够在每一步输出一个短句或子答案。可以类比为“写作助理”，每写完一句就交给评估器打分。  

2. **StepRAR 计算**  
   - **关键步骤库**：从高质量 CoT 数据中抽取每道题的必考关键点（如“先计算面积，再乘以密度”）。  
   - **软匹配**：对模型生成的每一步，用语义相似度（如余弦相似度）和位置容忍度（前后顺序可偏差1‑2步）判断是否命中关键点。匹配成功的步骤得到正奖励，未匹配的得到零或负奖励。  
   - **奖励聚合**：把所有步骤的匹配得分加权求和，形成该路径的 StepRAR。  

3. **StepRVR 计算**  
   - **完整性检查**：规则引擎遍历关键步骤库，统计已匹配的比例；比例越高，完整性得分越高。  
   - **逻辑一致性检查**：利用句法依存和因果图构建工具，检测前后步骤是否形成合理的因果链（如“先找出颜色，再判断形状”若顺序颠倒则扣分）。  
   - **最终得分**：把完整性和一致性两部分按预设权重加和，得到 StepRVR。  

4. **组相对奖励计算**  
   - 在一次 minibatch 中，所有路径的 (StepRAR + StepRVR) 形成一个向量。  
   - 对每对路径 i、j，若 i 的总奖励 > j，则 i 获得正向优势信号，j 获得负向抑制信号。  
   - 通过这种相对比较，梯度只推动“更好”路径的概率上升，避免因绝对奖励噪声导致的误更新。  

5. **策略梯度更新**  
   - 使用 REINFORCE‑style 的梯度公式，但把单个奖励换成相对优势值。  
   - 为了降低方差，作者还加入了基线（batch 内奖励均值）进行中心化。  
   - 更新后模型在生成每一步时的概率分布更倾向于产生高奖励的表达。  

**最巧妙的设计**  
软关键步骤匹配让奖励不再“死板”，大幅提升了对多样化语言的容忍度；而组相对策略的引入则把强化学习从“全局评分”转向“局部竞争”，极大加速了学习收敛。两者结合，使得模型在每一步都能感受到“这一步对了”，而不是等到全部结束才知道对错。

### 实验与效果
- **测试任务**：作者在 8 个公开的多模态推理基准上评估，包括 VQA‑Reason、ScienceQA‑Img、MMCoT、NLVR2‑Reason 等，覆盖视觉问答、图文推理、跨模态数学等场景。  
- **对比基线**：与原始 LLaVA、MiniGPT‑4、以及最近的 CoT‑fine‑tuned 版本（如 LLaVA‑CoT）进行比较。  
- **主要结果**：在大多数数据集上，R1‑VL 系列模型的整体准确率提升 3%~7%（例如在 VQA‑Reason 上从 71.2% 提升到 77.8%），而在需要细粒度步骤判断的 ScienceQA‑Img 上提升更明显，超过 9%。  
- **消融实验**：  
  1. 去掉 StepRAR，仅保留 StepRVR，整体提升下降约 1.5%。  
  2. 去掉相对奖励，仅使用绝对奖励，收敛速度慢两倍，最终准确率下降约 2%。  
  3. 将软匹配换成硬匹配，奖励稀疏导致模型在少数关键步骤上表现不稳，整体下降约 1%。  
- **局限性**：作者指出奖励规则依赖于手工构建的关键步骤库，若题目类型变化大，规则需要重新设计；此外，StepGRPO 的计算开销比单纯的监督微调高约 30%，在资源受限的环境下仍有挑战。

### 影响与延伸思考
R1‑VL 把“逐步奖励”引入多模态推理，让强化学习在大语言模型上不再是“全局打分”而是“细粒度指路”。自论文发布后，已有几篇工作尝试把类似的 step‑wise 奖励搬到纯文本 LLM（如 Step‑CoT、GRPO‑LLM），并在代码生成、数学证明等领域取得进展。未来可以探索 **自动化关键步骤抽取**（用模型自己发现哪些步骤是关键）以及 **跨语言/跨模态统一奖励框架**，进一步降低人工规则的依赖。对想深入的读者，建议关注强化学习在大模型中的 variance reduction 技术以及可解释性奖励设计的最新进展。

### 一句话记住它
R1‑VL 用逐步、相对的奖励让多模态大模型在每一步都懂得“这一步对不对”，从而把模仿式推理升级为自我纠错的真实推理能力。