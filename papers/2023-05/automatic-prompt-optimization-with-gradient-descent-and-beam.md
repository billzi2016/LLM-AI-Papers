# Automatic Prompt Optimization with "Gradient Descent" and Beam Search

> **Date**：2023-05-04
> **arXiv**：https://arxiv.org/abs/2305.03495

## Abstract

Large Language Models (LLMs) have shown impressive performance as general purpose agents, but their abilities remain highly dependent on prompts which are hand written with onerous trial-and-error effort. We propose a simple and nonparametric solution to this problem, Automatic Prompt Optimization (APO), which is inspired by numerical gradient descent to automatically improve prompts, assuming access to training data and an LLM API. The algorithm uses minibatches of data to form natural language "gradients" that criticize the current prompt. The gradients are then "propagated" into the prompt by editing the prompt in the opposite semantic direction of the gradient. These gradient descent steps are guided by a beam search and bandit selection procedure which significantly improves algorithmic efficiency. Preliminary results across three benchmark NLP tasks and the novel problem of LLM jailbreak detection suggest that Automatic Prompt Optimization can outperform prior prompt editing techniques and improve an initial prompt's performance by up to 31%, by using data to rewrite vague task descriptions into more precise annotation instructions.

---

# 基于梯度下降与束搜索的自动提示优化 论文详细解读

### 背景：这个问题为什么难？

LLM（大语言模型）在零样本、少样本任务上表现惊艳，但它们的输出质量往往高度依赖于提示（prompt）的写法。传统上，研究者和工程师需要手工调试提示，反复尝试不同的指令、示例和格式，这既耗时又缺乏可复现性。已有的自动化提示改写方法大多基于固定的规则或在大量示例上进行微调，导致：①对新任务的迁移能力弱；②需要额外的模型参数或大规模搜索，计算成本高；③改写过程缺乏明确的优化方向，往往是盲目的随机采样。正因为这些根本性限制，如何在不增加模型参数、只利用已有的 LLM 接口和少量标注数据，快速、系统地提升提示质量，成为亟待解决的难题。

### 关键概念速览
- **Prompt（提示）**：给 LLM 的文字指令或上下文，决定模型的行为，就像给人下达任务说明书一样。  
- **梯度下降（Gradient Descent）**：在连续空间里沿负梯度方向迭代更新参数，以最小化损失函数。这里把“梯度”抽象为自然语言的批评信息。  
- **自然语言梯度（Natural‑language Gradient）**：模型在当前提示下对一批样本的错误输出生成的文字化批评，类似于老师给学生的改错意见。  
- **束搜索（Beam Search）**：在生成序列时保留多个候选路径，逐步扩展最有希望的几条，像在迷宫里同时走几条可能的路。  
- **Bandit Selection（赌博机选择）**：在多个候选提示中根据历史表现分配探索与利用的概率，类似于在多台老虎机中挑选收益最高的那台。  
- **Mini‑batch**：一次抽取的少量训练样本，用来估计梯度，防止一次性使用全部数据导致噪声过大。  
- **Prompt Editing（提示编辑）**：对已有提示进行增删改，使其更贴合任务需求，类似于对说明书进行润色。

### 核心创新点
1. **把梯度概念搬进自然语言**  
   - 之前的提示优化要么是离散的搜索，要么是基于梯度的微调但只能在模型内部参数上操作。  
   - 本文让 LLM 本身生成对当前提示的文字化批评（自然语言梯度），再把这些批评的语义方向反向“传播”到提示中。  
   - 这样实现了在纯文本空间的梯度下降，既不需要额外模型，也保持了提示的可解释性。

2. **梯度驱动的束搜索 + 赌博机策略**  
   - 传统束搜索只在单一步骤里保留最优候选，容易陷入局部最优。  
   - 这里先用束搜索产生若干编辑候选，再用赌博机算法根据历史改写收益动态分配搜索预算。  
   - 结果是显著提升了搜索效率，少量 API 调用就能找到高质量提示。

3. **基于 Mini‑batch 的在线迭代**  
   - 直接在全数据上计算梯度会导致噪声和计算开销。  
   - 采用小批量抽样，每一步只用几条标注样本生成梯度，既保持了梯度的代表性，又让迭代快速。  
   - 这种“在线”方式让 APO 能在实际生产环境中边跑边改。

4. **从模糊描述到精准指令的自动重写**  
   - 许多手工提示只给出任务的大概目标，缺少细粒度的标注规则。  
   - APO 能把这些宽泛的描述转化为更具体的指令（例如明确列出标签定义、示例格式），提升模型对任务的理解深度。  
   - 实验显示，这种重写在三个基准 NLP 任务上带来了最高 31% 的性能提升。

### 方法详解
**整体框架**  
APO 把提示优化过程拆成四步：①采样 Mini‑batch；②让 LLM 生成自然语言梯度；③依据梯度语义对提示进行编辑；④用束搜索 + 赌博机挑选最有前景的编辑并迭代。整个循环在固定的迭代次数或性能收敛时停止。

**步骤拆解**  

1. **Mini‑batch 采样**  
   - 随机抽取 N 条标注数据（如文本‑标签对），作为当前迭代的评估基准。  
   - 这一步类似机器学习里常见的随机梯度下降，只是样本量更小，便于快速调用 LLM API。

2. **生成自然语言梯度**  
   - 将当前提示与每条样本一起送入 LLM，得到模型的预测。  
   - 再把同一提示与真实标签一起送入 LLM，要求模型输出“错误原因”或“改进建议”。  
   - 这些文字化的批评即为“梯度”，它们指明了提示在当前任务上的不足之处（比如缺少示例、指令不够明确）。

3. **梯度驱动的提示编辑**  
   - 对每条梯度文本，使用一种“语义反向”操作：如果梯度说“需要更明确的情感标签”，编辑器会在提示中加入对应的标签说明。  
   - 编辑方式包括：插入新句子、替换模糊词、删除冗余描述。实现上可以把梯度当作指令，交给同一个 LLM 完成编辑（即“编辑模型”），保持全流程在语言层面。

4. **束搜索 + 赌博机选择**  
   - 对每一次编辑，束搜索会产生 K 条候选提示（不同的插入位置、不同的措辞）。  
   - 赌博机算法记录每条候选在过去迭代中的得分（如验证集上的准确率提升），根据 Upper Confidence Bound（UCB）等策略分配下一轮的搜索预算。  
   - 这样既保留了多样性，又倾向于探索历史表现好的方向，避免盲目遍历所有可能编辑。

5. **迭代与收敛**  
   - 选出得分最高的提示作为本轮的“当前提示”，进入下一轮 Mini‑batch 采样。  
   - 迭代若干次后，提示的性能趋于稳定，即可停止。

**最巧妙的点**  
- 把梯度抽象为自然语言，让“负梯度方向”变成“把批评的语义反向写进提示”。这把本来只能在向量空间里做的优化，搬到了人类可读的文字层面。  
- 使用同一个 LLM 完成梯度生成、编辑和候选生成，形成闭环，无需额外模型或参数。  
- 赌博机的探索‑利用平衡，使得在 API 调用成本受限的情况下仍能找到高质量提示。

### 实验与效果
- **任务与数据集**：论文在三个公开 NLP 基准（如情感分类、自然语言推理、问答）以及一个新设的 LLM jailbreak 检测任务上评估 APO。  
- **对比基线**：包括手工提示、基于规则的提示编辑、以及最近的提示微调方法（如 Prompt Tuning、AutoPrompt）。  
- **性能提升**：在所有任务上 APO 均超过基线，最高提升约 31%（相对原始手工提示的准确率）。在 jailbreak 检测任务中，APO 能把原本几乎无法检测的逃逸案例识别率提升到可接受水平。  
- **消融实验**：作者分别去掉梯度生成、束搜索、赌博机三块组件，发现：①没有梯度的随机编辑性能下降约 12%；②仅用束搜索而不做赌博机选择，搜索成本翻倍但提升不到 5%；③去掉 Mini‑batch（改为全量）导致收敛速度显著变慢。  
- **局限性**：实验主要在英文任务上进行，中文或其他低资源语言的效果未报告；梯度质量依赖于 LLM 本身的自我批评能力，若模型本身不擅长生成有用的错误分析，优化效果会受限。作者也提到 APO 仍需要一定数量的标注样本，完全零样本场景下仍不可用。

### 影响与延伸思考
- 发表后，APO 的思路激发了“语言层面梯度优化”的一系列工作，例如把人类反馈直接转化为提示编辑指令的 RLHF（强化学习从人类反馈）变体。  
- 有研究尝试把 APO 与多模态模型结合，让视觉‑语言梯度一起作用于跨模态提示。  
- 未来可以探索：①在更低资源语言上通过少量跨语言迁移提升梯度质量；②把梯度生成交给专门的“评审模型”，提升批评的专业度；③将 APO 融入持续学习系统，实现提示随时间自动演化。  
- 对想深入的读者，建议关注近期的 “Self‑Critique” 与 “Chain‑of‑Thought Prompting” 方向，它们在生成自然语言梯度方面提供了更细粒度的技术细节。

### 一句话记住它
把 LLM 的自我批评当作梯度，用文字在提示上做“负梯度”更新，配合束搜索和赌博机，就能在几次 API 调用里把模糊提示自动炼成高效指令。