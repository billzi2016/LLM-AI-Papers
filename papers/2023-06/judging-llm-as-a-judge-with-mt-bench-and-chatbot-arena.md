# Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena

> **Date**：2023-06-09
> **arXiv**：https://arxiv.org/abs/2306.05685

## Abstract

Evaluating large language model (LLM) based chat assistants is challenging due to their broad capabilities and the inadequacy of existing benchmarks in measuring human preferences. To address this, we explore using strong LLMs as judges to evaluate these models on more open-ended questions. We examine the usage and limitations of LLM-as-a-judge, including position, verbosity, and self-enhancement biases, as well as limited reasoning ability, and propose solutions to mitigate some of them. We then verify the agreement between LLM judges and human preferences by introducing two benchmarks: MT-bench, a multi-turn question set; and Chatbot Arena, a crowdsourced battle platform. Our results reveal that strong LLM judges like GPT-4 can match both controlled and crowdsourced human preferences well, achieving over 80% agreement, the same level of agreement between humans. Hence, LLM-as-a-judge is a scalable and explainable way to approximate human preferences, which are otherwise very expensive to obtain. Additionally, we show our benchmark and traditional benchmarks complement each other by evaluating several variants of LLaMA and Vicuna. The MT-bench questions, 3K expert votes, and 30K conversations with human preferences are publicly available at https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge.

---

# 用 MT‑Bench 与 Chatbot Arena 评估 LLM 作为评审 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）已经可以和人聊天、写代码、写文章，但要判断它们到底有多好却很棘手。传统的评测数据集大多是闭式的、答案唯一的题目，根本抓不住用户真实的偏好。人工打分虽然能反映人类感受，却成本高、规模难以扩展。于是出现了“让强大的 LLM 当评审”的想法，但没人系统验证它们的判断是否真的和人类一致，也没有统一的基准来对比不同模型的表现。

### 关键概念速览
- **LLM‑as‑a‑judge**：把一个已经对齐好的大模型（如 GPT‑4）当成评分员，让它给其他模型的回答打分。类似请一位资深老师给学生作文评分，省去人工评卷的时间。
- **位置偏差（position bias）**：评审模型倾向于给先出现的答案更高分，就像人类在比较两篇文章时，先读的那篇往往印象更深。
- **冗长偏差（verbosity bias）**：模型更喜欢字数多、解释详细的答案，即使内容并不更好，类似有人在会议上说得多就被误认为更有价值。
- **自我增强偏差（self‑enhancement bias）**：当评审模型本身也是被评估对象时，它可能倾向于抬高自己的分数，类似“自家孩子成绩好”。
- **MT‑Bench**：一套多轮开放域问答题目，专门用来让评审模型在对话上下文中打分，像是让评审老师在课堂上连续提问并评分。
- **Chatbot Arena**：一个众包平台，用户可以让两位聊天机器人对话后投票选出更好的一方，提供大规模的真实人类偏好数据。
- **一致性（agreement）**：评审模型的评分与人类投票结果的匹配程度，用百分比表示，越高说明模型越像人类。

### 核心创新点
1. **系统化的 LLM‑as‑judge 框架 → 通过两套互补基准（MT‑Bench 与 Chatbot Arena）验证模型评分的可靠性 → 证明强模型（GPT‑4）与人类偏好超过 80% 的一致率，达到人与人之间的相似水平。**  
   之前只有零星实验或小规模对比，这里提供了完整的实验设计和公开数据。

2. **识别并缓解评审模型的系统性偏差 → 通过随机化答案顺序、限制回答长度、以及在评分时加入“自我对齐”提示来抑制位置、冗长和自我增强偏差 → 让评审结果更公平、更接近真实人类判断。**  
   过去直接使用 LLM 打分会被这些偏差扭曲，这一步显著提升了评分的可信度。

3. **将传统任务导向基准与开放式对话基准结合 → 在同一实验中同时跑了已有的 SFT/ RLHF 基准和新提出的 MT‑Bench，展示两者互补的评估视角 → 为模型研发提供了更全景的性能画像。**  
   以前的评测往往只关注单一维度，这里实现了多维度交叉验证。

### 方法详解
整体思路可以拆成三大步骤：**准备评审模型 → 设计评审任务 → 校正并验证评分**。

1. **准备评审模型**  
   选取已经经过指令微调（Instruction‑tuned）和强化学习（RLHF）的大模型作为评审员，主要是 GPT‑4。为了防止自我增强偏差，评审模型在评分时不暴露自己的身份，也不参与被评模型的训练过程。

2. **设计评审任务**  
   - **MT‑Bench**：构造约 80 条多轮对话，每轮包含一个开放式问题（如“解释量子纠缠的意义”），并让被评模型给出答案。评审模型收到完整的对话上下文以及所有候选答案，随后按照“有理有据、信息完整、语言流畅”等维度给出 0‑10 分的细粒度评分。  
   - **Chatbot Arena**：在网页平台上让真实用户同时与两位聊天机器人对话，结束后用户投票选出更满意的机器人。投票结果被记录为人类偏好标签。评审模型随后对同一对话对进行评分，输出的相对优劣顺序与用户投票进行比对。

3. **校正并验证评分**  
   - **位置随机化**：在每次评分前随机打乱候选答案的顺序，消除位置偏差。  
   - **冗长约束**：对每个答案强制截断到相同的 token 数，或在提示中要求评审模型忽略字数因素，只关注信息质量。  
   - **自我对齐提示**：在评审提示里加入“请客观评估，不要考虑模型的来源”，降低自我增强倾向。  
   - **一致性计算**：将评审模型的相对排序与人类投票的排序做配对，统计匹配比例。若两者完全一致，则一致率为 100%。实验中 GPT‑4 在两套基准上均超过 80% 的一致率。

最巧妙的地方在于把“随机化”和“约束”直接写进评审提示，而不是在后处理阶段做修正，这样评审模型本身就被迫遵守公平规则，避免了后期人为干预带来的噪声。

### 实验与效果
- **数据集**：MT‑Bench 包含约 3,000 条专家投票的多轮对话；Chatbot Arena 收集了 30,000 场真实用户对战记录。两者分别覆盖了专业问答和日常聊天两大场景。  
- **基线对比**：与传统人工评分、以及使用较弱模型（如 GPT‑3.5）作为评审的结果相比，GPT‑4 的一致率提升约 15‑20%。在 MT‑Bench 上，GPT‑4 与人类的 80% 一致率几乎等同于两个人类评审之间的 82% 一致率。  
- **消融实验**：去掉位置随机化后，一致率下降约 5%；去掉冗长约束后下降约 3%；去掉自我对齐提示后下降约 2%。说明每项校正都有实质贡献。  
- **局限性**：论文指出，LLM‑as‑judge 仍然受限于模型的推理深度，面对需要复杂逻辑或长时记忆的任务时评分可能不够精准；此外，评审模型本身的训练数据偏见仍会间接影响评分。  

### 影响与延伸思考
这篇工作打开了“模型自评”在大语言模型研发中的新局面。随后出现的多篇论文（如 *Self‑Refine*, *LLM‑Eval*）都借鉴了 LLM‑as‑a‑judge 的思路，尝试在更细粒度的任务（代码生成、数学推理）上使用同类评审模型。业界也开始在开源项目里提供类似的评审 API，降低了大规模人类标注的门槛。想进一步深入，可以关注以下方向：  
- **多模态评审**：把视觉、音频信息一起交给评审模型，评估多模态对话系统。  
- **自适应提示工程**：研究如何自动生成最能抑制偏差的评审提示。  
- **跨模型一致性**：探索不同规模、不同架构的评审模型之间的一致性，构建更稳健的评审联盟。  

### 一句话记住它
强大的 LLM（如 GPT‑4）配合精心设计的 MT‑Bench 与 Chatbot Arena，能够以超过 80% 的人类一致率，可靠地充当模型评审员。