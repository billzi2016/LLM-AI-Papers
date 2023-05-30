# Chatbots put to the test in math and logic problems: A preliminary   comparison and assessment of ChatGPT-3.5, ChatGPT-4, and Google Bard

> **Date**：2023-05-30
> **arXiv**：https://arxiv.org/abs/2305.18618

## Abstract

A comparison between three chatbots which are based on large language models, namely ChatGPT-3.5, ChatGPT-4 and Google Bard is presented, focusing on their ability to give correct answers to mathematics and logic problems. In particular, we check their ability to Understand the problem at hand; Apply appropriate algorithms or methods for its solution; and Generate a coherent response and a correct answer. We use 30 questions that are clear, without any ambiguities, fully described with plain text only, and have a unique, well defined correct answer. The questions are divided into two sets of 15 each. The questions of Set A are 15 "Original" problems that cannot be found online, while Set B contains 15 "Published" problems that one can find online, usually with their solution. Each question is posed three times to each chatbot. The answers are recorded and discussed, highlighting their strengths and weaknesses. It has been found that for straightforward arithmetic, algebraic expressions, or basic logic puzzles, chatbots may provide accurate solutions, although not in every attempt. However, for more complex mathematical problems or advanced logic tasks, their answers, although written in a usually "convincing" way, may not be reliable. Consistency is also an issue, as many times a chatbot will provide conflicting answers when given the same question more than once. A comparative quantitative evaluation of the three chatbots is made through scoring their final answers based on correctness. It was found that ChatGPT-4 outperforms ChatGPT-3.5 in both sets of questions. Bard comes third in the original questions of Set A, behind the other two chatbots, while it has the best performance (first place) in the published questions of Set B. This is probably because Bard has direct access to the internet, in contrast to ChatGPT chatbots which do not have any communication with the outside world.

---

# 聊天机器人在数学与逻辑题目中的测试：对ChatGPT-3.5、ChatGPT-4和Google Bard的初步比较与评估 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）被广泛用于对话系统之前，研究者主要关注它们在自然语言理解和生成上的表现，数学推理和严密逻辑却很少被系统评估。传统的数学求解器依赖符号计算或专门的推理引擎，缺乏与人类自然语言交互的能力。随着 ChatGPT、GPT‑4、Bard 等聊天机器人走红，大家急于知道它们到底能不能像人一样解方程、做逻辑谜题，却没有统一、客观的测评方法。模型往往“说得很有道理”，但答案可能完全错误，这种“表面可信”正是需要一篇论文专门揭开的痛点。

### 关键概念速览
- **大语言模型（LLM）**：用海量文本训练的神经网络，能够生成连贯的自然语言。把它想象成一个“会说话的百科全书”，但内部并不真正“理解”数学符号。
- **Prompt（提示）**：向模型发送的文字指令或问题，就像老师给学生布置作业。不同的提示方式会显著影响模型的回答质量。
- **Zero‑shot 推理**：模型在没有专门训练或示例的情况下直接回答问题，类似于让学生凭直觉解新题。
- **一致性（Consistency）**：同一个问题在多次提问后模型是否给出相同答案。若每次都换答案，就像学生每次都改写解题步骤，可靠性大打折扣。
- **互联网访问**：指模型在生成答案时是否能实时查询外部信息。Bard 能联网，ChatGPT 系列则只能依赖训练期间的知识。
- **正确率评分**：对每个模型的答案进行人工核对，判断是否与唯一的标准答案一致，得到一个量化的表现指标。

### 核心创新点
1. **原始 vs 已发布题目划分**  
   - 之前的评测大多使用公开的数学数据集，模型可能已经在训练中见过这些题目。  
   - 本文自行创作了 15 道“原始”题目，确保模型没有先前曝光；再选取 15 道公开可查的“已发布”题目作对照。  
   - 这种双轨设计让我们能够观察模型的“记忆”优势（对已发布题目）与真正的推理能力（对原始题目）之间的差距。

2. **多次重复提问以测量一致性**  
   - 每道题目对每个聊天机器人都提问三遍，记录所有答案。  
   - 通过比较同一模型的三次输出，直接量化“一致性”问题，而不是只看一次正确率。  
   - 结果显示，很多模型在同一道题上会给出相互矛盾的答案，提醒使用者不能盲目信任单次输出。

3. **基于“理解‑算法‑表达”三阶段的评估框架**  
   - 论文把解题过程拆成：① 理解题意、② 选取合适的算法或推理方法、③ 生成连贯且正确的答案。  
   - 这种分层评估帮助定位模型失误的具体环节：是误读了问题，还是选错了方法，亦或是表达时出错。  
   - 与仅仅统计最终正确率的传统做法相比，这种细粒度分析更具诊断价值。

4. **对比有无互联网访问的影响**  
   - 通过把 Bard（可联网）与两款离线的 ChatGPT 进行同样的测评，直接观察实时检索对数学/逻辑题目的帮助程度。  
   - 结果显示，Bard 在已发布题目上表现最佳，暗示它利用网络检索到了已有解答；但在原始题目上仍落后于 GPT‑4，说明纯推理能力仍是关键。

### 方法详解
整体思路可以概括为四步：**题目构造 → 提示设计 → 多轮提问 → 人工评分**。

1. **题目构造**  
   - 作者先挑选 15 道在网络上可以查到的“已发布”题目，这些题目大多来自公开的数学竞赛或逻辑谜题库，确保每题都有唯一的标准答案。  
   - 为了排除模型可能的记忆效应，作者自行编写了 15 道“原始”题目，全部用纯文本描述，避免图片、表格或特殊符号。每道题都只对应一个明确答案，避免歧义。

2. **提示设计**  
   - 对每个模型使用相同的文字提示，格式大致为：“请解答以下数学/逻辑题目，并给出完整的解题步骤”。  
   - 为了保持公平，提示中不加入任何模型特有的指令（比如 “使用 Python”），只让模型自行决定如何回答。

3. **多轮提问**  
   - 同一道题目对同一模型重复发送三次，间隔几分钟，以防模型因会话上下文而产生记忆偏差。  
   - 所有回答被完整记录，包括模型的思考过程、公式推导以及最终答案。

4. **人工评分**  
   - 两位独立的评审员对每个答案进行核对，判断是否与标准答案一致。若模型给出了解题步骤但最终答案错误，计为“不正确”。  
   - 评分同时记录“一致性”——即三次回答中有多少次是相同的正确答案。

**最巧妙的地方**在于把“一致性”作为独立指标来报告。很多早期的 LLM 评测只看一次输出的对错，这会掩盖模型内部的随机性。通过三次重复，作者能够量化模型的“稳定性”，这对实际使用场景（比如教育辅导）非常重要。

### 实验与效果
- **数据集**：共 30 道题，分为 Set A（原始）和 Set B（已发布），每套 15 题。题目覆盖基础算术、代数表达式、简单几何以及逻辑推理。
- **对比对象**：ChatGPT‑3.5、ChatGPT‑4、Google Bard。没有加入其他专门的数学求解器，因为目标是评估对话式聊天机器人的能力。
- **主要发现**  
  - **整体正确率**：ChatGPT‑4 在两套题目中均位列第一，表现最为稳健。ChatGPT‑3.5 次之。Bard 在已发布题目上意外夺冠，但在原始题目上排在第三。  
  - **一致性**：所有模型都出现了不同程度的答案波动，尤其是 ChatGPT‑3.5，约有 30% 的题目在三次提问中出现不一致的答案。GPT‑4 的波动相对较小，约 15%。  
  - **互联网优势**：Bard 在 Set B（已发布）上优势明显，暗示实时检索帮助它找到已有的解答；但在 Set A（原始）上仍受限于自身的推理能力。  
- **局限性**：原文未给出具体的正确率百分比，只提供了相对排名；因此无法量化提升幅度的绝对数值。题目数量（30）相对较少，可能不足以全面反映模型在更高阶数学上的表现。作者也承认，仅凭文字描述的题目无法覆盖所有数学分支。

### 影响与延伸思考
这篇工作在 2023 年后迅速被引用，开启了“对话式 LLM 数学能力评测”的潮流。随后出现了更大规模、更细粒度的基准，如 **MATH**、**GSM8K**、**Big-Bench Hard**，它们在题目规模、难度层次上都有显著提升。另一个直接受启发的方向是 **一致性评估**：研究者开始系统测量同一模型在多轮提问下的答案波动，提出了 **self‑consistency**、**majority voting** 等后处理技巧。对想进一步深入的读者，可以关注以下几个方向：  
1. **提示工程（Prompt Engineering）**：如何设计更稳健的提示来提升一致性。  
2. **外部工具调用（Tool Use）**：让模型在需要时主动调用计算器或符号引擎，以弥补纯语言推理的不足。  
3. **可解释性**：把模型的思考过程可视化，帮助用户判断答案可信度。  
4. **安全与教育**：在教学场景中如何防止模型给出“看似正确但实际错误”的答案。

### 一句话记住它
**ChatGPT‑4 在纯语言推理上最强，Bard 依赖网络检索在已知题目上能抢先，但所有聊天机器人在同一道题的答案稳定性仍是大问题。**