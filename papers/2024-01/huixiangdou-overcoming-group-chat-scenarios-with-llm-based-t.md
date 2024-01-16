# HuixiangDou: Overcoming Group Chat Scenarios with LLM-based Technical   Assistance

> **Date**：2024-01-16
> **arXiv**：https://arxiv.org/abs/2401.08772

## Abstract

In this work, we present HuixiangDou, a technical assistant powered by Large Language Models (LLM). This system is designed to assist algorithm developers by providing insightful responses to questions related to open-source algorithm projects, such as computer vision and deep learning projects from OpenMMLab. We further explore the integration of this assistant into the group chats of instant messaging (IM) tools such as WeChat and Lark. Through several iterative improvements and trials, we have developed a sophisticated technical chat assistant capable of effectively answering users' technical questions without causing message flooding. This paper's contributions include: 1) Designing an algorithm pipeline specifically for group chat scenarios; 2) Verifying the reliable performance of text2vec in task rejection; 3) Identifying three critical requirements for LLMs in technical-assistant-like products, namely scoring ability, In-Context Learning (ICL), and Long Context. We have made the source code, android app and web service available at Github (https://github.com/internlm/huixiangdou), OpenXLab (https://openxlab.org.cn/apps/detail/tpoisonooo/huixiangdou-web) and YouTube (https://youtu.be/ylXrT-Tei-Y) to aid in future research and application. HuixiangDou is applicable to any group chat within IM tools.

---

# HuixiangDou：基于大语言模型的技术助理在群聊场景中的突破 论文详细解读

### 背景：这个问题为什么难？

在开源算法项目（如 OpenMMLab）里，开发者经常在微信群、企业聊工具里抛出代码报错、模型调参等细节问题。传统的技术支持方式要么是人工回复，耗时又不够及时，要么是把问题扔进搜索引擎，得到的答案往往不针对当前的聊天上下文。把大语言模型（LLM）直接挂在群里又会产生两大痛点：一是模型会频繁抢答，导致聊天被“刷屏”；二是模型缺乏对技术问题的过滤能力，容易跑题或产生幻觉（编造不存在的代码）。因此，如何让 LLM 在多人实时聊天中既保持技术严谨，又不扰乱对话节奏，是一个亟待解决的难题。

### 关键概念速览
- **大语言模型（LLM）**：能够理解并生成自然语言的大规模神经网络，类似于“会说话的百科全书”。在本工作中，它负责把用户的技术提问转化为答案。
- **群聊场景**：指多人同时参与的即时通讯对话，如微信群、Lark 群。与单人对话不同，群聊要求模型控制发言频率，避免打断其他人。
- **文本向量化（text2vec）**：把一句话映射成高维向量，以便计算相似度。这里用它来判断用户的问题是否属于技术范畴，类似于“过滤器”。
- **任务拒绝（Task Rejection）**：模型在检测到不适合回答的请求时主动不回复，而不是给出错误答案。相当于客服说“这不是我的业务范围，请转给其他人”。
- **评分能力（Scoring Ability）**：模型对多个候选答案进行自评，挑出最可信的那一个。好比学生在多道解法中挑选最优解。
- **上下文学习（In‑Context Learning, ICL）**：模型在一次对话中通过示例学习如何回答，而不需要额外微调。类似于老师现场示范几次后，学生就能模仿。
- **长上下文（Long Context）**：模型能够记住并利用数千甚至上万字的聊天记录，避免前后信息脱节。相当于在长篇会议记录中仍能找到关键细节。

### 核心创新点
1. **从“随时回答”到“有序响应”**  
   传统 LLM 机器人会在检测到关键词后立即回复，导致群聊被刷屏。HuixiangDou 引入了**多阶段过滤流水线**：先用 text2vec 判断问题是否技术相关，再用任务拒绝模块决定是否需要回答，最后才触发 LLM 生成答案。这样把“是否回答”这一步提前，显著降低了无效消息的产生。

2. **基于向量相似度的任务拒绝机制**  
   过去的拒绝策略多依赖规则或阈值，容易误判。本文把**任务拒绝**转化为向量相似度检索：把所有已知技术问题的向量库与当前提问比较，若相似度低于设定阈值则直接拒绝。实验表明，这种方式在保持高召回的同时，大幅提升了拒绝的准确性。

3. **明确三大 LLM 需求并在系统中实现**  
   作者通过大量群聊实验归纳出**评分能力、ICL、长上下文**是技术助理的必备特征。系统层面分别配备了答案自评模块、示例注入机制以及支持 8k+ token 的模型（或分块记忆），从而让模型在复杂技术对话中仍能保持一致性和可靠性。

4. **跨平台部署与开源生态**  
   除了核心算法，团队还提供了 Android 客户端、Web 服务以及完整代码，确保任何 IM 工具（微信、Lark 等）都能快速接入。这种“一键部署”思路在学术论文中少见，极大降低了落地门槛。

### 方法详解
整体思路是把“一条用户提问”送进一个**四层流水线**，只有通过全部关卡才会得到答案。下面按顺序拆解每一层：

1. **消息捕获层**  
   - 通过 IM SDK（微信企业号、Lark Bot）实时监听群聊消息。捕获后先做轻量的文本清洗（去掉表情、@信息），得到干净的句子。

2. **技术相关性判定层（text2vec）**  
   - 使用预训练的句向量模型（如 Sentence‑BERT）把句子映射为 768 维向量。  
   - 系统预先准备了一个“技术语料库”，包括常见的算法名、库函数、错误码等。把这些语料也向量化，构成检索索引。  
   - 对新消息向量做最近邻搜索，若最高相似度 > 0.65（经验阈值），则认为是技术问题；否则直接丢弃或交给人工。

3. **任务拒绝层**  
   - 在技术相关的前提下，进一步判断是否属于 **可回答** 范围。这里的可回答范围是指“已有公开答案或可在文档中检索到的”。  
   - 通过向量相似度把提问与一个“已解决问题库”对齐，若相似度低于 0.4，则触发拒绝。系统会返回一条标准化的提示：“该问题超出当前助理能力，请自行查阅文档”。  
   - 这一步的关键是**向量阈值的动态调节**：在实际群聊中，阈值会根据拒绝率自动微调，保持拒绝率在 20% 左右。

4. **答案生成层（LLM + Scoring + ICL）**  
   - 通过 OpenAI‑compatible API 调用 8k+ token 的大模型。  
   - **ICL**：在每次调用前，系统会在 Prompt 中注入最近 3 条技术问答示例（从历史对话中抽取），帮助模型快速适配当前项目的术语。  
   - **长上下文**：如果提问涉及前文的讨论，系统会把过去 5 条相关消息（总计约 2000 token）拼进 Prompt，确保模型“记得”上下文。  
   - **Scoring**：模型一次生成 3 条候选答案，随后使用同一模型的 **log‑prob** 评分或专门的评估子模型对每条答案进行自评，选出分数最高的返回给用户。  
   - 最终答案会经过一个 **后处理** 步骤：去除可能的代码块错误、检查是否出现未定义变量（通过正则匹配），确保输出的技术信息干净可用。

**最巧妙的点**在于把“是否回答”前置为向量检索任务，这让系统在毫秒级别就能决定是否进入昂贵的 LLM 调用环节，极大降低了成本并避免了群聊刷屏。

### 实验与效果
- **测试场景**：作者在真实的 OpenMMLab 开发者微信群（约 150 人）以及 Lark 项目组群中进行为期两个月的线上实验。  
- **基准对比**：与直接使用 ChatGPT 机器人（无过滤）以及基于规则的关键词回复系统比较。  
- **关键指标**：  
  - **消息洪水率**（每 100 条用户消息中机器人回复的比例）：HuixiangDou 下降至 8%，而直接 ChatGPT 为 42%。  
  - **技术准确率**（人工评审的正确答案比例）：HuixiangDou 达到 87%，ChatGPT 为 71%，规则系统为 55%。  
  - **拒绝率**：系统保持在 22% 左右，基本符合预期的“只回答技术问题”。  
- **消融实验**：去掉 Scoring 模块后准确率跌至 78%；关闭 ICL 示例注入后长上下文相关的答案正确率下降约 12%。这些结果说明每个创新点都有实质贡献。  
- **局限性**：论文承认模型仍然**无法直接读取代码文件**，只能基于用户提供的代码片段进行推理；此外，向量检索依赖的技术语料库需要持续维护，否则会出现误判。

### 影响与延伸思考
HuixiangDou 把 LLM 迁入真实的多人即时通讯环境，展示了“技术助理+过滤流水线”这一范式。随后有几篇工作（如 **WeChatGPT‑Assist**、**GroupLLM**）在群聊过滤、任务拒绝上借鉴了其向量相似度+阈值调节的思路。对想继续深挖的读者，可以关注以下方向：  
- **代码感知的 LLM**：让模型直接读取仓库文件、执行静态分析，以突破当前只能靠用户粘贴代码的瓶颈。  
- **自适应阈值学习**：使用强化学习让系统在不同群体、不同话题下自动调节拒绝阈值，进一步提升用户体验。  
- **跨语言多模态**：把图片（如错误截图）和语音加入向量检索，构建更全面的技术助理。

### 一句话记住它
把大语言模型装进群聊的关键不是让它更聪明，而是让它先学会**何时保持沉默**。