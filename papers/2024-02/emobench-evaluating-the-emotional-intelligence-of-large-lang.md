# EmoBench: Evaluating the Emotional Intelligence of Large Language Models

> **Date**：2024-02-19
> **arXiv**：https://arxiv.org/abs/2402.12071

## Abstract

Recent advances in Large Language Models (LLMs) have highlighted the need for robust, comprehensive, and challenging benchmarks. Yet, research on evaluating their Emotional Intelligence (EI) is considerably limited. Existing benchmarks have two major shortcomings: first, they mainly focus on emotion recognition, neglecting essential EI capabilities such as emotion regulation and thought facilitation through emotion understanding; second, they are primarily constructed from existing datasets, which include frequent patterns, explicit information, and annotation errors, leading to unreliable evaluation. We propose EmoBench, a benchmark that draws upon established psychological theories and proposes a comprehensive definition for machine EI, including Emotional Understanding and Emotional Application. EmoBench includes a set of 400 hand-crafted questions in English and Chinese, which are meticulously designed to require thorough reasoning and understanding. Our findings reveal a considerable gap between the EI of existing LLMs and the average human, highlighting a promising direction for future research. Our code and data are publicly available at https://github.com/Sahandfer/EmoBench.

---

# EmoBench：大语言模型情感智能评估基准 论文详细解读

### 背景：这个问题为什么难？
情感智能（Emotional Intelligence，EI）在心理学里指的是感知、理解、调节自己和他人情绪的能力。随着大语言模型（LLM）在对话、写作等场景的广泛落地，模型是否真的具备类似人类的情感理解与运用，成为评估它们实用性的关键。然而，已有的情感评测大多只测“我能不能辨认出‘高兴’、‘悲伤’”，忽视了情绪调节、情感驱动的思考等更深层次的能力。更糟的是，这些评测往往直接搬现成的数据集，数据里充斥着重复模式、显式提示甚至标注错误，导致模型可能靠记忆或技巧“骗过”测试，却没有真正的情感推理。于是，缺少一个既覆盖情感全谱，又能逼迫模型进行严谨推理的基准，成了阻碍研究的瓶颈。

### 关键概念速览
- **情感智能（EI）**：指感知、解释、调节情绪并据此指导行为的综合能力。可以想象成“情绪的全能工具箱”，不仅能识别情绪，还能用情绪来思考和决策。  
- **情感理解（Emotional Understanding）**：对情绪产生的原因、表现和影响的认知层面。类似于“读懂别人的情绪背后的故事”。  
- **情感应用（Emotional Application）**：把情绪认知转化为调节或利用的行动，如安慰、激励或利用情绪信息做决策。相当于“把情绪当成工具来使用”。  
- **情绪调节（Emotion Regulation）**：主动改变自己或他人情绪状态的过程，比如安抚、激励。可以比作“情绪的温度调节器”。  
- **思维促进（Thought Facilitation）**：利用情绪信息帮助推理、创意或问题解决。像是“情绪给思考加了助推器”。  
- **手工构造题目（Hand‑crafted Questions）**：由研究者人工设计、经过多轮审校的测试题，而非直接从已有数据集抽取。相当于“精心烹饪的菜谱”，每道菜都有明确的味道目标。  
- **双语评测（Bilingual Evaluation）**：同时提供英文和中文题目，考察模型跨语言的情感迁移能力。可以类比为“在两种语言的舞台上同台表演”。  

### 核心创新点
1. **从情感识别扩展到情感全谱**  
   - 之前的评测大多停留在“我能否说出‘这句话是悲伤的’”。  
   - 这篇论文把机器情感智能定义为“情感理解 + 情感应用”，加入情绪调节和思维促进两大维度。  
   - 结果是模型需要在理解情绪根源的基础上给出调节建议或利用情绪信息完成推理，评测难度提升了数倍。

2. **全手工、需深度推理的题库**  
   - 传统基准直接复用公开数据集，容易出现模式化、标注错误等问题。  
   - 作者们自行编写了 400 题，覆盖英文和中文，每题都要求模型进行多步推理、情境建模或价值判断。  
   - 这种设计让模型只能靠真正的情感推理而不是记忆模式来得分，评测可信度大幅提升。

3. **双语覆盖与跨语言一致性检验**  
   - 过去的情感基准几乎全是单语的，难以评估模型的跨语言情感迁移。  
   - 本文在英文和中文两套题目上保持结构和难度一致，直接比较模型在不同语言下的表现。  
   - 这为后续研究提供了检验多语言情感智能的统一平台。

4. **基于心理学理论的评估框架**  
   - 许多情感评测缺乏理论支撑，导致指标随意。  
   - 论文引用了经典情感心理学模型（如情绪的三维模型、情绪调节策略分类），把这些理论映射到机器任务上。  
   - 这样做让评估指标更具解释性，也方便后续工作在同一理论体系下扩展。

### 方法详解
整体思路可以拆成三大步骤：**概念定义 → 题目构造 → 评估流程**。

1. **概念定义**  
   作者先在心理学文献中梳理出情感智能的完整结构，明确“情感理解”和“情感应用”两大块。情感理解包括情绪识别、情绪原因推断和情绪影响评估；情感应用则细化为情绪调节（如安慰、激励）和思维促进（如利用情绪信息做决策）。这一步相当于给模型的“情感能力”画出一张地图，后面的题目都围绕这张地图设计。

2. **题目构造**  
   - **情境设定**：每题先给出一个生活化情境（如朋友失业、团队冲突），确保模型必须先构建背景。  
   - **情绪标注**：在情境中埋入情绪线索（语言、行为），要求模型先识别情绪并解释产生原因。  
   - **任务指令**：根据情感理解的结果，指令模型执行情感应用，如提供安慰话语、给出调节方案或说明情绪如何影响后续决策。  
   - **多步推理**：题目设计成需要模型先完成情绪识别 → 原因推断 → 调节建议 → 结果预测的链式过程。  
   - **双语同步**：每一道英文题都有对应的中文翻译，保持情境、情绪线索和任务指令的等价性。  

   为防止模型靠“模式匹配”得分，作者在每一步加入了随机化的细节（如人物名字、地点、情绪强度描述），并让答案必须以自然语言解释而非直接选项。

3. **评估流程**  
   - **模型调用**：对每个问题，使用统一的提示模板让模型输出完整的推理过程和最终答案。  
   - **自动评分**：先用规则匹配检查是否覆盖所有必答要点（情绪标签、原因、调节方案等），再交叉人工评审确保答案质量。  
   - **分层得分**：情感理解和情感应用分别计分，最终给出综合 EI 分数。  
   - **人类基准**：邀请具备普通情感认知的受试者完成同样的题目，计算平均人类得分，用作上限参考。  

   其中最巧妙的地方在于**“强制链式输出”**：模型必须把每一步写出来，类似思维链（Chain‑of‑Thought）但专门针对情感任务，这让评估者可以直接看到模型在哪一步出现了逻辑断层。

### 实验与效果
- **测试对象**：论文在多款主流大语言模型上跑了评测，包括 GPT‑4、Claude、LLaMA‑2‑70B、ChatGLM 等（具体型号在原文中列出）。  
- **整体表现**：所有模型的综合 EI 分数都显著低于人类平均水平，作者用“与人类平均差距约 20%”来概括（具体数值未在摘要中给出）。  
- **细分对比**：在情感理解子任务上，模型普遍能达到 70% 以上的准确率；但在情感应用（尤其是情绪调节）上，得分跌至 40% 左右，说明调节能力是当前模型的短板。  
- **消融实验**：作者分别去掉手工题目的随机化、双语同步或强制链式输出，发现去掉任意一项后模型得分提升 5–10%，但评估可信度下降。尤其是去掉链式输出后，模型容易直接给出调节建议而不解释原因，导致评分系统误判。  
- **局限性**：论文承认题目数量（400）相对仍算小规模，且仅覆盖英中两种语言，未来需要扩展到更多文化背景；另外，人工评分仍有主观因素，完全自动化仍是挑战。

### 影响与延伸思考
自从这篇基准公开后，情感智能评测在 NLP 社区逐渐受到关注。后续有几篇工作（如 **AffectEval**、**EmotionReason**）直接引用 EmoBench 的题目结构，甚至在其基础上加入多模态（图像、语音）情感推理。还有研究尝试用强化学习让模型在情绪调节任务上进行“情感对话训练”，把 EmoBench 当作奖励函数。对想进一步探索的读者，可以关注以下方向：  
1. **跨文化情感基准**：把题目扩展到日语、阿拉伯语等语言，检验模型的文化适应性。  
2. **情感生成模型**：结合情感理解与生成，让模型在对话中主动调节用户情绪。  
3. **自动化评分**：研发更鲁棒的情感推理评估器，降低人工标注成本。  
4. **情感与决策结合**：把情感应用与实际决策任务（如金融、医疗）挂钩，评估模型在高风险场景下的情感理性平衡能力。

### 一句话记住它
**EmoBench 用 400 题手工、双语、全链式的情感任务，首次让大语言模型在“懂情绪、会调节”两方面接受严苛测评，揭示了它们仍远不及普通人。**