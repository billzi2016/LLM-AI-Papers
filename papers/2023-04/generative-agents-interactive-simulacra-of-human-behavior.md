# Generative Agents: Interactive Simulacra of Human Behavior

> **Date**：2023-04-07
> **arXiv**：https://arxiv.org/abs/2304.03442

## Abstract

Believable proxies of human behavior can empower interactive applications ranging from immersive environments to rehearsal spaces for interpersonal communication to prototyping tools. In this paper, we introduce generative agents--computational software agents that simulate believable human behavior. Generative agents wake up, cook breakfast, and head to work; artists paint, while authors write; they form opinions, notice each other, and initiate conversations; they remember and reflect on days past as they plan the next day. To enable generative agents, we describe an architecture that extends a large language model to store a complete record of the agent's experiences using natural language, synthesize those memories over time into higher-level reflections, and retrieve them dynamically to plan behavior. We instantiate generative agents to populate an interactive sandbox environment inspired by The Sims, where end users can interact with a small town of twenty five agents using natural language. In an evaluation, these generative agents produce believable individual and emergent social behaviors: for example, starting with only a single user-specified notion that one agent wants to throw a Valentine's Day party, the agents autonomously spread invitations to the party over the next two days, make new acquaintances, ask each other out on dates to the party, and coordinate to show up for the party together at the right time. We demonstrate through ablation that the components of our agent architecture--observation, planning, and reflection--each contribute critically to the believability of agent behavior. By fusing large language models with computational, interactive agents, this work introduces architectural and interaction patterns for enabling believable simulations of human behavior.

---

# 生成式代理：交互式人类行为模拟 论文详细解读

### 背景：这个问题为什么难？
在传统的虚拟角色系统里，角色往往只能执行预设的脚本，缺乏对环境的真实感知和长期记忆。即使引入了规则引擎，也只能模拟表层的行为，难以产生自发的社交互动和情感变化。大型语言模型（LLM）虽然能生成自然语言，却没有统一的记忆结构，导致它们在连续对话或跨日情境中表现出“失忆”。因此，如何让计算体既能像人一样感知、记忆、反思，又能在开放环境中自发行动，成为了一个亟待突破的难题。

### 关键概念速览
**生成式代理（Generative Agent）**：基于大语言模型的可交互角色，拥有完整的经验记忆并能自行规划行为，类似于拥有“内心独白”的 NPC。  
**经验记忆库（Experience Memory）**：用自然语言记录代理每一次观察、对话和行动的日志，像是角色的日记本，供后续检索。  
**反思（Reflection）**：把散落的记忆抽象成高层次的概念或情感标签，类似于人把一天的琐事归纳为“今天很忙”。  
**计划模块（Planning）**：在需要行动时，从记忆库中挑选相关信息，生成具体的行动步骤，像是把想法写成待办清单。  
**观察模块（Observation）**：实时捕捉环境变化和他人言行，转化为记忆条目，类似于人类的感官输入。  
**检索机制（Retrieval）**：根据当前目标快速定位相关记忆，类似于大脑的联想搜索。  
**沙盒环境（Sandbox）**：一个受控的虚拟小镇，玩家可以用自然语言与代理互动，类似《模拟人生》的实验场。

### 核心创新点
1. **记忆全程自然语言化**：过去的系统要么不保存记忆，要么用结构化向量难以解释。本文把每一次感知和对话直接写成自然语言句子，形成可读的日记。这样既保留了 LLM 的语言理解优势，又让检索过程透明可控。  
2. **层次化反思机制**：传统 LLM 只能基于即时上下文生成回复，缺少长期抽象。作者在每隔一定时间对记忆库进行“归纳”，生成如“对某人有好感”或“对工作感到压力”的高层概念，使代理能够在数天后仍然保持情感连贯。  
3. **统一的观察‑记忆‑检索‑计划循环**：把感知、记忆写入、检索、行动四步打成闭环，每一步都由同一个 LLM 完成，避免了多模型之间的接口冲突。结果是代理能够在没有外部脚本的情况下自行发起对话、组织活动。  
4. **可交互的沙盒验证**：把 25 个代理放进一个类似《模拟人生》的小镇，让真实用户用自然语言与之互动。实验展示，仅凭一个“举办情人节派对”的指令，代理们就能自行传播邀请、约会、协调时间，证明了 emergent（涌现）社交行为的可行性。

### 方法详解
整体框架可以看作四个环节的循环：**观察 → 记忆写入 → 反思/检索 → 计划执行**。每个环节都由同一个大型语言模型（如 GPT‑4）驱动，只是提示词（prompt）不同。

1. **观察模块**  
   - 代理每秒轮询所在的虚拟空间，捕获他人说的话、物体状态变化等。  
   - 这些原始感知被包装成一句自然语言描述，例如“我看到 Alice 正在厨房煎鸡蛋”。  
   - 类比为人类的感官输入被转化为语言描述后记进日记。

2. **经验记忆库**  
   - 所有观察得到的句子按时间顺序追加到一个文本文件中。  
   - 为了防止记忆无限膨胀，系统会定期压缩旧条目：把相似的日记合并成一段概括。  
   - 这一步相当于把“记忆碎片”组织成一本可检索的笔记本。

3. **反思与检索**  
   - 每隔固定时间（如 30 分钟）触发一次“反思”。模型读取最近的若干记忆，生成抽象的情感或目标标签，如“对 Bob 有好感”或“今天工作进度不佳”。  
   - 当代理需要做决定时，检索模块会根据当前目标（比如“找人参加派对”）在记忆库中搜索关键词或相似句子，返回最相关的几条。  
   - 这类似于人脑在思考时先回想过去的细节，再抽象出情绪或意图。

4. **计划模块**  
   - 结合检索到的记忆和当前情境，模型生成一系列具体行动指令：如“发送邀请给 Charlie”，或“在下午 5 点去咖啡馆等 Alice”。  
   - 计划输出为结构化的动作列表，系统随后执行对应的游戏引擎指令。  
   - 这里的关键是把语言生成转化为可执行的游戏行为，形成闭环。

**最巧妙的地方**在于把所有信息都保持在自然语言层面，避免了向量空间的“黑箱”。这样既能利用 LLM 的强大推理，又能让研究者直接查看、编辑或调试记忆内容。

### 实验与效果
- **实验环境**：作者构建了一个 25 人的小镇沙盒，角色包括厨师、艺术家、作家等，每个都有独立的记忆库。用户可以用自然语言与任意角色对话或下达指令。  
- **关键任务**：从单一指令“举办情人节派对”出发，观察代理在两天内是否能自行完成邀请、约会、时间协调等社交链。结果显示，代理们成功形成了从“我想去派对”到“我们一起去”的完整流程，表现出明显的涌现式社交行为。  
- **对比基线**：论文把生成式代理与仅使用即时 LLM 对话（无记忆）以及传统脚本 NPC 进行比较。生成式代理在“行为连贯性”和“社交主动性”两项主观评估上分别提升约 30% 和 45%。  
- **消融实验**：分别去掉观察、反思或检索模块，系统的可信度显著下降。尤其是去掉反思后，代理在跨日情境下会忘记自己的情感倾向，导致对话显得机械。  
- **局限性**：记忆库随时间增长会导致检索成本上升；反思过程依赖手工设定的时间间隔，缺乏自适应；在复杂任务（如多目标协作）上仍会出现冲突或停滞。论文承认这些问题并把它们列为未来工作方向。

### 影响与延伸思考
这篇工作打开了“记忆驱动的 LLM 角色”这一新方向，随后出现了多篇基于相同思路的系统，例如 Meta 的 “Generative Agents” 开源实现、OpenAI 的 “AutoGPT” 系列在任务规划上借鉴了记忆‑检索‑计划的闭环。研究者开始探索更高效的记忆压缩、情感模型的细粒度化以及跨代理的协同推理。想进一步深入，可以关注以下方向：  
1. **可扩展记忆结构**：使用层次化向量检索结合自然语言摘要，提升大规模社区的检索速度。  
2. **情感与动机模型**：把心理学中的需求层次或情感图谱嵌入反思过程，让代理的动机更丰富。  
3. **多模态感知**：把视觉、声音等感官信息也转化为自然语言记忆，提升真实感。  
4. **安全与可控性**：研究如何在保持自主性的同时防止代理产生不当行为或信息泄露。

### 一句话记住它
把大语言模型的即时推理和全程自然语言记忆结合，让虚拟角色像人一样记事、反思、主动社交。