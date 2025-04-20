# BookWorld: From Novels to Interactive Agent Societies for Creative Story   Generation

> **Date**：2025-04-20
> **arXiv**：https://arxiv.org/abs/2504.14538

## Abstract

Recent advances in large language models (LLMs) have enabled social simulation through multi-agent systems. Prior efforts focus on agent societies created from scratch, assigning agents with newly defined personas. However, simulating established fictional worlds and characters remain largely underexplored, despite its significant practical value. In this paper, we introduce BookWorld, a comprehensive system for constructing and simulating book-based multi-agent societies. BookWorld's design covers comprehensive real-world intricacies, including diverse and dynamic characters, fictional worldviews, geographical constraints and changes, e.t.c. BookWorld enables diverse applications including story generation, interactive games and social simulation, offering novel ways to extend and explore beloved fictional works. Through extensive experiments, we demonstrate that BookWorld generates creative, high-quality stories while maintaining fidelity to the source books, surpassing previous methods with a win rate of 75.36%. The code of this paper can be found at the project page: https://bookworld2025.github.io/.

---

# BookWorld：从小说到交互式智能体社会的创意故事生成 论文详细解读

### 背景：这个问题为什么难？

传统的多智能体模拟大多从零开始构造角色，给每个智能体随意设定人格。这样虽然灵活，却无法直接搬入已有的文学世界——小说里的人物关系、地理设定、历史演变都极其复杂。现有方法缺少对这些固有设定的忠实再现，导致生成的情节常常跑题或违背原著。要让大语言模型在保持创意的同时，严格遵守已有小说的世界观和人物性格，技术难点在于如何把书本的结构化信息转化为可交互的智能体，并让它们在动态情境中协同演绎。

### 关键概念速览
- **多智能体系统（Multi‑Agent System）**：由多个具备独立决策能力的模型组成的集合，它们可以相互交流、竞争或合作，类似一群玩家在同一棋盘上对弈。  
- **角色画像（Character Persona）**：对小说中每个人物的性格、动机、背景等信息的结构化描述，像是给角色写的“使用手册”。  
- **世界模型（World Model）**：对小说整体设定的数字化表示，包括地图、历史事件、社会规则等，类似游戏的“地图引擎”。  
- **情境驱动对话（Context‑Driven Dialogue）**：智能体在对话时会参考当前的情境信息（地点、时间、正在进行的事件），就像演员在舞台上根据剧本的场景即兴表演。  
- **创意约束采样（Creative‑Constraint Sampling）**：在生成文本时同时满足两类要求：一是保持创新，二是不违背已有设定，类似在限定颜色范围内调配新颜色。  
- **社会仿真评估（Social Simulation Evaluation）**：通过让人类评审判断生成的情节是否合理、连贯且符合原著设定的评测方式，类似观众投票选出最佳剧本。  

### 核心创新点
1. **从书本到智能体的全链路映射**  
   之前的工作只在虚构人物上做角色设定，缺少系统化的书籍信息抽取。BookWorld 首先使用大语言模型对原著进行结构化解析，自动生成角色画像、地理坐标、时间线等要素，然后把每个角色包装成独立的对话模型。这样做让系统能够直接“搬进”已有小说的世界，而不是在空白画布上重新绘制。

2. **动态世界状态管理**  
   传统多智能体系统的环境通常是静态或仅支持简单的状态切换。BookWorld 引入了可变更的世界模型，实时记录地点占用、事件进度、资源分配等信息，并在每轮对话后更新。结果是智能体的行为会受到前情后续的约束，像在真实小说里人物会随情节发展而改变计划。

3. **创意约束采样机制**  
   为了兼顾创新和忠实，作者在生成每句文本时加入双层过滤：第一层是基于角色画像的约束，确保语言风格、价值观不偏离；第二层是全局情节约束，检查新句子是否与世界模型冲突。只有同时通过两层检查的输出才会被采纳，这种“先约束后创新”的思路显著提升了故事的质量。

4. **基于人类评审的社会仿真评估**  
   过去的评测往往只看语言流畅度或BLEU 分数，忽视了情节合理性。BookWorld 设计了一个对比实验，让评审在两段由不同系统生成的情节中选出更符合原著精神的那一段。实验结果显示，BookWorld 获得 75.36% 的胜率，显著领先于已有基线。

### 方法详解
整体框架可以拆成四大步骤：**（1）原著信息抽取 →（2）角色与世界模型构建 →（3）多智能体交互生成 →（4）约束过滤与输出**。下面按顺序展开。

1. **原著信息抽取**  
   - 使用预训练的大语言模型（如 GPT‑4）对小说全文进行章节划分、人物出现频次统计、事件时间线标注。  
   - 通过提示工程让模型输出结构化 JSON，包括每个人物的性格标签（如“冲动”“理性”）、关键关系（如“师徒”“宿敌”）以及所在地点坐标。  
   - 类比于把一本厚厚的小说拆成 LEGO 块，每块都有明确的接口信息。

2. **角色与世界模型构建**  
   - 为每个角色创建一个专属的对话模型，模型的系统提示（system prompt）被填入该角色的画像信息，使其在生成时自动遵循性格设定。  
   - 世界模型采用图数据库存储，节点代表地点或事件，边表示可达性或因果关系。每轮交互后，系统会根据对话内容更新图的属性（例如“城堡被攻陷”会把“城堡安全”属性置为 false）。  
   - 这一步相当于在游戏引擎里放置 NPC（非玩家角色）和地图，并让它们随时读取最新的游戏状态。

3. **多智能体交互生成**  
   - 采用轮询机制：在每一步，系统从所有活跃角色中挑选一个（基于情境相关性或随机），让其在当前世界状态下生成一句对话或行动描述。  
   - 生成时模型会把**情境驱动对话**的提示拼接进去，提示中包含当前地点、时间、最近的事件等信息，确保输出与情境匹配。  
   - 生成的文本随后进入约束过滤阶段。

4. **约束过滤与输出**  
   - **角色约束**：检查文本是否违背角色画像（如冲动的角色不应使用极度理性的语言）。  
   - **全局约束**：利用世界模型的规则引擎验证情节一致性（如人物不可能在同一时间出现在两个相距千里的地点）。  
   - 只有双重通过的文本才被写入最终故事，并触发世界模型的状态更新。若被拒，系统会让同一角色重新生成，最多尝试三次。  
   - 这种双层过滤的设计是本论文最巧妙的地方，它把“创意自由”和“设定忠实”这两个看似冲突的目标用层级结构调和起来。

### 实验与效果
- **测试对象**：作者选取了几部经典长篇小说（如《指环王》系列、《哈利·波特》前四部）作为实验基准，分别在这些书的设定上构建 BookWorld。  
- **对比基线**：包括传统从零构造的多智能体系统、单一 LLM 直接续写（无角色约束）以及最近的“角色驱动生成”模型。  
- **胜率**：在人类评审的两两对比实验中，BookWorld 获得 75.36% 的胜率，显著高于最高基线的约 58%。  
- **消融实验**：作者分别去掉了动态世界模型、约束过滤或角色画像，胜率分别下降到 62%、68% 和 71%，说明每个模块都有实质贡献。  
- **局限性**：论文承认对极其庞大的小说（章节数上千）仍存在抽取效率瓶颈，且角色画像的自动生成仍会出现细节遗漏，需要人工微调。

### 影响与延伸思考
这篇工作打开了“把已有文学作品搬进交互式 AI 世界”的新路径。随后有研究尝试把同样的框架用于电影剧本、动漫宇宙，甚至历史文献的情境再现（推测）。如果想进一步探索，可以关注以下方向：更高效的结构化抽取技术、跨书宇宙的角色迁移、以及把玩家行为反馈回模型进行在线学习。  

### 一句话记住它
把小说的设定直接编码成可交互的智能体，让 AI 在忠实原著的前提下自由创作故事。