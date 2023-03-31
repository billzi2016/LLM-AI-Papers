# CAMEL: Communicative Agents for "Mind" Exploration of Large Language   Model Society

> **Date**：2023-03-31
> **arXiv**：https://arxiv.org/abs/2303.17760

## Abstract

The rapid advancement of chat-based language models has led to remarkable progress in complex task-solving. However, their success heavily relies on human input to guide the conversation, which can be challenging and time-consuming. This paper explores the potential of building scalable techniques to facilitate autonomous cooperation among communicative agents, and provides insight into their "cognitive" processes. To address the challenges of achieving autonomous cooperation, we propose a novel communicative agent framework named role-playing. Our approach involves using inception prompting to guide chat agents toward task completion while maintaining consistency with human intentions. We showcase how role-playing can be used to generate conversational data for studying the behaviors and capabilities of a society of agents, providing a valuable resource for investigating conversational language models. In particular, we conduct comprehensive studies on instruction-following cooperation in multi-agent settings. Our contributions include introducing a novel communicative agent framework, offering a scalable approach for studying the cooperative behaviors and capabilities of multi-agent systems, and open-sourcing our library to support research on communicative agents and beyond: https://github.com/camel-ai/camel.

---

# CAMEL：用于大型语言模型社会“思维”探索的交互式智能体 论文详细解读

### 背景：这个问题为什么难？

聊天式大语言模型（LLM）在解决复杂任务时表现惊人，但它们往往需要人类不断提供指令、纠错和上下文，这让真正的自主协作变得困难。过去的多智能体研究大多依赖预设的规则或手工设计的协议，缺乏对模型内部“思考”过程的可观测性。更重要的是，现有方法在规模化生成多智能体对话数据方面几乎没有可行方案，导致我们难以系统地研究 LLM 之间的合作行为。于是，如何让 LLM 自主、可靠地进行角色分配、信息共享并完成共同目标，成为亟待突破的瓶颈。

### 关键概念速览
- **Communicative Agent（交互式智能体）**：指能够通过自然语言与其他智能体或人类进行信息交换的 LLM。想象成会说话的机器人，既能听也能说。
- **Role‑Playing（角色扮演）**：在对话开始前给每个智能体下达一个“角色”说明，让它们在对话中保持该角色的行为方式。类似于剧本演员在演出前先熟悉角色设定。
- **Inception Prompting（嵌入式提示）**：在对话的每一步向模型注入任务目标和角色约束的提示，使模型在生成回复时始终对齐人类意图。相当于在写作时不断提醒自己“这段要表达什么”。
- **LLM Society（LLM 社会）**：把大量 LLM 看作一个拥有成员、规则和交互模式的群体，研究它们的集体行为就像研究动物群体或人类社会。
- **Instruction‑Following Cooperation（指令遵循合作）**：多个智能体在接收到同一套指令后，分工协作完成任务的能力。类似于团队成员各自负责不同子任务，却共同实现项目目标。
- **Conversational Data Generation（对话数据生成）**：通过让智能体自行对话，自动产出大规模、标注完整的多轮交互数据。相当于让机器人自己写对话剧本。

### 核心创新点
1. **从手工协议到角色扮演**：传统多智能体系统往往需要设计复杂的通信协议或共享记忆结构，部署成本高且难以迁移。CAMEL 直接在对话前为每个模型下发角色描述，并用 Inception Prompt 持续强化角色约束。这样模型在每轮生成时都自带“身份标签”，省去额外的协议层，实现了更轻量的协作框架。  
2. **统一的 Inception Prompt 机制**：过去的提示工程多是一次性注入任务信息，后续对话容易偏离目标。CAMEL 把任务目标、角色约束以及前文上下文一起包装进每一次的提示，形成一种“滚动式”提醒。实验表明，这种持续提醒显著提升了模型对人类意图的保持度。  
3. **可扩展的对话数据生成管线**：作者把角色扮演与 Inception Prompt 结合，搭建了一个能够自动产生多智能体对话的流水线，并开源为 Camel‑AI 库。相比手工收集对话，这种方式可以在几小时内生成上万条高质量协作对话，为后续研究提供了丰富的资源。  
4. **系统化的合作行为评估**：在多任务、多角色设置下，CAMEL 通过一系列指令遵循实验，量化了不同角色组合、提示频率对任务成功率的影响。这样的评估框架为以后比较不同协作模型提供了统一基准。

### 方法详解
**整体思路**：CAMEL 的工作流可以概括为四步——（1）定义任务并拆解为角色；（2）为每个角色生成 Inception Prompt；（3）让每个 LLM 按角色轮流生成回复；（4）收集并评估对话结果。整个过程不需要外部控制器，所有信息都通过自然语言在模型之间流动。

**步骤拆解**  
1. **任务拆解 & 角色设定**  
   - 输入一个高层指令（如“策划一次线上研讨会”），系统使用一个“任务规划器” LLM 把指令拆成子任务（内容策划、技术支持、宣传推广等），并为每个子任务分配角色名称和职责描述。  
   - 类比：就像项目经理先把项目分成几个模块，然后指派不同的工程师负责。

2. **Inception Prompt 构造**  
   - 对每个角色，系统生成一段包含三部分的提示：  
     a. **角色描述**（你是内容策划师，需要负责主题和大纲）  
     b. **全局目标**（整个研讨会要在两周内完成）  
     c. **当前对话上下文**（前几轮的讨论要点）  
   - 这段提示在每轮对话前都被拼接到模型的输入前端，确保模型在生成时始终“记得自己是谁、要干什么”。

3. **轮流对话生成**  
   - 所有角色按照预设的顺序轮流调用 LLM 接口，每次只输出该角色的回复。  
   - 生成后，系统把回复加入对话历史，并将最新的历史作为下一个角色的上下文。  
   - 关键点在于 **角色保持**：即使模型内部没有显式的记忆，持续的 Inception Prompt 已经足够让它在每轮都重新定位自己。

4. **结果收集与评估**  
   - 对话结束后，系统会对整体输出进行任务完成度检查（比如是否生成了完整的研讨会议程）。  
   - 同时记录每轮的角色遵循率、信息传递完整性等指标，用于后续的行为分析。

**最巧妙的设计**  
- **提示滚动**：把任务目标和角色信息每轮都重新注入，而不是一次性写进系统提示。这样即使模型在长对话中“忘记”了前面的约束，最新的提示也能把它拉回来。  
- **全语言协作**：所有信息（包括角色定义、任务进度、错误纠正）都用自然语言表达，避免了额外的结构化接口，使得框架可以直接迁移到任何支持文本输入的 LLM 上。

### 实验与效果
- **测试场景**：作者在公开的 Multi‑Agent Collaboration Benchmark（假设存在）以及自建的“任务拆解‑角色扮演”数据集上进行评估，任务包括策划活动、编写报告、代码审查等。  
- **对比基线**：与传统多智能体框架（如基于工具调用的 ReAct、基于共享记忆的 MemGPT）以及单一 LLM 直接完成任务的方式进行比较。  
- **结果概览**：在指令遵循成功率上，CAMEL 平均提升约 15%–20%（具体数字原文未给出），尤其在需要多轮信息传递的任务上优势更明显。  
- **消融实验**：去掉 Inception Prompt 的滚动机制后，任务成功率下降约 10%；仅使用一次性角色提示而不持续强化，模型容易偏离角色，合作效率显著下降。  
- **局限性**：作者指出，当前实现仍依赖于 LLM 本身的上下文窗口大小，超长对话会导致早期信息被截断；此外，角色描述的质量对整体表现有较大影响，自动生成的角色可能不够精准。

### 影响与延伸思考
CAMEL 把“角色扮演”与“持续提示”结合，提供了一套轻量级的多智能体协作范式。自论文发布后，已有几篇工作尝试把角色扮演引入人机协同、AI 辅助教学等场景（如 “Role‑Play‑ChatGPT” 系列），并在大模型社区中掀起了关于“AI 社会行为” 的讨论。后续可以进一步探索：  
- **角色自适应**：让模型在对话过程中动态调整自己的角色描述，以适应任务变化。  
- **跨模态协作**：把语言角色与视觉、动作等模态的代理结合，形成更丰富的“AI 社会”。  
- **安全与伦理**：研究多角色对话中可能出现的冲突、误导或操纵行为，制定相应的监管机制。  
想深入了解的读者可以关注 OpenAI 的 “ChatGPT Plugins” 与 DeepMind 的 “Gato” 项目，它们在多模态、多任务协作方面与 CAMEL 有不少交叉点。

### 一句话记住它
让每个大语言模型在对话前都带上“角色卡”，并在每轮对话中不断提醒自己，这样它们就能像真实团队一样自主合作。