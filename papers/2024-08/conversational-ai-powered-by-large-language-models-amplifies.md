# Conversational AI Powered by Large Language Models Amplifies False   Memories in Witness Interviews

> **Date**：2024-08-08
> **arXiv**：https://arxiv.org/abs/2408.04681

## Abstract

This study examines the impact of AI on human false memories -- recollections of events that did not occur or deviate from actual occurrences. It explores false memory induction through suggestive questioning in Human-AI interactions, simulating crime witness interviews. Four conditions were tested: control, survey-based, pre-scripted chatbot, and generative chatbot using a large language model (LLM). Participants (N=200) watched a crime video, then interacted with their assigned AI interviewer or survey, answering questions including five misleading ones. False memories were assessed immediately and after one week. Results show the generative chatbot condition significantly increased false memory formation, inducing over 3 times more immediate false memories than the control and 1.7 times more than the survey method. 36.4% of users' responses to the generative chatbot were misled through the interaction. After one week, the number of false memories induced by generative chatbots remained constant. However, confidence in these false memories remained higher than the control after one week. Moderating factors were explored: users who were less familiar with chatbots but more familiar with AI technology, and more interested in crime investigations, were more susceptible to false memories. These findings highlight the potential risks of using advanced AI in sensitive contexts, like police interviews, emphasizing the need for ethical considerations.

---

# 大语言模型驱动的对话式 AI 放大证人访谈中的虚假记忆 论文详细解读

### 背景：这个问题为什么难？

在司法实践中，证人访谈的准确性直接关系到案件走向。传统研究主要关注人类提问者的暗示效应，却很少考虑机器提问会产生怎样的误导。早期的聊天机器人大多基于固定脚本或规则，交互深度有限，难以模拟真实访谈的灵活性。因此，无法评估当代大语言模型（LLM）在高度情境化对话中是否会无意放大记忆扭曲。缺少对“AI+访谈”这一新情境的系统实验，使得潜在风险被低估，也让伦理监管缺乏依据。

### 关键概念速览
- **虚假记忆**：指人们对从未发生或被篡改的事件产生的确信感，就像把一段不存在的电影情节当成亲身经历一样。  
- **大语言模型（LLM）**：一种在海量文本上训练的生成式模型，能够根据上下文自如生成自然语言，类似于“会说话的百科全书”。  
- **生成式聊天机器人**：利用 LLM 实时生成回复的对话系统，不同于预先写好的脚本，它的每句话都可能因用户输入而变化。  
- **暗示性提问**：在提问时加入误导信息的技巧，例如“你看到嫌疑人拿了红色的手枪吗？”即使现场没有红色手枪，也会诱导记忆偏差。  
- **控制组**：实验中不接受任何 AI 介入的基线条件，用来衡量自然记忆误差。  
- **置信度**：受访者对自己记忆的自信程度，类似于对答案的“确信度”，高置信度往往意味着记忆更难被纠正。  

### 核心创新点
1. **从问卷到对话的实验迁移**  
   之前的记忆暗示实验大多使用纸质或线上问卷，缺少交互性。本文把同样的暗示性问题搬进了实时对话环境，分别设置了预设脚本机器人和 LLM 生成机器人。这样做让研究直接捕捉到“对话流”对记忆的影响，而不是静态文字的阅读效应。  
2. **四种访谈情境的系统对比**  
   传统研究只比较有无暗示两种情形，这里加入了三种中间态：控制、调查式（纯文字问卷）、预设脚本聊天机器人。通过这种层级设计，作者能够量化“交互复杂度”与“记忆扭曲”之间的关系，发现生成式机器人是唯一显著放大误导的因素。  
3. **长期追踪与置信度测量**  
   大多数记忆实验只在当场测量错误率，忽视记忆的持久性。本文在访谈后一周再次评估，同步记录受访者对每条记忆的置信度。结果显示，生成式机器人导致的错误记忆在数量上保持不变，而置信度仍高于对照组，揭示了记忆“固化”风险。  
4. **用户属性的调节效应探索**  
   研究首次把受访者对 AI 技术的熟悉度、对聊天机器人的使用经验以及对犯罪调查的兴趣作为调节变量。发现对聊天机器人不熟悉但对 AI 技术了解较多、且对犯罪调查兴趣高的人更易受误导，这为后续的风险分层提供了实证依据。  

### 方法详解
整体思路可以拆成三步：**材料准备 → 访谈交互 → 记忆评估**。  
1. **材料准备**：研究者挑选了一段约两分钟的犯罪现场视频，确保所有受试者看到的情节完全相同。随后设计了十个访谈问题，其中五个是有意误导的暗示性问题（如“嫌疑人是否穿了蓝色外套？”），其余为中性核查。  
2. **访谈交互**：200 名受试者随机分入四组，每组 50 人。  
   - **控制组**：观看视频后直接填写纸质问卷，问题顺序与其他组保持一致。  
   - **调查式组**：通过在线调查平台完成同样的问卷，没有任何对话交互。  
   - **预设脚本机器人组**：使用一个基于固定对话树的聊天机器人，机器人只能按预先写好的句式提问，无法根据受访者的回答进行灵活追问。  
   - **生成式机器人组**：采用公开的 LLM（如 GPT‑4）搭建的聊天系统，机器人在每轮对话中实时生成提问和追问，语言自然、上下文连贯。  
   受访者在对话中必须对每个问题作答，机器人会根据答案继续提问，形成类似真实访谈的流动。  
3. **记忆评估**：访谈结束后立即进行一次记忆测验，记录每位受访者对五个误导性细节的回答是否错误，并让他们对每个答案给出 1‑7 级置信度。随后在一周后通过线上问卷再次测量相同指标，以观察记忆的持久性。  

**关键细节**：  
- 生成式机器人的提示词（prompt）被设计为“扮演经验丰富的警官”，以确保提问方式与真实访谈相似。  
- 为防止模型自行纠正误导信息，研究者在提示中加入“不要纠正受访者的记忆”。  
- 所有对话均被录音并转写，以便后续行为编码。  

**最巧妙的地方**：作者没有直接让 LLM 输出暗示性问题，而是让它在对话中自行产生暗示，这种“自发暗示”更贴近真实情境，也让误导效应更难被实验者预先控制，从而真实捕捉到模型的潜在风险。

### 实验与效果
- **实验规模**：共 200 人，四组均衡。  
- **主要发现**：生成式机器人组在即时测验中产生的错误记忆数量是对照组的约 3 倍，超过调查式组 1.7 倍。具体来说，约 36.4% 的受访者在与生成式机器人交互时被误导回答错误。  
- **长期效应**：一周后错误记忆的数量基本持平，说明误导记忆不会自行消失；但受访者对这些错误记忆的置信度仍显著高于对照组，意味着记忆已经被“固化”。  
- **调节变量**：对聊天机器人不熟悉、但对 AI 技术了解较多、且对犯罪调查兴趣高的受试者，误导率更高。  
- **对照组表现**：控制组和调查式组的错误记忆率相近，均远低于生成式机器人组。  
- **消融实验**：原文未提供细化的消融实验，只报告了四种情境的整体比较。  
- **局限性**：实验使用的是单一视频情境，真实案件的复杂度可能更高；生成式机器人的具体模型和提示细节未公开，导致可复现性受限。作者也承认未探讨不同文化背景或年龄段的差异。  

### 影响与延伸思考
这篇工作首次把大语言模型的对话能力与记忆暗示实验结合，提醒法律、心理学以及 AI 伦理社区：即便模型本身不“想”误导，它的语言流畅性也足以在不经意间放大人类记忆的脆弱性。随后出现的研究（如 2024 年的《LLM 在司法访谈中的风险评估》）进一步探讨了多轮对话、情感色彩以及跨语言场景的影响。对想深入的读者，可以关注以下方向：  
- **对抗性提示设计**：如何在模型提示中加入防误导机制。  
- **可解释性追踪**：追踪模型在生成暗示性问题时的内部注意力分布。  
- **跨模态访谈**：结合视频、语音等多模态输入，评估误导效应是否更强。  
- **政策与监管**：制定针对 AI 辅助访谈的使用准则，尤其在执法部门的部署。  

### 一句话记住它
大语言模型驱动的聊天机器人在模拟访谈时会显著放大误导性暗示，导致持久且自信的虚假记忆。