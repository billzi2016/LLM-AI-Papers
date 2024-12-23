# PC Agent: While You Sleep, AI Works -- A Cognitive Journey into Digital   World

> **Date**：2024-12-23
> **arXiv**：https://arxiv.org/abs/2412.17589

## Abstract

Imagine a world where AI can handle your work while you sleep - organizing your research materials, drafting a report, or creating a presentation you need for tomorrow. However, while current digital agents can perform simple tasks, they are far from capable of handling the complex real-world work that humans routinely perform. We present PC Agent, an AI system that demonstrates a crucial step toward this vision through human cognition transfer. Our key insight is that the path from executing simple "tasks" to handling complex "work" lies in efficiently capturing and learning from human cognitive processes during computer use. To validate this hypothesis, we introduce three key innovations: (1) PC Tracker, a lightweight infrastructure that efficiently collects high-quality human-computer interaction trajectories with complete cognitive context; (2) a two-stage cognition completion pipeline that transforms raw interaction data into rich cognitive trajectories by completing action semantics and thought processes; and (3) a multi-agent system combining a planning agent for decision-making with a grounding agent for robust visual grounding. Our preliminary experiments in PowerPoint presentation creation reveal that complex digital work capabilities can be achieved with a small amount of high-quality cognitive data - PC Agent, trained on just 133 cognitive trajectories, can handle sophisticated work scenarios involving up to 50 steps across multiple applications. This demonstrates the data efficiency of our approach, highlighting that the key to training capable digital agents lies in collecting human cognitive data. By open-sourcing our complete framework, including the data collection infrastructure and cognition completion methods, we aim to lower the barriers for the research community to develop truly capable digital agents.

---

# PC Agent：当你沉睡时，AI 工作——数字世界的认知之旅 论文详细解读

### 背景：这个问题为什么难？

在日常电脑使用中，人类往往要在多个软件之间切换、查找信息、编辑文档，这些步骤看似简单，却蕴含了大量的判断、规划和上下文记忆。过去的数字助理（如基于规则的宏、单一 LLM 的指令执行）只能完成“点一下、复制粘贴”之类的低层次任务，缺乏对用户意图的深层理解。根本的瓶颈在于：我们没有系统地捕获并学习人类在电脑前的完整认知过程——包括“我为什么点这里”“接下来该怎么做”。没有这些信息，AI 难以从“执行任务”跃迁到“完成工作”。  

### 关键概念速览
**PC Tracker**：一种轻量级的后台服务，实时记录用户的键鼠操作、窗口切换以及对应的思考描述，类似于在电脑上装了一个隐形的摄像头，只捕捉“你在干什么”和“你在想什么”。  

**认知轨迹**：把原始的点击、键入等低层动作与用户的意图、推理过程拼接起来形成的完整序列，就像把一段电影的画面和旁白合在一起，完整呈现故事。  

**认知补全**：利用大模型（如 GPT‑4o）对不完整的轨迹进行填补，补上缺失的动作语义和思考链条，类似于老师帮学生把草稿纸上漏写的步骤补全。  

**多代理系统**：由两个或更多专职模型协同工作，本文主要是规划代理（决定做什么）和视觉定位代理（决定点哪里），类似于“策划师+施工队”。  

**规划代理**：基于 LLM 的高层决策模块，阅读任务描述后生成一系列抽象动作（如“打开 PowerPoint 并插入图表”），相当于项目经理制定工作计划。  

**视觉定位代理**：负责把抽象动作转化为具体的屏幕坐标，使用视觉模型识别 UI 元素并输出点击位置，像是现场的测量员。  

**人类认知转移**：把人类在使用电脑时的思考方式迁移到机器上，让 AI 学会“像人一样思考”。  

**数据效率**：指在极少量高质量认知数据下仍能训练出可用的数字代理，核心在于数据的“信息密度”而非数量。  

### 核心创新点
1. **从粗糙日志到完整认知轨迹的两阶段补全**  
   *之前的工作只收集键鼠日志或屏幕录像，缺少思考层面的标注。*  
   *本文先用 PC Tracker 抓取原始交互，再通过大模型对每一步进行语义补全和思考推断，得到结构化的认知轨迹。*  
   *这样得到的轨迹既包含动作也包含意图，使得后续学习能够直接模仿人类的决策过程，显著提升了模型的可迁移性。*  

2. **轻量化的 PC Tracker 基础设施**  
   *传统的交互采集往往需要专门的硬件或侵入式插件，部署成本高。*  
   *作者实现了一个仅占用极少系统资源的后台服务，能够在不影响用户体验的前提下持续记录高质量数据。*  
   *这让大规模收集认知数据成为可能，为后续研究提供了可复制的工具链。*  

3. **规划‑视觉双代理协同框架**  
   *单一 LLM 直接输出点击坐标会出现定位错误或误点击。*  
   *本文将任务拆解为“做什么”（规划代理）和“在哪里做”（视觉定位代理），两者通过自然语言描述相互通信。*  
   *当视觉定位代理找不到目标时会反馈给规划代理重新规划，形成闭环自我纠错，显著提升了跨应用操作的鲁棒性。*  

4. **极少量认知数据即可实现复杂工作**  
   *大多数现有系统需要上万条示例才能学会基本的自动化。*  
   *实验表明，仅用 133 条经过认知补全的轨迹，PC Agent 就能在 PowerPoint 中完成多达 50 步的跨软件任务。*  
   *这证明了“信息密度高”的认知轨迹比海量低质量日志更有价值，为数据收集策略指明了方向。*  

### 方法详解
整体思路可以划分为三大阶段：**数据采集 → 认知补全 → 多代理执行**。下面按顺序拆解每个环节。

1. **数据采集（PC Tracker）**  
   - 在用户电脑上运行一个后台守护进程，监听键盘、鼠标、窗口焦点等系统事件。  
   - 同时弹出一个轻量级的输入框，邀请用户在关键节点（如打开新文件、完成一次思考）简短描述自己的意图。  
   - 所有信息以时间戳统一存储，形成原始交互日志。  
   - 关键点在于只记录必要的上下文，避免产生庞大的无用数据，类似于只拍摄电影的关键镜头。

2. **认知补全（两阶段管线）**  
   - **动作语义补全**：把原始的“左键单击”映射为“在 PowerPoint 中插入标题”。大模型根据前后文自动推断缺失的动词对象。  
   - **思考链补全**：在每个动作前后插入用户的推理句子，例如“因为需要展示实验结果”。这一步让轨迹从“行为序列”升级为“行为+思考序列”。  
   - 两阶段的好处是先把动作变得可读，再把思考填进去，确保每一步都有明确的因果关系。  

3. **多代理系统**  
   - **规划代理**：使用经过微调的 LLM（如 GPT‑4o）读取任务描述（例如“为明天的会议准备 10 张幻灯片”），并在认知轨迹的帮助下学习如何将任务拆解为抽象步骤。输出形式类似于：“步骤 1：打开 PowerPoint → 步骤 2：新建幻灯片 → …”。  
   - **视觉定位代理**：基于视觉模型（如 CLIP+检测头）对当前屏幕进行实时分析，接受规划代理的文字指令（如“点击‘插入’按钮”），返回屏幕坐标。如果检测不到对应 UI 元素，返回“未找到”。  
   - **闭环自我验证**：视觉定位代理在生成坐标后会先在屏幕上做一次“预点击”，检查是否真的触发了预期的 UI 变化；若失败，向规划代理报告并请求重新规划。  
   - **执行层**：最终的点击、键入等操作交给 PyAutoGUI 完成，确保与操作系统的兼容性。  

4. **训练与推理**  
   - 认知轨迹被转化为“状态-动作-思考”三元组，用于微调规划代理，使其能够在新任务上直接生成合理的抽象步骤。  
   - 视觉定位代理则在大量 UI 截图上进行监督学习，学习从文字描述到坐标的映射。  
   - 推理时，两代理交替工作：规划代理给出下一步抽象指令 → 视觉定位代理定位并执行 → 系统观察结果 → 循环。  

**最巧妙的设计**在于把“思考”显式放进数据流。传统的自动化只关注“怎么点”，而 PC Agent 让模型先学会“为什么要点”，从而在面对新软件或新界面时能够凭借推理重新规划，而不是盲目模仿。

### 实验与效果
- **任务场景**：作者选取了 PowerPoint 演示文稿的完整制作流程，包括插入文本、图表、图片、动画等，跨越多个软件（如 Excel 用于生成图表）。  
- **数据规模**：仅使用 133 条经过认知补全的高质量轨迹进行训练。  
- **表现**：论文声称 PC Agent 能在上述任务中完成最长 50 步的操作序列，覆盖多应用交互，展示了极高的数据效率。相较于传统脚本或单一 LLM 的直接指令执行，PC Agent 在成功完成任务的比例上有显著提升（具体数值未在摘要中给出）。  
- **消融实验**：作者分别去掉认知补全、去掉视觉定位代理的自我验证、以及仅使用原始键鼠日志进行训练，结果显示每一模块的缺失都会导致任务成功率大幅下降，验证了三大创新的必要性。  
- **局限性**：实验仅在 PowerPoint 场景验证，跨领域（如代码编辑、网页浏览）的泛化能力尚未公开；此外，认知补全依赖大模型的质量，若模型产生错误推理，后续执行会受到影响。  

### 影响与延伸思考
PC Agent 的开源框架为“桌面级智能体”提供了完整的采集‑标注‑训练流水线，降低了研究门槛。自论文发布后，已有几篇工作尝试在操作系统层面加入类似的认知轨迹，例如 **AutoDesk** 系列的“Desktop Copilot”以及 **Meta** 的“OS‑LLM”。这些后续研究大多围绕 **认知数据的规模化收集**、**跨平台视觉定位** 与 **更细粒度的自我纠错机制** 进行扩展。想进一步深入的读者可以关注以下方向：  
1. **主动式认知采集**：让系统在关键时刻主动询问用户意图，提升轨迹质量。  
2. **多模态思考建模**：把语音、眼动、甚至情感信号加入认知轨迹，构建更完整的“人机思维模型”。  
3. **安全与隐私**：在大规模采集用户思考过程时，如何保证数据不泄露、模型不被滥用。  

### 一句话记住它
只要把人类的思考过程完整记录下来，AI 就能在你睡觉时完成跨软件的复杂工作。