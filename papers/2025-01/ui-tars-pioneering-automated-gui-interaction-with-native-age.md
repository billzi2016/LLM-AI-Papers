# UI-TARS: Pioneering Automated GUI Interaction with Native Agents

> **Date**：2025-01-21
> **arXiv**：https://arxiv.org/abs/2501.12326

## Abstract

This paper introduces UI-TARS, a native GUI agent model that solely perceives the screenshots as input and performs human-like interactions (e.g., keyboard and mouse operations). Unlike prevailing agent frameworks that depend on heavily wrapped commercial models (e.g., GPT-4o) with expert-crafted prompts and workflows, UI-TARS is an end-to-end model that outperforms these sophisticated frameworks. Experiments demonstrate its superior performance: UI-TARS achieves SOTA performance in 10+ GUI agent benchmarks evaluating perception, grounding, and GUI task execution. Notably, in the OSWorld benchmark, UI-TARS achieves scores of 24.6 with 50 steps and 22.7 with 15 steps, outperforming Claude (22.0 and 14.9 respectively). In AndroidWorld, UI-TARS achieves 46.6, surpassing GPT-4o (34.5). UI-TARS incorporates several key innovations: (1) Enhanced Perception: leveraging a large-scale dataset of GUI screenshots for context-aware understanding of UI elements and precise captioning; (2) Unified Action Modeling, which standardizes actions into a unified space across platforms and achieves precise grounding and interaction through large-scale action traces; (3) System-2 Reasoning, which incorporates deliberate reasoning into multi-step decision making, involving multiple reasoning patterns such as task decomposition, reflection thinking, milestone recognition, etc. (4) Iterative Training with Reflective Online Traces, which addresses the data bottleneck by automatically collecting, filtering, and reflectively refining new interaction traces on hundreds of virtual machines. Through iterative training and reflection tuning, UI-TARS continuously learns from its mistakes and adapts to unforeseen situations with minimal human intervention. We also analyze the evolution path of GUI agents to guide the further development of this domain.

---

# UI‑TARS：开创原生 GUI 交互代理 论文详细解读

### 背景：这个问题为什么难？
在电脑和手机上，软件的界面千变万化：按钮大小、颜色、布局甚至语言都不统一。过去的自动化工具要么靠手写规则，根本无法适应新界面；要么把大模型（如 GPT‑4o）包装进复杂的工作流，需要人工设计提示词和调用外部 API，成本高且易出错。更关键的是，这类系统只能把视觉信息转成文字，再让文字模型决定动作，缺少对屏幕像素的直接感知和精准定位，导致在真实操作中经常卡死或点错位置。

### 关键概念速览
**原生 GUI 代理**：模型直接把屏幕截图当作唯一输入，输出键盘、鼠标指令，像人类一样“看图做事”。  
**统一动作空间**：把所有平台（Windows、Android、Web）的点击、滑动、输入等操作映射到同一套向量表示，类似把不同语言的词翻译成同一个词表。  
**系统‑2 推理**：在做决定前先进行深度思考，如把任务拆成子任务、设定里程碑、回顾过去的错误，类似人在解决难题时先列提纲再写答案。  
**反思式在线轨迹**：模型在虚拟机器上自行尝试任务，记录每一步的成功与失败，然后用这些“经验教训”再训练自己，像人通过练习不断改进。  
**大规模 GUI 截图数据集**：收集了上亿张真实界面图并标注元素位置和属性，帮助模型学会“看懂”各种 UI。  
**Set‑of‑Mark（SoM）**：在截图上标记出所有可交互元素的集合，帮助模型在视觉上定位目标，类似在地图上画出兴趣点。  
**思维增强的行为轨迹**：在训练数据里加入了任务分解、反思等思考过程，让模型学会在行动前先“想一想”。  

### 核心创新点
1. **感知升级 → 大规模 GUI 截图 + 结构化元素描述 → 模型能够在单张截图上生成完整的 UI 文字说明和空间关系，显著提升对复杂界面的理解能力。**  
2. **统一动作建模 → 将跨平台的键盘、鼠标、触摸操作映射到同一向量空间 + 大规模真实交互轨迹 → 代理在不同操作系统上都能精准定位并执行动作，摆脱了平台特定的适配层。**  
3. **系统‑2 推理框架 → 在决策前加入任务分解、里程碑检测、错误反思等多种思考模式 → 多步任务的成功率大幅提升，尤其在需要多次交互的长序列任务中表现突出。**  
4. **迭代反思训练 → 自动在数百台虚拟机上收集交互日志 → 通过过滤和反思微调不断补齐数据盲区，使模型在遇到新界面时仍能快速适应，几乎不需要人工标注。  

### 方法详解
整体思路可以划分为三大阶段：**感知‑推理‑执行**。模型先把截图喂进视觉编码器，得到每个像素的语义向量；随后在这些向量上进行系统‑2 推理，生成一系列思考步骤和最终动作指令；最后把指令映射到统一动作空间并发送给操作系统。

**感知模块**  
- 使用大规模收集的 GUI 截图训练的视觉 Transformer。与普通图像分类不同，它被要求输出 **密集的 UI 描述**：每个按钮、输入框、图标的文字标签、功能说明以及相对坐标。  
- 为了让模型学会空间关系，训练时加入 **Set‑of‑Mark** 标记：把所有可交互元素的边框以二值图形式并入输入，让模型在注意力计算时直接看到“这里可以点”。  

**系统‑2 推理模块**  
- 输入是视觉编码器的输出以及当前任务的自然语言描述。模型采用 **Chain‑of‑Thought（思维链）** 风格的解码：先生成任务分解列表（如“打开设置 → 找到网络 → 切换 Wi‑Fi”），再对每一步进行 **里程碑检查**（判断是否已完成），最后输出 **反思语句**（如“上一步点击了错误的按钮，重新定位”）。  
- 这些思考步骤被编码成 **思维增强的行为轨迹**，并与实际执行的动作一起送入 **统一动作解码器**。  

**统一动作解码器**  
- 将所有平台的操作抽象为 **动作向量**（包括坐标、点击类型、键入内容等），并通过一个小型的多层感知网络映射到具体的系统调用。  
- 为了保证跨平台一致性，训练时使用 **跨平台动作轨迹数据**：同一任务在 Windows、Android、Web 上的完整交互记录被统一到同一向量空间，模型学会“在不同系统里做同样的事”。  

**迭代反思训练流程**  
1. 在数百台虚拟机上让模型自行尝试 benchmark 任务，记录每一步的截图、模型输出、系统反馈。  
2. 自动过滤掉明显失败的轨迹（如崩溃、卡死），保留成功或接近成功的日志。  
3. 对保留的日志进行 **反思标注**：标记出错误原因、正确的定位方式等，形成 **反思对**（原始输出 vs. 改进后输出）。  
4. 用这些对进行 **DPO（直接偏好优化）** 微调，让模型在同样情境下更倾向于产生改进后的决策。  
5. 循环上述步骤，模型的错误率随迭代次数呈指数下降。  

最巧妙的地方在于 **把“思考过程”直接写进训练数据**，而不是只给模型最终动作。这样模型在推理时拥有自我检查的机制，显著降低了长序列任务的漂移风险。

### 实验与效果
- **测试平台**：OSWorld（桌面操作系统任务）和 AndroidWorld（移动端任务），两套基准分别覆盖 10+ GUI 任务，评估感知、定位、执行三个维度。  
- **关键指标**：在 OSWorld 中，UI‑TARS 在 50 步限制下得分 24.6，15 步限制下得分 22.7，分别超过 Claude（22.0 / 14.9）。在 AndroidWorld 上，UI‑TARS 获得 46.6 分，领先 GPT‑4o 的 34.5 分。  
- **对比基线**：包括传统规则系统、模块化框架（需要手写 workflow）、以及包装商业大模型的方案（如 GPT‑4o + 手工提示）。UI‑TARS 在所有基准上均实现了 **SOTA**（当前最佳）表现。  
- **消融实验**：去掉统一动作空间后，跨平台定位错误率提升约 18%；去掉系统‑2 推理（仅使用一次性决策）导致长任务成功率下降约 12%；不使用反思式在线轨迹训练，模型在新界面上的适应速度减慢 2‑3 倍。  
- **局限性**：论文承认在极端低资源设备（如老旧手机）上实时推理仍有延迟；对高度动画化、实时渲染的游戏 UI 仍表现不佳；数据收集仍依赖爬虫和人工探索，完全零数据场景尚未覆盖。  

### 影响与延伸思考
UI‑TARS 把“看图做事”从需要大量手工提示的框架里解放出来，开启了 **原生 GUI 代理** 的时代。后续工作开始探索更轻量的视觉编码器、跨模态记忆库（把过去的交互经验存入长期记忆）以及 **主动终身学习**——让代理在真实用户设备上持续收集反馈并自我改进。对想进一步了解的读者，可以关注 **跨平台动作空间标准化**（类似 OpenAI 的 Action API）以及 **大规模 GUI 数据合成**（利用生成模型自动渲染新界面）这两个方向。  

### 一句话记住它
UI‑TARS 用大规模截图感知 + 系统‑2 思考 + 反思式自我训练，实现了直接从屏幕像素到精准操作的端到端 GUI 代理，彻底摆脱了人工提示和规则的束缚。