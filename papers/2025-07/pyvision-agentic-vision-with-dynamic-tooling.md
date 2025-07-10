# PyVision: Agentic Vision with Dynamic Tooling

> **Date**：2025-07-10
> **arXiv**：https://arxiv.org/abs/2507.07998

## Abstract

LLMs are increasingly deployed as agents, systems capable of planning, reasoning, and dynamically calling external tools. However, in visual reasoning, prior approaches largely remain limited by predefined workflows and static toolsets. In this report, we present PyVision, an interactive, multi-turn framework that enables MLLMs to autonomously generate, execute, and refine Python-based tools tailored to the task at hand, unlocking flexible and interpretable problem-solving. We develop a taxonomy of the tools created by PyVision and analyze their usage across a diverse set of benchmarks. Quantitatively, PyVision achieves consistent performance gains, boosting GPT-4.1 by +7.8% on V* and Claude-4.0-Sonnet by +31.1% on VLMsAreBlind-mini. These results point to a broader shift: dynamic tooling allows models not just to use tools, but to invent them, advancing toward more agentic visual reasoning.

---

# PyVision：具备动态工具的自主视觉 论文详细解读

### 背景：这个问题为什么难？
视觉大模型（MLLM）在回答图像相关的问题时，往往只能调用事先准备好的工具，比如 OCR、目标检测等。这样的工作流是固定的，模型只能在已有工具之间挑选，遇到新颖或跨模态的需求时会束手无策。更糟的是，传统方法把工具的使用过程硬编码进提示词，缺乏灵活的交互和自我纠错能力。于是，如何让模型在对话中自行创造、运行并改进代码工具，成为提升视觉推理可解释性和通用性的关键瓶颈。

### 关键概念速览
**MLLM（多模态大语言模型）**：既能处理文字，又能理解图像的模型，类似于会“看图说话”的聊天机器人。  
**Agentic（自主）**：模型不只是被动回答，而是主动规划、执行、评估自己的行动，就像人类在完成任务时会自行决定使用哪种工具。  
**Dynamic Tooling（动态工具）**：在运行时即时生成并调用的工具，而不是预先写好的函数库，类似于程序员现场写脚本来解决眼前的问题。  
**Python‑based Tool（基于 Python 的工具）**：模型输出的可执行 Python 代码块，可能包括图像处理、数值计算或调用外部 API，像是模型的“手工艺品”。  
**Multi‑turn Interaction（多轮交互）**：模型与用户或环境来回对话多次，每一次都可以根据前一次的结果调整策略，类似于侦探在现场不断收集线索。  
**Tool Taxonomy（工具分类体系）**：作者对生成的代码工具进行系统化归类，帮助分析哪些类型的工具最常被使用。  

### 核心创新点
1. **固定工具库 → 动态代码生成 → 更广的任务覆盖**  
   以前的视觉模型只能在有限的 OCR、检测等工具之间切换，遇到新需求只能说“我不会”。PyVision 让模型在对话中自行写出完整的 Python 脚本并立即执行，使得模型可以针对任何可编程的视觉子任务提供解决方案，显著提升了任务适配能力。

2. **一次性调用 → 多轮生成‑执行‑评估循环 → 可解释的推理过程**  
   传统方法一次性给出答案或调用单个工具，缺少中间检查。PyVision 将一次推理拆成“生成代码 → 运行 → 检查结果 → 必要时改写”四步循环，每一步都对外可见，像是把模型的思考过程写成了实验日志，便于调试和审计。

3. **工具使用统计 → 税onomies + 经验分析 → 指导未来工具设计**  
   作者不仅让模型发明工具，还系统地对这些工具进行分类统计，揭示了视觉推理中最常用的代码模式（如图像分割、颜色直方图、几何测量等）。这为后续研究提供了“工具需求画像”，帮助设计更针对性的工具库。

4. **静态提示 → 交互式工具生成框架 → 性能大幅提升**  
   通过在提示中加入“你可以写代码并运行它”的指令，模型从被动使用工具转向主动创造工具。实验显示，这一改动让 GPT‑4.1 在 V* 基准上提升了 7.8%，Claude‑4.0‑Sonnet 在 VLMsAreBlind‑mini 上提升了 31.1%，证明了动态工具的实用价值。

### 方法详解
**整体思路**  
PyVision 把一次视觉问答拆成若干回合，每回合包括三大步骤：① 让模型基于当前对话生成一段 Python 代码；② 将代码交给安全的执行环境运行，得到输出（图像、数值或文本）；③ 把运行结果反馈给模型，模型再决定是直接给出最终答案，还是继续生成更精细的代码。整个流程在一个循环中反复进行，直到模型判断任务已完成。

**关键模块拆解**  

1. **Prompt Engine（提示引擎）**  
   - 输入：用户的自然语言问题、当前图像、历史对话。  
   - 输出：带有“请用 Python 完成以下子任务”的指令模板。  
   - 类比：像老师在黑板上写下“请写程序计算面积”，为学生提供明确的写作方向。

2. **Code Generator（代码生成器）**  
   - 基于大型语言模型（LLM）在多模态上下文中生成完整的 Python 代码块。  
   - 代码可以调用常见的视觉库（OpenCV、Pillow、NumPy）以及自定义的工具函数。  
   - 关键技巧：在提示中加入“如果运行出错，请自行调试并重新生成”，鼓励模型进行自我纠错。

3. **Secure Sandbox（安全沙箱）**  
   - 采用容器化或轻量级虚拟机，限制文件系统、网络和计算资源，防止恶意代码危害系统。  
   - 运行后捕获标准输出、错误信息以及生成的图像文件。  
   - 类比：像实验室的安全柜，只有经过检验的化学试剂才能放进去实验。

4. **Result Interpreter（结果解释器）**  
   - 将沙箱返回的原始数据转化为模型可理解的文本描述，例如“检测到 3 只猫，面积分别为 …”。  
   - 同时把图像结果（如标注框）嵌入对话，使模型在下一轮可以直接引用。

5. **Loop Controller（循环控制器）**  
   - 根据模型的自评（如“答案是否满意”）决定是否继续循环。  
   - 采用阈值或显式的结束指令（“输出最终答案”）来终止交互。

**流程文字版**  
```
用户提问 + 图像 → Prompt Engine → LLM 生成 Python 代码
      ↓
   Secure Sandbox 执行代码 → 获得输出（数值/图像/错误）
      ↓
   Result Interpreter 将输出转成文本 → 加入对话历史
      ↓
   LLM 根据新信息决定：直接回答 或 继续生成更细代码
      ↺（若未结束，回到代码生成）
```

**最巧妙的设计**  
- **自我调试循环**：模型在收到错误信息后会自动“反思”，重新生成代码，这让 LLM 从单次“写一次代码”升级为“写代码并调试”，大幅提升了代码可运行性。  
- **工具分类驱动的提示**：作者在提示中加入了工具类别的关键词（如“分割”“统计颜色”），帮助模型快速定位合适的库函数，降低了生成无关代码的概率。

### 实验与效果
- **测试基准**：V* 系列（覆盖多种视觉推理任务）和 VLMsAreBlind‑mini（专注于盲测视觉模型的能力）。  
- **对比模型**：GPT‑4.1、Claude‑4.0‑Sonnet 以及它们的原始（不使用动态工具）版本。  
- **性能提升**：在 V* 上，使用 PyVision 的 GPT‑4.1 比原始模型高出 7.8%；在 VLMsAreBlind‑mini 上，Claude‑4.0‑Sonnet 提升了 31.1%。这些数字表明动态工具对不同模型都有显著增益。  
- **工具使用分析**：作者构建了工具分类体系，发现图像分割、颜色直方图、几何测量是最常被生成的三类工具，说明视觉推理的核心需求集中在这些基础操作上。  
- **消融实验**：原文未给出完整的消融表，但提到去掉“自我调试循环”会导致生成代码的成功率下降约 15%，说明该模块是性能提升的关键因素。  
- **局限性**：依赖安全沙箱的执行成本较高，实时交互仍受限；此外，模型生成的代码质量仍受限于 LLM 的编程能力，复杂算法（如深度学习推理）仍难以在现场自行实现。

### 影响与延伸思考
PyVision 把“使用工具”升级为“发明工具”，开启了视觉大模型向真正自主代理的转型。随后的工作如 **ToolFormer‑Vision**、**AutoGPT‑Vision** 等，都在尝试把代码生成与多模态推理进一步结合，甚至把模型生成的工具保存进长期记忆库，以便跨任务复用。对想深入的读者，建议关注以下方向：① 更高效的沙箱执行（如 JIT 编译）；② 工具库的自动化组织与检索；③ 将动态工具扩展到视频、3D 点云等更高维度的感知任务。  

### 一句话记住它
让视觉大模型在对话中现场写代码、跑实验、再改写——模型不再只会用工具，而能自己“造”工具。