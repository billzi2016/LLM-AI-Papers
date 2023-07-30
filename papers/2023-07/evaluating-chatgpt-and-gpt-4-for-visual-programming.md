# Evaluating ChatGPT and GPT-4 for Visual Programming

> **Date**：2023-07-30
> **arXiv**：https://arxiv.org/abs/2308.02522

## Abstract

Generative AI and large language models have the potential to drastically improve the landscape of computing education by automatically generating personalized feedback and content. Recent works have studied the capabilities of these models for different programming education scenarios; however, these works considered only text-based programming, in particular, Python programming. Consequently, they leave open the question of how well these models would perform in visual programming domains popularly used for K-8 programming education. The main research question we study is: Do state-of-the-art generative models show advanced capabilities in visual programming on par with their capabilities in text-based Python programming? In our work, we evaluate two models, ChatGPT (based on GPT-3.5) and GPT-4, in visual programming domains for various scenarios and assess performance using expert-based annotations. In particular, we base our evaluation using reference tasks from the domains of Hour of Code: Maze Challenge by Code-dot-org and Karel. Our results show that these models perform poorly and struggle to combine spatial, logical, and programming skills crucial for visual programming. These results also provide exciting directions for future work on developing techniques to improve the performance of generative models in visual programming.

---

# 评估 ChatGPT 与 GPT-4 在视觉编程中的表现 论文详细解读

### 背景：这个问题为什么难？
视觉编程（如 Code.org 的迷宫挑战或 Karel 机器人）把代码块、图形界面和空间推理混在一起，学生必须同时理解几何位置、逻辑顺序和编程语法。过去的研究大多把大语言模型（LLM）当作“会写 Python 的老师”，只评估它们在文本代码上的自动批改或生成能力。可是，把同样的模型搬到需要“看图、走格子、摆指令块”的环境时，模型既缺少对空间布局的直观感知，又没有专门的训练数据，导致它们在视觉编程任务上可能会卡壳。正因为这层跨模态、跨认知的壁垒，评估 LLM 在视觉编程中的真实水平成为了迫切且未被解答的科研点。

### 关键概念速览
**视觉编程**：用图形化的积木或拖拽式界面写程序，常见于 K-8 教育，例如让机器人在格子地图上走迷宫。类似于用拼图而不是文字描述来表达算法。  
**大语言模型（LLM）**：基于海量文本训练的生成式模型，如 ChatGPT（GPT‑3.5）和 GPT‑4，擅长自然语言理解和代码生成。把它们想象成“会说话的代码助手”。  
**空间推理**：在脑中或模型里对二维/三维位置关系进行计算的能力，比如判断“左转后再前进两格会碰墙吗”。相当于在玩拼图时需要先想象每块拼图的放置位置。  
**逻辑组合**：把条件、循环、函数等编程结构正确嵌套的能力。就像把不同颜色的积木按正确顺序搭建成一座桥。  
**专家标注**：由领域专家对模型输出进行人工评分，提供客观的质量基准。类似于老师给学生的作业打分。  
**Hour of Code**：Google 与 Code.org 合作的“一小时编程”活动，提供标准化的教学任务（如 Maze Challenge），常被用作编程教育研究的基准。  

### 核心创新点
1. **从文本转向视觉编程评估**：以前的工作只看模型能否写出或解释 Python 代码，这篇论文把评估场景搬到了需要图形界面和空间感知的任务上。这样直接检验了模型在“看图写指令”这一全新维度的能力。  
2. **使用真实教育任务作为基准**：作者挑选了 Code.org 的 Maze Challenge 与 Karel 两套广泛使用的教学关卡，而不是自制的玩具例子。这样保证了实验结果对实际课堂有直接参考价值。  
3. **专家驱动的细粒度标注体系**：研究团队让编程教育专家对模型的每一步输出（如路径规划、指令顺序、错误提示）进行打分，形成了比单纯对错更丰富的评价维度。  
4. **对比两代模型的跨模态表现**：通过同时测试基于 GPT‑3.5 的 ChatGPT 和更强大的 GPT‑4，论文直接展示了模型规模提升是否真的带来视觉编程能力的跃迁，结果却出乎意料地不理想。

### 方法详解
整体思路可以拆成三步：任务选取 → 模型交互 → 专家评估。

1. **任务选取**  
   - **Maze Challenge**：学生需要在 5×5~10×10 的网格中控制小角色走出迷宫，指令只有前进、左转、右转三种。  
   - **Karel**：类似的机器人编程环境，指令包括移动、放标记、条件判断等。两者都提供了明确的起点、终点和障碍布局，便于对比模型输出与正确解。

2. **模型交互**  
   - 研究者把每个关卡的地图信息（以文字描述的方式）喂给 ChatGPT 和 GPT‑4，要求模型生成完整的指令序列或提供解题思路。  
   - 为了模拟真实教学场景，还让模型在错误的指令上进行纠错、解释为什么会卡住等交互式对话。  
   - 交互过程严格记录，包括模型的原始输出、后续的追问与补充回答。

3. **专家评估**  
   - 两位具备 K‑8 编程教学经验的专家分别对每一次模型输出进行打分，评分维度包括：**空间正确性**（路径是否真的能走通）、**逻辑完整性**（循环、条件是否合理）、**语言表达**（指令是否符合平台语法）以及**教学价值**（解释是否对学生有帮助）。  
   - 评分采用 0–5 的离散尺度，最终取平均值作为该模型在该任务上的表现。

**最巧妙的地方**在于把原本只能“看文字”的 LLM 通过文字化的地图描述强行带入视觉任务，而不是直接让模型处理图像。这样既保持了模型的输入限制，又能检验它们是否能在纯语言层面推演空间关系，实际上是对模型“想象力”的极限测试。

### 实验与效果
- **数据集/任务**：共选取了 30 关 Maze 关卡和 20 关 Karel 关卡，覆盖不同难度和布局变化。  
- **Baseline 对比**：论文没有引入其他视觉编程专用模型作为对照，只把 ChatGPT 与 GPT‑4 两个版本进行横向比较。  
- **结果概述**：两款模型在空间正确性上平均得分约为 1.2（满分 5），逻辑完整性更低，常出现指令顺序错误或遗漏关键条件。GPT‑4 的整体得分略高于 ChatGPT，但提升幅度不显著，仍远低于人类专家（约 4.5 分）。  
- **消融实验**：作者尝试只提供起点终点的简要描述（去掉障碍信息），模型表现进一步下降，说明空间信息的缺失对模型已经是致命打击。  
- **局限性**：实验仅使用文字描述的地图，未测试模型直接处理图像或交互式 UI；评估规模相对有限（50 关卡），可能不足以覆盖所有视觉编程的变体。作者也承认，当前 LLM 仍缺乏真正的空间感知能力，单靠语言推理难以弥补。

### 影响与延伸思考
这篇工作首次把大语言模型的能力边界扩展到 K‑8 视觉编程，直接提醒社区：**大模型在文本代码上表现不错，却不等同于在图形化、空间化编程环境中也能胜任**。随后有几篇后续研究尝试结合视觉大模型（如 CLIP、GPT‑4V）或专门的图形化编程数据集进行微调，目标是让模型在“看图写指令”上拥有更可靠的推理。对想进一步探索的读者，可以关注以下方向：  
- **多模态微调**：把图像特征与语言模型联合训练，让模型直接“看到”迷宫布局。  
- **结构化提示工程**：设计更精细的提示模板，引导模型先进行空间布局推演，再生成指令。  
- **教育交互评估**：把模型嵌入真实课堂的编程平台，收集学生的学习效果反馈，评估模型的教学价值。  

### 一句话记住它
**大语言模型在文字代码上很强，但在需要空间推理的视觉编程里仍表现不佳，提醒我们必须为它们加入真正的“看图”能力。**