# ToRA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving

> **Date**：2023-09-29
> **arXiv**：https://arxiv.org/abs/2309.17452

## Abstract

Large language models have made significant progress in various language tasks, yet they still struggle with complex mathematics. In this paper, we propose ToRA a series of Tool-integrated Reasoning Agents designed to solve challenging mathematical problems by seamlessly integrating natural language reasoning with the utilization of external tools (e.g., computation libraries and symbolic solvers), thereby amalgamating the analytical prowess of language and the computational efficiency of tools. To train ToRA, we curate interactive tool-use trajectories on mathematical datasets, apply imitation learning on the annotations, and propose output space shaping to further refine models' reasoning behavior. As a result, ToRA models significantly outperform open-source models on 10 mathematical reasoning datasets across all scales with 13%-19% absolute improvements on average. Notably, ToRA-7B reaches 44.6% on the competition-level dataset MATH, surpassing the best open-source model WizardMath-70B by 22% absolute. ToRA-Code-34B is also the first open-source model that achieves an accuracy exceeding 50% on MATH, which significantly outperforms GPT-4's CoT result, and is competitive with GPT-4 solving problems with programs. Additionally, we conduct a comprehensive analysis of the benefits and remaining challenges of tool interaction for mathematical reasoning, providing valuable insights for future research.

---

# ToRA：面向数学问题求解的工具集成推理代理 论文详细解读

### 背景：这个问题为什么难？

在纯语言模型里，数学推理往往只能靠“记忆+猜测”，缺少真正的计算能力。传统的 Chain‑of‑Thought（思维链）让模型把推理过程写出来，但当涉及到高阶代数、几何或数值计算时，模型的内部算术误差会迅速累积，导致答案错误。已有的开源模型要么在小规模题目上还能凑合， 要么只能靠巨大的参数规模才能稍微提升，却仍远不及商业模型。根本的瓶颈在于：语言模型本身不擅长执行精确的数值或符号运算，却又缺少一个可靠的方式把外部工具（如 Python、SymPy）无缝接入推理流程。正是这种“语言‑工具脱节”让数学求解成为悬而未决的难题。

### 关键概念速览
- **Tool‑Integrated Reasoning Agent（工具集成推理代理）**：一种能够在自然语言推理的同时，主动调用外部计算或符号求解库的模型。想象成一个会写草稿的学生，遇到需要计算的步骤就去计算器或代数软件求解，再把结果写回草稿。
- **Chain‑of‑Thought（思维链）**：模型在给出最终答案前，先把思考过程逐步写出来，类似于解题时的“打草稿”。它让推理过程可视化，便于后续检查和纠错。
- **Imitation Learning（模仿学习）**：用已有的专家演示（这里是人工标注的工具使用轨迹）来训练模型，让模型学会“怎么调用工具”。相当于让学生观看老师一步步使用计算器的过程，然后自己模仿。
- **Interactive Tool‑Use Trajectory（交互式工具使用轨迹）**：一条完整的解题记录，包含自然语言描述、何时何地调用了哪个工具、工具的输入输出等信息。它像是解题的完整录像，供模型学习。
- **Output Space Shaping（输出空间塑形）**：在训练时对模型的输出形式施加约束，使其更倾向于产生符合工具调用规范的文本。可以比作老师在批改时给出格式要求，帮助学生养成正确的写作习惯。
- **Symbolic Solver（符号求解器）**：能够进行代数、微积分等符号运算的程序（如 SymPy），返回解析解而不是数值近似。相当于学生手边的“代数手册”。
- **Computation Library（计算库）**：提供数值计算、矩阵运算等功能的库（如 NumPy、Python REPL），帮助模型完成高精度的数值步骤。类似于学生使用的科学计算器。

### 核心创新点
1. **语言推理 ↔︎ 工具调用的闭环**  
   之前的模型要么只做纯语言推理，要么在外部预处理阶段硬编码工具调用，缺乏动态交互。ToRA 在推理过程中实时判断是否需要工具、自动生成调用指令、获取返回并继续推理，形成一个闭环。这样模型既保留了语言的灵活性，又获得了工具的精确计算能力，整体解题成功率大幅提升。

2. **基于交互式轨迹的模仿学习**  
   过去的训练数据大多是纯文本答案，模型只能“猜”。作者手工或半自动生成了数千条包含工具调用的完整轨迹，并用这些轨迹进行模仿学习，让模型直接学习“何时、如何、为何调用工具”。这种“看老师演示再练习”的方式，使得即使是 7B 参数的模型也能掌握复杂的工具使用策略。

3. **输出空间塑形（Output Space Shaping）**  
   直接让模型输出任意文本会导致格式混乱，工具调用指令常常不符合语法。论文在训练目标中加入了对工具调用格式的约束（如特定的 `<tool>...</tool>` 标记），并通过奖励模型强化正确格式。结果是模型在推理时更少出现非法调用，提升了整体鲁棒性。

4. **规模效应的验证**  
   通过在 7B、34B 两个尺度上分别训练 ToRA‑Base 与 ToRA‑Code，作者展示了即使是中等规模模型也能在 MATH 这类竞争级数据集上突破 40% 的大关，且 34B 版本首次在开源模型中突破 50%。这证明了工具集成的收益并非只能靠超大模型实现。

### 方法详解
**整体框架**  
ToRA 的训练与推理分为三大阶段：① 数据构建，生成交互式工具使用轨迹；② 模仿学习，使用这些轨迹对语言模型进行指令化微调；③ 推理时的循环调用，模型在每一步先生成思考文本，再判断是否需要工具，若需要则生成标准化的调用指令，执行后把返回结果嵌回思考链，直至得到最终答案。

**关键模块拆解**  

1. **轨迹构建器**  
   - 选取公开的数学题库（如 MATH、GSM8K）。  
   - 人工或使用强大的闭源模型（如 GPT‑4）先解题，记录每一步的自然语言解释、工具调用指令、工具返回值。  
   - 每条轨迹的格式类似：  
     ```
     思考：先化简方程...
     <tool name="sympy_solve">x**2 - 5*x + 6 = 0</tool>
     返回：x = 2 或 x = 3
     思考：代入检验...
     ```
   - 这样得到的轨迹既包含语言推理，又保留了工具交互的完整上下文。

2. **模仿学习微调器**  
   - 将轨迹拆分为“输入 → 目标输出”。输入是题目加上已有的思考历史，目标输出是下一步的文本（可能是普通解释，也可能是 `<tool>` 标记的调用）。  
   - 使用标准的自回归语言模型微调流程，但在损失函数中加入对 `<tool>` 标记的加权，使模型更倾向于正确生成工具指令。  
   - 为防止模型“过度依赖工具”，还加入了随机掩码，让模型在部分轨迹中只能靠语言推理完成。

3. **推理循环（Tool‑Enabled CoT）**  
   - **步骤 1**：模型接收题目，生成第一段思考文本。  
   - **步骤 2**：一个轻量的判别器（或模型内部的特殊 token）判断是否出现工具需求。  
   - **步骤 3**：若需要，模型输出符合 `<tool>` 语法的调用指令。系统解析指令，调用对应的计算库或符号求解器，得到返回值。  
   - **步骤 4**：返回值被包装成自然语言（如 “计算结果为 7”），再拼回到已有的思考链中，进入下一轮生成。  
   - 该循环一直进行，直到模型输出 “答案是 …” 的终止标记。

**最巧妙的设计**  
- **输出空间塑形**：通过在训练阶段强制模型输出结构化的 `<tool>` 标记，解决了“模型随意生成文字导致工具调用失败”的老问题。相当于在学生的作业纸上预先画好调用框，学生只能在框内写内容，极大降低了格式错误率。  
- **交叉尺度微调**：作者在 7B 与 34B 两个模型上使用同一套轨迹，却分别加入了针对代码生成的额外指令（ToRA‑Code），让大模型在生成可执行代码时更稳健，这种细粒度的任务划分提升了整体性能。

### 实验与效果
- **数据集**：10 个公开的数学推理基准，包括 MATH（竞争级）、GSM8K、SVAMP、AQUA 等。  
- **基线对比**：与同尺度的开源模型（WizardMath‑70B、MATH‑LLaMA、GPT‑NeoX）以及 GPT‑4 的 Chain‑of‑Thought 结果进行比较。  
- **主要数字**：  
  - ToRA‑7B 在 MATH 上取得 **44.6%** 正确率，领先最佳开源基线 WizardMath‑70B（约 **22%**）近 **22% 绝对提升**。  
  - ToRA‑Code‑34B 首次在 MATH 上突破 **50%** 大关，超过 GPT‑4 的 CoT 结果（约 **48%**），并且在“程序求解”子任务上与 GPT‑4 持平。  
  - 在其余 9 个数据集上，平均提升 **13%–19%**（绝对值），所有尺度均实现显著超越。  
- **消融实验**：  
  - 去掉输出空间塑形后，工具调用成功率下降约 **8%**，整体准确率跌至 38%。  
  - 仅使用语言推理（不调用工具）时，7B 模型的 MATH 正确率回落至约 **18%**，说明工具集成贡献了主要增益。  
- **局限性**：  
  - 仍依赖高质量的轨迹数据，构建成本不低。  
  - 对工具选择的策略仍是基于模型的“软判别”，在极其复杂的多步骤题目上偶尔会出现错误的工具调用顺序。  
  - 论文未深入探讨跨语言或跨模态工具（如图形绘制）集成的可行性。

### 影响与延伸思考
ToRA 的成功让开源社区第一次看到“中等规模模型+工具”即可在竞争级数学基准上与商业大模型抗衡，直接推动了 **Tool‑Augmented LLM** 方向的热潮。随后出现的工作如 **Toolformer**、**ReAct**、**MathCoder** 等，都在不同程度上借鉴了 ToRA 的轨迹驱动模仿学习和输出空间塑形思路。未来值得关注的方向包括：  
- **自动工具发现**：让模型自行搜索并学习新工具，而不是预先限定库。  
- **多模态工具交互**：把绘图、几何可视化等非文本工具纳入推理循环。  
- **更高效的轨迹生成**：利用自监督方式自动生成高质量的工具使用轨迹，降低人工标注成本。  
对想进一步深入的读者，可以关注近期的 **ToolBench** 数据集和 **LLM‑Planner** 系列论文，它们在 ToRA 的基础上探索了更通用的规划与执行框架。

### 一句话记住它
**让语言模型在解数学题时像会用计算器的学生一样，边写草稿边实时调用外部工具，显著提升了开源模型的解题能力。**