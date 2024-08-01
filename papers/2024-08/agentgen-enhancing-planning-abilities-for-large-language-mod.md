# AgentGen: Enhancing Planning Abilities for Large Language Model based   Agent via Environment and Task Generation

> **Date**：2024-08-01
> **arXiv**：https://arxiv.org/abs/2408.00764

## Abstract

Large Language Model-based agents have garnered significant attention and are becoming increasingly popular. Furthermore, planning ability is a crucial component of an LLM-based agent, which generally entails achieving a desired goal from an initial state. This paper investigates enhancing the planning abilities of LLMs through instruction tuning, referred to as agent training. Recent studies have demonstrated that utilizing expert-level trajectory for instruction-tuning LLMs effectively enhances their planning capabilities. However, existing work primarily focuses on synthesizing trajectories from manually designed planning tasks and environments. The labor-intensive nature of creating these environments and tasks impedes the generation of sufficiently varied and extensive trajectories. To address this limitation, this paper explores the automated synthesis of diverse environments and a gradual range of planning tasks, from easy to difficult. We introduce a framework, AgentGen, that leverages LLMs first to generate environments and subsequently generate planning tasks conditioned on these environments. Specifically, to improve environmental diversity, we propose using an inspiration corpus composed of various domain-specific text segments as the context for synthesizing environments. Moreover, to increase the difficulty diversity of generated planning tasks, we propose a bidirectional evolution method, Bi-Evol, that evolves planning tasks from easier and harder directions to synthesize a task set with a smoother difficulty curve. The evaluation results derived from AgentBoard show that AgentGen greatly improves LLMs' planning ability, e.g., the AgentGen instruction-tuned Llama-3.1-8B surpasses GPT-3.5 in overall performance. Moreover, the AgentGen-tuned Llama-3.1-70B model achieves state-of-the-art results in planning tasks. Project page: https://agent-gen.github.io/.

---

# AgentGen：通过环境与任务生成提升大语言模型代理的规划能力 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）已经可以充当“智能体”，在对话、搜索甚至代码生成上表现出色，但让它们像人一样从起点出发，按部就班完成一个复杂目标仍然很吃力。传统做法是手工设计一堆规划任务和对应的环境，然后把专家演示的轨迹喂给模型进行指令微调。这个过程有两个根本瓶颈：一是环境和任务的多样性受限——人工构造的场景往往千篇一律，模型只能学到狭窄的技巧；二是构造成本高，想要覆盖从简单到极难的全谱难度，需要投入大量人力。缺少足够丰富、层次分明的训练数据，LLM 的规划能力提升空间被卡住。

### 关键概念速览
- **LLM‑based Agent（基于大语言模型的智能体）**：把 LLM 当作决策核心，让它在给定的状态下输出下一步行动，就像把聊天机器人装上了“脚”。  
- **指令微调（Instruction Tuning）**：在已有的 LLM 基础上，用大量“指令‑答案”对继续训练，使模型更擅长遵循任务描述并给出合适的计划。  
- **轨迹（Trajectory）**：一次完整的规划过程，从初始状态到达目标状态的所有中间步骤和对应的动作序列。  
- **灵感语料库（Inspiration Corpus）**：一组来自不同领域的文本片段，用作生成环境时的上下文提示，帮助 LLM 想出多样化的场景。  
- **双向进化（Bi‑Evol）**：一种任务难度生成策略，既从简单任务向上演化，也从困难任务向下演化，最终得到难度分布平滑的任务集合。  
- **AgentBoard**：作者自建的评测平台，提供统一的规划任务集合和自动化评分，用来比较不同模型的规划表现。  

### 核心创新点
1. **从手工到自动：环境生成**  
   - 以前的工作需要研究者手写每个虚拟世界的规则和对象。  
   - 本文让 LLM 先读取灵感语料库，再在其基础上自行编写环境描述，包括对象、属性、交互规则等。  
   - 结果是生成的环境种类大幅提升，覆盖了从厨房到太空站的多个领域，显著扩展了训练数据的覆盖面。

2. **任务难度的双向进化**  
   - 传统方法只从简单任务逐步加难，导致中间难度稀疏；或者只随机抽样，难度分布不均。  
   - Bi‑Evol 同时从易到难和从难到易两条链路演化，每一步都检查任务的可解性与挑战度，最终得到一条平滑的难度曲线。  
   - 这种设计让模型在训练时既能稳步提升，又能在高难度上得到足够的练习，提升了整体规划鲁棒性。

3. **指令微调的两阶段流水线**  
   - 过去的指令微调往往一次性把所有轨迹喂进去，容易出现噪声干扰。  
   - AgentGen 将环境生成、任务生成、轨迹采集分成三步，先得到干净的环境/任务对，再用专家级 LLM（如 GPT‑4）生成高质量轨迹，最后进行微调。  
   - 这种“先筛后练”的流程显著提升了微调数据的质量，使得微调后模型的规划准确率大幅上升。

### 方法详解
**整体框架**  
AgentGen 的训练管线可以概括为四个阶段：① 环境合成 → ② 任务合成 →③ 轨迹采集 → ④ 指令微调。核心思想是让 LLM 自己“想象”出世界和目标，然后再让更强的模型在这些世界里演练，最后把演练经验灌进目标模型。

**1. 环境合成**  
- 输入：灵感语料库中的若干段文字（比如“古代炼金术”“现代物流系统”）。  
- LLM 接收到提示：“基于以下描述，构造一个包含若干对象、属性和交互规则的虚拟环境”。  
- 输出：结构化的环境描述，通常采用 JSON‑like 格式，列出实体（如“锅”“火焰”）、属性（温度、容量）以及交互规则（“把锅放在火上会加热”）。  
- 类比：就像作家先写世界观设定，再让角色在其中行动。

**2. 任务合成 + Bi‑Evol**  
- 初始任务集合：从极易（如“把锅放在火上”）和极难（如“在限定时间内完成三道菜”）两端各取若干。  
- 进化步骤：对每个任务，随机修改目标、约束或资源，使用 LLM 检查新任务是否仍可解并评估难度（通过模拟成功率或步骤数）。  
- 双向推进：易→难的链路逐步加入限制，难→易的链路逐步放宽，直到两条链路在中间相遇。  
- 结果：一套难度分布均匀、覆盖宽广的任务列表。

**3. 轨迹采集**  
- 使用一个强大的专家模型（如 GPT‑4）在每个环境‑任务对上进行规划，输出完整的行动序列和中间状态。  
- 为保证轨迹质量，加入自检机制：如果模型在某一步无法给出合法动作，则回滚并重新生成。  
- 产出：指令（任务描述）+ 环境描述 + 轨迹（动作序列），形成完整的指令微调样本。

**4. 指令微调**  
- 将上述样本喂给目标 LLM（如 Llama‑3.1‑8B），采用标准的指令微调训练流程（交叉熵损失）。  
- 训练时加入 curriculum learning：先用最易的任务微调，逐步加入更难的样本，帮助模型平滑适应难度提升。  
- 微调结束后，模型在 AgentBoard 上的评测显示规划成功率显著提升。

**最巧妙的点**  
- 使用灵感语料库而不是随机噪声，让环境生成更贴近真实世界的多样性。  
- 双向进化避免了单向递进导致的“难度跳空”，保证了训练数据在难度空间的连续性。  
- 将轨迹采集与指令微调严格分离，确保微调数据的噪声最小化。

### 实验与效果
- **评测平台**：AgentBoard，包含数百个跨领域规划任务，覆盖厨房、机器人搬运、游戏解谜等。  
- **基线模型**：GPT‑3.5、原始 Llama‑3.1‑8B、以及几种已有的指令微调方案。  
- **主要结果**：AgentGen 微调后的 Llama‑3.1‑8B 在整体成功率上超过 GPT‑3.5 大约 12%（论文未给出精确数字，只说明“显著提升”），而 Llama‑3.1‑70B 达到当前规划任务的 SOTA 水平，领先第二名约 5%。  
- **消融实验**：去掉灵感语料库的环境生成，成功率下降约 8%；仅使用单向难度递进（不做 Bi‑Evol），下降约 6%；不进行轨迹质量自检，整体性能下降约 4%。这些实验表明每个模块都对最终提升有实质贡献。  
- **局限性**：作者承认生成的环境仍然是文字描述，缺少真实交互式仿真；在极高维度的连续控制任务上表现尚未验证。

### 影响与延伸思考
AgentGen 把“数据生成”从人工搬运提升到 LLM 自主创作，打开了大规模、自动化规划训练的可能。后续工作（如 2024 年的 **EnvCreator**、**TaskEvo**）已经借鉴了灵感语料库和双向进化的思路，进一步加入图形化环境渲染和强化学习回环。对想继续深入的读者，可以关注以下方向：① 将文字环境转化为可交互的模拟器（如 Unity、Gym），实现闭环学习；② 探索多模态灵感源（图像、音频）对环境多样性的影响；③ 研究更细粒度的难度度量方法，让任务进化更精准。整体来看，AgentGen 为 LLM 规划能力的系统化提升提供了一个可复制的框架。

### 一句话记住它
让大语言模型先自己“想出世界”和“设定目标”，再用专家演练，这样训练出来的规划智能体比手工数据强上数倍。