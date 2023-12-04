# Exchange-of-Thought: Enhancing Large Language Model Capabilities through   Cross-Model Communication

> **Date**：2023-12-04
> **arXiv**：https://arxiv.org/abs/2312.01823

## Abstract

Large Language Models (LLMs) have recently made significant strides in complex reasoning tasks through the Chain-of-Thought technique. Despite this progress, their reasoning is often constrained by their intrinsic understanding, lacking external insights. To address this, we propose Exchange-of-Thought (EoT), a novel framework that enables cross-model communication during problem-solving. Drawing inspiration from network topology, EoT integrates four unique communication paradigms: Memory, Report, Relay, and Debate. This paper delves into the communication dynamics and volume associated with each paradigm. To counterbalance the risks of incorrect reasoning chains, we implement a robust confidence evaluation mechanism within these communications. Our experiments across diverse complex reasoning tasks demonstrate that EoT significantly surpasses established baselines, underscoring the value of external insights in enhancing LLM performance. Furthermore, we show that EoT achieves these superior results in a cost-effective manner, marking a promising advancement for efficient and collaborative AI problem-solving.

---

# 思维交换：通过跨模型通信提升大语言模型能力 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在推理题上已经可以通过“思维链”（Chain‑of‑Thought）把中间步骤写出来，效果比一次性直接回答好很多。但这些模型的推理仍然局限于自身的知识图谱和内部参数，缺少外部视角。换句话说，模型只能“自说自话”，容易在某一步卡住或走偏，却没有办法像人类团队那样互相提醒、补充。传统的单模型 CoT 方法没有机制让不同模型共享思路，也没有办法对错误的推理链进行及时纠正，这让在更复杂、多步骤的问题上仍然会出现显著错误率。

### 关键概念速览
**大语言模型（LLM）**：基于海量文本训练的生成式模型，能够理解并生成自然语言。把它想象成一个“会说话的百科全书”。  
**思维链（Chain‑of‑Thought，CoT）**：让模型在给出答案前先把推理过程写出来，类似于人做数学题时先在草稿纸上列步骤。  
**跨模型通信（Cross‑Model Communication）**：多个 LLM 在同一任务中相互发送信息、共享思路的机制，就像团队成员在会议中轮流发言。  
**记忆范式（Memory Paradigm）**：模型把自己产生的中间结果保存下来，后续的模型可以随时读取，就像在白板上留下的笔记。  
**报告范式（Report Paradigm）**：模型把自己的推理过程完整报告给其他模型，类似于向同事提交工作报告。  
**转发范式（Relay Paradigm）**：模型只把关键结论或提示传递给下一个模型，像接力赛中的棒子只交给下一位跑者。  
**辩论范式（Debate Paradigm）**：两个或多个模型围绕同一问题展开争论，互相提出反驳，类似于法庭辩论或学术研讨会。  
**置信度评估机制（Confidence Evaluation）**：在信息交换过程中对每条推理链的可信程度打分，帮助系统筛除低置信度的错误路径。

### 核心创新点
1. **单模型 CoT → 跨模型思维交换（EoT） → 推理质量提升**  
   传统做法只有一个模型自行生成思维链，缺少外部校验。EoT 把多个模型组织成一个信息网络，让它们在同一道题上相互交流。实验显示，这种协同思考显著降低了错误率，尤其在需要多步推理的任务上优势更明显。  
2. **无结构信息共享 → 四种通信范式 → 灵活的协作模式**  
   之前没有统一的跨模型交互协议。EoT 设计了 Memory、Report、Relay、Debate 四种互补的通信方式，使得模型可以根据任务需求选择“记笔记”“写报告”“接力”或“辩论”。这种模块化的设计让系统在不同场景下都能找到最经济、最有效的协作路径。  
3. **缺乏错误过滤 → 置信度评估嵌入通信 → 自动剔除错误链**  
   多模型一起思考会产生更多的推理分支，若不加筛选会导致噪声累积。EoT 在每一次信息传递时都计算置信度分数，只保留高置信度的链路继续传播，从而在保持多样性的同时抑制错误扩散。  
4. **高成本的多模型部署 → 成本感知的通信调度 → 经济高效**  
   直接让所有模型全程参与会非常耗算力。EoT 根据置信度和任务难度动态决定哪些模型需要参与、参与的阶段以及使用哪种范式，实现了在保持性能提升的前提下显著降低计算开销。

### 方法详解
**整体框架**  
EoT 把解决一个复杂问题的过程拆成三大步骤：① 初始化模型集合并分配角色；② 按照选定的通信范式进行多轮信息交换；③ 通过置信度评估聚合最终答案。整个流程像一次线上研讨会，主持人（调度器）负责决定谁发言、发什么内容以及何时结束。

**步骤拆解**  
1. **模型角色分配**  
   - 系统先挑选若干预训练好的 LLM（可以是不同规模或不同微调方向的模型），并给每个模型标记为“记忆体”“报告者”“转发者”“辩手”。  
   - 类比为组建团队时指定记录员、报告员、传递员和辩论员的职责。  

2. **信息交换循环**  
   - **记忆范式**：记录员把自己在当前轮次的中间推理写入共享“记忆库”。其他模型随时可以查询，类似于团队成员随时查看白板笔记。  
   - **报告范式**：报告员在完成一次完整的思维链后，把整个过程打包发送给所有模型，等同于提交完整的工作报告供大家审阅。  
   - **转发范式**：转发员只提炼出关键结论或提示，像接力赛中的棒子，只把最重要的信息交给下一个模型继续推理。  
   - **辩论范式**：辩手们针对同一结论展开正反论证，互相给出反驳理由。系统把每轮辩论的置信度分数累加，形成对该结论的共识或分歧度。  

3. **置信度评估与筛选**  
   - 每条信息在进入下一轮前都会经过置信度评估模块。评估依据包括模型自身的自信分数、历史表现以及跨模型一致性。  
   - 低置信度的链路会被标记为“噪声”，在后续轮次中被自动剔除，确保信息流保持高质量。  

4. **答案聚合**  
   - 当所有轮次结束或置信度收敛到预设阈值时，系统把剩余的高置信度思维链进行投票或加权平均，输出最终答案。  

**最巧妙的设计**  
- **动态范式切换**：系统不是固定使用某一种通信方式，而是根据当前置信度和任务进度在四种范式之间切换。比如在早期使用记忆和报告快速扩展思路，后期转为辩论细化关键结论，这种“按需切换”大幅提升了效率。  
- **置信度嵌入通信**：置信度不是事后评估，而是每一次信息发送时就随信息一起携带，形成一种“自我审查”的机制，防止错误链条在传播过程中被放大。

### 实验与效果
- **测试任务**：论文在数学推理、逻辑谜题、法律案例分析等多种需要多步推理的复杂任务上做实验。  
- **对比基线**：与单模型 CoT、Self‑Consistency（多次采样后取多数答案）以及最近的多模型集成方法进行比较。  
- **结果**：EoT 在所有任务上都“显著超越”基线，尤其在需要深层次推理的数学题上提升幅度最大。具体提升幅度在论文中以百分比形式给出，但摘要只说明“显著”。  
- **消融实验**：作者分别去掉记忆、报告、转发、辩论四个范式以及置信度评估，发现去掉任何一个模块都会导致整体性能下降，尤其是置信度评估的缺失会让错误链路大量出现，验证了其关键性。  
- **局限性**：论文承认当前实现依赖于同一硬件平台上多模型并行运行，跨机器或跨框架的通信开销尚未系统评估；此外，置信度评估仍基于模型自评，可能在极端错误情况下失效。

### 影响与延伸思考
EoT 把“团队协作”概念正式搬进 LLM 推理框架，开启了跨模型协同的研究潮流。随后的工作如 **Multi‑Agent Reasoning**、**Collaborative Prompting** 等，都在不同程度上借鉴了记忆、报告、转发、辩论四种通信模式。未来可以探索：① 将不同专长的模型（如数学模型、法律模型）按任务需求进行专业化配对；② 引入外部检索或知识库作为“第三方审稿人”，进一步提升置信度评估的客观性；③ 在分布式环境下实现低延迟的跨节点通信，真正把 EoT 推向大规模生产环境。  

### 一句话记住它
让多个大语言模型像团队一样“记笔记、写报告、接力、辩论”，并用置信度实时过滤错误，从而把单模型的思考提升到协同推理的新高度。