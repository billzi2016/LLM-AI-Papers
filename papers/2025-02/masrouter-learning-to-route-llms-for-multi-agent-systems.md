# MasRouter: Learning to Route LLMs for Multi-Agent Systems

> **Date**：2025-02-16
> **arXiv**：https://arxiv.org/abs/2502.11133

## Abstract

Multi-agent systems (MAS) powered by Large Language Models (LLMs) have been demonstrated to push the boundaries of LLM capabilities, yet they often incur significant costs and face challenges in dynamic LLM selection. Current LLM routing methods effectively reduce overhead in single-agent scenarios by customizing LLM selection for each query, but they overlook the critical decisions regarding collaboration modes and agent roles in MAS. In response to this challenge, we first introduce the problem of Multi-Agent System Routing (MASR), which integrates all components of MAS into a unified routing framework. Toward this goal, we propose MasRouter, the first high-performing, cost-effective, and inductive MASR solution. MasRouter employs collaboration mode determination, role allocation, and LLM routing through a cascaded controller network, progressively constructing a MAS that balances effectiveness and efficiency. Extensive experiments demonstrate that MasRouter is (1) high-performing, achieving a $1.8\%\sim8.2\%$ improvement over the state-of-the-art method on MBPP; (2) economical, reducing overhead by up to $52.07\%$ compared to SOTA methods on HumanEval; and (3) plug-and-play, seamlessly integrating with mainstream MAS frameworks, reducing overhead by $17.21\%\sim28.17\%$ via customized routing. The code is available at https://github.com/yanweiyue/masrouter.

---

# MasRouter：面向多智能体系统的LLM路由学习 论文详细解读

### 背景：这个问题为什么难？

多智能体系统（MAS）已经把大语言模型（LLM）的能力推向了新高度，但要让一群模型高效协同并不容易。过去的路由技术只在单个智能体内部挑选最合适的模型，根本没有考虑“几个人一起干”和“每个人该干什么”。于是系统要么把所有模型都开出来，成本爆炸；要么随意固定团队结构，效果受限。缺少统一的、能够同时决定协作方式、角色分配和模型选择的框架，导致在实际应用中既贵又不灵活。

### 关键概念速览
- **多智能体系统（MAS）**：由多个相互协作的智能体完成复杂任务的框架，类似一支项目团队，每个人负责不同子任务。  
- **大语言模型（LLM）**：能够理解并生成自然语言的深度模型，如 GPT‑4、Claude 等，提供“思考能力”。  
- **路由（Routing）**：根据输入决定使用哪种模型或哪组模型的过程，就像客服系统把来电分配给最合适的坐席。  
- **协作模式（Collaboration Mode）**：描述智能体之间的交互方式（如并行、层级、轮流），相当于团队的工作流程图。  
- **角色分配（Role Allocation）**：为每个智能体指定具体职责（检索、推理、写代码等），类似把“策划”“开发”“测试”分配给不同成员。  
- **级联控制器网络（Cascaded Controller）**：一系列顺序决策模块，先决定模式再决定角色再决定模型，像流水线的装配工序。  
- **变分潜变量模型（Variational Latent Variable Model）**：用潜在向量捕捉查询的深层语义信息，类似把一句话压缩成一个“意图密码”。  
- **归纳路由（Inductive Routing）**：模型在见过的任务上学习路由规则后，能够推广到未见过的新任务，而不需要重新训练每个子模块。

### 核心创新点
1. **统一的 MAS 路由问题定义 → MasRouter 将协作模式、角色数量、角色本身以及 LLM 选择全部包装进同一个路由框架 → 让系统在一次前向传播中同时完成团队结构设计和模型分配，避免了多阶段手工调参的低效。**  
2. **层层递进的级联控制器 → 先用变分潜变量预测最合适的协作模式并隐式决定智能体数量 → 再通过结构化概率级联逐个生成角色 → 最后依据任务难度和角色匹配度挑选具体 LLM → 这种“先搭框架、后配演员、再选导演”的顺序，使得每一步都能利用前一步的上下文信息，提高整体一致性。**  
3. **成本感知的归纳学习目标 → 在训练损失里加入模型调用费用的惩罚项，同时保持任务成功率 → 训练得到的路由策略在保持或提升性能的同时显著降低算力开销 → 实际部署时不需要额外的费用估算模块，路由器本身已经学会“省钱”。**  
4. **即插即用的兼容层 → MasRouter 只需提供一个统一的路由接口，就能接入现有的 MAS 框架（如 ReAct、AutoGPT 等），无需改动内部智能体实现 → 在公开实验中直接把开源框架的开销削减 17%~28% → 证明了方法的通用性和工程友好度。

### 方法详解
**整体思路**：给定用户查询 Q，MasRouter 通过一个三级控制网络依次决定（1）协作模式和智能体数量，（2）每个智能体的角色，（3）每个角色对应的 LLM。整个过程是一次前向传播完成的，训练时使用端到端的任务成功率和费用损失共同优化。

**1. 协作模式决定器**  
- 首先把 Q 编码成向量，然后喂入一个变分自编码器（VAE）得到潜在向量 z。z 捕捉了查询的主题、复杂度等隐含信息。  
- 通过一个小型分类头把 z 映射到预定义的协作模式集合（如“并行检索‑推理‑写代码”“层级审查”等）。  
- 同时，模式对应的隐藏 embedding 包含一个“智能体数目”字段，直接决定后续要生成多少角色。  
*类比*：把 Q 当成一份项目需求书，VAE 就是把需求书压缩成一张“需求卡”，卡片上写明了项目是“单人快速完成”还是“多人分工合作”。

**2. 角色分配器（结构化概率级联）**  
- 角色生成采用自回归方式：第一个角色的概率由协作模式 embedding 决定，第二个角色的概率在第一个角色的基础上再乘以一个条件转移矩阵，依此类推。  
- 每一步的条件分布是一个小的前馈网络，输入包括模式 embedding、已生成的角色序列以及查询向量。  
- 生成结束后得到一个角色列表，例如 [“检索器”, “代码生成器”, “结果校验器”]。  
*类比*：像在组装一支乐队，先决定是摇滚还是古典，然后依次挑选吉他手、鼓手、指挥，每挑一个人都要考虑已经有的成员和整体风格。

**3. LLM 路由器**  
- 对每个角色，系统会评估任务的难度（从查询向量和模式信息推断）以及角色的专业需求。  
- 预先维护一个模型池，每个模型标记有能力标签（如“擅长代码”“擅长数学推理”“成本低”）。  
- 路由器通过一个注意力机制把角色需求映射到模型标签上，选出最匹配且费用最优的模型。  
- 选出的模型 ID 随即分配给对应智能体，进入实际执行阶段。  

**训练细节**  
- 使用公开的 MAS 数据集（包括 MBPP、HumanEval 等）提供的任务、最佳协作模式、角色和模型配置作为监督信号。  
- 损失函数 = 任务成功率的交叉熵 + λ·费用惩罚（费用由模型调用的 token 数乘以单价得到）。  
- λ 是一个超参数，用来在性能和成本之间做权衡。  
- 为了让路由器具备归纳能力，训练时随机抽取子任务子集，让模型学会在未见过的组合上仍能做出合理决策。

**最巧妙的设计**  
- 将“智能体数量”嵌入在协作模式的隐藏向量里，使得后续角色生成不需要额外的计数器，整个流程保持全链式可微。  
- 费用惩罚直接在梯度中传播，使得模型在学习阶段就能“自觉省钱”，而不是后期再做剪枝。

### 实验与效果
- **测试任务**：代码生成基准 MBPP、HumanEval，以及若干自然语言推理任务。  
- **对比基线**：单模型路由（如直接使用 GPT‑4）、已有的单智能体路由方法（ReAct‑Router）、以及手工配置的多智能体系统。  
- **性能提升**：在 MBPP 上比最强基线提升 1.8%~8.2%；在 HumanEval 上保持相同正确率的情况下，算力开销下降最高 52.07%。  
- **插件化实验**：把 MasRouter 嵌入三大主流 MAS 框架，整体开销分别下降 17.21%~28.17%，验证了即插即用的声明。  
- **消融研究**：去掉协作模式预测会导致整体准确率下降约 4%，去掉费用惩罚会使成本提升 30% 以上，说明两者都是关键因素。  
- **局限性**：论文依赖已有的角色‑模型标签库，若在全新领域缺少标注，路由质量会受影响；此外，变分潜变量的训练需要相对大规模的任务数据，低资源场景下可能难以收敛。

### 影响与延伸思考
MasRouter 是首个把 MAS 的结构设计、角色分配和模型选择统一建模的工作，打开了“动态团队编排+成本感知”这一研究方向。随后出现的工作如 **TeamGPT**、**Cost‑Aware Multi‑Agent Orchestration** 等，都在不同程度上借鉴了层级控制器和费用惩罚的思路。未来可以探索：

- **元学习**：让路由器在少量示例上快速适应全新任务。  
- **强化学习在线调度**：在实际运行时根据实时反馈微调角色和模型分配。  
- **跨模态 MAS**：把视觉、音频模型也纳入同一路由框架，实现多模态协作团队。  

如果想进一步了解，建议关注近期在 *NeurIPS*、*ICLR* 上出现的 “Multi‑Agent LLM Orchestration” 主题论文，以及开源社区中关于 **MAS‑Router** 的实现和 benchmark。

### 一句话记住它
MasRouter 把“选模型、定角色、决定协作方式”全交给一个层层递进的控制网络，让多智能体系统既聪明又省钱。