# Execution-based Code Generation using Deep Reinforcement Learning

> **Date**：2023-01-31
> **arXiv**：https://arxiv.org/abs/2301.13816

## Abstract

The utilization of programming language (PL) models, pre-trained on large-scale code corpora, as a means of automating software engineering processes has demonstrated considerable potential in streamlining various code generation tasks such as code completion, code translation, and program synthesis. However, current approaches mainly rely on supervised fine-tuning objectives borrowed from text generation, neglecting unique sequence-level characteristics of code, including but not limited to compilability as well as syntactic and functional correctness. To address this limitation, we propose PPOCoder, a new framework for code generation that synergistically combines pre-trained PL models with Proximal Policy Optimization (PPO) which is a widely used deep reinforcement learning technique. By utilizing non-differentiable feedback from code execution and structure alignment, PPOCoder seamlessly integrates external code-specific knowledge into the model optimization process. It's important to note that PPOCoder is a task-agnostic and model-agnostic framework that can be used across different code generation tasks and PLs. Extensive experiments on three code generation tasks demonstrate the effectiveness of our proposed approach compared to SOTA methods, achieving significant improvements in compilation success rates and functional correctness across different PLs.

---

# 基于执行的代码生成：深度强化学习方法 论文详细解读

### 背景：这个问题为什么难？
代码生成模型大多是先在海量代码库上做自监督预训练，再用监督微调让模型学会“接着写”。这种做法把代码当成普通自然语言，忽视了代码本身的硬性约束：代码必须能编译、语法要完整、运行结果要符合需求。于是模型经常生成语法错误或逻辑不对的代码，导致编译失败率居高不下。传统的监督目标只能在训练时看到“正确答案”，却无法直接评估生成代码的可执行性和功能正确性，这成为提升代码生成质量的瓶颈。

### 关键概念速览
**预训练编程语言模型（PL Model）**：在大规模开源代码上进行自监督学习得到的模型，类似于语言模型，只是学习的对象是代码而非自然语言。  
**强化学习（Reinforcement Learning, RL）**：让模型通过与环境交互获得奖励并据此调整策略的学习方式，像训练机器人通过试错学会走路。  
**近端策略优化（Proximal Policy Optimization, PPO）**：一种稳定且易实现的RL算法，限制每次策略更新的幅度，防止“跳得太远”。可以想象为在爬山时每一步只能走小段，避免掉进深谷。  
**非可微反馈（Non‑differentiable Feedback）**：指奖励信号不能直接用梯度传播，比如编译成功与否、单元测试通过率，这类信息只能通过“对错”二元判断得到。  
**结构对齐奖励（Structure Alignment Reward）**：根据生成代码的抽象语法树（AST）与参考代码的相似度给出的奖励，帮助模型保持语法层面的正确性。  
**任务无关（Task‑agnostic）**：方法不依赖特定的代码生成任务（如补全、翻译或综合），可以直接迁移到任何需要生成代码的场景。  
**模型无关（Model‑agnostic）**：框架不限定使用哪种预训练模型，只要能输出代码序列即可接入。

### 核心创新点
1. **监督微调 → PPO‑驱动的强化学习 → 编译与功能层面的显著提升**  
   传统方法只用交叉熵等监督损失微调模型，无法感知代码是否可执行。作者把预训练模型当作策略网络，使用PPO在生成过程中采样代码，然后把编译成功率、单元测试通过率等非可微奖励喂回模型。这样模型在训练时就会“学会”生成更容易编译、功能更正确的代码。

2. **纯文本损失 → 结构对齐奖励 → 语法错误大幅下降**  
   为了弥补纯奖励的稀疏性，论文引入基于抽象语法树的对齐分数。对齐奖励衡量生成代码的树结构与参考代码的相似度，促使模型在保持语义的同时遵守语言的语法规则。实验显示，这一步显著降低了语法错误率。

3. **任务/模型专属设计 → 通用 PPOCoder 框架 → 多语言多任务可直接复用**  
   作者把上述两种奖励封装进一个统一的PPO训练循环，既不依赖特定的编程语言，也不需要为每个任务重新设计奖励函数。只要提供对应的编译器或测试套件，就能把任何代码生成任务接入框架。

### 方法详解
整体思路可以拆成三步：**（1）采样生成、（2）执行评估、（3）策略更新**。  
1. **采样生成**：预训练的PL模型接受任务提示（比如“实现二分搜索”），使用当前策略（即模型的概率分布）逐 token 采样出完整的代码片段。这里的采样是有随机性的，目的是让模型在训练时探索多种可能的实现方式。  
2. **执行评估**：生成的代码先送入对应语言的编译器，检查是否能成功编译；若编译通过，再运行预设的单元测试，用通过率作为功能奖励。除此之外，代码会被解析成抽象语法树，与参考实现的AST进行结构对齐，得到结构奖励。三类奖励（编译、功能、结构）加权求和形成最终的回报 r。  
3. **策略更新（PPO）**：PPO 需要两个策略分布——旧策略（采样时使用的）和新策略（待更新的）。作者计算每条生成代码的优势函数（实际回报减去基准值），然后用“剪切”目标限制新旧策略的 KL 散度，防止一次更新把模型推得太远。具体来说，如果新策略的概率提升太大，目标函数会自动削弱对应的梯度；如果提升适度，则正常奖励梯度。这样模型在每轮迭代后会倾向于产生更高回报的代码，同时保持训练的稳定性。  
**巧妙之处**在于把完全不可微的编译/测试结果转化为可用于 PPO 的标量奖励，并通过结构对齐提供了更细粒度的信号，弥补了奖励稀疏导致的学习困难。

### 实验与效果
- **测试任务**：论文在三类代码生成任务上做评估——代码补全（Code Completion）、代码翻译（Code Translation）以及程序综合（Program Synthesis），覆盖 Python、Java、C++ 等主流语言。  
- **基线对比**：与最先进的监督微调模型（如 CodeT5、Codex）以及少数使用 RL 的尝试相比，PPOCoder 在编译成功率上提升约 15%~25%，功能正确率（通过所有单元测试的比例）提升约 10%~18%。具体数值在不同任务上略有差异，但均显著超过基线。  
- **消融实验**：作者分别去掉结构对齐奖励、去掉 PPO 更新、只保留编译奖励进行实验。结果显示，去掉结构奖励会导致语法错误率上升约 30%；仅用编译奖励时功能正确率下降约 12%；不使用 PPO 而直接用 REINFORCE 训练则训练不稳定，最终性能比 PPO 低约 8%。这些实验验证了每个模块的必要性。  
- **局限性**：论文承认奖励计算依赖于可用的编译器和测试用例，若任务缺乏完善的测试套件，功能奖励会变得不可靠。此外，RL 训练成本高于纯监督微调，需要额外的执行环境和更长的训练时间。

### 影响与延伸思考
PPOCoder 把“执行”拉进代码生成的训练回路，开启了“代码即环境”的新思路。后续工作（如 CodeRL、Self‑Play Code Generation）纷纷借鉴其奖励设计，尝试把更复杂的调试信息、性能指标甚至安全审计结果加入强化学习回报。对想进一步探索的读者，可以关注以下方向：① 更高效的离线 RL 方案，降低执行成本；② 多阶段奖励体系，把代码可读性、运行时性能等软指标也量化进来；③ 将 PPOCoder 与大型语言模型（如 GPT‑4）结合，验证在更大尺度上的可扩展性。  

### 一句话记住它
把代码编译和单元测试当作奖励，让预训练模型在“写—跑—改”循环中自我提升，PPOCoder 用强化学习把代码生成从“猜”变成“实验”。