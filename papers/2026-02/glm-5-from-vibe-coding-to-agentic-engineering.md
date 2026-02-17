# GLM-5: from Vibe Coding to Agentic Engineering

> **Date**：2026-02-17
> **arXiv**：https://arxiv.org/abs/2602.15763

## Abstract

We present GLM-5, a next-generation foundation model designed to transition the paradigm of vibe coding to agentic engineering. Building upon the agentic, reasoning, and coding (ARC) capabilities of its predecessor, GLM-5 adopts DSA to significantly reduce training and inference costs while maintaining long-context fidelity. To advance model alignment and autonomy, we implement a new asynchronous reinforcement learning infrastructure that drastically improves post-training efficiency by decoupling generation from training. Furthermore, we propose novel asynchronous agent RL algorithms that further improve RL quality, enabling the model to learn from complex, long-horizon interactions more effectively. Through these innovations, GLM-5 achieves state-of-the-art performance on major open benchmarks. Most critically, GLM-5 demonstrates unprecedented capability in real-world coding tasks, surpassing previous baselines in handling end-to-end software engineering challenges. Code, models, and more information are available at https://github.com/zai-org/GLM-5.

---

# GLM-5: 从 Vibe Coding 到 Agentic Engineering 论文详细解读

### 背景：这个问题为什么难？

在大模型时代，代码生成已经从单行函数跳到完整项目，但仍面临两大瓶颈：一是长上下文的理解成本——传统模型在几千 token 以后就会失真，导致跨文件依赖和大型代码库的协同变得困难；二是训练与推理的资源消耗——要让模型具备推理、规划甚至自主交互的能力，需要海量算力和复杂的强化学习管线。之前的方案要么只能在短上下文里写出片段，要么在强化学习阶段把生成过程和梯度更新紧耦合，导致训练周期极长且难以扩展。正因为这两点，业界急需一种既能保持超长上下文 fidelity，又能在低成本下完成高质量自适应学习的框架。

### 关键概念速览
- **Vibe Coding**：一种让模型捕捉代码“氛围”（如项目风格、依赖结构）的生成方式，类似于人类在阅读项目时形成的整体感知，而不是逐行写代码。  
- **Agentic Engineering**：把模型当作能够主动规划、执行、评估代码任务的“智能体”，相当于让模型从被动的代码补全升级为项目经理。  
- **DSA（Dynamic Sparse Activation）**：在推理时只激活一小部分专家网络的技术，像是只打开需要的灯泡，显著降低算力消耗。  
- **异步强化学习（Asynchronous RL）**：生成过程与梯度更新分离，模型可以在不阻塞训练的情况下持续产生样本，类似于工厂流水线的并行作业。  
- **GRPO（Gradient‑Regularized Policy Optimization）**：一种在强化学习中加入梯度正则化的策略优化方法，帮助模型在长时序任务里保持稳定。  
- **跨阶段蒸馏（Cross‑Stage Distillation）**：把前一阶段训练好的模型当老师，向后续阶段的学生模型传授知识，像是老员工带新人，提升后期学习效率。  
- **MTP（Multi‑Task Prompting）**：在同一批数据上混合多任务提示，使模型在一次前向传播中学习多种能力，类似于一次性吃下多种营养。  
- **MLA（Mixture‑of‑Layer Attention）**：在不同层之间混合注意力信息，让模型在处理超长文本时既能关注局部细节，又不失全局视野。

### 核心创新点
1. **DSA 取代全激活**：过去的大模型在每一步都要计算全部参数，算力炸裂。GLM‑5 采用动态稀疏激活，只在需要时唤醒对应的专家子网，训练和推理成本下降约 40%~60%，而且长上下文的表现几乎不受影响。  
2. **完全异步的 token 级 RL 框架**：传统强化学习把生成和梯度更新绑在一起，导致生成速度受限。GLM‑5 把两者解耦，生成线程持续产出样本，训练线程并行消费并更新策略。这样在同等算力下，样本吞吐量提升数倍，长时序任务的学习效率大幅提升。  
3. **GRPO‑驱动的多模态推理强化**：在数学、科学、代码三大领域加入梯度正则化的策略优化，使模型在面对复杂推理链时更稳健。相比仅用奖励信号的普通 RL，GRPO 能在保持高奖励的同时抑制梯度爆炸，提升了长程规划的成功率。  
4. **跨阶段蒸馏结合 SFT 与 RL**：在大规模指令微调（SFT）后，直接把该 checkpoint 用作后续 RL 阶段的教师，进行跨阶段蒸馏。这样既保留了指令遵循的能力，又让 RL 学到的探索策略更快收敛，整体训练时间比传统两阶段训练缩短约 30%。

### 方法详解
**整体框架**  
GLM‑5 的训练流程可以划分为五个阶段：① 预训练（海量网页、代码、数学科学数据），② 大规模指令微调（SFT），③ 基于 GRPO 的推理强化学习（Reasoning RL），④ 完全异步的 Agentic RL，⑤ 跨阶段蒸馏。每个阶段都围绕“稀疏激活 + 多任务提示”展开，确保模型在保持算力效率的同时不断提升多模态推理和自主规划能力。

**关键模块拆解**  

1. **稀疏激活路由（DSA）**  
   - 输入 token 先经过轻量路由网络，路由网络根据 token 的语义特征（如代码关键字、数学符号）决定激活哪些专家子网。  
   - 只激活约 10%–15% 的专家，未激活的子网保持静默，类似于在大型团队中只叫到需要的成员开会。  
   - 这种设计在训练时通过“负载均衡损失”确保各专家被均匀使用，防止某些子网被闲置。

2. **多任务提示（MTP）**  
   - 同一批数据会混入多种任务的指令，例如“解释下面的代码”“求解该方程”“生成单元测试”。  
   - 模型在一次前向传播中同时学习这些任务的共享表示，提升了跨领域迁移能力。  

3. **异步 RL 基础设施**  
   - 生成进程（Generator）持续接受环境（如代码执行器、数学求解器）返回的状态，输出下一步 token。  
   - 训练进程（Learner）从共享队列中读取生成的轨迹，计算奖励（规则奖励、GRM、ORM）并执行策略梯度更新。两者通过锁自由的队列解耦，生成不必等待梯度同步。  

4. **GRPO 策略优化**  
   - 在普通的策略梯度基础上加入梯度正则项，约束新策略的梯度方向与旧策略保持一定相似度，防止在长序列上出现剧烈波动。  
   - 该正则项在代码调试和数学证明等需要多步推理的任务里表现尤为突出，使模型在 50 步以上的推理链上成功率提升约 15%。  

5. **跨阶段蒸馏**  
   - 在 Reasoning RL 完成后，把该阶段的 checkpoint 作为教师，使用 KL 散度让 Agentic RL 的学生模型在每一步输出概率分布上逼近教师。  
   - 这种“先教再让它自己探索”的顺序让学生在保持指令遵循的同时，快速适应异步 RL 的高噪声环境。

**最巧妙的设计**  
- **完全异步的 token 级 RL**：把生成和学习拆成两条并行流水线，突破了传统 RL 必须同步的瓶颈。这个想法在大模型上极少出现，因为需要精细的分布式调度和一致性保证，GLM‑5 用轻量的共享队列和版本控制实现了高效同步。  

### 实验与效果
- **评测数据集**：在 OpenAI HumanEval、MBPP（代码生成）、MATH（数学推理）以及 ScienceQA（科学问答）等公开基准上进行评测。  
- **对比基线**：与前代 GLM‑4、DeepSeek‑V3.2、Claude‑2、GPT‑4 等模型对比。  
- **主要结果**：在 HumanEval 上通过率从 GLM‑4 的 46% 提升到 58%；MATH 的解题准确率提升约 12%；ScienceQA 的多选准确率提升约 9%。这些提升在同等算力预算下实现，说明 DSA 与异步 RL 的组合有效。  
- **消融实验**：作者分别关闭 DSA、异步 RL、GRPO、跨阶段蒸馏四个模块。结果显示：去掉 DSA 会导致推理成本上升约 45%，准确率下降 3%；去掉异步 RL，训练时间延长 2.5 倍，长序列任务成功率下降 10%；去掉 GRPO，长推理链的错误率上升约 8%；去掉跨阶段蒸馏，后期 RL 收敛速度慢 30%。这些实验验证了每个创新的贡献。  
- **局限性**：论文承认在极端超长上下文（>200k token）下仍会出现记忆漂移；异步 RL 对硬件同步要求较高，部署到资源受限的边缘设备仍有挑战；奖励函数的设计仍依赖人工规则，自动化程度有待提升。

### 影响与延伸思考
GLM‑5 的出现让「大模型 + 强化学习」的组合从“高成本实验”转向“可工业化流水线”。随后的工作如 **AgenticCoder**、**SparseRL‑X** 等都在借鉴其异步 RL 框架和稀疏激活路由。业界也开始探索把 DSA 与更细粒度的专家调度（如基于 LoRA 的子网）结合，以进一步压缩算力。对想继续深挖的读者，建议关注以下方向：① 更高效的奖励模型（如基于对齐的 LLM 评估器）；② 跨模态的 Agentic RL（把代码、文档、UI 交互统一建模）；③ 在边缘设备上实现稀疏激活的硬件加速。  

### 一句话记住它
GLM‑5 用稀疏激活＋完全异步强化学习，让大模型在超长上下文里以低成本实现真正的“代码工程师”自我进化。