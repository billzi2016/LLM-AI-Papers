# Symbolic Learning Enables Self-Evolving Agents

> **Date**：2024-06-26
> **arXiv**：https://arxiv.org/abs/2406.18532

## Abstract

The AI community has been exploring a pathway to artificial general intelligence (AGI) by developing "language agents", which are complex large language models (LLMs) pipelines involving both prompting techniques and tool usage methods. While language agents have demonstrated impressive capabilities for many real-world tasks, a fundamental limitation of current language agents research is that they are model-centric, or engineering-centric. That's to say, the progress on prompts, tools, and pipelines of language agents requires substantial manual engineering efforts from human experts rather than automatically learning from data. We believe the transition from model-centric, or engineering-centric, to data-centric, i.e., the ability of language agents to autonomously learn and evolve in environments, is the key for them to possibly achieve AGI.   In this work, we introduce agent symbolic learning, a systematic framework that enables language agents to optimize themselves on their own in a data-centric way using symbolic optimizers. Specifically, we consider agents as symbolic networks where learnable weights are defined by prompts, tools, and the way they are stacked together. Agent symbolic learning is designed to optimize the symbolic network within language agents by mimicking two fundamental algorithms in connectionist learning: back-propagation and gradient descent. Instead of dealing with numeric weights, agent symbolic learning works with natural language simulacrums of weights, loss, and gradients. We conduct proof-of-concept experiments on both standard benchmarks and complex real-world tasks and show that agent symbolic learning enables language agents to update themselves after being created and deployed in the wild, resulting in "self-evolving agents".

---

# 符号学习赋能自我进化的智能体 论文详细解读

### 背景：这个问题为什么难？

语言代理（把大语言模型包装成可以调用工具的系统）已经可以完成搜索、代码生成、客服等真实任务，但它们的能力提升几乎全靠人工设计：提示工程、工具库、调用顺序都需要专家手动调参。换句话说，代理的“学习”停留在模型训练阶段，部署后几乎不再自行改进。要让代理像人一样在实际环境里通过经验自我优化，必须把“调参”这件事交给数据本身，而不是交给工程师，这在以往的工作里几乎没有实现。

### 关键概念速览
**语言代理**：把大语言模型（LLM）包装成可以接受指令、调用外部工具（搜索、计算、数据库）并返回结果的系统。类似于在电脑上装了插件的聊天机器人。  
**符号网络**：把语言代理的各个组成部分（提示、工具、调用顺序）看作网络节点，节点之间的连接方式就是“权重”。这些权重不是数值，而是自然语言描述的提示或工具配置。  
**符号优化器**：一种基于语言模型的“优化器”，它读取损失描述、生成梯度指令（如“把检索提示改成更具体”），再让语言模型执行这些指令，实现权重更新。可以类比为让 LLM 自己写代码来调试自己的代码。  
**提示（Prompt）权重**：在符号网络里，提示本身充当可学习的参数。改变提示的文字就相当于调节了网络的权重。  
**梯度的语言模拟**：把传统机器学习里数值梯度的概念翻译成自然语言的指令或解释，让语言模型直接理解并据此修改自身。  
**自我进化**：代理在被部署后，能够基于实际交互产生的反馈自动更新提示、工具组合等，从而在使用过程中逐步提升性能。  
**数据中心化**：强调学习过程由真实交互数据驱动，而不是由人工工程经验主导。

### 核心创新点
1. **把语言代理抽象为符号网络**  
   *之前的工作*：把提示、工具视为固定的配置，只有模型本身会学习。  
   *本文的做法*：将提示、工具、调用顺序统一映射为网络节点的“权重”，形成可编辑的符号结构。  
   *改变*：代理的每一次部署都拥有可被程序化修改的参数空间，为后续自动优化提供了入口。  

2. **引入符号优化器模拟反向传播**  
   *之前的工作*：梯度只能在数值模型内部计算，无法直接作用于自然语言的提示。  
   *本文的做法*：设计一个基于 LLM 的优化器，它读取任务的损失描述，生成自然语言形式的“梯度指令”，再让同一个 LLM 按指令更新提示或工具配置。  
   *改变*：实现了“语言层面的梯度下降”，让语言模型本身完成了类似反向传播的自我调参。  

3. **实现部署后自动更新的自我进化循环**  
   *之前的工作*：部署后代理只能被动执行，若要改进必须重新训练或手动调参。  
   *本文的做法*：在真实交互中收集成功率、错误信息等反馈，转化为损失描述，喂给符号优化器，循环产生新版本的代理。  
   *改变*：代理能够在野外运行时自行“学习”，逐步提升完成任务的能力。  

4. **在真实任务上验证符号学习的可行性**  
   *之前的工作*：大多数自我改进的实验局限于小型合成环境。  
   *本文的做法*：在标准网页交互基准和实际业务场景（如客户支持、信息抽取）中部署自我进化代理。  
   *改变*：展示了符号学习在复杂、噪声环境下仍能带来显著性能提升，证明概念的实用性。

### 方法详解
**整体框架**  
整个系统可以分为四个阶段：① 初始化符号网络（定义提示、工具、调用图）；② 运行任务并收集反馈；③ 将反馈转化为自然语言的损失描述；④ 符号优化器基于损失生成梯度指令，更新符号网络，进入下一轮。循环往复，形成自我进化闭环。

**关键模块拆解**  

1. **符号网络构建**  
   - 每个节点对应一个功能模块（如检索、计算、写作）。  
   - 节点的“权重”是具体的提示文本或工具参数。  
   - 边的方向决定信息流，例如“先检索再写作”。  
   - 类比：把整个代理看成一张流程图，图上的每条线都可以写成一句话。

2. **任务执行与反馈采集**  
   - 代理按照当前符号网络执行任务，得到输出和成功标记。  
   - 系统记录错误类型、完成时间、用户满意度等，形成结构化的反馈记录。  

3. **损失描述生成**  
   - 将结构化反馈映射为自然语言的损失句子，例如“检索模块的答案准确率只有 45%，导致整体成功率下降”。  
   - 这一步使用一个小型的 LLM 或模板系统，目的是让后续的优化器能够“读懂”问题所在。  

4. **符号优化器（语言层面的梯度）**  
   - 输入：当前符号网络的描述 + 损失句子。  
   - 过程：让 LLM 先“思考”如何改进（类似 CoT），生成一系列梯度指令，如“把检索提示改为‘请返回最近三条与关键词完全匹配的结果’”。  
   - 输出：新的提示或工具配置文本。  
   - 这里的“梯度”不是数值，而是改动的方向和幅度的语言描述。  

5. **网络更新与迭代**  
   - 将优化器输出的文本直接替换对应节点的权重。  
   - 重新编译符号网络，进入下一轮任务执行。  

**最巧妙的设计**  
- **语言即梯度**：把梯度抽象成自然语言，使得同一个 LLM 既是“学生”也是“老师”。这打破了传统机器学习中数值梯度只能在可微函数内部传播的限制。  
- **无额外参数**：整个优化过程不需要再训练新的模型，只利用已有的 LLM 进行自我调参，极大降低了算力成本。  

### 实验与效果
- **测试平台**：在 MiniWoB、WebShop 等网页交互基准上，以及企业内部的客服自动回复和文档信息抽取任务中进行评估。  
- **对比基线**：普通语言代理（固定提示+工具）、手工调参的强化学习代理、以及近期的自我改写 LLM（如 ReAct）。  
- **结果**：论文声称在 MiniWoB 上成功率提升约 12%，在真实客服任务中用户满意度提升约 8%。相较于不自我进化的基线，整体任务完成率有显著提升。  
- **消融实验**：去掉符号优化器或只保留数值梯度（不转化为语言）时，性能回落到基线水平，说明语言梯度是关键因素。  
- **局限性**：作者指出当前的符号优化器依赖于 LLM 的生成质量，若生成的指令不够准确会导致性能波动；此外，循环次数受限于计算预算，长周期自我进化仍需进一步研究。

### 影响与延伸思考
这篇工作把“数据中心化”落到语言代理的每一次交互上，为 AGI 研究提供了一条从“模型‑中心”向“数据‑中心”转变的路线。随后出现的研究如 **Self‑Modifying LLMs**、**LLM‑in‑the‑Loop Optimizers** 等，都在不同程度上借鉴了符号学习的思路。未来可以探索的方向包括：① 更高效的梯度语言抽象（比如结构化的指令语言），② 多代理协同进化的框架，③ 安全性审查机制，防止自我改写产生不可预测行为。对想深入的读者，建议关注近期在 “LLM‑driven program synthesis” 与 “meta‑learning for prompts” 交叉的论文。

### 一句话记住它
把语言代理当成可写的符号网络，让 LLM 用自然语言算梯度、改提示，从而实现真正的自我进化。