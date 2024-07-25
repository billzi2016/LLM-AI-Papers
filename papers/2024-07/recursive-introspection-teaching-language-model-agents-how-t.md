# Recursive Introspection: Teaching Language Model Agents How to   Self-Improve

> **Date**：2024-07-25
> **arXiv**：https://arxiv.org/abs/2407.18219

## Abstract

A central piece in enabling intelligent agentic behavior in foundation models is to make them capable of introspecting upon their behavior, reasoning, and correcting their mistakes as more computation or interaction is available. Even the strongest proprietary large language models (LLMs) do not quite exhibit the ability of continually improving their responses sequentially, even in scenarios where they are explicitly told that they are making a mistake. In this paper, we develop RISE: Recursive IntroSpEction, an approach for fine-tuning LLMs to introduce this capability, despite prior work hypothesizing that this capability may not be possible to attain. Our approach prescribes an iterative fine-tuning procedure, which attempts to teach the model how to alter its response after having executed previously unsuccessful attempts to solve a hard test-time problem, with optionally additional environment feedback. RISE poses fine-tuning for a single-turn prompt as solving a multi-turn Markov decision process (MDP), where the initial state is the prompt. Inspired by principles in online imitation learning and reinforcement learning, we propose strategies for multi-turn data collection and training so as to imbue an LLM with the capability to recursively detect and correct its previous mistakes in subsequent iterations. Our experiments show that RISE enables Llama2, Llama3, and Mistral models to improve themselves with more turns on math reasoning tasks, outperforming several single-turn strategies given an equal amount of inference-time computation. We also find that RISE scales well, often attaining larger benefits with more capable models. Our analysis shows that RISE makes meaningful improvements to responses to arrive at the correct solution for challenging prompts, without disrupting one-turn abilities as a result of expressing more complex distributions.

---

# 递归自省：教语言模型代理自我提升 论文详细解读

### 背景：这个问题为什么难？

当前的大语言模型（LLM）在一次性给出答案时已经相当强大，但它们很少会在发现错误后主动“回头改”。即使在提示模型“你错了”时，模型往往仍然坚持原答案，缺乏持续改进的能力。根本原因在于训练阶段只让模型学习一次性映射——从提示到答案，而没有让它学会把“失败的尝试”当作新的信息源来重新推理。缺少这种递归的自我审视，使得模型在需要多轮思考、纠错或交互的场景（比如复杂数学、代码调试）里表现不佳，这也正是本文想要突破的瓶颈。

### 关键概念速览
- **递归自省（Recursive Introspection）**：模型在一次生成后，检查自己的输出并基于检查结果再生成一次或多次，类似人类写完答案后再回头检查并改正的过程。  
- **单轮 vs 多轮推理**：单轮指一次前向传播直接得到答案；多轮指模型可以在同一次任务中进行多次交互，每次都把上一次的输出当作新输入。  
- **Markov 决策过程（MDP）**：把一次对话看成状态转移的序列，当前提示是起始状态，模型的每一次生成是一次“动作”，环境反馈（如是否正确）是奖励信号。  
- **在线模仿学习（Online Imitation Learning）**：在训练时让模型不断模仿自己在新状态下的“理想”行为，类似教会模型在每一步都知道该怎么改。  
- **强化学习（Reinforcement Learning）**：模型通过奖励信号学习策略，这里用来鼓励模型在后续轮次产生更正确的答案。  
- **CoT（思维链）**：让模型先写出推理步骤再给出结论，帮助模型把思考过程显式化，便于后续检查。  
- **自我纠错（Self‑Correction）**：模型在发现自己前一步出错后，主动生成修正版本，而不是依赖外部人工标注。  

### 核心创新点
1. **把单轮微调重新表述为多轮 MDP**  
   - 之前的微调大多把“提示 → 答案”当作一次性映射。  
   - 本文把同一个提示视作 MDP 的起始状态，模型的每一次生成视作一次决策，环境反馈（是否正确）作为奖励。  
   - 这样一来，微调目标不再是一次性预测，而是学习在任意中间状态下如何纠错，从而让模型具备递归自省的能力。

2. **基于在线模仿学习的多轮数据采集策略**  
   - 传统数据采集只收集一次性正确答案，缺少错误案例。  
   - 作者让模型先自行尝试解决难题，记录失败的答案，然后在同一提示下让模型生成“纠错”版本，形成“错误 → 修正”对。  
   - 这种自生成的错误-修正对让模型在训练时直接看到自己的缺陷，显著提升了纠错学习的效率。

3. **将强化学习奖励嵌入微调目标**  
   - 仅靠模仿学习会让模型倾向于复制已有的纠错模式，缺乏探索。  
   - 论文在微调损失中加入基于是否得到正确答案的奖励项，鼓励模型在每一轮都争取更高的正确率。  
   - 结果是模型在多轮推理时能够主动尝试不同的思路，而不是死守原来的错误路径。

4. **规模化验证：从 Llama‑2 到 Mistral 的一致提升**  
   - 过去的自我纠错方法往往只在小模型上有效，放大到更大模型会出现“忘记单轮能力”的问题。  
   - RISE 在 Llama‑2、Llama‑3、Mistral 等不同规模的模型上都保持或提升了单轮表现，同时在多轮推理上获得更大收益。  
   - 这说明递归自省的训练框架与模型规模兼容，具备良好的可扩展性。

### 方法详解
**整体思路**：把一次任务看成一个多步的决策过程。先让模型尝试解决任务，记录它的错误；再让模型在同一提示下基于错误进行纠正。整个循环形成训练数据，微调时把每一步都当作 MDP 的状态‑动作对，并加入奖励信号，引导模型学会“发现‑改正”循环。

**步骤拆解**：

1. **初始提示准备**  
   - 给模型一个标准的任务提示（例如一道数学题），这就是 MDP 的起始状态。

2. **第一次生成（尝试）**  
   - 让模型在常规的单轮设置下生成答案。此时模型并未接受任何纠错信息，输出可能正确也可能错误。

3. **错误检测与环境反馈**  
   - 使用外部评估器（如数学求解器、代码单元测试）判断答案是否正确。若错误，标记为“负奖励”，若正确则给出“正奖励”。这一步相当于环境向模型提供即时反馈。

4. **纠错提示构造**  
   - 将原始提示、模型的错误答案以及环境反馈拼接成新的多轮提示，明确告诉模型“你刚才的答案有问题”。这一步类似于人类在纸上写下“我算错了，哪里错了？”的自我提醒。

5. **第二次生成（纠正）**  
   - 在新提示下让模型重新生成答案。此时模型可以利用自己的错误信息和环境提示进行重新推理。

6. **多轮循环（可选）**  
   - 如果第二次仍未通过评估，可继续重复步骤 3‑5，形成任意长度的纠错链。实际训练中作者通常限制在 2‑3 轮，以平衡计算成本。

7. **数据收集与标注**  
   - 将每一次状态‑动作对（提示、模型输出、奖励）保存下来，形成“递归自省数据集”。关键是保留错误答案本身，而不是直接丢弃。

8. **微调目标设计**  
   - 损失函数由两部分组成：  
     a) **模仿损失**：让模型在每个状态下尽可能产生“理想的纠正答案”。  
     b) **奖励加权**：根据环境给出的正负奖励，对模仿损失进行加权，奖励正确的纠正、惩罚仍然错误的输出。  
   - 这种设计让模型在学习纠错的同时，也被鼓励去探索更有效的纠正策略。

9. **训练与推理**  
   - 使用上述数据对 LLM 进行常规的梯度微调。推理时，模型可以自行进入递归自省模式：先给出初始答案，再在内部循环中检查并改进，直至达到预设的轮数或满足评估标准。

**巧妙之处**：  
- **自生成错误**：不需要人工标注大量错误案例，模型自己产生错误，再自行纠正，极大降低数据成本。  
- **MDP 视角**：把单轮任务升级为序列决策，使得强化学习的奖励机制自然嵌入微调过程。  
- **兼容单轮能力**：通过奖励加权而非直接替换原始答案，确保模型在只需要一次输出的场景下仍保持原有水平。

### 实验与效果
- **测试任务**：主要在数学推理基准（如 GSM‑8K、MATH）上评估，还包括少量代码调试任务。  
- **对比基线**：包括普通单轮微调、CoT（思维链）+ 单轮微调、以及已有的自我纠错方法（如 Self‑Consistency）。  
- **论文声称**：在相同的推理计算预算下，RISE 能让 Llama‑2、Llama‑3、Mistral 等模型在多轮推理上显著超越所有单轮基线，且提升幅度随模型规模增大而更明显。  
- **消融实验**：作者分别去掉奖励加权、去掉错误‑纠正对的多轮采集、只保留单轮微调，结果显示每一项都对最终性能有贡献，尤其奖励机制对高难度题目的提升最为关键。  
- **局限性**：RISE 仍依赖外部评估器提供即时反馈；在没有明确奖励信号的开放域对话或创意写作中，递归自省的收益尚未得到验证。作者也指出，过多轮的递归可能导致推理时间线性增长，需要在实际系统中权衡。

### 影响与延伸思考
RISE 把“自我纠错”从概念验证提升到可系统化的训练框架，打开了让大模型具备“反思—改进”循环的可能。后续工作已经开始在以下方向借鉴其思路：  
- **开放域对话**：利用用户反馈或情感评分作为环境奖励，让聊天机器人在对话中自行纠正不恰当回复。  
- **程序合成**：把单元测试结果当作奖励，让代码生成模型在多轮迭代中自动调试。  
- **元学习**：将递归自省视为一种元学习任务，让模型学会快速适应新任务的纠错策略。  
如果想进一步深入，可以关注 **“自监督纠错数据生成”** 与 **“基于人类反馈的递归强化学习”** 两条路线，这两者正是 RISE 之后的自然延伸。

### 一句话记住它
让语言模型把每一次错误当作新线索，像人类一样在同一道题上反复检查并改进——这就是 RISE 的核心。