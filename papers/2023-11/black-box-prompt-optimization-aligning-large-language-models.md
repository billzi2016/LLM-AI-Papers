# Black-Box Prompt Optimization: Aligning Large Language Models without   Model Training

> **Date**：2023-11-07
> **arXiv**：https://arxiv.org/abs/2311.04155

## Abstract

Large language models (LLMs) have shown impressive success in various applications. However, these models are often not well aligned with human intents, which calls for additional treatments on them; that is, the alignment problem. To make LLMs better follow user instructions, existing alignment methods primarily focus on further training them. However, the extra training of LLMs is usually expensive in terms of GPU computing; even worse, some LLMs are not accessible for user-demanded training, such as GPTs. In this work, we take a different perspective -- Black-Box Prompt Optimization (BPO) -- to perform alignments. The idea is to optimize user prompts to suit LLMs' input understanding, so as to best realize users' intents without updating LLMs' parameters. BPO leverages human preferences to optimize prompts, thus making it superior to LLM (e.g., ChatGPT) as a prompt engineer. Moreover, BPO is model-agnostic, and the empirical results demonstrate that the BPO-aligned ChatGPT yields a 22% increase in the win rate against its original version and 10% for GPT-4. Notably, the BPO-aligned LLMs can outperform the same models aligned by PPO and DPO, and it also brings additional performance gains when combining BPO with PPO or DPO. Code and datasets are released at https://github.com/thu-coai/BPO.

---

# 黑盒提示优化：在不进行模型训练的情况下对齐大型语言模型 论文详细解读

### 背景：这个问题为什么难？
大型语言模型（LLM）在生成文本、写代码等任务上已经很强，但它们常常会偏离用户真实意图，出现不恰当或不安全的回答。传统的对齐手段几乎都要求在原模型上继续微调——比如使用强化学习（RLHF）或直接的指令微调。这类方法需要大量 GPU 计算，成本高昂，而且很多商业模型（如 OpenAI 的 GPT 系列）根本不开放参数，普通研究者和开发者根本动手不了。于是出现了一个尴尬的局面：模型好用但难以让它们“听话”，而让它们听话的唯一途径又被高昂的算力和封闭的接口挡住了。

### 关键概念速览
**黑盒模型**：指我们只能通过输入输出交互使用的模型，内部参数和结构对使用者不可见。就像只能在柜子外面投硬币让机器吐出糖果的自动售货机。  
**提示（Prompt）**：给模型的文字指令或上下文，决定模型的思考方向。相当于对话中的开场白，开得好，后面的聊天更顺畅。  
**人类偏好数据**：收集的“哪种回答更好”标签，通常由人工评审给出，用来教系统判断答案质量。类似于让人挑选最满意的菜品来训练推荐系统。  
**序列到序列（Seq2Seq）模型**：一种把一段文字映射成另一段文字的网络结构，常用于机器翻译。这里把原始提示当作“源语言”，把优化后的提示当作“目标语言”。  
**PPO（近端策略优化）**：强化学习里一种常用的算法，用来在保持旧策略不变太多的前提下提升新策略的奖励。可以想象为在不把车子开得太离谱的情况下，慢慢调校驾驶技巧。  
**DPO（直接偏好优化）**：直接利用人类偏好对模型输出进行梯度更新的办法，省去奖励模型的训练环节。像是直接给厨师“这道菜要更咸”而不是先让他学会怎么评估咸淡。  

### 核心创新点
1. **从微调模型转向微调提示**：传统对齐把目标放在模型参数上，这篇论文把目标搬到提示本身，用一个小型的 Seq2Seq 网络来生成更“懂模型”的指令。这样既省掉了巨大的算力，又绕开了封闭模型的限制。  
2. **把人类偏好直接用于提示搜索**：以前人类偏好主要用来训练奖励模型或直接更新 LLM 参数。这里把偏好当作评价函数，驱动提示生成器迭代改进，使得每一次生成的提示都更可能让模型给出用户满意的答案。  
3. **模型无关的黑盒优化框架**：方法只依赖于模型的输入输出接口，没有任何假设关于模型内部结构或训练方式。实验表明，同一套提示优化器可以同时提升 ChatGPT、GPT‑4 等不同模型的表现。  
4. **与传统 RL 对齐方法的互补**：作者把 BPO 与 PPO、DPO 组合使用，发现两者相加的效果超过单独使用任意一种。提示层面的改进为后续的参数层面微调提供了更好的起点。  

### 方法详解
整体思路可以拆成三步：**收集偏好、训练提示生成器、迭代评估**。下面用“写信”这个日常场景来比喻每一步。

1. **收集偏好**  
   研究者先准备一批任务（比如回答问题、写代码），让模型在原始提示下生成答案。随后让人工评审对同一任务的两种答案进行二选一，标记哪一个更符合意图。这样得到的“哪种回答更好”数据，就是后面所有优化的基准。

2. **训练提示生成器（Seq2Seq）**  
   把原始提示视作信的“收件人地址”，让一个小型的 Seq2Seq 网络学习把它“翻译”成更适合模型的版本。训练目标是：**在同一任务上，用新提示得到的答案要在偏好评审中赢得更多票**。实现上，作者把每一次生成的提示喂给黑盒 LLM，得到答案后用偏好评分（比如 0/1 交叉熵）来计算损失，反向传播更新 Seq2Seq 参数。因为只更新提示生成器，算力需求只相当于普通的文本生成模型训练，远低于对 LLM 本体的微调。

3. **迭代评估与采样**  
   训练过程中会不断采样新的提示并让模型生成答案，交给偏好评审打分。若新提示的胜率提升，就保留；否则继续搜索。作者采用了类似强化学习的“策略梯度”思想，但梯度直接作用在提示生成器上，而不是模型本身。整个循环在几千次采样后即可收敛到一个稳健的提示集合。

**关键细节**  
- **黑盒调用**：所有交互都通过标准的 API（如 `chat.completions`）完成，代码里没有任何模型内部的调用。  
- **偏好函数**：作者使用了对比式的二元交叉熵损失，把“更好”标记转化为概率目标，避免了需要训练单独的奖励模型。  
- **采样策略**：在生成提示时加入温度噪声，保持一定的探索性，防止陷入局部最优。  
- **多模型共享**：同一个提示生成器可以在不同 LLM 上重复使用，只要把对应的 API 替换即可，这正是“模型无关”的来源。

最让人意外的地方是：**只调一个几百万参数的 Seq2Seq，就能让数十亿参数的 LLM 在特定任务上提升 10%‑20% 的胜率**。这说明提示本身的表达方式对模型行为的影响比我们想象的更大。

### 实验与效果
- **测试任务**：论文选用了公开的指令遵循基准（如 Alpaca、OpenAI Evals）以及一些对话安全评测，覆盖问答、代码生成、情感引导等多种场景。  
- **对比基线**：包括原始模型（未对齐）、使用 PPO 微调的模型、使用 DPO 微调的模型，以及最直接的手工提示改写。  
- **主要结果**：在 ChatGPT 上，BPO 生成的提示使胜率提升约 **22%**；在更强大的 GPT‑4 上提升约 **10%**。更惊人的是，BPO‑aligned 的模型在同等评测下跑赢了 PPO 和 DPO 单独微调的版本。把 BPO 与 PPO 或 DPO 组合后，整体提升幅度进一步扩大（具体数字论文中给出）。  
- **消融实验**：作者分别去掉了偏好驱动的损失、温度采样、以及多轮迭代，发现每一项都对最终提升有显著贡献，尤其是去掉偏好损失后提升几乎消失，说明人类偏好是核心驱动力。  
- **局限性**：实验主要集中在指令遵循和安全对齐任务，尚未验证在大规模开放域对话或长文本生成上的效果；此外，提示生成器本身仍需要一定的训练数据和算力，虽然远低于全模型微调，但对资源极度匮乏的场景仍有门槛。  

### 影响与延伸思考
这篇工作打开了“提示层面对齐”的新视角，随后出现了多篇围绕黑盒优化的研究，比如利用进化算法搜索提示、把多模态输入也纳入提示优化、以及在检索增强生成（RAG）系统中加入 BPO 思路。对工业界来说，BPO 为那些只能通过 API 调用的商用模型提供了一条成本可控的提升路径。想进一步深入，可以关注以下方向：  
- **跨任务通用提示生成器**：是否能训练一个一次性适用于所有指令的提示优化器？  
- **自动化偏好收集**：利用用户交互日志自动生成偏好标签，降低人工标注成本。  
- **与检索/工具使用结合**：把提示优化与外部知识检索或工具调用一起考虑，提升模型在复杂任务中的表现。  

### 一句话记住它
只要把“怎么叫模型听话”这件事交给一个小的提示生成网络，就能在不动大模型参数的前提下，让它们显著更符合人类意图。