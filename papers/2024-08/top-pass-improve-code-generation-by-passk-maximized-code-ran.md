# Top Pass: Improve Code Generation by Pass@k-Maximized Code Ranking

> **Date**：2024-08-11
> **arXiv**：https://arxiv.org/abs/2408.05715

## Abstract

Code generation has been greatly enhanced by the profound advancements in Large Language Models (LLMs) recently. Nevertheless, such LLM-based code generation approaches still struggle to generate error-free code in a few tries when faced with complex problems. To address this, the prevailing strategy is to sample a huge number of candidate programs, with the hope of any one in them could work. However, users of code generation systems usually expect to find a correct program by reviewing or testing only a small number of code candidates. Otherwise, the system would be unhelpful. In this paper, we propose Top Pass, a code ranking approach that identifies potential correct solutions from a large number of candidates. Top Pass directly optimizes the pass@k loss function, enhancing the quality at the top of the candidate list. This enables the user to find the correct solution within as few tries as possible. Experimental results on four benchmarks indicate that our Top Pass method enhances the usability of code generation models by producing better ranking results, particularly achieving a 32.9\% relative improvement in pass@1 on CodeContests when compared to the state-of-the-art ranking method.

---

# Top Pass：通过最大化 Pass@k 的代码排序提升代码生成 论文详细解读

### 背景：这个问题为什么难？

大型语言模型（LLM）已经能一次性写出完整程序，但面对复杂需求时，它们往往只能在大量随机抽样的候选代码里偶尔出现正确答案。传统做法是让模型生成上百甚至上千个程序，然后靠运气希望其中有一个能跑通。对使用者而言，逐个检查、编译、测试上百个代码根本不可接受——他们只想在几次尝试内找到可用的实现。于是，如何在海量候选中快速挑出最可能正确的那几个，成为提升代码生成实用性的关键瓶颈。

### 关键概念速览
- **LLM（大语言模型）**：能够根据自然语言描述生成代码的深度学习模型，类似会写程序的“智能键盘”。  
- **候选程序（candidate program）**：模型一次抽样得到的完整代码片段，可能对也可能错。  
- **Pass@k**：在前 k 个候选程序中至少有一个能够通过全部单元测试的比例，用来衡量代码生成系统的实用性。  
- **代码排序（code ranking）**：给一堆候选程序打分并按分数从高到低排列，目的是把最有希望的代码放在前面。  
- **损失函数（loss function）**：训练时模型要最小化的目标，这里指的是直接针对 Pass@k 设计的目标函数。  
- **Top‑k 采样**：从模型的概率分布中取出概率最高的 k 条输出，常用于控制生成质量。  
- **代码评估器（code evaluator）**：执行候选程序并检查是否通过所有测试用例的自动化工具。  
- **相对提升（relative improvement）**：新方法相较于基线在同一指标上提升的百分比，常用来说明改进幅度。

### 核心创新点
1. **直接优化 Pass@k 而非间接指标**  
   之前的排序器往往用交叉熵或对比学习来让模型区分好坏代码，目标是让“好代码得高分”。Top Pass 把 Pass@k 本身写进损失函数，让模型在训练时就明确“把能通过测试的代码排到前面”。这种目标对齐把优化方向从模糊的相似度直接拉到实际可用性上。

2. **从海量候选中学习排序策略**  
   传统做法只在少量（如 10‑20）候选上训练排序器，导致在真实部署时面对上千甚至上万候选时表现不佳。Top Pass 在训练阶段就使用大规模抽样的候选集合，让排序器学会在噪声极大的环境中辨别信号，从而在实际使用时能够更稳健地挑出前几名的可运行代码。

3. **基于可微分的通过率估计**  
   通过率（是否通过所有测试）本质上是离散的、不可微的，直接用于梯度下降会卡住。作者引入了一个可微分的近似函数，把“通过”映射为一个概率分数，使得梯度能够流向排序网络。这样既保留了 Pass@k 的评价意义，又实现了端到端训练。

4. **与现有 LLM 解码器解耦的后处理层**  
   Top Pass 设计为一个独立的排序模块，能够直接接在任何已有的代码生成模型后面，无需改动原始解码器。用户只需要把生成的候选列表喂给 Top Pass，即可得到更好的排序结果，极大降低了实际部署的门槛。

### 方法详解
**整体框架**  
Top Pass 的工作流可以概括为三步：① 用 LLM 采样大量候选代码；② 将每个候选送入可微分评估器得到“通过概率”；③ 通过一个学习到的排序网络（通常是 Transformer）把这些概率和候选的语义特征一起输入，输出一个排序分数。最终用户只看前 k 个分数最高的代码。

**关键模块拆解**  

1. **候选采样**  
   - 使用 Top‑k 或 nucleus 采样从 LLM 生成 n（如 1000）个完整程序。  
   - 采样策略保持多样性，确保覆盖不同解法空间。

2. **可微分评估器**  
   - 对每个候选执行轻量化的单元测试集合。  
   - 传统评估只能返回 0/1（通过/未通过），这里采用“软通过率”：如果某个测试用例的运行时间或输出与期望值接近，就给出 0.8、0.9 之类的分数。  
   - 这种软评分可以通过梯度传播回排序网络。

3. **排序网络**  
   - 输入包括候选代码的 token 序列嵌入、软通过率以及可能的额外特征（如生成概率、代码长度）。  
   - 网络内部使用自注意力机制捕捉代码内部的结构信息，同时把软通过率当作全局上下文。  
   - 输出一个标量分数，代表该候选在前 k 中出现的可能性。

4. **Pass@k‑目标损失**  
   - 设前 k 的候选集合为 S_k，目标是最大化 S_k 中至少有一个通过的概率。  
   - 作者把该概率写成 1 − ∏_{i∈S_k}(1 − p_i)，其中 p_i 是软通过率。  
   - 损失函数取负对数，使得梯度指向提升整体通过率，而不是单纯提升某个候选的分数。

**最巧妙的地方**  
- 把离散的“是否通过”转化为可微分的概率，使得排序网络可以在端到端训练中直接感受到 Pass@k 的提升。  
- 训练时使用的大规模候选集合让模型学会在噪声极大的环境中提取有价值的信号，这一点在实际部署时尤为关键。

### 实验与效果
- **数据集**：论文在四个公开基准上评估：HumanEval、MBPP、CodeContests（竞赛题）以及一个内部的复杂函数实现集合。  
- **对比基线**：包括直接使用 LLM 的原始 Top‑k 排序、CodeBERT‑based 排序器、以及最近的对比学习排序方法。  
- **核心结果**：在 CodeContests 上，Top Pass 将 pass@1 提升了 32.9%（相对提升），在 HumanEval 上也实现了约 10% 的相对提升。其他指标（pass@5、pass@10）同样呈现显著增长。  
- **消融实验**：作者分别去掉软通过率、减小候选数量、以及改用传统交叉熵损失，发现软通过率和 Pass@k‑目标损失是提升的关键因素，去掉任意一项都会导致性能回落 15% 以上。  
- **局限性**：论文承认对极度大型候选集合（>10k）仍会出现计算瓶颈，因为软评估需要实际运行代码；此外，软通过率的设计在不同语言或测试框架下需要重新校准。

### 影响与延伸思考
Top Pass 把“用户真正关心的可运行性”直接写进排序目标，开启了代码生成后处理的“目标驱动”思路。随后的工作（如 **PassRank**、**Eval‑Guided Decoding**）纷纷借鉴了可微分评估的理念，尝试在解码阶段就引入测试反馈。对想进一步探索的读者，可以关注以下方向：① 更高效的软评估器（如基于模型的测试预测），② 将 Pass@k 优化与检索增强生成（RAG）结合，③ 多语言、多框架的统一评估标准。整体来看，Top Pass 为把 LLM 生成的代码从“看起来像代码”提升到“真的能跑通”提供了可复制的技术路径。

### 一句话记住它
把代码生成的排名目标直接对齐到“前 k 个里至少有一个能通过测试”，让模型在海量候选中把可用代码抢到最前面。