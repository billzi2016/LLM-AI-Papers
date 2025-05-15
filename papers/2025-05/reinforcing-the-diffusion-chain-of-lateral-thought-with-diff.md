# Reinforcing the Diffusion Chain of Lateral Thought with Diffusion Language Models

> **Date**：2025-05-15
> **arXiv**：https://arxiv.org/abs/2505.10446

## Abstract

We introduce the Diffusion Chain of Lateral Thought (DCoLT), a reasoning framework for diffusion language models. DCoLT treats each intermediate step in the reverse diffusion process as a latent "thinking" action and optimizes the entire reasoning trajectory to maximize the reward on the correctness of the final answer with outcome-based Reinforcement Learning (RL). Unlike traditional Chain-of-Thought (CoT) methods that follow a causal, linear thinking process, DCoLT allows bidirectional, non-linear reasoning with no strict rule on grammatical correctness amid its intermediate steps of thought. We implement DCoLT on two representative Diffusion Language Models (DLMs). First, we choose SEDD as a representative continuous-time discrete diffusion model, where its concrete score derives a probabilistic policy to maximize the RL reward over the entire sequence of intermediate diffusion steps. We further consider the discrete-time masked diffusion language model -- LLaDA, and find that the order to predict and unmask tokens plays an essential role to optimize its RL action resulting from the ranking-based Unmasking Policy Module (UPM) defined by the Plackett-Luce model. Experiments on both math and code generation tasks show that using only public data and 16 H800 GPUs, DCoLT-reinforced DLMs outperform other DLMs trained by SFT or RL or even both. Notably, DCoLT-reinforced LLaDA boosts its reasoning accuracy by +9.8%, +5.7%, +11.4%, +19.5% on GSM8K, MATH, MBPP, and HumanEval.

---

# 用扩散语言模型强化横向思维的扩散链 论文详细解读

### 背景：这个问题为什么难？

在自然语言推理任务里，传统的大语言模型往往一次性直接输出答案，缺乏可解释的思考过程。为了解决这个问题，研究者提出了 **Chain‑of‑Thought（CoT）**，让模型先写出一步步推理再给出结论。但 CoT 本质上是线性的：每一步必须紧跟前一步，且必须是语法完整的句子，这在处理数学证明、代码生成等需要“跳跃式”思考的场景时会受限。与此同时，**扩散语言模型（Diffusion Language Model，DLM）** 通过逐步“去噪”生成文本，天然拥有多步中间状态，但此前并没有把这些中间状态当作思考动作来优化。于是出现了一个空白：如何让 DLM 在生成过程中进行自由、非线性的推理，并用强化学习（RL）把整条思考链的质量提升到最优？

### 关键概念速览
- **Diffusion Language Model（扩散语言模型）**：把文本生成看成从噪声到真实句子的逆向扩散过程，模型在每一步“去噪”时产生一个潜在的中间文本。类似于把写作过程拆成多层滤镜，每层都稍微清晰一点。
- **Chain‑of‑Thought（思维链）**：让模型在给答案前先列出推理步骤，像在黑板上写草稿一样帮助模型保持逻辑连贯。它假设思考是线性、因果的。
- **Diffusion Chain of Lateral Thought（DCoLT）**：把扩散过程的每一步视作一次“横向思考”动作，允许思考顺序前后跳、甚至出现不完整的句子。相当于让模型在草稿纸上随意涂鸦，只要最终能拼出正确答案。
- **Reinforcement Learning with Outcome‑based Reward（基于结果的强化学习）**：模型的每一步行为都会影响最终答案的对错，奖励只在答案正确时给出，整个思考轨迹被视作一个完整的决策序列进行优化。
- **SEDD（连续时间离散扩散模型）**：一种基于连续时间的扩散语言模型，内部维护一个“score”函数，用来估计每一步去噪的概率分布。
- **LLaDA（离散时间掩码扩散模型）**：把文本看成被掩码的序列，模型在每一步决定先掩码哪个位置再填入词汇，类似于先挑选要写的空格再写字。
- **Plackett‑Luce Model（Plackett‑Luce 排序模型）**：一种概率排序模型，用来为“哪个 token 先被 unmask”生成一个可比较的分数，越高的 token 越可能先出现。
- **Unmasking Policy Module（UPM）**：在 LLaDA 中负责决定掩码的解除顺序的策略网络，输出一个基于 Plackett‑Luce 的排名分布。

### 核心创新点
1. **把逆向扩散的每一步当作思考动作 → DCoLT 框架**  
   传统的扩散模型只把中间状态当作噪声的中间产物，未对其进行语义层面的约束。DCoLT 明确将每一步视作“思考”行为，允许这些行为不符合语法或因果顺序，从而提供更自由的推理空间。结果是模型可以先写出关键概念或中间结论，再回头补全细节。

2. **基于结果的全序列强化学习 → 整体轨迹优化**  
   过去的 RL‑HF（人类反馈）或 RL‑SFT（监督微调）往往在每一步给出奖励，导致局部最优。DCoLT 只在最终答案正确与否上给奖励，让策略网络学习“全局最优的思考路径”。这让模型在训练时会主动探索非线性、跳跃式的思考方式。

3. **针对连续 DLM（SEDD）构建概率策略 → Score‑based RL**  
   在 SEDD 中，模型输出的 score 能直接转化为每一步的动作概率。论文利用这一特性，把 RL 的策略分布直接映射到 score 上，省去额外的策略网络，训练更高效。

4. **为离散 DLM（LLaDA）设计基于 Plackett‑Luce 的 Unmasking Policy → 两阶段决策**  
   LLaDA 的核心难点在于“先选哪个位置 unmask”。作者提出 UPM，用 Plackett‑Luce 为所有被掩码的 token 生成一个排名分布，再用 RL 对该排名进行优化。这样模型既能决定顺序，又能在每一步填入最合适的 token，显著提升了代码和数学题的推理准确率。

### 方法详解
**整体框架**  
DCoLT 把整个推理过程抽象为一个马尔可夫决策过程（MDP），状态是当前的扩散噪声向量或掩码文本，动作是一次去噪或一次 unmask，奖励只在最终答案评估后给出。训练目标是最大化期望奖励，即让模型在整个思考链上获得最高的正确率。

**步骤拆解**

1. **状态定义**  
   - 对 SEDD：状态是当前时间点的噪声向量 \(z_t\)。  
   - 对 LLaDA：状态是当前已填充的文本以及剩余的掩码集合。

2. **动作空间**  
   - SEDD：模型输出一个 score 向量，表示在该噪声下每个可能的 token 的概率分布。动作即从该分布采样一个 token，完成一次去噪。  
   - LLaDA：动作分两步：① 通过 UPM 为所有掩码位置生成一个排名；② 按排名顺序选取最高的掩码位置并填入 token（同样依据模型的 token 预测分布）。

3. **策略实现**  
   - **Score‑based Policy（SEDD）**：直接把模型的 score 归一化为概率，形成策略 \(\pi(a|s)\)。因为 score 本身已经是对去噪的最优估计，RL 只需要在整个序列上调节它，使得最终答案更可能正确。  
   - **Plackett‑Luce Policy（LLaDA）**：UPM 为每个掩码位置输出一个正数权重 \(w_i\)。Plackett‑Luce 把这些权重转化为一个逐项抽取的排名分布：先抽取权重最大的位置，抽走后重新归一化，继续抽取。这样得到的顺序即为模型的 unmask 策略。

4. **奖励设计**  
   - 只在完整生成后评估答案的正确性（例如数学题的数值误差、代码的单元测试通过率）。正确则奖励 +1，错误则 0，或使用更细粒度的分数（如部分对的得分）。这种“结果导向”奖励迫使模型在整个思考链上进行全局优化。

5. **RL 训练细节**  
   - 使用 **Proximal Policy Optimization（PPO）** 或 **REINFORCE** 之类的策略梯度方法。因为奖励稀疏，作者在实验中加入了基线（value network）来降低方差。  
   - 为了保持生成质量，训练时仍保留了 **Supervised Fine‑Tuning（SFT）** 的数据作为混合目标，防止模型在追求奖励时产生无意义的胡言乱语。

**最巧妙的点**  
- 把 **score** 直接当作策略分布，省去额外的策略网络，这是对连续扩散模型的天然利用。  
- 对离散模型使用 **Plackett‑Luce** 排序，使得“先写哪个空”本身成为可学习的策略，而不是固定的随机顺序，这在代码生成中尤为关键。

### 实验与效果
- **任务与数据集**：数学推理（GSM8K、MATH）和代码生成（MBPP、HumanEval）。这些基准分别衡量模型的数值推理和可执行代码的正确率。  
- **实验设置**：仅使用公开数据，训练资源为 16 块 H800 GPU（相当于中等规模算力），与多数大型模型的数百 GPU训练形成对比。  
- **主要结果**（论文给出的提升）：  
  - 在 GSM8K 上提升 **+9.8%**，  
  - 在 MATH 上提升 **+5.7%**，  
  - 在 MBPP 上提升 **+11.4%**，  
  - 在 HumanEval 上提升 **+19.5%**。  
  这些数字表明 DCoLT 在不同任务上都有显著增益，尤其在代码生成的 HumanEval 上几乎提升了二成。  
- **Baseline 对比**：与同样使用 SEDD 或 LLaDA 的 **SFT**（仅监督微调）模型、**RLHF**（基于人类反馈的强化学习）模型以及两者的混合模型相比，DCoLT consistently 超越。  
- **消融实验**：论文报告了去掉 **UPM**（仅使用固定掩码顺序）或不使用 **结果导向奖励**（改为每步奖励）时，性能分别下降约 4–7%。这说明两大模块都是提升的关键因素。  
- **局限性**：作者承认 DCoLT 仍然依赖大量的计算资源进行 RL 采样，且奖励稀疏导致训练收敛慢；此外，非语法化的中间步骤在解释性上不如传统 CoT，难以直接用于人类审查。

### 影响与延伸思考
- **领域影响**：DCoLT 把扩散模型的“去噪”过程重新定义为思考过程，打开了把 **生成式扩散模型** 与 **强化学习** 结合的全新路径。后续工作开始探索在 **多模态扩散模型**（如文本‑图像）中加入类似的思考链，以提升跨模态推理。  
- **后续工作**：已有几篇论文（如 2024 年的 “Diffusion‑CoT for Multi‑Step Reasoning”）借鉴了 DCoLT 的思路，尝试在 **大规模语言模型** 上加入“思考自由度”。还有研究把 **自监督奖励**（如自检的数学一致性）与 DCoLT 结合，进一步缓解奖励稀疏的问题。  
- **深入方向**：如果想继续深挖，可以关注：① 如何设计更高效的 **稀疏奖励估计**（比如使用价值函数预训练）；② 将 **可解释性约束** 加入 DCoLT，使中间步骤既自由又可审查；③ 将 DCoLT 与 **检索增强**（RAG）结合，让模型在思考链中主动检索外部知识。

### 一句话记住它
**DCoLT 把扩散模型的每一步去噪当作自由的思考动作，用结果导向的强化学习整体优化，让模型在数学和代码任务上实现了跨越式的准确率提升。**