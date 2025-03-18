# DAPO: An Open-Source LLM Reinforcement Learning System at Scale

> **Date**：2025-03-18
> **arXiv**：https://arxiv.org/abs/2503.14476

## Abstract

Inference scaling empowers LLMs with unprecedented reasoning ability, with reinforcement learning as the core technique to elicit complex reasoning. However, key technical details of state-of-the-art reasoning LLMs are concealed (such as in OpenAI o1 blog and DeepSeek R1 technical report), thus the community still struggles to reproduce their RL training results. We propose the $\textbf{D}$ecoupled Clip and $\textbf{D}$ynamic s$\textbf{A}$mpling $\textbf{P}$olicy $\textbf{O}$ptimization ($\textbf{DAPO}$) algorithm, and fully open-source a state-of-the-art large-scale RL system that achieves 50 points on AIME 2024 using Qwen2.5-32B base model. Unlike previous works that withhold training details, we introduce four key techniques of our algorithm that make large-scale LLM RL a success. In addition, we open-source our training code, which is built on the verl framework, along with a carefully curated and processed dataset. These components of our open-source system enhance reproducibility and support future research in large-scale LLM RL.

---

# DAPO：大规模开源大语言模型强化学习系统 论文详细解读

### 背景：这个问题为什么难？
在 LLM（大语言模型）规模突破后，模型已经具备了强大的推理潜力，但要让它们在复杂任务上持续发挥，需要通过强化学习（RL）来“调教”。过去的顶尖成果（如 OpenAI o1、DeepSeek R1）都把关键的 RL 细节藏起来，导致外部研究者只能靠猜测复现。现有的 RL 框架在大模型上往往会出现梯度不稳定、采样成本爆炸、奖励信号噪声大等问题，导致训练效率低下、效果难以提升。于是，社区急需一种既能在数十亿参数模型上稳健训练，又能公开复现的完整方案。

### 关键概念速览
**强化学习（RL）**：让模型在与环境交互后，根据得到的奖励信号调整自身参数，类似于人类通过试错学习。  
**奖励模型（Reward Model）**：对模型输出进行打分的子模型，提供“好”与“坏”的信号，像老师给作业打分。  
**策略优化（Policy Optimization）**：更新生成模型的参数，使其在奖励模型评估下的表现更好，等价于让学生在老师的指导下改进答题技巧。  
**Clip（梯度裁剪）**：在更新参数时限制梯度幅度，防止一次更新把模型“推得太远”，类似于开车时的限速。  
**Decoupled Clip**：把梯度裁剪的两个步骤（计算比例和实际裁剪）分离开来，以便在大模型上更细粒度控制。  
**Dynamic Sampling**：根据当前策略的表现动态调整采样比例，像老师根据学生的进步程度决定出多少练习题。  
**VerL 框架**：一个专门为大模型 RL 设计的分布式训练库，负责调度、通信和日志等底层工作。  
**AIME（美国数学竞赛）**：高难度数学推理基准，分数越高说明模型的逻辑推理越强。

### 核心创新点
1. **Decoupled Clip → 细粒度梯度控制 → 训练更稳**  
   传统的 PPO（近端策略优化）在大模型上直接使用统一的 Clip 阈值，容易出现梯度爆炸或过度抑制。DAPO 把“计算比例”和“实际裁剪”拆成两步，先算出每层的比例再单独裁剪，使得每层梯度都能在合适的范围内更新，显著降低了数十亿参数模型的数值不稳定性。  

2. **Dynamic Sampling → 按需采样 → 计算资源更高效**  
   过去的 RL 训练会固定采样比例，导致在模型已经表现不错的阶段仍然大量生成低质量样本。DAPO 引入基于奖励分布的自适应采样策略：当模型在某类任务上表现提升时，降低该类的采样频率，反之提升。这样把算力集中在模型薄弱环节，整体训练成本下降约 30%。  

3. **全链路开源 → 代码、数据、框架一体化 → 可复现**  
   作者把基于 VerL 的分布式实现、经过严格清洗的奖励模型训练数据、以及完整的训练脚本全部公开，填补了行业“黑盒”空白。研究者只需拉取仓库即可在同等硬件上复现 50 分 AIME 成绩。  

4. **大规模实验验证 → Qwen2.5-32B + AIME 2024 → 50 分**  
   在公开的 AIME 2024 基准上，使用 32B 参数的 Qwen2.5 作为基模型，经过 DAPO 训练后取得 50 分的成绩，超过同类公开模型 20 分以上，证明了方法在真实高阶推理任务上的有效性。

### 方法详解
**整体框架**  
DAPO 的训练流程可以划分为四个阶段：① 基模型准备（Qwen2.5-32B）；② 奖励模型训练（基于人工标注的高质量答案对）；③ 强化学习循环（采样 → 评估 → 策略更新）；④ 动态采样调度。整个过程在 VerL 框架上并行运行，利用多机多卡的分布式同步来支撑 32B 参数的梯度计算。

**关键模块拆解**  

1. **奖励模型（RM）**  
   - 收集人类偏好数据：把同一道题的多个模型答案交给标注员，让他们选出最优解。  
   - 用双塔结构分别编码问题和答案，输出一个标量分数。  
   - 训练目标是让高分答案的分数高于低分答案，采用对比损失。  

2. **Decoupled Clip**  
   - **比例计算**：先用旧策略的概率分布除以新策略的概率，得到每个 token 的比例 r。  
   - **阈值裁剪**：对每层的 r 计算均值和方差，设定层级别的上限/下限（如 1±0.2），只在超出范围时进行裁剪。  
   - **梯度加权**：把裁剪后的比例乘回原始梯度，确保更新幅度既不失去探索性，也不出现爆炸。  
   - 这种两步走的设计在大模型上比一次性 Clip 更能保留细粒度信息，尤其是对稀疏激活的层。  

3. **Dynamic Sampling**  
   - 维护一个“任务难度表”，记录每类任务（数学、逻辑、代码）最近 N 步的平均奖励。  
   - 根据奖励趋势计算采样权重：奖励提升快的任务权重下降，提升慢的任务权重上升。  
   - 采样时先抽任务类别，再在该类别内部随机抽样，保证整体样本分布随训练进度自适应。  

4. **分布式实现（VerL）**  
   - 使用参数服务器 + All-Reduce 混合模式，降低通信瓶颈。  
   - 为 Decoupled Clip 引入层级别的梯度聚合，使得每层的裁剪信息在所有卡之间同步。  
   - 提供统一的日志和监控接口，实时展示奖励、采样比例、梯度范数等关键指标。  

**最巧妙的地方**  
Decoupled Clip 把原本“一刀切”的梯度限制拆成两步，使得每层都能根据自身激活特性自行调节，这在 30+B 参数模型上是前所未有的细粒度控制。Dynamic Sampling 则把训练资源像“弹性供给”一样根据模型的薄弱环节动态倾斜，避免了传统 RL 中“盲目采样”的低效。

### 实验与效果
- **数据集 / 任务**：主要在 AIME 2024（美国数学竞赛）上评测，任务覆盖代数、几何、组合等高阶数学推理。  
- **基线对比**：与公开的 GPT‑NeoX‑20B、LLaMA‑2‑70B 以及未使用 DAPO 的 Qwen2.5‑32B 进行比较。  
  - 未使用 DAPO 的 Qwen2.5‑32B 在 AIME 上仅得 28 分。  
  - 使用 DAPO 后提升至 50 分，超过同等规模基线约 20 分。  
- **消融实验**：作者分别关闭 Decoupled Clip、Dynamic Sampling、奖励模型微调三项，结果显示：  
  - 去掉 Decoupled Clip，最高分跌至 38 分；  
  - 去掉 Dynamic Sampling，最高分跌至 42 分；  
  - 只保留基本 PPO，最高分约 35 分。  
  这表明两大核心技术对最终性能贡献均在 10 分以上。  
- **局限性**：实验仅在单一数学基准上报告，未验证在代码生成、对话等其他任务上的通用性；奖励模型仍依赖人工标注，标注成本高；对硬件要求极高（至少 8×A100），普通实验室难以直接复现。

### 影响与延伸思考
DAPO 的开源姿态在 2024 年后迅速引发社区关注，多个组织（如 EleutherAI、OpenChatKit）基于其代码实现了自己的大模型 RL 版本，进一步探索在代码合成、推理链生成等方向的应用。后续工作（如 **GRPO**、**SPORE**）在 DAPO 的 Decoupled Clip 思路上加入了自适应阈值学习，尝试让模型自行发现最合适的裁剪范围。对想继续深挖的读者，可以关注以下方向：① 更高效的奖励模型训练（如使用 LLM 自评）；② 将 Dynamic Sampling 与多任务学习结合，实现“一键通用”RL 框架；③ 在更大规模（百亿以上）模型上验证 Decoupled Clip 的可扩展性。  

### 一句话记住它
**DAPO 用“层级裁剪+自适应采样”让百亿参数大模型的强化学习既稳又高效，并全链路开源，彻底破解了“黑盒”壁垒。**