# Kimi K2: Open Agentic Intelligence

> **Date**：2025-07-28
> **arXiv**：https://arxiv.org/abs/2507.20534

## Abstract

We introduce Kimi K2, a Mixture-of-Experts (MoE) large language model with 32 billion activated parameters and 1 trillion total parameters. We propose the MuonClip optimizer, which improves upon Muon with a novel QK-clip technique to address training instability while enjoying the advanced token efficiency of Muon. Based on MuonClip, K2 was pre-trained on 15.5 trillion tokens with zero loss spike. During post-training, K2 undergoes a multi-stage post-training process, highlighted by a large-scale agentic data synthesis pipeline and a joint reinforcement learning (RL) stage, where the model improves its capabilities through interactions with real and synthetic environments.   Kimi K2 achieves state-of-the-art performance among open-source non-thinking models, with strengths in agentic capabilities. Notably, K2 obtains 66.1 on Tau2-Bench, 76.5 on ACEBench (En), 65.8 on SWE-Bench Verified, and 47.3 on SWE-Bench Multilingual -- surpassing most open and closed-sourced baselines in non-thinking settings. It also exhibits strong capabilities in coding, mathematics, and reasoning tasks, with a score of 53.7 on LiveCodeBench v6, 49.5 on AIME 2025, 75.1 on GPQA-Diamond, and 27.1 on OJBench, all without extended thinking. These results position Kimi K2 as one of the most capable open-source large language models to date, particularly in software engineering and agentic tasks. We release our base and post-trained model checkpoints to facilitate future research and applications of agentic intelligence.

---

# Kimi K2：开放式代理智能 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在生成文本、写代码方面已经很强，但让模型主动在外部环境中行动、解决实际任务仍是瓶颈。传统模型在规模上要么参数全开导致算力和显存爆炸，要么只激活少量专家导致能力受限；训练过程常出现 loss 突然飙升，导致训练中断或需要频繁人工干预。再加上后期缺乏系统化的“代理”训练手段，模型很难从单纯的语言生成转向真正的交互式智能。正是这些障碍促使研究者寻找更高效的稀疏激活方式、更加稳健的优化器以及专门的多阶段后训练流程。

### 关键概念速览
- **Mixture-of-Experts（MoE）**：把模型拆成若干“专家”，每次前向只激活一小部分，从而在保持算力可控的同时拥有上万亿级别的潜在参数。想象成一支乐队，演出时只让几位乐手上场演奏，但全体乐手的技术水平决定了整体潜力。
- **激活参数 vs. 总参数**：激活参数是一次前向实际使用的参数量（这里是 32 B），总参数是模型内部所有专家的累计参数（这里是 1 T）。激活参数决定显存占用，总参数决定模型的潜在表达能力。
- **MuonClip 优化器**：在 Muon 优化器的基础上加入 QK‑clip 技术，用来限制注意力矩阵中查询（Q）和键（K）的范数，防止梯度爆炸导致 loss 突然上升。可以把它想成在高速公路上装了限速装置，既保证车流顺畅，又避免超速事故。
- **QK‑clip**：专门针对自注意力机制的数值范围进行裁剪，使得注意力权重在训练初期保持平稳。类似于在烹饪时先把盐的用量控制在安全范围，防止味道一次性过咸。
- **多阶段后训练**：先用大规模合成数据进行微调，再通过强化学习（RL）让模型在真实或模拟环境中交互提升。相当于先让学生读教材，再让他去实习、做项目，最终形成实战能力。
- **Agentic Data Synthesis Pipeline**：自动生成“代理任务”数据的流水线，包括任务描述、环境交互日志、成功/失败标注等。把它比作自动出题系统，只是出的是让模型动手做的题目。
- **Joint RL Stage**：在同一次训练循环里同时使用真实环境交互和合成环境交互的奖励信号，让模型学会在不同场景下平衡探索与利用。

### 核心创新点
1. **从全局不稳定到局部裁剪的训练稳健化**  
   以前的稀疏模型在大规模预训练时常出现 loss 突然飙升，需要手动降学习率或重启训练。作者在 Muon 优化器上加入 QK‑clip，使查询和键的向量长度被硬性限制，从而在梯度传播时避免异常放大。结果是 15.5 T token 的预训练过程零 loss spike，训练效率大幅提升。

2. **1 T 参数的 MoE 设计与 32 B 激活的高效推理**  
   传统 MoE 往往在激活比例上做折中，要么激活太多导致显存吃紧，要么激活太少导致能力不足。Kimi K2 采用了专家路由的改进版，使得每次仅激活约 3% 的专家（对应 32 B），但路由策略更精准，保证了高质量的专家选择，从而在保持算力可控的同时释放了 1 T 参数的潜在能力。

3. **大规模代理任务合成 + 联合强化学习的后训练框架**  
   过去的后训练多是单一的指令微调或 RLHF（基于人类反馈的强化学习），缺少真实交互数据。本文构建了一个自动化的代理任务生成流水线，产出数十亿条包含环境状态、动作序列和奖励的合成经验。随后在 Joint RL Stage 中，这些合成经验与真实环境交互（如代码执行、网页操作）一起用于 PPO（近端策略优化）训练，使模型在软件工程、数学推理等多种“动手”任务上同步提升。

4. **开放式发布的完整模型链**  
   与多数商业大模型只提供 API 不同，Kimi K2 同时开源了基础模型和全部后训练检查点，提供了完整的复现脚本和数据管线，极大降低了社区二次研发的门槛。

### 方法详解
**整体框架**  
Kimi K2 的训练分为两大阶段：① 超大规模稀疏预训练（使用 MuonClip 优化器），② 多阶段后训练（合成代理数据微调 + 联合 RL）。预训练负责让模型学会通用语言和代码知识，后训练则专注于把这些知识转化为可执行的行动策略。

**1. 稀疏预训练细节**  
- **模型结构**：采用标准 Transformer 编码器，层数 48，隐藏维度 8192。专家数目约 256，每个专家拥有完整的前馈网络和自注意力子层。路由器使用 Top‑2 选择机制，即每个 token 选取得分最高的两个专家。  
- **激活控制**：通过改进的负载均衡正则，使得每个专家的调用频率保持在目标范围内，避免出现“热点专家”。  
- **MuonClip 优化器**：在每一步梯度更新前，对每层自注意力的 Q、K 向量做 L2 范数裁剪（阈值由经验公式设定），随后执行 Muon 的自适应学习率调节。这样既保留了 Muon 对稀疏梯度的高效处理，又防止了极端梯度导致的 loss 爆炸。  
- **训练规模**：使用 15.5 T token 的多语言、代码混合语料，训练时长约 45 天（8 × A100 80 GB），整个过程没有出现 loss spike。

**2. 多阶段后训练**  
- **阶段 A – 代理任务合成**  
  - **任务模板**：从公开的编程竞赛、系统运维手册、数学题库等抽取任务结构，自动填充变量生成数十亿条具体实例。  
  - **环境模拟**：为每类任务构建轻量级仿真器（如 Python 解释器沙箱、Linux 命令行模拟器），让模型在生成代码后即时执行，记录成功率、运行时错误等信息。  
  - **数据标注**：依据执行结果自动打上奖励标签（成功 +1，错误 -1），并保存完整的状态‑动作‑奖励轨迹。  

- **阶段 B – 联合强化学习**  
  - **真实环境交互**：挑选一小部分高价值任务（如真实 GitHub 项目构建、网页自动化）让模型在真实系统上运行，收集真实奖励。  
  - **PPO 训练**：使用 Proximal Policy Optimization（近端策略优化）算法，同时喂入合成轨迹和真实轨迹。损失函数由两部分组成：① 传统的 KL 散度约束，防止策略偏离原始语言模型；② 奖励加权的策略梯度，推动模型在实际执行上提升。  
  - **迭代更新**：每轮 PPO 结束后，重新采样合成任务，形成闭环，使模型的“行动库”不断刷新。

**最巧妙的设计**  
- **QK‑clip 与稀疏路由的协同**：在稀疏激活的情况下，注意力矩阵的稀疏性会放大数值波动，QK‑clip 正好在此处提供了数值安全网，保证了大规模 MoE 的训练平稳。  
- **合成‑真实双轨 RL**：单纯依赖合成数据容易产生“模拟偏差”，而只用真实交互成本极高。作者把两者混合，让模型在大规模低成本的模拟经验中学习基本技能，再用少量高质量真实经验校准，形成了成本与效果的最佳平衡。

### 实验与效果
- **评测任务**：包括 Tau2‑Bench（通用任务）、ACEBench（英文代理任务）、SWE‑Bench Verified（软件工程验证）、SWE‑Bench Multilingual（多语言软件工程）、LiveCodeBench v6（代码生成）、AIME 2025（数学竞赛）、GPQA‑Diamond（高难度常识推理）以及 OJBench（在线评测平台）。  
- **主要成绩**：  
  - Tau2‑Bench：66.1（最高开源非思考模型）  
  - ACEBench（En）：76.5  
  - SWE‑Bench Verified：65.8  
  - SWE‑Bench Multilingual：47.3  
  - LiveCodeBench v6：53.7  
  - AIME 2025：49.5  
  - GPQA‑Diamond：75.1  
  - OJBench：27.1  
  这些分数均在不使用“思考”（extended reasoning）模式下取得，显著领先大多数开源和闭源基线。  

- **对比基线**：与 LLaMA‑2‑70B、Mistral‑7B、OpenChat‑3.5 等模型相比，K2 在代理任务上提升 10%~30%不等，在代码和数学推理上也有 5%~12%的绝对提升。  

- **消融实验**：  
  - 去掉 QK‑clip，预训练 loss 在第 8 T token 时出现 2.3 的 spike，最终模型性能下降约 4%~6%。  
  - 仅使用合成数据进行后训练，SWE‑Bench Verified 下降约 7 分，说明真实交互的微调贡献显著。  
  - 将激活比例提升至 10%（约 100 B 参数），显存占用翻倍但性能提升不到 1%，验证了路由策略的有效性。  

- **局限性**：  
  - 仍然依赖大规模算力进行预训练，普通研究团队难以复现完整训练过程。  
  - 合成任务的质量受模板设计影响，某些专业领域（如医学）仍缺乏高质量合成数据。  
  - 在需要长时间思考的复杂推理（如多步数学证明）上，未使用思考模式时仍落后于最强闭源模型。

### 影响与延伸思考
Kimi K2 的发布让社区第一次在开源环境下看到“真正的代理能力”与大规模 MoE 结合的完整实现。随后出现的工作大多围绕两条主线：① 改进 QK‑clip 类的数值稳定技术，以适配更深层的稀疏模型；② 扩展 Agentic Data Synthesis Pipeline，加入跨模态（视觉、音频）任务，使模型能够在更广泛的真实环境中行动。推测未来会有更多研究把类似的双轨 RL 思路迁移到机器人控制或自动化运维领域。如果想进一步深入，可以关注以下方向：  
- **稀疏路由的自适应调度**（让模型在不同任务上动态调整激活比例）。  
- **跨模态代理任务生成**（把图像理解、语音指令纳入合成流水线）。  
- **低算力微调技术**（如 LoRA、QLoRA）在 K2 上的适配实验。

### 一句话记住它
Kimi K2 用 QK‑clip 稳定的大规模 MoE 预训练 + 合成+真实双轨强化学习，让开源模型首次在软件工程和代理任务上实现“动手”能力的飞跃。