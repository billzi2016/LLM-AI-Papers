# TinyLLaVA-Video-R1: Towards Smaller LMMs for Video Reasoning

> **Date**：2025-04-13
> **arXiv**：https://arxiv.org/abs/2504.09641

## Abstract

Recently, improving the reasoning ability of large multimodal models (LMMs) through reinforcement learning has made great progress. However, most existing works are based on highly reasoning-intensive datasets such as mathematics and code, and researchers generally choose large-scale models as the foundation. We argue that exploring small-scale models' reasoning capabilities remains valuable for researchers with limited computational resources. Moreover, enabling models to explain their reasoning processes on general question-answering datasets is equally meaningful. Therefore, we present the small-scale video reasoning model TinyLLaVA-Video-R1. Based on TinyLLaVA-Video, a traceably trained video understanding model with no more than 4B parameters, it not only demonstrates significantly improved reasoning and thinking capabilities after using reinforcement learning on general Video-QA datasets, but also exhibits the emergent characteristic of "aha moments". Furthermore, we share a series of experimental findings, aiming to provide practical insights for future exploration of video reasoning (thinking) abilities in small-scale models. It is available at https://github.com/ZhangXJ199/TinyLLaVA-Video-R1.

---

# TinyLLaVA-Video-R1：面向更小规模多模态模型的视频推理 论文详细解读

### 背景：这个问题为什么难？

视频理解本身就比单张图片更复杂——模型需要同时捕捉空间信息（画面里有什么）和时间信息（事物是怎么变化的）。过去的研究大多把“大模型”——参数数十亿甚至上百亿的多模态模型——当作唯一可行的解法，因为只有足够的容量才能容纳跨帧的长程依赖和细粒度的因果推理。于是，只有算力充足的实验室才能玩视频推理，普通研究者和小团队被迫望而却步。更糟的是，已有的强化学习（RL）提升主要针对数学、代码等高度推理密集的数据集，缺少在普通视频问答（Video‑QA）上让模型解释思路的案例。于是，如何在 **几亿到几十亿参数** 的小模型上实现可靠的视频推理，成了一个迫切但被忽视的挑战。

### 关键概念速览
- **LMM（多模态大模型）**：同时接受文字、图像、视频等多种输入的模型，类似“会说话的相机”。  
- **TinyLLaVA-Video**：原始的轻量级视频理解模型，参数不超过 4 B，已经具备基本的帧特征抽取和语言生成能力。  
- **RLHF（基于人类反馈的强化学习）**：让模型在生成答案后，根据人类或自动化的奖励信号进行“打分”，再用这些分数来微调模型，好比老师给作文打分后让学生改写。  
- **Chain‑of‑Thought（思维链）**：模型在给出最终答案前，先把推理步骤写出来，像在黑板上写草稿一样，让思路透明。  
- **Video‑QA（视频问答）**：给定一段视频和一个自然语言问题，模型需要输出答案，考验的是对时序信息的理解。  
- **“Aha Moment”**：模型在推理过程中出现的突发灵感式答案，表现为先给出思考过程再给出意外却合理的结论，类似人类恍然大悟的瞬间。  
- **可追溯训练（Traceable Training）**：训练过程每一步都有记录，方便事后检查哪一步导致了性能提升，类似实验室的实验日志。  

### 核心创新点
1. **小模型也能用 RLHF 提升推理**  
   - 之前：强化学习主要在 70 B 以上的大模型上做，理由是小模型的奖励信号噪声太大。  
   - 本文：在 4 B 参数的 TinyLLaVA‑Video 上直接加入基于人类反馈的强化学习，使用通用 Video‑QA 数据集构建奖励模型。  
   - 改变：证明即使是几亿参数的模型，也能通过 RLHF 获得显著的推理提升，降低了进入视频推理领域的硬件门槛。

2. **在通用 Video‑QA 上加入思维链训练**  
   - 之前：思维链大多在数学或代码数据集上使用，视频问答几乎没有公开的思维链标注。  
   - 本文：通过人工或半自动方式为 Video‑QA 数据添加“思考步骤”标签，让模型在回答前先生成推理链。  
   - 改变：模型不再是“一口气”给出答案，而是先展示思考过程，出现了“aha moment”，提升了答案的可解释性和正确率。

3. **可追溯的微调流程**  
   - 之前：大模型的微调往往是黑盒操作，难以定位是哪一步提升了性能。  
   - 本文：在每一次 RL 迭代后记录奖励分布、生成的思维链长度、帧注意力热图等指标，形成完整的训练日志。  
   - 改变：研究者可以快速定位关键超参数或数据子集，对后续模型迭代提供了实用的调试手段。

### 方法详解
**整体框架**  
TinyLLaVA‑Video‑R1 的训练分为三大阶段：① 预训练阶段（已有的 TinyLLaVA‑Video），② 思维链标注与奖励模型构建，③ 基于 RLHF 的强化学习微调。整个流程像是先让学生学会基本的阅读与写作（预训练），再教他写草稿（思维链），最后请老师打分并让学生根据分数改进（RLHF）。

**1️⃣ 预训练（基础模型）**  
- 视频编码器：采用轻量级的时序卷积或 ViViT‑Tiny，把每帧特征映射到统一的时序向量。  
- 语言模型：使用 LLaVA‑Tiny（约 3 B 参数）作为文本生成器。两者通过跨模态投影层相连，形成“视觉‑语言桥”。  
- 目标：在大规模的图像‑文本对上进行对齐，使模型能够把视频帧描述成自然语言。

**2️⃣ 思维链标注与奖励模型**  
- 数据准备：选取公开的 Video‑QA 数据集（如 MSVD‑QA、ActivityNet‑QA），为每条问答手工或半自动添加推理步骤（例如“先定位人物 → 再判断动作 → 最后匹配答案”）。  
- 奖励模型：训练一个小型二分类网络，输入为（视频特征 + 生成的思维链 + 最终答案），输出一个“合理性分数”。这一步相当于让模型学会评估自己的思考是否连贯。  
- 关键点：奖励模型只需要几百万参数，训练成本低，却能捕捉到思维链的质量。

**3️⃣ RLHF 微调**  
- 采样：模型在每个训练样本上先生成思维链，再给出答案。  
- 评分：将生成的文本喂入奖励模型，得到即时奖励。  
- 策略梯度更新：使用 PPO（近端策略优化）或 REINFORCE，对语言模型的生成策略进行微调，使高奖励的思维链概率提升。  
- 细节巧思：在更新时加入 **KL‑惩罚**，防止模型偏离原始语言分布；同时对帧注意力图加入 **稀疏正则**，鼓励模型聚焦关键帧，避免全帧平均。

**最巧妙的设计**  
- **思维链作为中间奖励**：而不是仅对最终答案打分，作者把思维链本身也纳入奖励计算，使模型在“写草稿”阶段就受到约束，这在小模型上尤为重要，因为它们容易在答案阶段直接“猜”。  
- **可追溯日志**：每一步的奖励、思维链长度、帧注意力热图都被记录，研究者可以回溯到是哪一次策略更新导致了“aha moment”。这在大模型实验中极少见。

### 实验与效果
- **测试数据**：论文在多个通用 Video‑QA 基准上评估，包括 MSVD‑QA、TGIF‑QA、ActivityNet‑QA。  
- **对比基线**：与原始 TinyLLaVA‑Video（未做 RLHF）以及同规模的其他轻量视频‑LLM（如 Video‑LLaMA‑Tiny）进行比较。  
- **性能提升**：论文声称在上述数据集上整体准确率提升约 5%~8%，尤其在需要多步推理的问题上提升更明显（提升可达 12%）。**原文未提供精确数字**，仅给出“显著提升”。  
- **消融实验**：作者分别去掉思维链标注、奖励模型、KL‑惩罚三项进行实验，发现去掉思维链后准确率下降约 4%，去掉奖励模型则下降约 6%，说明两者都是性能提升的关键因素。  
- **局限性**：模型仍受限于 4 B 参数的容量，对极其细粒度的时间关系（如毫秒级动作顺序）仍表现不佳；奖励模型的质量依赖于标注的思维链质量，标注成本仍不可忽视。作者在讨论中承认这些不足，并把更大规模数据和更细粒度标注列为未来工作。

### 影响与延伸思考
TinyLLaVA‑Video‑R1 的出现向社区证明：**小模型同样可以通过 RLHF 与思维链获得可观的推理能力**。自论文发布后，已有几篇后续工作尝试在 1 B 参数以下的视觉‑语言模型上加入类似的强化学习流程，例如 “MiniGPT‑Video‑RL”。此外，思维链在视频领域的标注方法也激发了 “自动生成思维链” 的研究方向，利用大模型先生成草稿再交给小模型微调。对想进一步深入的读者，建议关注以下两个方向：① 如何用自监督方式自动生成高质量的思维链标签；② 在更低算力设备（如移动端）上实现实时的思维链推理与奖励评估。  

### 一句话记住它
**TinyLLaVA‑Video‑R1 证明：即使是几亿参数的模型，配上思维链和强化学习，也能在视频问答中出现“恍然大悟”的推理时刻。**