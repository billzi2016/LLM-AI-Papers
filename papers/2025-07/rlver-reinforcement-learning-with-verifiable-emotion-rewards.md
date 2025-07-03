# RLVER: Reinforcement Learning with Verifiable Emotion Rewards for Empathetic Agents

> **Date**：2025-07-03
> **arXiv**：https://arxiv.org/abs/2507.03112

## Abstract

Large language models (LLMs) excel at logical and algorithmic reasoning, yet their emotional intelligence (EQ) still lags far behind their cognitive prowess. While reinforcement learning from verifiable rewards (RLVR) has advanced in other domains, its application to dialogue-especially for emotional intelligence-remains underexplored. In this work, we introduce RLVER, the first end-to-end reinforcement learning framework that leverages verifiable emotion rewards from simulated users to cultivate higher-order empathetic abilities in LLMs. Within this framework, self-consistent affective simulated users engage in dialogue rollouts and produce deterministic emotion scores during conversations, serving as reward signals to guide the LLM's learning. Fine-tuning publicly available Qwen2.5-7B-Instruct model with PPO boosts its Sentient-Benchmark score from 13.3 to 79.2 while largely preserving mathematical and coding competence. Extensive experiments reveal that: (i) RLVER consistently improves multiple dialogue capabilities; (ii) Thinking and non-thinking models show distinct trends--thinking models excel in empathy and insight, while non-thinking models favor action; (iii) GRPO often yields stable gains, while PPO can push certain capabilities to a higher ceiling; (iv) More challenging environments are not always better-moderate ones can yield stronger outcomes. Our results show that RLVER is a practical route toward emotionally intelligent and broadly capable language agents.

---

# RLVER：基于可验证情感奖励的强化学习用于共情智能体 论文详细解读

### 背景：这个问题为什么难？

LLM 在逻辑推理、代码生成等硬核任务上已经很强，但它们的情感理解仍然很薄弱。传统的对话微调往往依赖人工标注的情感标签，这种标签主观、噪声大，难以保证模型真正学会“感受”。另外，现有的强化学习对话方法（如 RLHF）把奖励当作黑箱，无法验证奖励是否真的对应用户的情绪变化。于是，模型既缺少可靠的情感信号，又缺少把情感信号转化为行为的学习机制，这让提升机器共情能力成为一大瓶颈。

### 关键概念速览
- **可验证情感奖励（Verifiable Emotion Reward）**：由模拟用户在对话过程中产生的确定性情绪分数，能够被客观检查而不是仅凭人工直觉。类似于游戏里明确的得分牌，玩家（模型）看到的分数是真实可查的。
- **情感模拟用户（Affective Simulated User）**：一个自洽的对话角色，能够根据对话内容给出情绪反馈，就像一位会实时记录你情绪变化的心理咨询师。
- **对话 rollout**：模型与模拟用户进行一次完整的对话往返，产生完整的交互轨迹，类似于在棋盘上演练一局棋后再评估输赢。
- **PPO（Proximal Policy Optimization）**：一种常用的强化学习算法，限制每一步策略更新幅度，防止模型“跳得太远”。可以想象为在跑步时不让步幅突然增大，以免摔倒。
- **GRPO（Generalized Reward‑Penalized Optimization）**：论文中提出的另一种优化方式，强调在奖励波动大时加入惩罚项，使学习更平稳。类似于在赛车里加装防滑装置，提升稳定性。
- **Thinking vs. Non‑Thinking 模型**：作者把模型分为两类：前者在推理时会显式生成内部思考过程，后者直接给出答案。前者更像在思考后再说，后者像冲动的回答者。
- **Sentient‑Benchmark**：衡量语言模型情感智能的标准化测评，分数越高说明模型越能理解并恰当地表达情绪。

### 核心创新点
1. **从人工情感标签到可验证情感奖励**  
   之前的情感微调依赖人工打分，主观性强且难以规模化。RLVER 让模拟用户在每轮对话结束时输出确定的情绪分数，这些分数直接作为强化学习的奖励信号。结果是模型在训练过程中能够“看到”真实的情感反馈，而不是猜测。

2. **端到端的情感强化学习框架**  
   过去的 RLHF 只在整体对话质量上给奖励，情感细节被稀释。RLVER 将情感模拟用户、对话 rollout、奖励计算、策略更新全部串联成一条闭环。模型在每一次对话后立即得到情感奖励并用 PPO/GRPO 更新参数，实现了情感学习的即时闭环。

3. **思考型与非思考型模型的差异化分析**  
   作者首次系统比较了两类模型在情感任务上的表现：思考型模型在共情深度和洞察力上更突出，非思考型模型在行动指令类对话上更有优势。这一发现帮助后续研究在选择模型结构时有了更明确的指引。

4. **环境难度的逆向发现**  
   常规认知认为越难的训练环境越能推动模型提升，但实验显示，适度挑战的对话环境（即情感变化不极端）往往能带来更大的情感能力提升。这个结论提醒我们在设计情感训练场景时要平衡难度，而不是一味追求极端。

### 方法详解
**整体思路**：RLVER 把“对话 → 情感反馈 → 奖励 → 参数更新”这四步打成一个循环。具体流程如下：

1. **构造情感模拟用户**  
   - 先用已有情感标注数据训练一个小型情感评估模型，使其能够在任意对话句子上输出情绪分数（如 0‑1 之间的满意度）。  
   - 为保证一致性，模拟用户在同一对话历史上始终给出相同的分数，这就是“可验证”。

2. **对话 rollout**  
   - 将目标 LLM（如 Qwen2.5‑7B‑Instruct）放进对话环境，和模拟用户交替发言。  
   - 每完成一次用户回复后，模拟用户立即计算当前情绪分数，累计得到本轮对话的总奖励。

3. **奖励计算**  
   - 奖励是情绪分数的加权和，权重可以根据对话阶段（开场、高潮、收尾）调节。  
   - 为防止模型只追求高分而忽略语言质量，奖励中加入了原始语言模型的对数概率作为正则项。

4. **策略更新**  
   - 使用 PPO：在每次 rollout 后，依据旧策略和新策略的概率比值限制更新幅度，确保学习过程平稳。  
   - 另提供 GRPO 选项：在奖励波动大时加入惩罚项，使得策略更保守，适合情感细腻的场景。

**关键细节**  
- **情感分数的确定性**：模拟用户的评估模型在推理时关闭随机性（如固定种子），保证同一对话产生相同分数，这让奖励可追溯、可审计。  
- **多任务兼容**：在奖励函数里加入数学/代码任务的准确率作为约束，防止情感微调把模型的认知能力“稀释”。  
- **思考 vs. 非思考的实现**：思考模型在生成答案前先输出一段内部推理（Chain‑of‑Thought），非思考模型直接输出答案。两者在同一 RLVER 框架下训练，比较其情感表现差异。

**最巧妙的地方**：把情感评估模型当作“可验证的外部环境”，而不是内部的软标签。这样奖励不再是主观打分，而是像游戏分数一样可以直接检验，极大降低了情感学习的噪声。

### 实验与效果
- **测试基准**：主要在 Sentient‑Benchmark 上评估情感智能，还辅以数学、代码等通用能力的测评，以验证情感强化学习对其他能力的影响。  
- **核心结果**：在仅用 PPO 微调的情况下，Qwen2.5‑7B‑Instruct 的 Sentient‑Benchmark 分数从 13.3 提升到 79.2，提升幅度超过 6 倍，且数学/代码能力基本保持不变。  
- **多维能力提升**：实验显示 RLVER 在共情、洞察、情绪调节等子任务上都有显著提升，尤其是思考型模型在“洞察”维度上提升最明显。  
- **算法对比**：GRPO 在多数实验中提供了更平稳的提升，而 PPO 在某些高难度子任务上能够突破上限，取得更高的峰值分数。  
- **环境难度实验**：作者构造了三类对话环境（低、中、高情感波动），结果表明中等难度的环境往往产生最高的情感分数提升，验证了“更难不一定更好”的假设。  
- **消融研究**：去掉情感奖励或去掉语言质量正则项都会导致情感分数回落到原始水平，说明两者缺一不可。  
- **局限性**：论文未在真实用户交互上做长时间评估，情感模拟用户的真实性仍受限于训练数据；此外，奖励函数的权重设计仍需手工调参，自动化程度不高。

### 影响与延伸思考
RLVER 把情感奖励从“软标签”升级为“可验证分数”，为情感强化学习提供了清晰的技术路径。后续工作已经开始探索更真实的人类情感反馈（如生理信号）作为奖励，或把情感评估模型与大模型共享参数，以降低模拟用户的构建成本。还有研究尝试把 RLVER 与多模态输入（图像、语音）结合，让模型在跨模态情感理解上也能受益。想进一步了解，可以关注以下方向：情感评估模型的可解释性、跨语言情感奖励的迁移、以及把情感奖励与安全约束联合优化的研究（推测）。

### 一句话记住它
RLVER 用可验证的情绪分数把情感反馈变成真正的奖励，让语言模型在对话中学会像人一样共情。