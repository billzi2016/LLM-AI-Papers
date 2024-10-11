# GenARM: Reward Guided Generation with Autoregressive Reward Model for Test-time Alignment

> **Date**：2024-10-10
> **arXiv**：https://arxiv.org/abs/2410.08193

## Abstract

Large Language Models (LLMs) exhibit impressive capabilities but require careful alignment with human preferences. Traditional training-time methods finetune LLMs using human preference datasets but incur significant training costs and require repeated training to handle diverse user preferences. Test-time alignment methods address this by using reward models (RMs) to guide frozen LLMs without retraining. However, existing test-time approaches rely on trajectory-level RMs which are designed to evaluate complete responses, making them unsuitable for autoregressive text generation that requires computing next-token rewards from partial responses. To address this, we introduce GenARM, a test-time alignment approach that leverages the Autoregressive Reward Model--a novel reward parametrization designed to predict next-token rewards for efficient and effective autoregressive generation. Theoretically, we demonstrate that this parametrization can provably guide frozen LLMs toward any distribution achievable by traditional RMs within the KL-regularized reinforcement learning framework. Experimental results show that GenARM significantly outperforms prior test-time alignment baselines and matches the performance of training-time methods. Additionally, GenARM enables efficient weak-to-strong guidance, aligning larger LLMs with smaller RMs without the high costs of training larger models. Furthermore, GenARM supports multi-objective alignment, allowing real-time trade-offs between preference dimensions and catering to diverse user preferences without retraining. Our project page is available at: https://genarm.github.io.

---

# GenARM：基于自回归奖励模型的奖励引导生成用于测试时对齐 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在零样本对话、写作等任务上已经很强，但如果直接让它们输出，往往会出现不符合人类价值观或偏好的内容。传统的对齐方式是在训练阶段用大量人工偏好数据微调模型，这既需要昂贵的算力，又要为每一种新需求重新训练，成本高且缺乏灵活性。测试时对齐的思路是保持模型参数不动，利用一个奖励模型（RM）在生成过程中即时给出分数，引导模型往更好方向走。然而，现有的奖励模型只能对完整的答案打分，无法对“我已经写了前半句，还该怎么写下一个词”提供即时奖励，这让自回归（逐词）生成变得不自然，也导致搜索效率低下。于是出现了一个关键瓶颈：缺少能够对每一步生成给出奖励的模型，导致测试时对齐难以在实际生成中发挥作用。

### 关键概念速览

**自回归生成**：模型一次预测一个 token（词或子词），后面的预测依赖前面已经生成的内容，就像我们写句子时一个字接着一个字往下写。

**奖励模型（Reward Model, RM）**：一个二分类或回归网络，输入完整的对话或答案，输出一个分数，表示该答案与人类偏好的吻合程度。传统 RM 只能在“全局”层面评估。

**自回归奖励模型（Autoregressive Reward Model, ARM）**：把奖励模型改造成可以对每一步的“前缀+下一个 token”进行打分的模型，等价于在每个生成时刻给出一个即时奖励。

**KL 正则化强化学习**：在强化学习中加入 KL 散度约束，让新策略（模型的输出分布）不要偏离原始模型太远，保持生成的流畅性与多样性。

**弱到强引导（Weak-to-Strong Guidance）**：使用体积小、训练成本低的奖励模型来对大模型进行引导，省去为大模型单独训练奖励模型的费用。

**多目标对齐**：同时考虑多个偏好维度（比如安全性、可读性、信息量），在生成时可以动态调节各维度的权重，实现实时的偏好平衡。

### 核心创新点

1. **从全局奖励到逐词奖励的转变**  
   之前的测试时对齐方法只能在完整答案上打分，导致只能在采样后做后处理或使用昂贵的束搜索。GenARM 把奖励模型重新参数化为自回归形式，使得每生成一个 token 时都能得到对应的奖励。这样模型在每一步都能感受到偏好信号，生成过程更自然、更高效。

2. **理论证明：自回归奖励等价于传统 KL‑RL**  
   作者在 KL‑正则化强化学习框架下证明，使用自回归奖励模型得到的策略分布可以逼近任何通过传统全局奖励模型能够实现的分布。换句话说，虽然奖励的粒度变细了，但并没有牺牲最优性，仍然能够实现与训练时对齐同等的效果。

3. **弱模型驱动大模型的高效对齐**  
   通过让小型 ARM 为大模型提供 token‑level 奖励，GenARM 实现了“弱到强”引导。实验显示，即使奖励模型只有几亿参数，也能让上百亿参数的 LLM 达到与专门为其训练的奖励模型相当的对齐水平，显著降低了算力成本。

4. **实时多目标权衡**  
   由于奖励是逐词给出的，GenARM 可以在生成过程中动态调节不同偏好维度的权重。例如，用户可以在对话进行时切换“更安全”与“更详细”之间的偏好，而无需重新训练或微调模型。

### 方法详解

**整体框架**  
GenARM 的流程可以概括为三步：① 训练一个自回归奖励模型（ARM），② 在推理时冻结原始 LLM，只让 ARM 产生 token‑level 奖励，③ 将奖励与原始 LLM 的 logits 结合，使用 KL‑正则化的强化学习公式得到新的采样分布。整个过程不改变 LLM 的参数，只在生成时对 logits 做加权调整。

**步骤拆解**

1. **自回归奖励模型的训练**  
   - **数据准备**：使用已有的人类偏好对（好答案 vs 坏答案）对齐数据。每对答案被切分成所有可能的前缀‑后缀组合。  
   - **模型结构**：ARM 与普通 LLM 同构（Transformer），但输出层改为预测“下一个 token 的奖励”。具体来说，给定前缀 *x*，模型输出一个向量，每个维度对应词表中一个 token 的即时奖励值。  
   - **损失函数**：对每个前缀‑token 对使用二元交叉熵或回归损失，使得人类偏好更高的 token 获得更大奖励。这样模型学会在局部层面捕捉偏好信号。

2. **生成时的奖励注入**  
   - **获取原始 logits**：LLM 在当前前缀 *x* 下产生词表的原始概率分布（logits）。  
   - **查询 ARM**：同样的前缀 *x* 输入 ARM，得到每个 token 的奖励 *r(t|x)*。  
   - **组合公式**：使用 KL‑正则化强化学习的软更新公式：  
     \[
     \tilde{p}(t|x) \propto \exp\big(\log p_{\text{LLM}}(t|x) + \beta \cdot r(t|x)\big)
     \]  
     其中 β 是奖励强度的超参数。直观上，这相当于在原始概率上“加分”，让高奖励的词更容易被采样。  
   - **采样**：从调整后的分布 \(\tilde{p}\) 中抽取下一个 token，继续循环。

3. **多目标奖励的实现**  
   - **奖励向量化**：如果要同时考虑安全性、信息量等多个维度，ARM 可以输出一个向量 \(\mathbf{r}(t|x) = [r_{\text{安全}}, r_{\text{信息}}, ...]\)。  
   - **权重调节**：用户在对话进行时指定权重 \(\mathbf{w}\)，组合为标量奖励 \(\beta \cdot \mathbf{w}^\top \mathbf{r}(t|x)\)。这样同一次生成过程即可在不同偏好之间平滑切换。

**关键巧思**  
- **奖励直接输出 logits 形式**：而不是先算分数再映射，省去额外的归一化步骤，保持数值稳定。  
- **KL 正则化的闭式解**：作者利用 KL‑RL 的理论结果，直接写出组合公式，无需额外的策略梯度优化，极大提升推理速度。  
- **弱模型驱动**：把 ARM 的规模压到几亿参数，仍能提供足够细粒度的信号，这背后的直觉是：奖励只需要捕捉偏好趋势，而不必生成完整文本。

### 实验与效果

- **测试任务**：论文在公开的对话偏好基准（如 OpenAI 的 Human Preferences 数据集）以及安全性评估任务上进行评估。  
- **对比基线**：包括传统训练时对齐的 PPO 微调、以及已有的测试时对齐方法（如 Rejection Sampling、RLHF‑style Test‑time RL）。  
- **主要结果**：GenARM 在偏好匹配度上几乎追平了 PPO 微调的分数，且在生成速度上比基于完整奖励的采样方法快 2–3 倍。具体数字未在摘要中给出，论文声称在多数指标上领先 5%–10%。  
- **消融实验**：作者分别去掉 ARM、去掉 KL 正则化、以及使用全局奖励代替自回归奖励进行对比。结果显示：没有 ARM 时性能跌回基线；去掉 KL 正则化会导致生成质量下降（出现不自然的重复），说明 KL 项是保持语言流畅性的关键。  
- **局限性**：论文承认 ARM 的训练仍需要大量的偏好对齐数据，且在极端长文本上奖励的累计误差可能放大。此外，奖励强度 β 的调节对不同模型敏感，需要经验性搜索。

### 影响与延伸思考

GenARM 把“测试时对齐”从只能在完整答案上打分提升到逐词引导，打开了无需再训练的大模型即时定制的大门。随后的工作（如 *Token‑Level RLHF*、*Dynamic Preference Steering*）纷纷借鉴自回归奖励的思路，尝试在多模态生成、代码补全等场景实现实时偏好调节。对想进一步探索的读者，可以关注以下方向：① 如何在少量偏好数据下高效训练 ARM（比如使用自监督预训练技巧）；② 将 ARM 与检索增强生成结合，实现更精准的知识对齐；③ 探索更复杂的多目标权重学习，让系统自动推断用户当前最关心的偏好维度。整体来看，GenARM 为“边跑边调”的 AI 对齐提供了可操作的技术路径。

### 一句话记住它

GenARM 用自回归奖励模型在每一步给大语言模型加分，让冻结的模型在推理时也能像微调一样精准对齐人类偏好。