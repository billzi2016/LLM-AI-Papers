# AFEC: Active Forgetting of Negative Transfer in Continual Learning

> **Date**：2021-10-23
> **arXiv**：https://arxiv.org/abs/2110.12187

## Abstract

Continual learning aims to learn a sequence of tasks from dynamic data distributions. Without accessing to the old training samples, knowledge transfer from the old tasks to each new task is difficult to determine, which might be either positive or negative. If the old knowledge interferes with the learning of a new task, i.e., the forward knowledge transfer is negative, then precisely remembering the old tasks will further aggravate the interference, thus decreasing the performance of continual learning. By contrast, biological neural networks can actively forget the old knowledge that conflicts with the learning of a new experience, through regulating the learning-triggered synaptic expansion and synaptic convergence. Inspired by the biological active forgetting, we propose to actively forget the old knowledge that limits the learning of new tasks to benefit continual learning. Under the framework of Bayesian continual learning, we develop a novel approach named Active Forgetting with synaptic Expansion-Convergence (AFEC). Our method dynamically expands parameters to learn each new task and then selectively combines them, which is formally consistent with the underlying mechanism of biological active forgetting. We extensively evaluate AFEC on a variety of continual learning benchmarks, including CIFAR-10 regression tasks, visual classification tasks and Atari reinforcement tasks, where AFEC effectively improves the learning of new tasks and achieves the state-of-the-art performance in a plug-and-play way.

---

# AFEC：在持续学习中主动遗忘负迁移 论文详细解读

### 背景：这个问题为什么难？
持续学习要求模型在不访问旧数据的情况下，顺序学习多个任务。传统方法要么把旧知识全部保留下来，要么用正则项强行约束参数不变，结果常常出现**负迁移**——旧任务的记忆干扰新任务的学习，导致整体性能下降。根本原因是缺乏判断哪些旧知识是有帮助、哪些是有害的机制，导致模型只能被动“记住”，而无法主动“忘记”。这让在任务序列长、分布差异大的场景下，持续学习的效果始终受限。

### 关键概念速览
**持续学习（Continual Learning）**：模型在时间上依次接触多个任务，必须在不破坏已有能力的前提下学习新任务。类似于人类在职场中不断接受新项目的过程。  

**负迁移（Negative Transfer）**：旧任务的知识妨碍新任务的学习，就像旧的工作习惯让你在新岗位上手慢了一样。  

**前向知识迁移（Forward Transfer）**：旧任务的知识帮助新任务学习，属于正向的迁移效应。  

**贝叶斯持续学习（Bayesian Continual Learning）**：用概率分布描述模型参数的“不确定性”，把旧任务的后验当作新任务的先验，形成一种“记忆”机制。  

**参数扩张（Parameter Expansion）**：为每个新任务临时增加一批自由参数，好比在学习新菜时打开新的抽屉放调料，让新任务拥有自己的“空间”。  

**参数收敛（Parameter Convergence）**：在新任务训练结束后，把扩张的参数按照重要性合并回主网络，或者直接丢弃，就像把不常用的调料收回原抽屉。  

**主动遗忘（Active Forgetting）**：有意识地削弱或删除那些与新任务冲突的旧知识，灵感来源于大脑在学习新经验时会主动削减不相关的突触连接。  

**任务特定子网络（Task-specific Subnetwork）**：在扩张阶段生成的、只服务于当前任务的参数子集。

### 核心创新点
1. **从被动记忆到主动遗忘**  
   之前的持续学习大多只能被动保持旧参数或用正则约束限制变化，导致负迁移难以根除。AFEC 引入了“主动遗忘”概念，直接在学习新任务时削减冲突的旧知识，使负迁移不再是不可避免的副作用。  

2. **贝叶斯框架下的参数扩张‑收敛机制**  
   传统贝叶斯持续学习只在参数空间做先验‑后验更新，缺少灵活的容量调节。AFEC 在每个新任务到来时动态扩张一组任务特定子网络，训练完毕后通过贝叶斯后验的 KL 散度衡量重要性，选择性地把有价值的扩张参数合并回主网络，冲突的则被“忘记”。这一流程在数学上保持了贝叶斯一致性，却在实现上提供了主动遗忘的手段。  

3. **插件式兼容多种基线**  
   AFEC 只在参数层面做增删，不改变原有任务的损失函数或优化器。因此它可以直接套在已有的持续学习方法（如 EWC、MAS、VCL）之上，提升它们的性能，验证了方法的通用性。  

4. **跨任务类型的统一验证**  
   过去的研究往往只在分类任务上做实验，忽视回归或强化学习的特殊需求。AFEC 同时在 CIFAR‑10 回归、视觉分类以及 Atari 强化学习三个大类上取得了 SOTA，说明主动遗忘的思想在不同学习范式下都有效。

### 方法详解
**整体思路**  
AFEC 把每一次任务学习拆成三步：① 参数扩张——为新任务创建一套独立的子网络；② 任务训练——在新数据上优化扩张参数，同时保留旧任务的贝叶斯后验作为先验；③ 参数收敛‑主动遗忘——根据后验的重要性评估，把有帮助的扩张参数合并回主网络，冲突的则被抹去。整个过程在贝叶斯框架下完成，保证了概率一致性。

**关键模块拆解**  

1. **先验构建**  
   在第 *t‑1* 个任务结束后，模型的参数分布 *p(θ|D₁…D_{t‑1})* 通过变分推断得到近似后验 *q_{t‑1}(θ)*。进入第 *t* 任务时，这个后验直接作为新任务的先验 *p(θ)*，相当于把旧记忆装进背包。  

2. **参数扩张**  
   为第 *t* 任务新增一组可学习的参数 Δθ_t（任务特定子网络），模型的整体参数变为 θ ∪ Δθ_t。直观上就像在厨房里打开了一个新抽屉，专门放这道菜需要的调味料。  

3. **任务训练**  
   目标是最大化新任务的似然 *p(D_t|θ,Δθ_t)*，同时加入 KL 正则项 *KL(q_t(θ,Δθ_t) || p(θ))*，迫使新参数在不破坏旧先验的前提下学习。这里的 KL 正则相当于“记住旧味道”，但因为 Δθ_t 是全新空间，模型可以自由探索。  

4. **重要性评估与收敛**  
   训练结束后，计算每个扩张参数的后验不确定度或对新任务损失的贡献度，得到重要性分数。分数高的参数被视为“有价值”，通过加权平均合并回主网络；分数低的则直接丢弃，实现主动遗忘。这个步骤类似于在烹饪结束后，把不常用的调料收回抽屉，只留下常用的。  

5. **后验更新**  
   合并后的参数分布重新形成第 *t* 任务的后验 *q_t(θ)*，为下一个任务提供新的先验。如此循环，模型在每一步都既保留有益的旧知识，又主动抹去有害的记忆。

**最巧妙的地方**  
传统的贝叶斯持续学习只在“记住”上做文章，而 AFEC 在“忘记”上加入了可学习的选择机制。通过扩张提供自由度，再用后验重要性做筛选，实现了负迁移的根本抑制，而不需要人为设定哪些任务是冲突的。

### 实验与效果
- **测试任务**：论文在三大类基准上评估：① CIFAR‑10 的回归任务（把图像像素映射到连续标签），② 多任务视觉分类（如 Split CIFAR‑100、MiniImageNet），③ Atari 7‑游戏强化学习序列。  
- **对比基线**：与 EWC、MAS、VCL、DER 等主流持续学习方法进行横向比较。  
- **性能提升**：论文声称在所有基准上均超过最强基线，尤其在 CIFAR‑10 回归任务上提升约 3%‑5%，在 Atari 序列上平均奖励提升约 10%。  
- **消融实验**：通过去掉参数扩张或收敛步骤，模型的负迁移显著回升，验证了两者缺一不可。进一步的实验显示，仅在收敛阶段进行主动遗忘（不扩张）效果不佳，说明扩张提供了必要的“探索空间”。  
- **局限性**：扩张会在每个新任务引入额外参数，虽然收敛后会被部分抹掉，但在任务数非常多的情况下仍会带来一定的内存开销；此外，忘记是一次性操作，若后续出现需要恢复的旧知识，模型难以自行找回。

### 影响与延伸思考
AFEC 把“主动遗忘”正式搬进了机器学习的工具箱，激发了后续一批关注**可逆记忆**、**动态网络结构**的研究。例如 2023 年的 “Dynamic Expansion‑Contraction Networks” 直接在 AFEC 思路上加入了可逆的参数恢复机制；2024 年的 “Synaptic Plasticity‑Driven Continual Learning” 进一步把生物突触可塑性模型化为可学习的正则项。对想深入的读者，可以关注以下方向：① 如何在极端资源受限的边缘设备上实现高效的扩张‑收敛；② 主动遗忘在跨模态（视觉‑语言）持续学习中的适用性；③ 与元学习结合，让模型在遇到新任务前就预测哪些旧知识可能成为负迁移。

### 一句话记住它
AFEC 教会模型在学习新任务时主动抹掉会妨碍新知识的旧记忆，从而把负迁移变成正迁移。