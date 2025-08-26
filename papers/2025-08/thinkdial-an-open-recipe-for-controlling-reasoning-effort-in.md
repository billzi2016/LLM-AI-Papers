# ThinkDial: An Open Recipe for Controlling Reasoning Effort in Large Language Models

> **Date**：2025-08-26
> **arXiv**：https://arxiv.org/abs/2508.18773

## Abstract

Large language models (LLMs) with chain-of-thought reasoning have demonstrated remarkable problem-solving capabilities, but controlling their computational effort remains a significant challenge for practical deployment. Recent proprietary systems like OpenAI's gpt-oss series have introduced discrete operational modes for intuitive reasoning control, but the open-source community has largely failed to achieve such capabilities. In this paper, we introduce ThinkDial, the first open-recipe end-to-end framework that successfully implements gpt-oss-style controllable reasoning through discrete operational modes. Our system enables seamless switching between three distinct reasoning regimes: High mode (full reasoning capability), Medium mode (50 percent token reduction with <10 percent performance degradation), and Low mode (75 percent token reduction with <15 percent performance degradation). We achieve this through an end-to-end training paradigm that integrates budget-mode control throughout the entire pipeline: budget-mode supervised fine-tuning that embeds controllable reasoning capabilities directly into the learning process, and two-phase budget-aware reinforcement learning with adaptive reward shaping. Extensive experiments demonstrate that ThinkDial achieves target compression-performance trade-offs with clear response length reductions while maintaining performance thresholds. The framework also exhibits strong generalization capabilities on out-of-distribution tasks.

---

# ThinkDial：可控推理开源方案 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在加入思维链（CoT）后能像人一样一步步推理，解决了很多复杂任务。但推理过程往往会生成大量冗余的 token，导致算力和响应时间飙升。商业系统（如 OpenAI 的 gpt‑oss）已经提供了“高/中/低”三档的推理模式，让用户可以在速度和准确率之间权衡；开源社区却一直缺少类似的可控机制。之前的办法要么是手动截断输出，要么是通过提示词让模型自行压缩，这会导致性能不可预测、难以在生产环境中保证服务质量。因此，如何在保持可接受的解题水平的同时，显式控制推理所消耗的计算资源，成为了一个迫切需要系统性解决的问题。

### 关键概念速览
- **大语言模型（LLM）**：能够生成自然语言的深度神经网络，规模从几百亿到上千亿参数不等。把它想象成一个“会说话的百科全书”，可以根据输入生成连贯的文字。
- **思维链（CoT）**：让模型在给出答案前先写出推理步骤，类似于学生解题时的草稿，能够显著提升复杂问题的正确率。
- **离散操作模式（Operational Mode）**：预设的几种推理配置（High、Medium、Low），每种模式对应不同的 token 使用上限和性能目标，就像手机的省电、均衡、性能三档。
- **预算模式监督微调（Budget‑Mode Supervised Fine‑Tuning）**：在有标签的训练数据上加入“模式标签”，让模型在学习答案的同时学会在指定预算下生成答案。
- **预算感知强化学习（Budget‑Aware Reinforcement Learning）**：把推理过程视为一个决策序列，奖励函数同时考虑答案正确性和 token 消耗，模型在试错中学会在不同预算下做最优取舍。
- **自适应奖励塑形（Adaptive Reward Shaping）**：根据当前模式的目标（如 50% token 减少）动态调节奖励的权重，使模型在不同模式之间切换时不会出现突兀的性能跌落。
- **Out‑of‑Distribution（OOD）任务**：训练时未见过的任务类型，检验模型的泛化能力。相当于让学生在全新科目上仍能运用已有的解题技巧。

### 核心创新点
1. **从手工提示到模式标签的转变**  
   之前的开源方案只能通过提示词让模型自行压缩输出，效果不稳定。ThinkDial 在监督微调阶段直接在每条训练样本上标记“High/Medium/Low”，让模型在学习答案的同时内化不同预算的生成策略。这样模型在推理时只需切换模式标签，就能自动调节生成长度，省去了繁琐的提示工程。

2. **两阶段预算感知强化学习**  
   传统的强化学习只奖励答案正确性，忽视了计算开销。ThinkDial 在第二阶段引入预算感知奖励：正确答案得高分，超出预算的 token 会被扣分。通过两轮训练（先让模型熟悉任务，再让它学会在预算约束下优化），实现了在保持 90%+ 正确率的前提下显著削减 token。

3. **自适应奖励塑形机制**  
   直接把 token 数量写进奖励会导致梯度噪声大，训练不收敛。作者设计了一个随模式动态调节的奖励权重，使得在 High 模式下几乎不惩罚 token，而在 Low 模式下惩罚力度加大。这个技巧让模型在不同模式之间切换时表现平滑，避免了“高模式好、低模式崩”的现象。

4. **统一端到端的开源实现**  
   过去的可控推理大多是闭源或仅提供模型权重，缺少完整的训练、评估流水线。ThinkDial 把预算模式标签、监督微调、强化学习、奖励塑形全部封装在一个可复现的代码库里，真正实现了“开源配方”。这让社区可以直接在自己的模型上复刻或改进，而不必从头实现每个子模块。

### 方法详解
ThinkDial 的整体思路可以拆成三大步骤：**数据标记 → 预算模式监督微调 → 预算感知强化学习**。下面按顺序展开每一步的细节。

1. **数据标记（Budget‑Mode Tagging）**  
   - 选取已有的 CoT 训练集（如 GSM‑8K、MathQA），每条样本都包含问题、思维链、答案。  
   - 根据经验设定三种 token 上限：High（不限制），Medium（目标比原始长度削减 50%），Low（目标削减 75%）。  
   - 为每条样本生成三版输出：原始完整 CoT、压缩版（通过人工或自动摘要得到），以及极简版（只保留关键步骤）。  
   - 将对应的模式标签（`<mode:high>`、`<mode:mid>`、`<mode:low>`）插入输入前缀，使模型在训练时看到“我现在要在 High 模式下思考”。

2. **预算模式监督微调（Budget‑Mode SFT）**  
   - 使用标准的自回归语言模型微调流程，目标是最大化标签化数据的似然。  
   - 关键在于**模式标签的嵌入**：模型的词表里新增三个特殊 token，训练时它们的梯度会影响后续生成的长度分布。  
   - 为防止模型在 Medium/Low 模式下仍产生冗余，训练时加入**长度正则项**：对生成的 token 数做轻微惩罚，但只在非 High 模式下激活。  
   - 结果是模型在不同模式下已经具备了“粗略的预算感知”，即在 Medium 模式会倾向于省略次要步骤，在 Low 模式会只保留核心推理。

3. **预算感知强化学习（Budget‑Aware RL）**  
   - **环境设计**：每一次推理被视为一次交互，模型在每一步生成一个 token，环境返回两个信号：① 当前答案的正确性（通过答案比对或自动评估器），② 已使用的 token 数。  
   - **奖励函数**：`R = α * correctness - β * (tokens_used / token_budget)`。α、β 随模式动态调节：High 模式 α≈1, β≈0；Medium 模式 α≈0.9, β≈0.1；Low 模式 α≈0.8, β≈0.3。这样模型在 Low 模式下会更在意节约 token。  
   - **自适应奖励塑形**：在训练初期，β 设得较小，防止模型只追求短句而牺牲正确率；随着训练进度，β 按预设曲线上升，使模型逐步学会在预算约束下保持答案质量。  
   - **两阶段训练**：第一阶段只用 SFT 参数做初始化，第二阶段用 PPO（Proximal Policy Optimization）进行策略微调。PPO 能在保持旧策略不变的前提下安全探索新策略，适合这种兼顾正确率和资源的多目标优化。

4. **推理时的模式切换**  
   - 用户只需在请求体中指定 `mode=high|mid|low`，系统在前缀插入相应的模式 token，随后模型按照训练时学到的预算策略生成答案。  
   - 为防止极端情况下仍超预算，系统在生成过程中监控 token 数，一旦接近上限会强制触发 **截断策略**（如提前结束句子、返回当前最可信的答案），保证硬性资源约束。

**最巧妙的点**在于把预算约束直接写进奖励函数并随模式自适应，而不是事后硬截。这样模型内部已经形成了“在 Low 模式下思考更简洁”的习惯，截断只是一种保险措施。

### 实验与效果
- **测试任务**：作者在数学推理（GSM‑8K、MathQA）、逻辑推理（ARC‑E）以及代码生成（HumanEval）上评估。所有任务都采用标准的 CoT 基准，比较不同模式下的解答质量和 token 使用量。  
- **对比基线**：与原始未压缩的 CoT 模型、仅使用提示词压缩的开源方案以及 OpenAI gpt‑oss 的官方报告进行对比。  
- **主要结果**：论文声称在 Medium 模式下平均 token 数下降约 50%，正确率下降不足 10%；在 Low 模式下 token 数下降约 75%，正确率下降不足 15%。在所有测试集上，ThinkDial 的压缩效果与闭源 gpt‑oss 相当，但保持了完全开源的可复现性。  
- **消融实验**：作者分别去掉预算模式标签、去掉长度正则、以及只使用单阶段强化学习。结果显示，去掉标签会导致 Low 模式下的 token 减少率跌至 30% 以下，正确率下降超过 25%；去掉自适应奖励塑形会使模型在 Low 模式下出现“过度截断”，正确率骤降 20%。这些实验表明每个模块都是实现目标的关键。  
- **局限性**：论文承认在极端长文本（如 10k token 以上的文档摘要）上，模式切换的效果仍不稳定；此外，奖励函数中的 α、β 参数需要手动调节，尚未实现自动元学习。  

### 影响与延伸思考
ThinkDial 首次提供了完整的开源实现，让社区能够在自己的 LLM 上快速部署可控推理功能。自发布后，多个开源项目（如 LLaMA‑Adapter、OpenChatKit）开始加入类似的“预算模式”插件，推动了资源敏感型 AI 服务的落地。后续研究可能会在以下方向继续深化：  
- **自动预算调度**：根据实时负载或用户付费等级动态决定使用哪种模式。  
- **多模态预算控制**：把 token 预算扩展到视觉或音频生成的计算预算。  
- **元强化学习**：让模型自行学习 α、β 的最优组合，进一步降低人工调参成本。  

如果想深入了解，可关注近期在 *NeurIPS*、*ICLR* 上出现的 “Efficient LLM Inference” 系列论文，它们在算力约束下的训练技巧与 ThinkDial 的思路高度相似。

### 一句话记住它
ThinkDial 用模式标签和预算感知强化学习，让开源大模型在高、低算力之间自由切换，既省钱又不掉太多分。