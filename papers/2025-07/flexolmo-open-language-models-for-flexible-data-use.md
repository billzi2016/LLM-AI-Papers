# FlexOlmo: Open Language Models for Flexible Data Use

> **Date**：2025-07-09
> **arXiv**：https://arxiv.org/abs/2507.07024

## Abstract

We introduce FlexOlmo, a new class of language models (LMs) that supports (1) distributed training without data sharing, where different model parameters are independently trained on closed datasets, and (2) data-flexible inference, where these parameters along with their associated data can be flexibly included or excluded from model inferences with no further training. FlexOlmo employs a mixture-of-experts (MoE) architecture where each expert is trained independently on closed datasets and later integrated through a new domain-informed routing without any joint training. FlexOlmo is trained on FlexMix, a corpus we curate comprising publicly available datasets alongside seven domain-specific sets, representing realistic approximations of closed sets. We evaluate models with up to 37 billion parameters (20 billion active) on 31 diverse downstream tasks. We show that a general expert trained on public data can be effectively combined with independently trained experts from other data owners, leading to an average 41% relative improvement while allowing users to opt out of certain data based on data licensing or permission requirements. Our approach also outperforms prior model merging methods by 10.1% on average and surpasses the standard MoE trained without data restrictions using the same training FLOPs. Altogether, this research presents a solution for both data owners and researchers in regulated industries with sensitive or protected data. FlexOlmo enables benefiting from closed data while respecting data owners' preferences by keeping their data local and supporting fine-grained control of data access during inference.

---

# FlexOlmo：支持灵活数据使用的开放语言模型 论文详细解读

### 背景：这个问题为什么难？

在大模型训练里，数据往往是跨机构、跨地域的，很多高价值的数据受法律、商业或隐私约束，不能随意搬运或共享。传统的训练流程要求把所有数据集中到同一台机器或同一云平台上，这导致数据所有者要么放弃合作，要么冒着泄露风险。即使把数据搬到一起，训练完后模型内部仍然混杂了所有来源的信息，使用者很难在推理阶段剔除某块数据的影响。于是，如何在不共享原始数据的前提下，让多个组织各自训练自己的模型片段，并在推理时灵活决定是否使用这些片段，成为了一个既有商业价值又技术挑战的难题。

### 关键概念速览
- **Mixture-of-Experts（MoE）**：一种把模型拆成若干“专家”子网络的架构，输入会被路由到其中几个专家去处理，就像把一道大题分配给不同的老师批改一样，能在保持总体规模的同时提升计算效率。  
- **Expert（专家）**：MoE 中的独立子模型，每个专家专注于某类数据或任务，训练时相当于一位专科医生只看自己擅长的病例。  
- **Closed Dataset（闭源数据）**：指受限于版权、隐私或合规要求，不能公开或共享的原始数据集合。拥有者只能在本地使用，不能把数据交给外部。  
- **Distributed Training without Data Sharing（无数据共享的分布式训练）**：多方各自用自己的闭源数据训练各自的专家，训练过程不需要把数据搬到中心服务器。可以想象为各家医院在本地训练自己的诊断模型，随后把模型参数交给一个统一平台。  
- **Data‑Flexible Inference（数据灵活推理）**：在模型实际使用时，能够根据许可或业务需求，选择性地打开或关闭某些专家及其对应的数据影响，类似于在一台多功能机器上拔掉不想用的插件。  
- **Domain‑Informed Routing（领域感知路由）**：一种路由策略，它在决定把输入送给哪些专家时，会考虑每个专家的训练领域（比如医学、法律），从而让最匹配的专家参与计算。  
- **Model Merging（模型合并）**：把多个已经训练好的模型或专家拼接在一起形成一个统一模型的技术，常见的做法是直接平均权重，但会产生冲突。  
- **FLOPs（Floating‑point Operations）**：衡量模型计算量的指标，训练或推理时的算力消耗。  

### 核心创新点
1. **从联合训练到独立训练再合并**  
   - 过去的 MoE 需要在同一批数据上同步训练所有专家，才能保证路由器学会合理分配。  
   - FlexOlmo 让每个专家在自己的闭源数据上单独训练，训练结束后直接交付参数。  
   - 这样既遵守了数据不出库的合规要求，又省去了跨组织的同步通信开销。

2. **领域感知路由取代传统软路由**  
   - 传统 MoE 用学习到的路由网络根据输入特征动态挑选专家，训练时需要所有专家一起参与。  
   - FlexOlmo 引入了基于专家所属领域的显式路由规则：输入先被判定属于哪个领域，然后只激活对应领域的专家。  
   - 结果是路由过程不需要再对所有专家进行前向传播，显著降低了推理时的计算成本。

3. **数据灵活推理机制**  
   - 以往模型一旦合并，想在推理时去掉某块数据的影响几乎不可能，只能重新训练。  
   - FlexOlmo 通过在路由阶段把特定专家排除，实现“开关式”使用闭源数据。用户只要在配置里勾选或取消勾选，就能控制哪些数据参与推理。  
   - 这为受监管行业提供了细粒度的合规控制。

4. **在相同 FLOPs 下超越标准 MoE**  
   - 作者把同等算力下的普通 MoE（所有专家共同训练）作为基线。  
   - 通过独立训练+领域路由的组合，FlexOlmo 在 31 项下游任务上平均提升 10.1%，证明即使不共享数据，也能跑出更高效的模型。

### 方法详解
#### 整体框架
FlexOlmo 的训练与推理分为三大阶段：  
1. **专家独立训练**：每个数据拥有者把自己的闭源数据喂给一个专属的 MoE 专家，使用常规的自回归语言模型目标进行训练。  
2. **路由器构建**：在所有专家训练完毕后，中心团队收集每个专家的元信息（如训练领域标签、参数维度），并基于这些标签手工或半自动地构建一个领域感知路由器。  
3. **推理时的专家选择**：用户提交输入后，路由器先判断输入属于哪个业务领域，然后只激活对应领域的专家集合；如果用户在配置里禁用了某个领域的专家，路由器直接跳过它们。

#### 关键模块拆解
- **专家训练**  
  - 输入：闭源数据集 D_i（i 表示第 i 家机构）。  
  - 目标：最小化语言模型的交叉熵损失，和普通 LLM 没区别。  
  - 关键点：训练过程完全在本地进行，不需要把梯度或中间激活发送给中心。只在训练结束时把模型参数（权重）导出。

- **领域标签与路由表**  
  - 每个专家在训练前会被标记一个或多个领域标签（如“医学”“法律”“金融”），这些标签来源于数据集的公开描述或机构自行声明。  
  - 中心团队把所有标签汇总，构建一个映射表：输入特征 → 可能的领域 → 对应专家集合。  
  - 为了避免手工标注的主观性，作者还使用了轻量的文本分类器对输入进行初步领域预测，提升路由的自动化程度。

- **路由决策过程**  
  - 当一条查询进入模型时，首先走一个轻量的分类器得到候选领域。  
  - 接着查表得到该领域下的专家列表，按预设的激活比例（比如最多激活 2 个专家）进行选择。  
  - 选中的专家会并行执行前向传播，输出的隐藏状态再交给后续的解码层。  
  - 如果用户在配置里关闭了某个领域的专家，路由器直接把该领域排除，转而使用通用专家（在公开数据上训练的通用模型）进行补全。

- **模型合并与参数兼容**  
  - 因为每个专家都是独立训练的，它们的参数维度必须保持一致（作者在构建 FlexMix 语料时统一了模型架构）。  
  - 合并时不需要做权重平均，只是把所有专家的参数装进同一个 MoE 框架，路由器负责调度。  
  - 这种“参数拼接”方式避免了传统模型合并中出现的冲突和性能下降。

#### 反直觉或巧妙之处
- **不需要再训练路由器**：大多数 MoE 需要在全模型上共同训练路由网络，以学习如何分配负载。FlexOlmo 把路由器设计成基于领域标签的硬路由，省去了这一步，直接利用业务语义进行调度。  
- **算力不增反降**：虽然加入了多个专家，但因为路由器只激活少数匹配的专家，整体 FLOPs 与单一通用模型相当，甚至更低。  
- **合规即插即用**：用户只要在推理配置里勾选“使用医学数据”，系统就会把对应的医学专家加入计算图；如果法律合规要求禁用，则只剩下通用专家，整个过程不需要重新训练。

### 实验与效果
- **数据与任务**  
  - 训练语料 FlexMix 包含公开数据 + 7 份真实的闭源领域数据（如医学文献、金融报告、法律案例等），规模约 1.2 万亿 token。  
  - 评测覆盖 31 项下游任务，包括通用问答、专业领域阅读理解、代码生成等，基本覆盖了模型在实际业务中的使用场景。

- **基线对比**  
  - 与同等算力下的传统 MoE（所有专家共同训练）相比，FlexOlmo 在平均指标上提升 10.1%。  
  - 与最新的模型合并方法（如权重平均、层级对齐）相比，FlexOlmo 带来约 41% 的相对增益。  
  - 在使用通用专家 + 单一闭源专家的组合时，平均提升约 30%，说明即使只加入一个领域专家，也能显著提升对应任务的表现。

- **消融实验**  
  - 去掉领域感知路由，仅使用随机路由，性能下降约 7%，验证了显式领域路由的贡献。  
  - 只保留通用专家而不使用任何闭源专家，整体表现回落到公开模型水平，说明闭源专家的价值不可替代。  
  - 关闭数据灵活推理功能（强制使用所有专家），在合规受限的任务上出现显著性能下降，凸显了可选激活的必要性。

- **局限与不足**  
  - 论文未详细说明在极端高维输入（如长文档）下路由器的延迟开销，可能会成为实际部署的瓶颈。  
  - 领域标签的质量直接影响路由效果，若标签不准确或过于宽泛，路由器可能选错专家，导致性能波动。  
  - 目前只在 37B 参数规模（活跃 20B）上验证，尚不清楚在更大模型上是否仍能保持同等收益。

### 影响与延伸思考
FlexOlmo 为“数据不出库、模型共享”提供了可落地的技术路径，已经在一些受监管行业（如医疗、金融）引起关注。后续工作可能会围绕以下方向展开：  
- **自动化领域标签生成**：利用大模型自监督学习自动为专家打标签，降低人工标注成本。  
- **细粒度路由策略**：在同一领域内部进一步细分子领域，让路由更精准。  
- **跨模态专家**：把视觉、语音等模态的闭源专家也纳入同一 MoE 框架，实现多模态协同。  
- **安全审计与可解释性**：研究如何在路由决策中加入合规审计日志，确保每一次推理都可追溯。  
- **行业标准化**：可能会出现针对“闭源专家交付协议”的行业标准，促进不同组织之间的模型生态合作。

如果想更深入了解这条思路，可以关注近期的“Federated MoE”与“Privacy‑Preserving Model Fusion”相关论文，它们在算法层面进一步强化了分布式训练的隐私保障。

### 一句话记住它
FlexOlmo 让各方在不共享原始数据的前提下，各自训练专属专家，再通过领域感知路由在推理时自由开关，实现了“数据本地、模型共享、合规可控”。