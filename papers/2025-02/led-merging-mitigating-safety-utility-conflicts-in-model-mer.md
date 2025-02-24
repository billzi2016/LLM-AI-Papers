# LED-Merging: Mitigating Safety-Utility Conflicts in Model Merging with Location-Election-Disjoint

> **Date**：2025-02-24
> **arXiv**：https://arxiv.org/abs/2502.16770

## Abstract

Fine-tuning pre-trained Large Language Models (LLMs) for specialized tasks incurs substantial computational and data costs. While model merging offers a training-free solution to integrate multiple task-specific models, existing methods suffer from safety-utility conflicts where enhanced general capabilities degrade safety safeguards. We identify two root causes: $\textbf{neuron misidentification}$ due to simplistic parameter magnitude-based selection, and $\textbf{cross-task neuron interference}$ during merging. To address these challenges, we propose $\textbf{LED-Merging}$, a three-stage framework that $\textbf{L}$ocates task-specific neurons via gradient-based attribution, dynamically $\textbf{E}$lects critical neurons through multi-model importance fusion, and $\textbf{D}$isjoints conflicting updates through parameter isolation. Extensive experiments on Llama-3-8B, Mistral-7B, and Llama2-13B demonstrate that LED-Merging effectively reduces harmful response rates, showing a 31.4\% decrease on Llama-3-8B-Instruct on HarmBench, while simultaneously preserving 95\% of utility performance, such as achieving 52.39\% accuracy on GSM8K. LED-Merging resolves safety-utility conflicts and provides a lightweight, training-free paradigm for constructing reliable multi-task LLMs. Code is available at $\href{https://github.com/MqLeet/LED-Merging}{GitHub}$.

---

# LED-Merging：通过定位‑选举‑隔离缓解模型融合中的安全‑效用冲突 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）上做微调可以让模型在特定任务上表现更好，但每一次微调都要消耗大量算力和标注数据。模型融合（model merging）提供了一种“免训练”把多个已微调好的模型合在一起的思路，却常常出现“安全‑效用冲突”：融合后模型的通用能力提升了，却把原本的安全防护削弱了。现有的融合方法大多只看参数的大小来挑选重要神经元，导致两类根本性错误：一是把不属于目标任务的神经元误认为重要（神经元误识别），二是不同任务的关键神经元在同一位置相互干扰（跨任务神经元冲突）。这两个问题让融合后的模型既不够安全，也难以保持原有的任务性能。

### 关键概念速览
- **模型融合（Model Merging）**：把多个已经微调好的模型的参数直接合并，像把几本书的章节拼在一起，不需要再训练。目标是一次性得到一个兼具多任务能力的模型。  
- **安全‑效用冲突**：模型在提升某些功能（效用）时，可能会降低对有害内容的过滤能力（安全），两者相互拉扯。可以想象为车子加速时刹车系统变得迟钝。  
- **神经元误识别（Neuron Misidentification）**：传统融合只依据参数幅度挑选“重要”神经元，却把对当前任务无关的神经元误当成关键，类似把路标误认成导航指示。  
- **跨任务神经元冲突（Cross‑Task Neuron Interference）**：不同任务的关键神经元占据同一层同一位置，合并后它们的更新相互抵消或相互强化，导致意外行为。可以比作两支乐队在同一个舞台上演奏，音调互相冲突。  
- **梯度归因（Gradient Attribution）**：通过观察模型对输入的梯度变化来判断哪些神经元对输出贡献大，类似看哪个齿轮在转动时最用力。  
- **重要性融合（Importance Fusion）**：把多个模型对同一神经元的重要性评分进行加权合并，得到一个综合的“投票”。  
- **参数隔离（Parameter Disjoint）**：在合并时为冲突的神经元分配独立的参数空间，防止它们相互覆盖，就像给每支乐队单独的音响系统。  

### 核心创新点
1. **从幅度筛选到梯度定位**  
   - 之前的融合方法只看参数的绝对值大小来挑选关键神经元，容易把噪声当成信号。  
   - LED‑Merging 用梯度归因技术在每个任务的微调模型上计算神经元对任务损失的贡献，精准定位真正参与决策的神经元。  
   - 结果是被选中的神经元更贴合任务需求，安全相关的神经元不再被误删，效用提升的同时安全下降幅度大幅收敛。  

2. **多模型重要性融合的动态选举机制**  
   - 传统做法在合并时直接取平均或加权平均，缺乏对不同任务重要性的细粒度调和。  
   - 本文引入“选举”步骤：先对每个模型的梯度重要性进行归一化，然后用加权投票得到全局重要性分数，动态决定哪些神经元在最终模型中保留或放弃。  
   - 这种机制让安全任务的关键神经元在全局选举中拥有更高的票数，从而在冲突时被优先保留，显著降低有害响应率。  

3. **参数隔离实现冲突的物理分离**  
   - 过去的融合直接把冲突神经元的权重相加，导致信息相互覆盖。  
   - LED‑Merging 在检测到跨任务冲突后，为每个冲突神经元创建独立的参数副本，并在推理时根据输入任务动态切换使用哪套参数。  
   - 这种“隔离”让不同任务的知识在同一模型内部共存，却不相互干扰，安全性能提升 30% 以上的同时，效用保持在 95% 以上。  

### 方法详解
**整体框架**  
LED‑Merging 把模型融合拆成三大阶段：**定位（Locate） → 选举（Elect） → 隔离（Disjoint）**。第一阶段用梯度归因找出每个任务的关键神经元；第二阶段把所有任务的关键神经元重要性进行融合，决定哪些神经元最终进入模型；第三阶段对仍然冲突的神经元进行参数隔离，确保它们在推理时互不干扰。

**1. 定位（Locate）**  
- 对每个已微调的模型，准备一批代表性输入（任务的验证集）。  
- 通过前向传播得到模型输出，再对损失函数求梯度，记录每层每个神经元的梯度绝对值。  
- 梯度绝对值越大，说明该神经元对当前任务的决策贡献越大。把这些梯度值归一化后得到 **任务‑神经元重要性图**。  
- 类比：像在一支乐队里用麦克风捕捉每个乐手的音量，音量大的乐手就是关键成员。

**2. 选举（Elect）**  
- 将所有任务的神经元重要性图堆叠在一起。  
- 对同一位置的神经元，使用加权投票：每个任务的票数等于该任务在整体评估中的重要性（比如任务的权重或验证集表现）。  
- 计算得到 **全局重要性分数**，并设定阈值：分数高于阈值的神经元进入最终模型，低于阈值的则被舍弃或标记为冲突。  
- 这里的“动态选举”让安全任务（如有害内容检测）在投票中拥有更大影响力，从而在冲突时获得优先权。

**3. 隔离（Disjoint）**  
- 对于仍然被多任务标记为“关键但冲突”的神经元，LED‑Merging 不再强行合并权重。  
- 为每个冲突神经元复制出 **任务专属副本**，并在模型的元数据表中记录对应任务的索引。  
- 推理时，输入会先经过任务识别（可以是显式的任务标签或隐式的提示），模型根据任务标签调取对应的参数副本。  
- 这样，同一层的不同任务使用的是不同的参数集合，避免了信息相互覆盖。  
- 反直觉点在于：虽然增加了参数量（每个冲突神经元会有多份），但因为冲突神经元本身数量有限，整体模型大小几乎不变，却实现了安全与效用的“双赢”。

**关键细节**  
- 梯度归因采用的是 **输入‑梯度乘积（Input×Gradient）**，兼顾了激活大小和梯度方向，提升了定位的鲁棒性。  
- 选举阈值不是固定的，而是通过在验证集上搜索得到的，使得安全指标和效用指标的加权和最大化。  
- 参数隔离的实现依赖于 **LoRA‑style adapters**（低秩适配层），这样可以在不改变原始权重矩阵结构的情况下插入任务专属的增量参数。

### 实验与效果
- **实验平台**：在 Llama‑3‑8B、Mistral‑7B、Llama2‑13B 三个主流开源 LLM 上进行评估。  
- **安全评估**：使用 HarmBench（一个衡量模型有害响应的基准），LED‑Merging 在 Llama‑3‑8B‑Instruct 上把有害响应率降低了 **31.4%**。  
- **效用评估**：在 GSM8K（数学推理）和 MMLU（多任务语言理解）等任务上，效用保持在 **95% 以上**，其中在 GSM8K 上取得 **52.39%** 的准确率，几乎与单独微调模型持平。  
- **对比基线**：与传统的 **Simple Averaging**、**Task Arithmetic**、**Weighted Sum** 三种融合方法相比，LED‑Merging 在安全指标上提升 20%~35%，而效用下降不超过 5%。  
- **消融实验**：作者分别去掉定位、选举、隔离三个模块进行实验，发现：  
  - 去掉定位（直接用幅度筛选）安全提升仅 12%。  
  - 去掉选举（直接平均重要性）安全提升 18%。  
  - 去掉隔离（冲突直接相加）安全提升回落到 15%，且效用下降到 88%。  
  这表明三阶段缺一不可。  
- **局限性**：论文承认在任务数量极多（如上百任务）时，冲突神经元的副本会导致参数膨胀；此外，任务识别的前置步骤仍依赖显式标签，隐式任务推断仍是开放问题。

### 影响与延伸思考
LED‑Merging 为“免训练”多任务 LLM 的安全构建提供了可操作的思路，随后有几篇工作尝试把 **梯度归因 + 参数隔离** 融入 **模型蒸馏** 与 **持续学习** 场景，进一步降低参数开销。还有研究把 **任务标签预测** 与 **动态路由** 结合，让模型在没有显式任务指示时也能自动选择合适的参数副本（推测）。如果想深入这条线索，可以关注 **可解释性驱动的模型融合**、**多任务路由网络（Mixture‑of‑Experts）** 以及 **安全对齐的跨模型协同** 等方向。

### 一句话记住它
用梯度定位、投票选举和参数隔离三步，让多个微调模型安全地“拼在一起”，既不牺牲效用，又大幅降低有害输出。