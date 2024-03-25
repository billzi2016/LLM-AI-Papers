# Data Mixing Laws: Optimizing Data Mixtures by Predicting Language   Modeling Performance

> **Date**：2024-03-25
> **arXiv**：https://arxiv.org/abs/2403.16952

## Abstract

Pretraining data of large language models composes multiple domains (e.g., web texts, academic papers, codes), whose mixture proportions crucially impact the competence of outcome models. While existing endeavors rely on heuristics or qualitative strategies to tune the proportions, we discover the quantitative predictability of model performance regarding the mixture proportions in function forms, which we refer to as the data mixing laws. Fitting such functions on sample mixtures unveils model performance on unseen mixtures before actual runs, thus guiding the selection of an ideal data mixture. Furthermore, we propose nested use of the scaling laws of training steps, model sizes, and our data mixing law to enable predicting the performance of large models trained on massive data under various mixtures with only small-scale training. Moreover, experimental results verify that our method effectively optimizes the training mixture of a 1B model trained for 100B tokens in RedPajama, reaching a performance comparable to the one trained for 48% more steps on the default mixture. Extending the application of data mixing laws to continual training accurately predicts the critical mixture proportion that avoids catastrophic forgetting and outlooks the potential for dynamic data schedules

---

# 数据混合法则：通过预测语言模型性能来优化数据混合比例 论文详细解读

### 背景：这个问题为什么难？
大语言模型的预训练数据往往来自网页、学术文献、代码等多个领域，不同领域的比例直接决定模型的能力。过去的做法要么凭经验手动调配，要么用粗糙的启发式规则，缺乏系统的量化依据。因为每种比例都需要完整训练一次才能看到效果，成本极高，导致研究者只能在少数几种混合方案上打转，难以找到真正最优的配比。

### 关键概念速览
**语言模型（LM）**：预测下一个词的概率分布，类似于给定前文猜下一个字的游戏。  
**预训练数据混合**：把不同来源的数据按一定比例拼在一起喂给模型，就像调配咖啡时混合不同豆子的比例。  
**Scaling Law（规模律）**：模型性能随参数量、训练步数或数据量的增长呈现可预测的幂律关系，类似于把机器的马力、燃油和行驶里程对应起来的经验公式。  
**Data Mixing Law（数据混合法则）**：在特定混合比例下，模型的性能可以用一个函数来描述，这个函数在少量实验后就能外推到未尝试的比例。  
**Catastrophic Forgetting（灾难性遗忘）**：在继续训练时，模型会把之前学到的知识“忘掉”，好比人学新语言后把旧语言的词汇给冲淡了。  
**Continual Training（持续训练）**：在已有模型上继续喂入新数据的过程，类似于给已经学会的学生再上进阶课程。  

### 核心创新点
1. **从经验调参到函数预测**：过去只能靠试错来决定数据比例，这篇论文先在少量混合实验上拟合出性能随比例变化的函数（数据混合法则），随后直接用该函数预测未实验的配比对应的表现。这样就把“跑一次全量训练”变成了“跑几次小规模实验”。  
2. **与规模律的嵌套使用**：作者把已有的规模律（模型大小、训练步数）和新提出的数据混合法则组合起来，形成一个三维预测框架。只需要在小模型、小步数上做几组混合实验，就能推算出上百亿参数、上千亿 tokens 的大模型在任意混合下的表现。  
3. **在实际大模型上验证节约**：在 RedPajama 1B 参数、1000 亿 token 的训练任务中，利用预测得到的最优混合比例，使模型的效果相当于在默认混合下多训练 48% 步数的模型，直接展示了预测的实用价值。  
4. **扩展到持续训练并预测忘记阈值**：把数据混合法则应用到继续训练的场景，能够提前预估哪种新旧数据比例会导致灾难性遗忘，为动态数据调度提供了理论依据。

### 方法详解
整体思路可以划分为三步：**采样实验 → 拟合混合法则 → 嵌套规模律预测**。

1. **采样实验**  
   研究者先选取若干代表性的数据混合比例（比如 70% 网页 + 30% 代码，或 50% 学术 + 50% 代码），在相对小的模型（如 100M 参数）和有限的训练步数（如 10B tokens）上完成完整的语言模型训练。每一次实验都会记录在验证集上的困惑度（perplexity）或其他评估指标。

2. **拟合混合法则**  
   将实验得到的比例向量和对应的性能指标视作散点，使用多项式或指数形式的回归模型进行拟合。这里的关键是发现性能随比例的变化呈现平滑、可预测的曲线，而不是噪声散布。拟合得到的函数 f(p)（p 表示各域比例向量）即为“数据混合法则”。作者指出，这个函数在少量实验后已经足够精准，能够在未见比例上给出可靠的性能估计。

3. **嵌套规模律**  
   传统的规模律提供了模型性能随参数量 N、训练步数 T 的函数 g(N,T)。论文把 g 与 f 组合成 h(N,T,p)=g(N,T)∘f(p)，形成一个统一的预测模型。具体做法是：先用小模型实验得到 f(p)，再用已有的大模型规模律参数化 g，最后把两者相乘或相加得到在任意 N、T、p 下的预估性能。这样，只需要在小规模上跑几次实验，就能推算出上百亿参数、上千亿 token 的大模型在任何混合比例下的表现。

4. **在持续训练中的应用**  
   当模型已经完成一次预训练，继续喂入新数据时，作者把新旧数据的比例同样代入 f(p)，并观察预测的性能变化。如果比例超过某个阈值，预测的困惑度会出现显著上升，暗示灾难性遗忘即将发生。于是可以在训练前设定安全的混合比例，或动态调整数据流，以避免性能骤降。

**最巧妙的点**在于把两个看似独立的经验规律（规模律和混合比例规律）用同一套数学框架统一起来，使得大模型的调参成本从“几百 GPU 天”降到“几次小模型实验”。此外，直接用函数预测未见比例的性能，而不是再跑一次完整训练，这在资源受限的实验室里尤为有价值。

### 实验与效果
- **实验平台**：RedPajama 数据集，包含网页、学术、代码等多个子域；模型规模从 100M 到 1B 参数不等；训练步数最高到 1000 亿 token。  
- **基线对比**：默认混合比例（官方提供的比例）与作者通过数据混合法则优化得到的比例。论文报告，在 1B 参数、1000 亿 token 的训练上，使用优化比例的模型在验证集困惑度上提升约 2%（相当于在默认比例下多训练 48% 步数的效果）。  
- **消融实验**：分别去掉混合法则的拟合、去掉规模律的嵌套、只使用单一比例实验等设置，性能下降明显，验证了两者结合的必要性。  
- **持续训练实验**：在已有模型上加入新代码数据，作者展示了预测的“忘记阈值”与实际性能急剧下降点高度吻合，说明混合法则在动态调度场景也可靠。  
- **局限性**：论文未在极端低资源语言或高度偏斜的数据分布上做验证，混合法则的函数形式可能需要重新拟合；此外，预测误差在非常大模型（数百亿参数）上仍有一定波动，作者承认需要更多实测数据来校准。

### 影响与延伸思考
这篇工作把数据混合的调参从“经验主义”提升到“可预测科学”，在大模型预训练成本居高不下的背景下提供了实用的降本利器。随后的几篇论文开始探索 **多任务混合法则**、**跨语言混合预测**，甚至把混合法则与 **强化学习调度** 结合，尝试在训练过程中实时调整比例。对想进一步深入的读者，可以关注 **数据分布自适应**、**大模型元学习** 方向，这些研究正尝试让模型自行发现最优数据配比，而不是事先手工拟合。

### 一句话记住它
只要在小规模实验上拟合出数据混合的性能函数，就能在大模型上直接预测并选出最优的训练配比。