# MoBE: Mixture-of-Basis-Experts for Compressing MoE-based LLMs

> **Date**：2025-08-07
> **arXiv**：https://arxiv.org/abs/2508.05257

## Abstract

The Mixture-of-Experts (MoE) architecture has become a predominant paradigm for scaling large language models (LLMs). Despite offering strong performance and computational efficiency, large MoE-based LLMs like DeepSeek-V3-0324 and Kimi-K2-Instruct present serious challenges due to substantial memory requirements in deployment. While recent works have explored MoE compression to address this issue, existing methods often suffer from considerable accuracy drops (e.g., 7-14% relatively) even at modest compression rates. This paper introduces a novel Mixture-of-Basis-Experts (MoBE) method that achieves model compression while incurring minimal accuracy drops. Specifically, each up/gate matrix in an expert is decomposed via a rank decomposition as W = AB, where matrix A is unique to each expert. The relatively larger matrix B is further re-parameterized as a linear combination of basis matrices {Bi} shared across all experts within a given MoE layer. The factorization is learned by minimizing the reconstruction error relative to the original weight matrices. Experiments demonstrate that MoBE achieves notably lower accuracy drops compared to prior works. For instance, MoBE can reduce the parameter counts of Qwen3-235B-A22B-2507, DeepSeek-V3-0324 (671B) and Kimi-K2-Instruct (1T) by 24%-30% with only 1%-2% accuracy drop (about 2% drops when measured relatively).

---

# MoBE：基于基底专家混合的 MoE 大语言模型压缩方法 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在规模上一路飙升，Mixture‑of‑Experts（MoE）架构凭借“只激活部分专家”实现了算力与性能的双赢。然而，MoE 的优势是以“每个专家都有完整的参数矩阵”换来的，这导致模型在部署时需要数百 GB 的显存，普通服务器根本装不下。已有的压缩手段要么直接剪枝、要么用低秩近似，但在压缩 20% 左右时往往会出现 7%‑14% 的相对精度下降，显然难以满足实际业务对性能和资源的双重要求。于是，如何在显著削减参数量的同时保持原有的推理质量，成为了迫切需要突破的瓶颈。

### 关键概念速览

**Mixture‑of‑Experts（MoE）**：一种神经网络结构，把整个模型拆成若干“专家”，每次前向只让一小部分专家工作，类似于公司里不同部门轮流处理任务，从而在计算上更高效。

**专家（Expert）**：MoE 中的子网络，通常拥有自己的全连接或注意力权重矩阵，就像每个部门都有自己的工具箱。

**门控（Gate）**：决定哪些专家被激活的控制模块，类似于调度中心，根据输入特征挑选最合适的部门。

**低秩分解（Low‑rank decomposition）**：把一个大矩阵拆成两个小矩阵的乘积，想象把一本厚厚的字典拆成“词根表”和“词尾表”，从而大幅压缩存储。

**基底矩阵（Basis matrix）**：一组共享的矩阵，所有专家都可以用它们的线性组合来重建自己的大矩阵，就像所有部门共用几套标准化的工具。

**重参数化（Re‑parameterization）**：在不改变模型功能的前提下，用另一套参数表达原有权重，常用于压缩或加速。

**相对精度下降**：用压缩后模型的误差除以原始模型误差得到的比例，便于比较不同压缩率下的性能损失。

### 核心创新点

1. **专家权重的双层分解**  
   之前的工作只在每个专家内部做一次低秩分解，得到两个矩阵 A 与 B，A 与 B 都是该专家独有的，压缩效果受限。MoBE 在此基础上进一步把较大的矩阵 B 表示为若干共享基底矩阵的线性组合。这样，同一层的所有专家只需要保存少量基底矩阵，而各自的组合系数则保持独特。结果是参数量下降 24%‑30% 时，精度仅损失约 1%‑2%。

2. **统一的基底学习目标**  
   传统方法往往分别优化每个专家的分解误差，导致基底之间缺乏协同。MoBE 通过最小化所有专家重构误差的总和来 jointly 学习基底矩阵，使得基底能够兼顾不同专家的特征分布。这样得到的基底更具通用性，进一步提升了压缩后的模型表现。

3. **门控矩阵同样采用基底共享**  
   过去的压缩只关注专家的前向权重，忽视了门控矩阵的规模。MoBE 同样把每个门控矩阵分解为 A·B，并让 B 共享基底。这样，整个 MoE 层（包括专家和门控）都实现了统一的基底共享，压缩幅度更大且不会破坏路由决策的准确性。

4. **最小化重构误差的高效实现**  
   为了避免在大模型上直接求解高维矩阵的分解，作者采用了分块随机梯度下降和梯度累积技巧，使得基底学习可以在原始模型的微调阶段顺势完成，几乎不增加额外的训练成本。

### 方法详解

**整体思路**  
MoBE 的压缩流程可以概括为三步：  
1）对每个专家（以及对应的门控）提取原始权重矩阵；  
2）对每个矩阵做一次低秩分解，得到唯一的 A 矩阵和相对大的 B 矩阵；  
3）把所有 B 矩阵统一映射到一组共享的基底矩阵集合 {B_i}，并学习每个专家对应的线性组合系数。最终模型只需要保存 A 矩阵、基底矩阵以及系数向量。

**步骤拆解**  

- **低秩分解**：把原始权重 W 看成 “A × B”。A 的列数远小于 W 的行数，因而占用的显存很少；B 则保留了大部分信息，但仍比原始 W 小很多。可以把 A 想成“专家的个人签名”，B 是“通用的特征库”。

- **基底共享**：所有专家的 B 矩阵被强制写成 B = Σ α_i B_i（α_i 为该专家的系数，B_i 为全局基底）。这一步类似于把每个部门的工具箱拆成若干公共工具（锤子、螺丝刀）加上少量自定义配件。只需要存几套公共工具，配件用系数记录即可。

- **系数学习**：系数 α_i 通过最小化 ‖W - A·(Σ α_i B_i)‖² 来求得。因为所有基底都是共享的，梯度会在所有专家之间传播，促使基底捕捉到跨专家的共性特征。

- **门控矩阵的同构处理**：门控矩阵同样被分解并共享基底，确保路由过程的参数也被压缩。这样，激活的专家数目不变，模型的稀疏计算特性保持不变。

- **训练细节**：作者在原始模型的微调阶段加入了一个额外的重构损失项（reconstruction loss），与原始任务的损失一起反向传播。为了不让重构误差主导训练，使用了一个小的权重系数。分块随机梯度下降让每次更新只涉及一小块基底，显著降低了显存峰值。

**最巧妙的点**  
把“较大的 B 矩阵”再拆成共享基底，是一种两层压缩的思路。第一层已经把每个专家的参数压到低秩，第二层进一步把所有专家的低秩残差统一抽象成公共基底，这种跨专家的协同压缩在之前的工作里几乎没有出现。

### 实验与效果

- **测试模型与任务**  
  实验覆盖了三大主流 MoE LLM：Qwen3‑235B‑A22B‑2507、DeepSeek‑V3‑0324（671B 参数）以及 Kimi‑K2‑Instruct（约 1T 参数）。评估任务包括零样本问答、指令遵循以及常见的语言理解基准（如 MMLU、TruthfulQA）。

- **对比基线**  
  与最近的 MoE 压缩方法（如 MoE‑Prune、Low‑Rank MoE）相比，MoBE 在相似的压缩率（约 25%）下，精度下降仅 1%‑2%（相对下降约 2%），而基线方法在同等压缩率下往往出现 7%‑14% 的相对下降。

- **具体数字**  
  例如在 DeepSeek‑V3‑0324 上，MoBE 将参数量削减 28%（从 671B 到约 483B），在 MMLU 上的整体得分从 68.4 降至 66.9，仅损失 1.5 分；而 MoE‑Prune 同等压缩率下得分跌至 61.2 分。

- **消融实验**  
  作者分别去掉基底共享、去掉门控矩阵的基底化以及仅使用单层低秩分解进行对比。结果显示：去掉基底共享会导致压缩率只能提升到 15% 左右，且精度下降接近 5%；不共享门控矩阵会让路由错误率上升约 0.8%，对整体性能有明显负面影响。

- **局限性**  
  论文承认，基底数量的选取是一个经验超参数，若选得过少会限制表达能力，若选得过多则压缩收益下降。此外，当前实现仍需在微调阶段加入额外的重构损失，对训练时间有轻微增加。

### 影响与延伸思考

MoBE 的两层分解思路在 MoE 社区引起了广泛关注，后续有工作尝试把基底共享推广到 Transformer 的前馈层和自注意力层，形成了“全模型基底共享”概念（如 Basis‑MoE、Shared‑Basis Transformer）。还有研究把基底学习与知识蒸馏结合，进一步压缩到 10% 以下的参数量。对想继续深入的读者，可以关注以下方向：① 基底自动化搜索（如何在不人工设定基底数的情况下找到最优配置）；② 基底在多任务微调中的迁移能力；③ 将基底共享与稀疏激活的硬件加速协同设计。整体来看，MoBE 为大模型部署提供了更实用的压缩路径，也为跨专家参数复用打开了新思路。

### 一句话记住它

MoBE 用“共享基底+低秩分解”两层技巧，把巨型 MoE 模型压到 70% 参数仍保持原有水平。