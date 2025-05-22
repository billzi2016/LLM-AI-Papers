# Dimple: Discrete Diffusion Multimodal Large Language Model with Parallel Decoding

> **Date**：2025-05-22
> **arXiv**：https://arxiv.org/abs/2505.16990

## Abstract

In this work, we propose Dimple, the first Discrete Diffusion Multimodal Large Language Model (DMLLM). We observe that training with a purely discrete diffusion approach leads to significant training instability, suboptimal performance, and severe length bias issues. To address these challenges, we design a novel training paradigm that combines an initial autoregressive phase with a subsequent diffusion phase. This approach yields the Dimple-7B model, trained on the same dataset and using a similar training pipeline as LLaVA-NEXT. Dimple-7B ultimately surpasses LLaVA-NEXT in performance by 3.9%, demonstrating that DMLLM can achieve performance comparable to that of autoregressive models. To improve inference efficiency, we propose a decoding strategy termed confident decoding, which dynamically adjusts the number of tokens generated at each step, significantly reducing the number of generation iterations. In autoregressive models, the number of forward iterations during generation equals the response length. With confident decoding, however, the number of iterations needed by Dimple is even only $\frac{\text{response length}}{3}$. We also re-implement the prefilling technique in autoregressive models and demonstrate that it does not significantly impact performance on most benchmark evaluations, while offering a speedup of 1.5x to 7x. Additionally, we explore Dimple's capability to precisely control its response using structure priors. These priors enable structured responses in a manner distinct from instruction-based or chain-of-thought prompting, and allow fine-grained control over response format and length, which is difficult to achieve in autoregressive models. Overall, this work validates the feasibility and advantages of DMLLM and enhances its inference efficiency and controllability. Code and models are available at https://github.com/yu-rp/Dimple.

---

# Dimple：离散扩散多模态大语言模型与并行解码 论文详细解读

### 背景：这个问题为什么难？

多模态大语言模型（MLLM）需要同时理解文字和图像，传统上依赖自回归（autoregressive）方式逐词生成答案。自回归虽然稳健，却在推理时必须一步步展开，导致响应越长耗时越多，而且模型对生成长度极度敏感，容易出现“长度偏置”。最近有人尝试把离散扩散（discrete diffusion）引入 MLLM，理论上可以一次性生成多个 token，提升并行度。但纯离散扩散训练极不稳定，性能不如自回归，且在长文本上表现更差。于是出现了一个关键难点：如何让离散扩散既保持并行优势，又克服训练不稳和长度偏置。

### 关键概念速览
- **离散扩散（Discrete Diffusion）**：把文本看成离散的符号序列，先把它们随机扰动成噪声，再让模型学会一步步去噪恢复原句，类似把文字“洗白”再“洗回”。  
- **自回归（Autoregressive）**：模型每次只预测下一个 token，后面的预测依赖已经生成的内容，就像写文章时只能看前面的文字。  
- **长度偏置（Length Bias）**：模型倾向于生成短句或特定长度的答案，长答案的概率被系统性压低。  
- **并行解码（Parallel Decoding）**：一次性生成多个 token，而不是逐个生成，类似一次性打印整段文字。  
- **可信解码（Confident Decoding）**：在扩散过程中根据模型对每个位置的置信度动态决定是否继续迭代，置信高的 token 直接确定，置信低的再去噪。  
- **前填（Prefilling）**：在自回归模型中把已经确定的前缀一次性喂入模型，以减少重复计算的技巧。  
- **结构先验（Structure Priors）**：在提示中显式给出答案的格式或层次（如 JSON、表格），让模型在生成时遵循这些结构约束。  

### 核心创新点
1. **先自回归后扩散的双阶段训练 → 先让模型学会稳健的单词预测，再让它在此基础上学会去噪恢复 → 训练过程从不稳定的纯扩散转为平滑，最终模型在同等数据上比纯自回归提升 3.9%。**  
2. **可信解码策略 → 在每轮扩散结束后检查每个 token 的置信度，置信度高的直接写入答案，置信度低的继续迭代 → 生成迭代次数从“等于响应长度”降到约三分之一，大幅加速推理。**  
3. **重新实现前填技术并迁移到扩散模型 → 把已经确定的前缀一次性喂入扩散网络，避免在每轮迭代都重复计算 → 在多数基准上几乎不影响准确率，却能带来 1.5×~7× 的速度提升。**  
4. **结构先验驱动的可控生成 → 通过在提示里嵌入结构模板，让模型在扩散恢复时遵循固定的格式 → 能实现比传统指令或思维链更细粒度的长度和布局控制，这在自回归模型里几乎不可能。**  

### 方法详解
整体框架分为两个阶段：**预训练阶段**和**推理阶段**。  
1. **预训练阶段**  
   - **自回归阶段**：模型先用标准的自回归目标（预测下一个 token）在多模态数据上训练，确保它能够稳健地把图像特征映射到语言空间。  
   - **离散扩散阶段**：在自回归权重的基础上，引入离散噪声过程。具体做法是把已经学好的 token 序列随机替换成噪声 token（比如用词表中的随机词），然后让模型学习在多个噪声步长上逐步恢复原始序列。这里的关键是把自回归的输出作为“干净”目标，扩散过程只负责去噪，不需要重新学习语言本身。  
   - **双阶段切换**：训练时先跑若干 epoch 的自回归，然后切换到扩散，二者共享同一 Transformer 参数，避免了两套模型的额外开销。  

2. **推理阶段**  
   - **初始化**：给定图像和用户指令后，模型先生成一个全噪声的 token 序列。  
   - **可信解码循环**：每一步扩散网络输出每个位置的置信度分布。若置信度超过预设阈值，直接把该 token 固定下来；否则保留噪声继续下一轮去噪。这样高置信度的 token 在少数几轮后就确定，低置信度的才会被多轮细化。  
   - **并行生成**：因为每轮都在整个序列上并行计算，实际的前向传播次数远小于自回归的响应长度。实验显示，平均只需要响应长度的约三分之一轮即可完成。  
   - **前填优化**：在每轮去噪前，把已经确定的前缀一次性喂入模型的注意力层，省去重复计算的开销。  
   - **结构先验注入**：如果用户在提示里提供了 JSON、表格或固定段落标题等结构，模型在去噪时会把这些结构 token 视为不可动摇的“锚点”，其余位置围绕锚点进行填充，从而实现严格的格式控制。  

最让人意外的设计是**先自回归后扩散**的顺序。直觉上，扩散本应是从噪声到文本的完整过程，但作者发现直接从噪声学习语言会导致梯度不稳。把自回归当作“语言基底”，再让扩散只负责“去噪”，既保留了扩散的并行优势，又继承了自回归的语言稳健性。

### 实验与效果
- **数据与任务**：在与 LLaVA‑NEXT 相同的多模态指令遵循数据集上训练，评测覆盖视觉问答、图像描述、复杂推理等 12 项公开基准。  
- **整体表现**：Dimple‑7B 在综合得分上比 LLaVA‑NEXT 高出 3.9%，在视觉问答和长文本生成任务上尤为突出。  
- **推理速度**：使用可信解码后，平均迭代次数约为响应长度的 1/3；结合前填后，整体推理速度提升 1.5×~7×，在长答案场景下可实现接近实时的响应。  
- **消融研究**：作者分别去掉自回归预热、可信阈值、前填以及结构先验进行实验。结果显示，去掉自回归预热会导致训练不收敛，去掉可信解码会把迭代次数拉回到与自回归相当，前填的去除使推理慢 2‑3 倍，结构先验的缺失则让格式化任务的准确率下降约 12%。  
- **局限性**：论文承认在极端超长生成（超过 500 token）时仍会出现少量残留噪声，可信阈值需要手动调节；此外，离散扩散对词表大小敏感，词表扩展会显著增加噪声空间的复杂度。  

### 影响与延伸思考
Dimple 把离散扩散正式带入多模态大语言模型，证明了“先自回归后扩散”是一条可行的训练路线。随后的工作开始探索更细粒度的噪声调度、跨模态噪声共享以及在更大模型上复用该框架（如 34B、70B 规模）。对想进一步了解的读者，可以关注 **Diffusion Transformers**、**Parallel Decoding for LLMs** 以及 **Structure‑guided Generation** 方向的最新论文和开源实现。  

### 一句话记住它
先让模型学会稳健的自回归，再用离散扩散并行去噪——这样既保留语言质量，又把生成速度压到原来的三分之一。