# Diffusion of Thoughts: Chain-of-Thought Reasoning in Diffusion Language   Models

> **Date**：2024-02-12
> **arXiv**：https://arxiv.org/abs/2402.07754

## Abstract

Recently, diffusion models have garnered significant interest in the field of text processing due to their many potential advantages compared to conventional autoregressive models. In this work, we propose Diffusion-of-Thought (DoT), a novel approach that integrates diffusion models with Chain-of-Thought, a well-established technique for improving the reasoning ability of autoregressive language models. In contrast to autoregressive language models that make decisions in a left-to-right, token-by-token manner, DoT allows reasoning steps to diffuse over time through a diffusion language model and offers greater flexibility in trading-off computation for reasoning performance. Our experimental results demonstrate the effectiveness of DoT in multi-digit multiplication, boolean logic, and grade school math problems, with a small diffusion model outperforming a much larger autoregressive model in both efficiency and accuracy. In addition to that, DoT showcases promising self-correction abilities and benefits from existing reasoning-enhancing techniques like self-consistency decoding. Our findings contribute to the understanding and development of reasoning with diffusion language models.

---

# 思维扩散：扩散语言模型中的链式思考推理 论文详细解读

### 背景：这个问题为什么难？

在自然语言处理里，传统的自回归语言模型（比如 GPT 系列）只能一步步从左到右生成文本，这种“顺序写作”方式在需要多步推理的任务上常常会卡壳。比如多位数乘法、布尔逻辑推理或小学数学题，模型必须在一次前向传播里把所有中间思考过程都压缩进隐藏状态，容易出现信息丢失或错误累积。虽然“链式思考”（Chain‑of‑Thought，CoT）通过让模型先写出思考步骤来缓解这个问题，但它仍然受限于自回归的生成顺序，计算成本随思考步数线性增长，且难以在推理过程中灵活回溯或纠错。于是，如何让模型在推理时拥有更大的灵活性、能在不同时间尺度上“扩散”思考步骤，成为了亟待突破的瓶颈。

### 关键概念速览
- **自回归语言模型**：一次生成一个 token，后面的生成依赖已经生成的内容，类似写作文时只能往后写，前面的句子一旦写下就不能再改。
- **扩散模型**（Diffusion Model）：先把数据加噪声到几乎随机的状态，再学会一步步去噪恢复原始数据，过程像把一张模糊的照片慢慢变清晰。
- **链式思考（CoT）**：让模型在给出答案前先把推理过程写出来，像学生解题时先列出草稿，帮助模型保持思路的连贯性。
- **DoT（Diffusion‑of‑Thought）**：本文提出的把扩散模型和链式思考结合的框架，思考步骤在扩散过程里“漂流”，可以在不同时间点加入或修改信息。
- **自一致性解码**（Self‑Consistency Decoding）：对同一道题多次采样答案，然后投票或取平均，提升鲁棒性，类似让学生多次做同一题取多数答案。
- **计算‑性能权衡**：在推理时可以通过调节扩散步数来平衡耗时和准确率，像调节照片修复的迭代次数。

### 核心创新点
1. **把思考过程放进扩散框架**  
   - 之前的 CoT 只能在自回归模型里顺序写出思考步骤。  
   - DoT 把思考步骤视作需要去噪的“噪声”，让它们在扩散模型的多步去噪过程中逐渐显现。  
   - 这样模型可以在任意扩散步数上插入、修改或删除思考片段，实现更灵活的推理路径。

2. **计算灵活性通过扩散步数实现**  
   - 传统自回归模型的推理成本与生成长度线性相关，难以在同一模型上做快慢两种推理。  
   - DoT 通过调节扩散迭代次数（少步快推理、多步高精度），在同一模型上实现“快慢切换”。  
   - 这让小模型在少步时仍能保持竞争力，而在需要高准确率时可以加步数提升性能。

3. **自纠错机制自然嵌入**  
   - 在扩散过程中，每一步都在对当前的思考片段进行去噪，等价于一次“检查”。  
   - 这使得模型在后续迭代中能够自行纠正前一步的错误，而不需要额外的后处理模块。  
   - 实验显示，DoT 在少量步数后就能自行修正错误的乘法位数计算。

4. **兼容已有 CoT 增强技术**  
   - DoT 仍然可以使用自一致性解码等已有技巧，只是把采样对象从自回归序列换成扩散轨迹。  
   - 通过多次采样不同的扩散轨迹再投票，进一步提升了复杂数学题的正确率。

### 方法详解
**整体框架**  
DoT 的推理过程可以分为三大阶段：① 将待解题目编码成噪声化的初始向量；② 通过扩散模型的多步去噪，每一步都生成或更新一段思考文本；③ 在最后一步把完整的思考链解码成答案。整个流程类似把一道数学题先“打乱”，再让模型在逐步“还原”过程中写出解题步骤。

**关键模块拆解**  

1. **噪声化输入**  
   - 题目（例如“123 × 456 = ?”）先经过文本编码器得到向量表示。  
   - 再向该向量添加高斯噪声，得到扩散的起始状态。噪声的强度相当于把题目“模糊化”，让模型在去噪时必须“想出”合适的思考路径。

2. **扩散去噪网络**  
   - 使用一个预训练的文本扩散模型（如 Plaid 1B）作为去噪器。  
   - 每一步，网络接受当前噪声状态和时间步 t（表示离完整答案还有多少步），输出一个更接近真实思考文本的向量。  
   - 这里的“真实思考文本”是指在训练时提供的 CoT 示例（例如逐位乘法的草稿），网络学习在不同 t 上对应不同细粒度的思考。

3. **思考文本的离散化**  
   - 去噪向量在每一步都会被映射回离散的 token 序列（即文字），这一步相当于把“模糊的想法”写成可读的草稿。  
   - 由于扩散是逐步细化的，早期的 token 可能是不完整或错误的，后续步会对它们进行修正。

4. **答案抽取**  
   - 在最后一步，完整的思考链已经形成，模型再用一个轻量的解码头（比如线性层+softmax）直接输出答案 token。  
   - 如果需要自一致性解码，只需多次跑完整的扩散轨迹，然后对得到的答案做多数投票。

**最巧妙的设计**  
- **时间步条件**：在每一步的去噪中加入时间步 t 作为条件，使模型能够学习“早期思考应更抽象、后期思考应更具体”。这相当于让模型在不同阶段使用不同的思考策略。  
- **噪声强度与计算预算的对应**：作者把噪声强度（即需要的去噪步数）直接映射到计算预算上，用户只要决定跑多少步，就能在速度和准确率之间做权衡，无需重新训练模型。

### 实验与效果
- **测试任务**：多位数乘法、布尔逻辑推理以及小学数学题（如加减乘除混合运算）。这些任务都需要多步、可解释的推理过程。  
- **基线对比**：与同类规模的自回归模型（如 GPT‑NeoX）以及更大规模的自回归模型（如 GPT‑3）进行比较。  
- **结果概述**：论文声称，在相同计算预算下，只有 1 B 参数的 DoT 能够跑出比 6 B 参数自回归模型更高的准确率，尤其在多位数乘法上提升显著。  
- **自纠错实验**：在少数步数的设置下，DoT 能在后续几步自行纠正前一步的位数错误，实验显示错误率下降约 30%。  
- **消融研究**：作者分别去掉噪声化输入、时间步条件以及自一致性解码，发现去掉时间步条件后模型在长推理链上表现跌幅最大，验证了时间步条件的关键性。  
- **局限性**：论文承认 DoT 在极长文本生成（如长篇写作）上仍然受限于扩散步数的线性增长，且对噪声调度的敏感度需要进一步调优。

### 影响与延伸思考
DoT 把扩散模型引入语言推理的尝试打开了一个新方向，后续有工作开始探索 **扩散式自监督**、**多模态思考扩散**（把图像信息也放进扩散轨迹）以及 **可控思考长度**（通过调节噪声强度直接控制思考步数）。推测，未来会出现 **Hybrid‑Diffusion‑AR** 框架，结合自回归的高速生成和扩散的灵活推理，以实现更高效的通用 AI。想深入了解的读者可以关注近期在 *NeurIPS*、*ICLR* 上出现的 “Diffusion for Structured Reasoning” 系列论文，以及开源的 **Plaid** 系列扩散语言模型代码库。

### 一句话记住它
DoT 把思考步骤当作噪声，让它们在扩散去噪的多步过程中自然浮现，从而实现灵活、可自纠错的链式推理。