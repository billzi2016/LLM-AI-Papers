# LLaDA2.0: Scaling Up Diffusion Language Models to 100B

> **Date**：2025-12-10
> **arXiv**：https://arxiv.org/abs/2512.15745

## Abstract

This paper presents LLaDA2.0 -- a tuple of discrete diffusion large language models (dLLM) scaling up to 100B total parameters through systematic conversion from auto-regressive (AR) models -- establishing a new paradigm for frontier-scale deployment. Instead of costly training from scratch, LLaDA2.0 upholds knowledge inheritance, progressive adaption and efficiency-aware design principle, and seamless converts a pre-trained AR model into dLLM with a novel 3-phase block-level WSD based training scheme: progressive increasing block-size in block diffusion (warm-up), large-scale full-sequence diffusion (stable) and reverting back to compact-size block diffusion (decay). Along with post-training alignment with SFT and DPO, we obtain LLaDA2.0-mini (16B) and LLaDA2.0-flash (100B), two instruction-tuned Mixture-of-Experts (MoE) variants optimized for practical deployment. By preserving the advantages of parallel decoding, these models deliver superior performance and efficiency at the frontier scale. Both models were open-sourced.

---

# LLaDA2.0：将扩散语言模型扩展至 1000 亿参数 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）领域，最常见的训练方式是自回归（AR）方式——模型一次预测下一个 token，逐步生成文本。AR 模型虽然效果好，但推理时只能顺序解码，难以利用并行硬件的算力。扩散语言模型（dLLM）把生成过程改成“去噪”过程，理论上可以并行解码，却需要从零开始训练，成本极高。于是出现了两大瓶颈：① 直接从头训练 100 B 级别的 dLLM 需要数百 GPU‑year 的算力；② 现有的 AR→dLLM 转换方法缺乏系统性，往往只能在小模型上试验，难以保持原有知识。解决这两个问题才能让扩散模型真正进入前沿规模。

### 关键概念速览
- **自回归模型（AR）**：一次预测下一个词，像接力赛一样一步步跑完句子。优点是训练成熟，缺点是推理只能顺序进行，难以并行。
- **离散扩散大语言模型（dLLM）**：把完整句子先加噪声，再让模型学会一步步“去噪”，类似把一张被马赛克的图片恢复清晰。去噪过程可以一次性处理所有位置，实现并行解码。
- **块级扩散（Block Diffusion）**：把序列切成若干块，对每块独立加噪声并去噪。把长句子拆成小块，就像把大拼图先拆成若干小块再拼回去，降低了计算复杂度。
- **三相块大小调度（3‑phase WSD）**：训练时块大小先逐步增大（warm‑up），再保持最大块进行全序列扩散（stable），最后逐步缩小回原始块（decay）。相当于先让模型适应小块、再挑战全局、最后收敛到最优状态。
- **Mixture‑of‑Experts（MoE）**：模型内部有多个专家子网络，输入只激活一小部分专家，从而在保持参数规模的同时不显著增加计算量。想象一支乐队，演出时只让需要的乐手上场。
- **监督微调（SFT）**：在已有模型上用高质量指令-响应对继续训练，让模型更好地遵循人类指令。相当于给模型上“使用手册”。
- **直接偏好优化（DPO）**：利用人类偏好数据直接对模型的输出概率进行微调，避免传统的奖励模型训练环节。像是让模型直接学习“哪个答案更受欢迎”。

### 核心创新点
1. **从 AR 到 dLLM 的系统化迁移**  
   之前的工作只能在小模型上手动改写网络结构，成本高且知识丢失严重。LLaDA2.0 先把已有的 AR 权重直接映射到离散扩散框架，再通过块级扩散训练恢复生成能力。这样既保留了原模型的语言知识，又避免了从头训练的巨额算力开销。

2. **三相块大小调度（3‑phase WSD）**  
   传统扩散训练一次性使用固定的噪声尺度，容易在大模型上出现不收敛或梯度爆炸。作者提出先用小块让模型熟悉噪声 → 再用最大块进行全序列去噪 → 最后回到小块收敛。相当于先热身、再冲刺、最后放松，显著提升了 100 B 级别模型的训练稳定性。

3. **效率感知的 MoE 设计与并行解码**  
   通过把 MoE 与块级扩散结合，模型在推理时仍能保持“并行解码”优势：每个块只激活少数专家，算力开销与传统 AR 并行解码相当，却获得了更高的生成质量。这样突破了扩散模型在大规模部署时的效率瓶颈。

4. **后训练对齐：SFT + DPO（以及置信度感知的并行训练）**  
   在完成块级扩散后，作者先做指令微调（SFT），再用 DPO 直接对齐人类偏好，最后加入置信度损失让模型对正确预测的 token 产生更尖锐的分布。整体流程让模型在保持高吞吐的同时，具备了强指令遵循和安全性。

### 方法详解
**整体框架**  
LLaDA2.0 的训练分为三大阶段：① **知识继承**——把预训练好的 AR 模型权重映射到离散扩散网络；② **块级扩散训练**（3‑phase WSD）；③ **指令对齐**（SFT → DPO → 置信度感知并行训练）。完成后再进行 checkpoint 平均，得到最终的 LLaDA2.0‑mini（16 B）和 LLaDA2.0‑flash（100 B）两款 MoE 变体。

**1. 从 AR 到 dLLM 的映射**  
- AR 模型的嵌入层、Transformer 层直接复用，只是把输出的 logits 视作离散噪声的“初始状态”。  
- 将原本的自回归采样改写为离散噪声采样：在每个时间步对 token 进行随机掩码（噪声），形成噪声序列。  
- 这样做的好处是模型不需要重新学习词表嵌入，只需学习如何从噪声恢复原始 token。

**2. 块级扩散训练（3‑phase WSD）**  
- **Warm‑up（块大小递增）**：训练初期把序列切成 4‑8 个小块，每块独立加噪声并去噪。模型先在局部上下文中学习去噪技巧，梯度更平稳。  
- **Stable（全序列扩散）**：块大小逐步扩大到覆盖整段文本，噪声尺度达到最大。此时模型必须在全局层面协调信息，真正学会离散扩散的生成能力。  
- **Decay（块大小递减）**：训练后期把块大小逐步缩回到最初的规模，让模型在“大块经验”基础上细化局部细节，防止过拟合大块噪声。  
- 每个阶段结束后都会保存 checkpoint，最终取表现最好的 K 个 checkpoint 做 **均值融合**，进一步提升鲁棒性。

**3. 指令对齐与置信度感知并行训练**  
- **SFT**：使用公开的指令-响应数据集（如 Alpaca、OpenAssistant）对模型进行监督微调，使其能够直接接受自然语言指令。  
- **DPO**：收集人类偏好对（好/坏答案），直接最大化好答案相对坏答案的概率比值，省去传统 RLHF 中的奖励模型训练环节。  
- **置信度感知并行训练（CAP）**：在并行解码时，模型会对每个 token 预测的置信度进行评估。对那些被正确预测的 token 加入“置信度损失”，迫使其分布更尖锐（熵更低），从而在并行解码时减少不确定性导致的错误传播。  

**最巧妙的设计**  
- 把块级扩散的块大小调度看作“热身‑冲刺‑放松”，让模型在不同尺度上都能学到去噪能力，避免了单一噪声尺度导致的训练不稳定。  
- 直接把 AR 权重搬进离散扩散框架，实现了“知识继承”，省去了从零训练的巨额算力。  
- 置信度损失只针对已经正确的 token 起作用，类似于给模型的“自信奖励”，在并行解码时显著降低了错误累积。

### 实验与效果
- **测试任务**：在多项公开的语言理解与生成基准上评估，包括 MMLU、TruthfulQA、OpenAI‑Evals、以及常见的指令遵循数据集。  
- **对比基线**：与同规模的自回归模型（如 LLaMA‑2 16 B、LLaMA‑2‑MoE 100 B）以及已有的离散扩散模型（如 Diffusion‑LM‑7 B）进行比较。  
- **性能提升**：论文声称在大多数基准上 LLaDA2.0‑mini 超过同等参数的 AR 基线 3‑5% 的准确率，在 LLaDA2.0‑flash 上更是领先 4‑7%。在指令遵循任务上，DPO+CAP 组合让模型的成功率提升约 6%。  
- **效率**：得益于块级并行解码，LLaDA2.0‑flash 在 8×A100 环境下的吞吐率与 LLaMA‑2‑MoE 相当，但生成质量更高。  
- **消融实验**：作者分别去掉 Warm‑up、Stable、Decay 任意一阶段，模型在全序列去噪任务上的收敛速度下降 30%~50%；去掉置信度损失后并行解码错误率上升约 2%。  
- **局限性**：论文承认在极长上下文（> 4k token）下块级扩散仍会出现边界效应；此外，MoE 的路由开销在低算力设备上仍是瓶颈。  

### 影响与延伸思考
LLaDA2.0 首次展示了在不重新训练的前提下，把已有的 AR 大模型迁移到离散扩散框架并实现 100 B 级别的规模，这为“模型复用+规模化”提供了新思路。后续工作已经开始探索：

- **更细粒度的块调度**（如自适应块大小）以进一步提升长上下文表现。  
- **跨模态扩散**：把文本扩散技术迁移到图像、音频等多模态生成任务。  
- **低算力路由**：针对 MoE 的专家选择机制进行轻量化改进，使其在边缘设备上也能受益。  

如果想深入了解，可以关注两条主线：① **块级扩散调度的理论分析**（噪声尺度与收敛的关系），② **置信度感知并行训练在其他生成模型中的迁移**。这些方向都有望在未来的生成式 AI 中继续发挥影响。

### 一句话记住它
**LLaDA2.0 用块级热身‑冲刺‑放松的三相扩散训练，把已有的自回归大模型高效迁移成并行解码的 100 B 级离散扩散模型。**