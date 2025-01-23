# Can We Generate Images with CoT? Let's Verify and Reinforce Image Generation Step by Step

> **Date**：2025-01-23
> **arXiv**：https://arxiv.org/abs/2501.13926

## Abstract

Chain-of-Thought (CoT) reasoning has been extensively explored in large models to tackle complex understanding tasks. However, it still remains an open question whether such strategies can be applied to verifying and reinforcing image generation scenarios. In this paper, we provide the first comprehensive investigation of the potential of CoT reasoning to enhance autoregressive image generation. We focus on three techniques: scaling test-time computation for verification, aligning model preferences with Direct Preference Optimization (DPO), and integrating these techniques for complementary effects. Our results demonstrate that these approaches can be effectively adapted and combined to significantly improve image generation performance. Furthermore, given the pivotal role of reward models in our findings, we propose the Potential Assessment Reward Model (PARM) and PARM++, specialized for autoregressive image generation. PARM adaptively assesses each generation step through a potential assessment approach, merging the strengths of existing reward models, and PARM++ further introduces a reflection mechanism to self-correct the generated unsatisfactory image, which is the first to incorporate reflection in autoregressive image generation. Using our investigated reasoning strategies, we enhance a baseline model, Show-o, to achieve superior results, with a significant +24% improvement on the GenEval benchmark, surpassing Stable Diffusion 3 by +15%. We hope our study provides unique insights and paves a new path for integrating CoT reasoning with autoregressive image generation. Code and models are released at https://github.com/ZiyuGuo99/Image-Generation-CoT

---

# 我们能用思维链生成图像吗？逐步验证与强化图像生成 论文详细解读

### 背景：这个问题为什么难？

在传统的自回归图像生成模型里，模型一次生成一个像素或一个块，整个过程像“盲打”——只靠一次前向传播决定每一步的输出。虽然大模型已经能生成高质量图片，但它们缺乏对生成过程的自我检查和纠错能力，导致出现结构错位、颜色失真等低级错误。现有的提升手段大多是扩大模型容量或加长采样步数，却没有办法让模型在生成时主动评估自己的决定是否合理。于是，如何让模型像人类写草稿一样，先思考、再验证、再修正，成为了一个未被系统探索的空白。

### 关键概念速览
**思维链（CoT，Chain‑of‑Thought）**：让模型在给出最终答案前，先把推理步骤写出来，类似解数学题时先列出计算过程，便于检查和纠错。  
**自回归图像生成**：模型按顺序生成图像的每个 token（像素块），每一步都依赖前面的输出，就像一句话一个字地写。  
**测试时计算扩展（Test‑time Computation Scaling）**：在推理阶段投入更多算力进行多次采样或迭代检查，而不是只在训练时加大模型。  
**直接偏好优化（DPO，Direct Preference Optimization）**：直接用人类或模型偏好数据对生成策略进行微调，使模型的输出更符合期望的“好”。  
**奖励模型（Reward Model）**：给每一步生成的结果打分的网络，帮助模型判断哪条生成路径更优。  
**潜力评估奖励模型（PARM）**：一种专门为自回归图像生成设计的奖励模型，能够在每一步动态评估“潜在质量”。  
**反射机制（Reflection）**：模型在发现生成的某一步不满意后，主动回顾并重新生成该步或后续步，类似人写完一段文字后回头改写。  
**GenEval 基准**：衡量生成模型整体质量的评测套件，覆盖清晰度、结构一致性等多维指标。

### 核心创新点
1. **把思维链搬进图像生成**：过去的 CoT 只在语言理解任务里出现，作者把它迁移到像素级生成上。具体做法是让模型在每一步先生成一个“思考稿”（潜在评估分），再决定正式输出。这样模型在生成前就有了自我审视的机会，显著降低了结构错误。  
2. **测试时计算扩展用于验证**：传统做法是一次前向得到结果，这里改为在每一步进行多轮评估——先快速生成候选，随后用更大的算力对候选进行打分和筛选。相当于在写作时先写草稿，再用放大镜仔细检查。实验表明，这一步提升了整体图像质量。  
3. **将 DPO 与自回归生成对齐**：作者把直接偏好优化的目标从文本转移到图像生成的每一步，利用人类偏好数据或高质量模型的输出，对模型的“思考稿”进行微调，使其更倾向于产生高分的候选。这样模型的内部偏好与外部评价保持一致。  
4. **提出 PARM 与 PARM++ 两种奖励模型**：PARM 通过潜力评估在每一步给出细粒度分数，融合了已有奖励模型的优点；PARM++ 在此基础上加入反射机制，允许模型在检测到低分后主动重新生成，首次实现了自回归生成中的“自我纠错”。  

### 方法详解
整体思路可以拆成三层循环：**生成 → 评估 → 纠正**。  
1. **初始生成**：模型（以 Show‑o 为基线）按自回归方式逐块生成图像，每块输出前先产生一个“思考向量”。这个向量会送入 PARM 进行潜力评估。  
2. **潜力评估（PARM）**：PARM 接收当前已生成的图像上下文和思考向量，输出一个潜在质量分数。分数越高，说明该块在整体结构、颜色一致性等方面更有潜力。  
3. **多轮验证（测试时计算扩展）**：对每块，系统会生成 N（如 4）个候选思考向量并分别打分，选出最高分的候选继续生成。这里的 N 可以在推理时动态调节，算力充足时取更大值，得到更稳健的决定。  
4. **偏好对齐（DPO）**：在训练阶段，收集大量高分候选与低分候选的对比数据，用 DPO 直接最小化低分候选的概率，最大化高分候选的概率。这样模型在生成思考向量时会倾向于产生更容易得到高分的候选。  
5. **反射纠错（PARM++）**：当 PARM 检测到某一步的潜力分数低于阈值，PARM++ 会触发反射：模型重新进入该步的生成流程，使用更高的计算预算（更多候选、更深的网络层）重新生成思考向量并重新评估。若仍不满意，系统可以回溯到前几步进行整体调整。  
6. **最终输出**：经过上述循环后，所有块的正式像素值被写入图像，得到完整的生成结果。

**最巧妙的地方**在于把奖励模型嵌入生成的每一步，而不是仅在完整图像后打分。这样模型的“思考稿”本身就带有质量信号，能够在生成过程中即时纠错，类似人写作时边写边检查的习惯。

### 实验与效果
- **测试数据**：作者在 GenEval 基准上评估，GenEval 包含清晰度、结构完整性、颜色一致性等多维指标。  
- **对比基线**：Show‑o（原始自回归模型）、Stable Diffusion 3、以及其他主流扩散模型。  
- **核心结果**：在加入 CoT 思维链、PARM、PARM++ 以及 DPO 对齐后，Show‑o 的 GenEval 综合得分提升了约 24%。相较于 Stable Diffusion 3，提升约 15%。  
- **消融实验**：论文分别去掉（1）测试时计算扩展、（2）DPO 对齐、（3）PARM++ 反射机制，发现每去掉一项整体得分下降 4%~7%，其中反射机制对低分块的纠错贡献最大。  
- **局限性**：作者指出，计算扩展会显著增加推理时的算力需求，尤其在移动端或实时场景下难以直接部署；此外，PARM 的训练依赖大量高质量偏好数据，数据稀缺时效果会受限。  

### 影响与延伸思考
这篇工作首次把思维链的“写草稿、检查、改写”流程系统化到自回归图像生成，打开了“生成过程可解释、可纠错”的新方向。随后的研究开始探索在扩散模型、视频生成甚至 3D 生成中加入类似的逐步评估与反射机制（如 “Iterative Refinement with Reward Feedback”）。如果想进一步深入，可以关注以下几个方向：  
1. **轻量化的测试时计算扩展**：如何在算力受限的设备上实现多轮评估。  
2. **跨模态奖励模型**：把文本、音频的偏好信息融合进图像的潜力评估。  
3. **自监督的反射学习**：让模型在没有显式低分标签的情况下自行发现并纠正错误。  

### 一句话记住它
把思维链搬进自回归图像生成，让模型在每一步都“先想、再评、再改”，实现了显著的质量提升和首次可视化的自我纠错。