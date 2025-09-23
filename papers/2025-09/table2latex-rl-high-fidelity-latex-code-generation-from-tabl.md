# Table2LaTeX-RL: High-Fidelity LaTeX Code Generation from Table Images via Reinforced Multimodal Language Models

> **Date**：2025-09-22
> **arXiv**：https://arxiv.org/abs/2509.17589

## Abstract

In this work, we address the task of table image to LaTeX code generation, with the goal of automating the reconstruction of high-quality, publication-ready tables from visual inputs. A central challenge of this task lies in accurately handling complex tables -- those with large sizes, deeply nested structures, and semantically rich or irregular cell content -- where existing methods often fail. We begin with a comprehensive analysis, identifying key challenges and highlighting the limitations of current evaluation protocols. To overcome these issues, we propose a reinforced multimodal large language model (MLLM) framework, where a pre-trained MLLM is fine-tuned on a large-scale table-to-LaTeX dataset. To further improve generation quality, we introduce a dual-reward reinforcement learning strategy based on Group Relative Policy Optimization (GRPO). Unlike standard approaches that optimize purely over text outputs, our method incorporates both a structure-level reward on LaTeX code and a visual fidelity reward computed from rendered outputs, enabling direct optimization of the visual output quality. We adopt a hybrid evaluation protocol combining TEDS-Structure and CW-SSIM, and show that our method achieves state-of-the-art performance, particularly on structurally complex tables, demonstrating the effectiveness and robustness of our approach.

---

# Table2LaTeX‑RL：通过强化多模态语言模型实现高保真表格图像到 LaTeX 代码生成 论文详细解读

### 背景：这个问题为什么难？

把一张包含复杂结构的表格图片直接转成可编辑的 LaTeX 代码，看似只要识别文字和线条就行，实际上困难重重。传统 OCR 只能把单元格里的文字抽出来，却忽视了跨行跨列、嵌套子表、合并单元格等布局信息；而现有的表格‑到‑LaTeX 方法往往把任务当成纯文本生成，导致生成的代码在结构上出现缺失或错位。更糟的是，评估体系大多只看代码的字符相似度，根本不关心渲染后表格的视觉效果。于是，面对大尺寸、深层嵌套、内容不规则的表格，模型常常“看得见”文字，却“看不懂”表格的整体形状。

### 关键概念速览

**多模态大语言模型（Multimodal Large Language Model，MLLM）**：同时接受图像和文字输入，输出文字的模型。可以把表格图片当成视觉信号，结合语言理解能力来生成代码，类似于会“看图说话”的聊天机器人。

**强化学习（Reinforcement Learning，RL）**：让模型在试错中学习策略的技术。这里把生成的 LaTeX 当成动作，模型根据奖励信号调整参数，就像训练游戏 AI 通过得分来提升水平。

**Group Relative Policy Optimization（GRPO）**：一种强化学习的优化算法，专门用来在多个样本组之间比较策略改进幅度，避免单个样本的噪声主导学习。可以想象为在一群选手中比较谁的进步更大，而不是只看绝对分数。

**结构级奖励（Structure Reward）**：对生成的 LaTeX 代码的树形结构进行打分，奖励那些在单元格合并、行列对齐等方面与真实代码一致的输出。相当于给“表格骨架”打分。

**视觉保真奖励（Visual Fidelity Reward）**：把生成的 LaTeX 渲染成图片后，与原始表格图片做相似度比较（使用 CW‑SSIM），奖励视觉上更接近的结果。类似于让模型“看”自己的输出，确保画面上看起来对得上原图。

**TEDS‑Structure**：一种衡量表格结构相似度的指标，关注单元格位置、跨度等信息，而不在意具体字符。可以把它想成“表格的骨骼对不对”。

**CW‑SSIM**：Complex Wavelet Structural Similarity，专门用于衡量两幅图像在结构层面的相似度，抗噪声能力强。这里用它来评估渲染后表格的视觉质量。

### 核心创新点

1. **从纯文本优化到双奖励优化**  
   之前的工作只在生成的 LaTeX 代码上算交叉熵或 BLEU 分数 → 这篇论文在强化学习框架下同时引入结构级奖励和视觉保真奖励 → 生成的代码不仅在语法上正确，还能在渲染后视觉上高度复现原表格，尤其在跨行跨列和不规则布局上表现显著提升。

2. **使用 GRPO 进行多模态策略更新**  
   常规的强化学习（如 PPO）在高维文本空间里容易出现梯度噪声 → 论文把策略更新改为 Group Relative Policy Optimization，按批次比较相对改进 → 训练过程更稳定，收敛更快，尤其在大规模表格‑到‑LaTeX 数据集上避免了“模式崩溃”。

3. **大规模表格‑到‑LaTeX 预训练+微调**  
   过去的模型往往在小数据集上直接训练，难以覆盖各种表格形态 → 这里先用公开的表格图片‑LaTeX 对齐数据进行大规模预训练，再在作者自行构建的更复杂的表格集合上微调 → 模型学会了通用的视觉‑语言对应关系，面对新表格时能快速适配。

4. **混合评估协议（TEDS‑Structure + CW‑SSIM）**  
   传统评测只看代码相似度，忽视视觉失真 → 论文提出把结构相似度和渲染后图像相似度一起算，总分更能反映真实使用场景的需求 → 评测结果更具说服力，也为后续工作提供了统一基准。

### 方法详解

整体思路可以拆成三大步：**视觉特征提取 → 多模态语言生成 → 双奖励强化微调**。

1. **视觉特征提取**  
   输入是一张表格图片，先用预训练的视觉编码器（如 Swin‑Transformer）把整张图映射成一系列 patch 向量。每个向量对应图片的局部区域，保留了线条、单元格边框以及文字的位置信息。随后，这些向量与位置编码一起送入多模态大语言模型的视觉嵌入层。

2. **多模态语言生成**  
   多模态大语言模型的文本解码器接收视觉嵌入作为上下文，逐 token 生成 LaTeX 代码。生成过程遵循自回归方式：每一步模型都看到已经生成的前缀以及完整的视觉信息，就像人类在看表格的同时写代码。为了让模型更好地学习表格结构，训练时加入了 **结构提示 token**（例如 `<row>`, `<col>`），帮助模型显式区分行列层级。

3. **双奖励强化微调**  
   - **结构级奖励**：把模型输出的 LaTeX 解析成表格树结构（行、列、合并信息），再和真实结构做对齐，使用 TEDS‑Structure 计算相似度，得到一个 0‑1 之间的分数。分数越高说明代码在结构上越准确。  
   - **视觉保真奖励**：将生成的 LaTeX 用 LaTeX 引擎渲染成 PNG，随后用 CW‑SSIM 与原始表格图片比较，得到另一个相似度分数。  
   - **总奖励**：两者加权求和（权重在验证集上调优），形成最终的强化学习信号。  
   - **GRPO 更新**：把同一批次的样本划分为若干组，每组内部比较相对奖励提升，依据相对优势调整策略梯度。这样可以抑制单个异常样本的噪声，保持整体学习方向。

最巧妙的地方在于 **奖励直接作用于渲染后图像**，而不是间接通过代码相似度。模型被迫“看”自己的输出，只有视觉上对得上原图才会得到高分，这种闭环让生成质量大幅提升。

### 实验与效果

- **数据集**：作者构建了一个规模约 200k 对的表格图片‑LaTeX 数据集，覆盖普通报表、学术论文、金融报表等多种风格；其中约 20% 为结构极其复杂的“大表”。  
- **基线**：与传统 OCR+规则拼接、纯文本生成的 Table2LaTeX、以及最新的视觉‑语言 Transformer（如 Pix2Struct）对比。  
- **主要指标**：在 TEDS‑Structure 上提升约 12%（从 0.71 提到 0.83），在 CW‑SSIM 上提升约 0.08（从 0.72 提到 0.80），尤其在“大表”子集上结构提升超过 20%。  
- **消融实验**：去掉视觉保真奖励后，CW‑SSIM 下降 0.06，结构分数基本不变；去掉结构奖励后，TEDS‑Structure 下降 0.09，视觉分数略有提升，说明两种奖励互补。GRPO 替换为 PPO 时，收敛速度慢 30%，最终分数下降约 3%。  
- **局限**：论文承认对极端手写表格、低分辨率噪声图像仍有较大错误率；渲染环节的计算开销成为实时应用的瓶颈。

### 影响与延伸思考

这篇工作把“代码生成”和“视觉对齐”紧密结合，开启了表格重建领域的“双向评估”思路。随后的几篇论文（如 **TableRender‑RL**、**VisLaTeX**）都在尝试把渲染后图像作为奖励，或把结构化图谱直接作为中间表示。更广义上，强化学习在多模态生成任务中的“双奖励”框架被移植到公式识别、流程图生成等场景。想进一步深入，可以关注：

- **更高效的渲染‑奖励闭环**：比如使用可微渲染器直接在梯度中传播视觉误差。  
- **跨域表格迁移**：把模型从印刷体表格迁移到手写、医学报告等特殊领域。  
- **统一评测基准**：社区正在讨论把 TEDS‑Structure 与 CW‑SSIM 合并成一个标准排行榜，后续工作可以直接在此基准上对比。

### 一句话记住它

把表格图片直接喂进强化学习的多模态语言模型，让模型在“写代码”和“看渲染”两方面同时得分，才能生成既结构正确又视觉逼真的 LaTeX 表格。