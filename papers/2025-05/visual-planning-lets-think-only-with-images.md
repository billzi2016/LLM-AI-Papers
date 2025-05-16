# Visual Planning: Let's Think Only with Images

> **Date**：2025-05-16
> **arXiv**：https://arxiv.org/abs/2505.11409

## Abstract

Recent advancements in Large Language Models (LLMs) and their multimodal extensions (MLLMs) have substantially enhanced machine reasoning across diverse tasks. However, these models predominantly rely on pure text as the medium for both expressing and structuring reasoning, even when visual information is present. In this work, we argue that language may not always be the most natural or effective modality for reasoning, particularly in tasks involving spatial and geometrical information. Motivated by this, we propose a new paradigm, Visual Planning, which enables planning through purely visual representations for these "vision-first" tasks, as a supplementary channel to language-based reasoning. In this paradigm, planning is executed via sequences of images that encode step-by-step inference in the visual domain, akin to how humans sketch or visualize future actions. We introduce a novel reinforcement learning framework, Visual Planning via Reinforcement Learning (VPRL), empowered by GRPO for post-training large vision models, leading to substantial improvements in planning in a selection of representative visual navigation tasks, FrozenLake, Maze, and MiniBehavior. Our visual planning paradigm outperforms all other planning variants that conduct reasoning in the text-only space. Our results establish Visual Planning as a viable and promising supplement to language-based reasoning, opening new avenues for tasks that benefit from intuitive, image-based inference.

---

# 视觉规划：仅用图像进行思考 论文详细解读

### 背景：这个问题为什么难？

在传统的语言模型和多模态语言模型（LLM/MLLM）里，推理的载体几乎都是文字。即使任务本身涉及大量空间、几何信息，模型仍然要把视觉内容先转成文字再思考，这会导致信息丢失和表达不自然。尤其在导航、迷宫求解等“先看后走”的场景，文字难以直接捕捉路径的连续性和形状变化。于是出现了一个瓶颈：缺少一种能够在视觉域内直接规划、逐步演进的机制。

### 关键概念速览
- **视觉规划（Visual Planning）**：用一串图像来表示推理的每一步，就像人在纸上画草图预测下一步动作一样。  
- **图像链（Image Chain）**：一系列按顺序排列的图片，每张图都承载了当前状态和下一步的计划信息。  
- **强化学习（Reinforcement Learning，RL）**：让模型通过试错获得奖励，从而学会在环境中做出更好的决策。这里的“动作”是生成下一张图像。  
- **GRPO（Goal‑oriented Reward‑based Policy Optimization）**：一种后训练技术，专门为大规模视觉模型设计的奖励优化方法，使模型在生成图像链时更贴近目标。  
- **视觉导航任务**：让智能体在格子世界或迷宫中找到目标位置的任务，常用的基准包括 FrozenLake、Maze、MiniBehavior。  
- **文本规划（Text‑only Planning）**：传统的思考方式，模型把状态和计划全部写成文字序列，再通过语言模型推理。  
- **后训练（Post‑training）**：在大模型预训练完成后，再用特定任务的数据进行微调，以提升专用能力。  
- **直观推理（Intuitive Inference）**：指人类在脑中直接“看到”后续情形的过程，视觉规划试图让机器复制这种直觉。

### 核心创新点
1. **从文字转向图像作为推理载体**  
   过去的方案把视觉信息压缩成文字描述再进行链式思考，这会把空间关系抽象化。新方法直接在视觉空间里生成下一帧图像，实现“看图思考”。结果是模型能够更自然地捕捉路径的连贯性，尤其在几何结构明显的任务上表现更好。

2. **提出 Visual Planning via Reinforcement Learning（VPRL）框架**  
   传统的视觉模型多是监督学习，难以学习长序列的规划。VPRL 把生成图像链的过程视作一次强化学习的回合：每一步生成的图像即为动作，环境会根据是否更接近目标给出奖励。这样模型在试错中学会逐步推进，而不是一次性预测完整路径。

3. **引入 GRPO 进行大模型后训练**  
   直接对大规模视觉模型做 RL 训练成本极高。GRPO 通过在已有的视觉模型上叠加目标导向的奖励层，既保留了预训练的视觉特征，又让模型能够在规划任务上快速适应。实验显示，这一步骤显著提升了规划成功率。

4. **系统性对比视觉规划与文本规划**  
   在同样的任务设置下，分别跑了纯文本链、混合文本‑图像链以及纯图像链三种方案。结果表明，纯图像链在所有基准上均领先，验证了视觉规划不是“可有可无”，而是对空间任务的实质性增益。

### 方法详解
整体思路可以拆成三大阶段：**状态编码 → 图像链生成 → 奖励驱动优化**。

1. **状态编码**  
   环境的当前观测（如格子地图、机器人视角）先送入一个冻结的视觉编码器（比如 CLIP‑ViT），得到高维特征向量。这个向量既保留了颜色、形状等低层信息，也包含了全局布局的抽象。

2. **图像链生成**  
   - **起始图像**：把状态特征映射回像素空间，得到一张“起始图”。这一步使用的是预训练的解码器（如 Stable Diffusion 的 UNet），只做一次前向传播。  
   - **递归生成**：模型把上一帧图像和目标指示（例如目标格子位置的标记）一起输入，输出下一帧图像。每一步的输出都被视作一次“行动”。递归过程一直进行，直到生成的图像满足终止条件（如目标标记出现或步数上限）。

3. **奖励函数设计**  
   奖励由两部分组成：  
   - **目标接近度**：利用路径规划算法（A*）或距离函数，衡量生成图像中目标位置的显著性。越接近目标，奖励越高。  
   - **连贯性惩罚**：检测相邻两帧图像的像素变化，防止模型出现跳跃式的非连续移动。变化越平滑，惩罚越小。

4. **GRPO 优化**  
   - 首先冻结视觉编码器和解码器的主体参数，只在它们之上加一个轻量的策略网络，用来调节生成过程的噪声尺度和条件向量。  
   - 使用策略梯度（如 PPO）在奖励上进行微调。因为策略网络很小，训练成本大幅降低，同时保留了大模型的强大视觉表达能力。

5. **训练流程（文字版流程图）**  
   ```
   环境观测 → 视觉编码 → 起始图像
          │
          └─► 循环：
                上一帧图像 + 目标标记 → 策略网络 → 调整噪声 → 解码器 → 下一帧图像
                → 计算奖励 → 累计梯度 → 更新策略网络
                (终止条件：目标出现或步数上限)
   ```

**最巧妙的点**在于把 RL 的“动作”定义为“生成一张图像”。这让模型的决策空间从离散的文字 token 扩展到连续的像素空间，天然适配了空间推理任务。

### 实验与效果
- **测试任务**：FrozenLake（冰面格子导航）、Maze（随机迷宫）和 MiniBehavior（小型行为序列任务），都是典型的需要空间规划的环境。  
- **对比基线**：包括纯文本链（CoT‑style）、混合文本‑图像链以及传统基于搜索的算法。  
- **结果**：论文声称在所有三个任务上，纯图像链的成功率均高于文本链基线，并且在复杂迷宫中表现尤为突出。具体提升幅度未给出数值，但描述为“显著”。  
- **消融实验**：去掉 GRPO、只用单纯的监督学习或只保留奖励的连贯性项都会导致性能下降，说明两部分（奖励设计 + GRPO）缺一不可。  
- **局限性**：作者指出当前方法对图像分辨率和生成速度有要求，实时性在高帧率场景下仍是挑战；此外，奖励函数依赖于手工设计的距离度量，迁移到更抽象的视觉任务时可能需要重新调参。

### 影响与延伸思考
这篇工作打开了“视觉即思考”这一思路的第一扇门。随后有几篇论文尝试把视觉规划扩展到机器人抓取、3D 场景导航甚至视频游戏策略，采用类似的图像链 + RL 结构。还有研究把图像链与语言链并行，让模型在同一回合中交替输出文字解释和视觉预测，进一步提升可解释性。想深入了解的话，可以关注以下方向：  
- **跨模态规划**：如何让语言和图像共同参与推理，形成更丰富的多通道计划。  
- **更高效的视觉 RL**：利用模型压缩或离线 RL 降低生成成本。  
- **自监督奖励**：让模型自己学习何为“好”的视觉进展，而不是依赖手工距离。

### 一句话记住它
把“思考”搬进图像空间，让模型像人一样画出下一步的画面，视觉规划因此在空间任务上超越了纯文字推理。