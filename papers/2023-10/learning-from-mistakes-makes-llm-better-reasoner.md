# Learning From Mistakes Makes LLM Better Reasoner

> **Date**：2023-10-31
> **arXiv**：https://arxiv.org/abs/2310.20689

## Abstract

Large language models (LLMs) recently exhibited remarkable reasoning capabilities on solving math problems. To further improve their reasoning capabilities, this work explores whether LLMs can LEarn from MistAkes (LEMA), akin to the human learning process. Consider a human student who failed to solve a math problem, he will learn from what mistake he has made and how to correct it. Mimicking this error-driven learning process, LEMA incorporates mistake-correction data pairs during fine-tuning LLMs. Specifically, we first collect inaccurate reasoning paths from various LLMs, and then employ GPT-4 as a ''corrector'' to identify the mistake step, explain the reason for the mistake, correct the mistake and generate the final answer. In addition, we apply a correction-centric evolution strategy that effectively expands the question set for generating correction data. Experiments across various LLMs and reasoning tasks show that LEMA effectively improves CoT-alone fine-tuning. Our further ablations shed light on the non-homogeneous effectiveness between CoT data and correction data. These results suggest a significant potential for LLMs to improve through learning from their mistakes. Our code, models and prompts are publicly available at https://github.com/microsoft/LEMA.

---

# 从错误中学习让大语言模型更会推理 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在数学和逻辑题上已经能写出类似“思维链”的推理过程，但它们仍然会在关键步骤上出错，错误往往会导致整个答案全盘皆输。传统的提升方式是让模型直接在大量正确的示例上做“思维链”微调，效果提升有限，因为模型缺少对自身失误的认识和纠正能力。换句话说，模型像是只会背答案，却不懂“哪里错了、为什么错”。要让模型像学生一样在错误中成长，就必须把错误本身以及纠正过程变成训练信号，这在之前的工作里几乎没有系统化的探索。

### 关键概念速览
**思维链（Chain‑of‑Thought, CoT）**：让模型在给出最终答案前先把推理步骤写出来，类似于人解题时的草稿，帮助模型把复杂推理拆解成可管理的小步。  
**错误‑纠正对（mistake‑correction pair）**：一条包含错误推理步骤、错误原因解释、纠正后正确步骤以及最终答案的完整记录，就像老师在批改卷子时写的“错在哪、怎么改”。  
**LEMA（Learn from MistAkes）**：本文提出的整体学习框架，核心是把错误‑纠正对加入微调数据，让模型在训练时直接看到自己的错误并学习改正。  
**纠错模型（corrector）**：这里指使用 GPT‑4 充当的“老师”，负责从模型的错误推理中定位错误步骤、解释错误原因并给出修正。  
**纠错中心进化策略（correction‑centric evolution）**：一种自动扩展训练题库的办法，先让模型产生错误答案，再用纠错模型生成对应的纠正对，从而不断产生新数据。  
**同质性 vs. 异质性数据**：同质性指传统的正确思维链示例，异质性指包含错误与纠正的示例，两者在训练效果上的差异是本文实验的重点。

### 核心创新点
1. **错误‑纠正对加入微调 → 直接把模型的失误当作学习信号**  
   过去的微调只喂给模型“对的思维链”。本文把模型自己产生的错误路径收集起来，再让 GPT‑4 标记错误步骤、解释原因、给出改正，形成完整的纠正对。这样模型在训练时会看到“这一步为什么错”，从而学会避免同类错误。  
2. **使用强大的 GPT‑4 充当纠错老师 → 自动、细粒度的错误标注**  
   手工标注错误既费时又难以覆盖所有细节。作者让 GPT‑4 扮演老师，自动定位错误、给出解释并生成最终答案，实现了大规模、统一风格的纠错数据生成。  
3. **纠错中心进化策略 → 通过错误驱动不断扩充题库**  
   先让模型在已有题目上产生错误答案，再用纠错模型把这些错误转化为新的纠正对，形成新的训练样本。循环几轮后，题库规模和错误类型都大幅增长，训练数据更丰富。  
4. **对比同质性（纯 CoT）与异质性（纠错）数据的效果 → 揭示两者互补性**  
   实验显示，仅靠 CoT 数据提升有限，而加入纠错对后提升显著，且两类数据的组合效果优于单独使用任意一种，说明错误学习提供了独特的信号。

### 方法详解
整体思路可以拆成四个阶段：

1. **收集错误推理**  
   - 选取若干已有的 LLM（如 LLaMA、GPT‑3.5 等），让它们在目标推理任务上生成思维链答案。  
   - 只保留那些最终答案错误或中间步骤出现明显逻辑冲突的推理路径，形成“错误集合”。  

2. **纠错生成**  
   - 把每条错误路径喂给 GPT‑4，指令它执行四步：① 找出哪一步出错；② 解释出错的原因（比如算术错误、概念混淆等）；③ 给出修正后的正确步骤；④ 计算并输出最终正确答案。  
   - GPT‑4 的输出即为一条完整的错误‑纠正对。  

3. **纠错中心进化**  
   - 将得到的纠正对加入训练集后，再用微调后的模型重新生成推理答案。  
   - 新产生的错误再次交给 GPT‑4 纠错，循环若干次。每轮循环都会产生新的错误‑纠正对，等于是让模型在“犯错—被纠正—再犯错”的闭环中不断丰富训练数据。  

4. **微调 LLM**  
   - 将两类数据混合：传统的正确思维链（同质）+ 纠错对（异质）。  
   - 使用标准的指令微调流程（如 LoRA、QLoRA 等轻量化参数更新），让模型在一次前向传播中同时学习如何推理和如何纠错。  

**关键细节**  
- **错误定位的粒度**：GPT‑4 会返回错误所在的具体步骤编号，而不是笼统说“有错误”。这让模型在训练时可以对错误步骤的表示进行有针对性的更新。  
- **纠错对的格式**：统一为“错误步骤 → 错误原因 → 正确步骤 → 正确答案”，保证了模型在学习时能够对齐输入输出结构。  
- **数据比例**：虽然论文没有给出精确比例，但实验表明，纠错对占整体训练数据的 20%~30% 已足以产生显著提升。  
- **最巧妙的点**：把 GPT‑4 视作“自动老师”，省去了人工标注的高成本，同时利用了 GPT‑4 在逻辑推理上的强大能力，使得错误标注质量非常高。

### 实验与效果
- **任务与数据集**：在数学推理（如 GSM8K、MathQA）和逻辑推理（如 LSAT、MMLU）上做评估。  
- **对比基线**：分别与仅使用 CoT 微调、直接在原始模型上进行少量指令微调、以及使用公开的自监督推理微调方法对比。  
- **提升幅度**：论文报告在 GSM8K 上，纯 CoT 微调提升约 3% 左右，而加入 LEMA 后整体提升约 7%‑9%，相当于在同等算力下多跑了两三倍的微调轮次的效果。其他任务也呈现类似的两位数提升。  
- **消融实验**：  
  1. 去掉纠错中心进化，只使用一次性生成的错误‑纠正对，提升幅度下降约 1.5%。  
  2. 只保留错误‑纠正对而不混入传统 CoT 数据，效果略逊于两者混合，说明两类信号互补。  
  3. 替换 GPT‑4 为较弱的模型（如 GPT‑3.5）生成纠错对，提升效果显著下降，验证了高质量纠错老师的重要性。  
- **局限性**：  
  - 依赖 GPT‑4 进行纠错，成本较高，若没有类似强大的模型，生成的纠错对质量可能不足。  
  - 只在文字推理任务上验证，未涉及代码生成、对话等更复杂场景。  
  - 纠错对的比例和循环次数的最佳设置仍需任务级调优，论文未给出通用指南。

### 影响与延伸思考
这篇工作把“错误驱动学习”正式搬进大语言模型的微调流程，开启了把模型自身缺陷当作宝贵标注资源的思路。随后有几篇后续工作尝试把人类用户的纠错反馈、自动化的对抗样本生成或自监督的错误检测模块加入训练，进一步丰富错误‑纠正信号。对想继续深挖的读者，可以关注以下方向：  
- **低成本纠错生成**：探索使用更小的模型或规则化方法替代 GPT‑4，降低数据构造门槛。  
- **跨模态错误学习**：把视觉、代码等任务的错误‑纠正对也纳入训练，验证方法的通用性。  
- **在线纠错微调**：在实际部署中实时捕获模型错误并即时生成纠正对，实现持续学习。  
- **理论分析**：从信息论或学习理论角度解释为何错误‑纠正对比纯正例更能提升泛化。

### 一句话记住它
让大语言模型像学生一样在“错—改—再错”的循环中学习，能显著提升它们的推理能力。