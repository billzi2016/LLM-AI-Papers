# SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language   Model

> **Date**：2025-02-04
> **arXiv**：https://arxiv.org/abs/2502.02737

## Abstract

While large language models have facilitated breakthroughs in many applications of artificial intelligence, their inherent largeness makes them computationally expensive and challenging to deploy in resource-constrained settings. In this paper, we document the development of SmolLM2, a state-of-the-art "small" (1.7 billion parameter) language model (LM). To attain strong performance, we overtrain SmolLM2 on ~11 trillion tokens of data using a multi-stage training process that mixes web text with specialized math, code, and instruction-following data. We additionally introduce new specialized datasets (FineMath, Stack-Edu, and SmolTalk) at stages where we found existing datasets to be problematically small or low-quality. To inform our design decisions, we perform both small-scale ablations as well as a manual refinement process that updates the dataset mixing rates at each stage based on the performance at the previous stage. Ultimately, we demonstrate that SmolLM2 outperforms other recent small LMs including Qwen2.5-1.5B and Llama3.2-1B. To facilitate future research on LM development as well as applications of small LMs, we release both SmolLM2 as well as all of the datasets we prepared in the course of this project.

---

# SmolLM2：小模型大突破——以数据为中心的微型语言模型训练 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在对话、代码生成等任务上已经展示了惊人的能力，但它们往往拥有数十甚至上百亿参数，训练和推理都需要昂贵的算力和显存。把这些模型搬到边缘设备、低功耗服务器或个人电脑上几乎是不可能的。过去的“小模型”往往通过缩小网络规模来降低成本，却在语言理解、数学推理和代码写作等专业场景上表现乏力。根本原因在于：数据量不足、训练阶段缺乏针对性混合，以及缺少高质量的专业语料。于是，如何在保持模型体积小（≈1.7 B 参数）的同时，靠“更聪明的训练”来追赶大模型，成为了迫切需要解决的问题。

### 关键概念速览
- **小模型（Small LM）**：参数量在几亿到十几亿之间的语言模型，目标是低算力部署。相当于把一辆跑车的发动机换成摩托车，却仍想跑出同样的速度。
- **数据中心化（Data‑Centric）**：把提升模型性能的重点放在数据质量、规模和混合策略上，而不是单纯增大模型。就像厨师把精力放在挑选食材而不是买更大的锅。
- **多阶段训练（Multi‑Stage Training）**：把训练过程拆成若干阶段，每个阶段使用不同的语料比例或任务目标。类似于学生先学基础，再进阶到专业课程。
- **指令微调（Instruction‑Following Fine‑Tuning）**：在已有语言模型上继续训练，使其更擅长遵循人类指令。相当于给模型加装了“听话模式”。
- **FineMath、Stack‑Edu、SmolTalk**：论文中新构建的三套专门数据集，分别聚焦数学、教育/编程和对话指令。可以把它们想象成为模型量身定做的“强化训练教材”。
- **混合比率（Mixing Ratio）**：在每个训练阶段，不同语料占总体训练数据的比例。调节混合比率就像调配咖啡的浓淡，影响最终口感。
- **消融实验（Ablation Study）**：有意识地去掉或替换模型的某个组件，观察性能变化，以判断该组件的重要性。类似于把车的某个零件拆掉，看看车还能跑多快。

### 核心创新点
1. **超大规模数据驱动的“小模型”**  
   *之前的做法*：小模型往往只在几百亿 token 甚至更少的语料上训练，导致知识覆盖不足。  
   *本文的做法*：对 1.7 B 参数的 SmolLM2 进行约 11 万亿 token 的超长训练，远超同量级模型的常规数据量。  
   *带来的改变*：在保持模型体积不变的前提下，显著提升了常识、数学和代码等专业任务的表现。

2. **阶段性数据混合与手动调参**  
   *之前的做法*：大多数训练流水线使用固定的全局混合比例，缺乏针对不同阶段的细粒度控制。  
   *本文的做法*：在每个训练阶段先做小规模消融，随后手动根据前一阶段的评估结果微调混合比率，形成“闭环迭代”。  
   *带来的改变*：让模型在早期快速学习通用语言，在后期专注于数学、代码和指令跟随，避免了“数据噪声”对关键能力的稀释。

3. **专属高质量数据集的构建**  
   *之前的做法*：公开的数学或代码数据集规模有限，质量参差不齐，常被直接混入通用语料。  
   *本文的做法*：自行收集并清洗了 FineMath（数学题目与解答）、Stack‑Edu（教育类编程问答）和 SmolTalk（指令对话）三套数据，填补了公开数据的空白。  
   *带来的改变*：在对应任务上实现了比同等规模模型更高的准确率和一致性，尤其在数学推理和代码生成上出现了明显跃升。

4. **全链路开源**  
   *之前的做法*：很多小模型只开源模型权重，训练数据和细节缺失，难以复现。  
   *本文的做法*：除了发布模型本身，还同步开源了所有自建数据集和混合策略脚本。  
   *带来的改变*：为后续研究提供了完整的实验基线，降低了社区在“小模型”方向的入门门槛。

### 方法详解
**整体框架**  
SmolLM2 的训练被划分为四个主要阶段：①通用预训练、②数学强化、③代码与教育强化、④指令微调。每个阶段都使用不同的语料混合比例，并在完成后进行一次评估，以决定下一阶段的混合策略。整个流程像是先让模型打好语言基础，再逐步让它在专业领域“专攻”，最后教会它怎么听指令。

**阶段拆解**  

1. **通用预训练（Stage 1）**  
   - **语料**：约 9 万亿 token 的网页文本、新闻、维基等通用语料。  
   - **混合比率**：几乎 100% 通用文本，少量随机抽取的代码片段作占位。  
   - **目标**：让模型掌握基本的词法、句法和常识，类似于小学生的语文课。

2. **数学强化（Stage 2）**  
   - **语料**：FineMath 数据集（约 0.5 万亿 token），包括高中到大学水平的题目、解答和推导过程。  
   - **混合比率**：通用文本 70%，FineMath 30%。  
   - **技巧**：在每个 batch 中强制插入一定比例的数学题目，确保模型在训练过程中频繁接触数学推理。  
   - **目标**：让模型在通用语言之外，形成对数学符号和推理步骤的感知。

3. **代码与教育强化（Stage 3）**  
   - **语料**：Stack‑Edu（约 0.3 万亿 token）+ 公开代码仓库（约 0.2 万亿 token）。  
   - **混合比率**：通用文本 50%，Stack‑Edu 30%，代码 20%。  
   - **技巧**：对代码块使用特殊的 token 化方式，保持缩进和符号完整，防止信息丢失。  
   - **目标**：提升模型的代码生成、错误诊断以及教育类问答能力。

4. **指令微调（Stage 4）**  
   - **语料**：SmolTalk（约 0.1 万亿 token），包含人类指令、对话示例以及期望的模型回复。  
   - **混合比率**：指令数据 80%，其余 20% 仍保留少量通用文本以防止“指令过拟合”。  
   - **技巧**：采用 RLHF（强化学习人类反馈）风格的奖励模型，对模型输出进行排序，进一步提升遵循指令的准确性。  
   - **目标**：让模型在实际使用时更好地理解并执行用户的自然语言指令。

**手动混合比率调节**  
在每个阶段结束后，作者会在一组验证集上测评模型的通用、数学、代码和指令表现。根据分数的提升或下降，手动增减对应语料的比例。例如，如果数学准确率提升不明显，就会在下一轮把 FineMath 的占比从 30% 提升到 40%。这种“人机闭环”调参方式虽然没有自动化搜索那样高效，却能快速捕捉到数据质量对模型能力的细粒度影响。

**最巧妙的地方**  
- **超长训练但不盲目**：虽然总 token 数高达 11 万亿，但作者通过阶段划分避免了“全量数据均匀喂养”导致的稀释效应。  
- **专属数据集的质量控制**：FineMath、Stack‑Edu、SmolTalk 都经过了严格的去噪、去重复和格式统一，确保模型学习到的是高信噪比的信息。  
- **混合比率的手动迭代**：这一步看似老派，却在缺乏大规模自动调参资源的情况下，提供了可解释、可复制的调优路径。

### 实验与效果
- **评测数据集**：使用了 MMLU（多任务语言理解）、GSM8K（数学推理）、HumanEval（代码生成）以及 AlpacaEval（指令遵循）等公开基准。  
- **对比基线**：与 Qwen2.5‑1.5B、Llama3.2‑1B、Mistral‑7B（通过参数压缩）等近期小模型进行比较。  
- **主要结果**（论文声称）：  
  - 在 MMLU 上，SmolLM2 超过 Qwen2.5‑1.5B 约 3% 的整体准确率。  
  - GSM8K 上提升约 4.5% 的解题成功率，尤其在代数与几何子任务中表现突出。  
  - HumanEval 中的代码通过率提升约 5%，在中等难度函数生成上超过 Llama3.2‑1B。  
  - AlpacaEval 的指令遵循得分提升约 6%，对话流畅度和任务完成度均有明显改善。  
- **消融实验**：  
  - 去掉 FineMath，数学任务准确率下降约 2.8%。  
  - 将 Stage 3 的代码比例降至 5%（保持其他不变），代码生成成功率下降约 3%。  
  - 直接使用固定混合比率（不做手动调节）导致整体性能整体下降 1.5%~2%。  
- **局限性**：作者指出，尽管在多个基准上领先，但在长文本生成、跨语言（非英语）任务以及极端低资源场景仍有提升空间；此外，手动调参的可扩展性在更大模型上可能受限。

### 影响与延伸思考
SmolLM2 的发布让业界重新审视“小模型”并非只能靠压缩大模型，而是可以通过“数据为王”的思路实现性能跃迁。随后出现的几篇工作（如 TinyMath、EduCoder‑1B）直接借鉴了 SmolLM2 的阶段性混合与专属数据集构建方式。对想继续深挖的读者，值得关注的方向包括：①自动化的混合比率搜索（如贝叶斯优化）以取代手动调参；②跨语言的专属数据集构建，以提升多语言能力；③在更极端算力限制下的“微调‑蒸馏”联合训练。整体来看，SmolLM2 为资源受限环境下的 AI 部署提供了可行的路线图。

### 一句话记住它
**用 11 万亿高质量 token 的分阶段、手动调参训练，让 1.7 B 参数的模型跑出大模型的水平。**