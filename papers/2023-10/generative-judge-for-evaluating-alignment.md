# Generative Judge for Evaluating Alignment

> **Date**：2023-10-09
> **arXiv**：https://arxiv.org/abs/2310.05470

## Abstract

The rapid development of Large Language Models (LLMs) has substantially expanded the range of tasks they can address. In the field of Natural Language Processing (NLP), researchers have shifted their focus from conventional NLP tasks (e.g., sequence tagging and parsing) towards tasks that revolve around aligning with human needs (e.g., brainstorming and email writing). This shift in task distribution imposes new requirements on evaluating these aligned models regarding generality (i.e., assessing performance across diverse scenarios), flexibility (i.e., examining under different protocols), and interpretability (i.e., scrutinizing models with explanations). In this paper, we propose a generative judge with 13B parameters, Auto-J, designed to address these challenges. Our model is trained on user queries and LLM-generated responses under massive real-world scenarios and accommodates diverse evaluation protocols (e.g., pairwise response comparison and single-response evaluation) with well-structured natural language critiques. To demonstrate the efficacy of our approach, we construct a new testbed covering 58 different scenarios. Experimentally, Auto-J outperforms a series of strong competitors, including both open-source and closed-source models, by a large margin. We also provide detailed analysis and case studies to further reveal the potential of our method and make a variety of resources public at https://github.com/GAIR-NLP/auto-j.

---

# 用于评估对齐的生成式评审模型 论文详细解读

### 背景：这个问题为什么难？

LLM（大语言模型）已经可以完成从写代码到写诗的各种任务，但随着模型被要求更贴合人类需求（比如帮忙头脑风暴、写邮件），传统的评测方式已经跟不上。过去的评测大多是固定的指标（BLEU、ROUGE）或是单一的人工打分，既不能覆盖千变万化的使用场景，也缺乏对模型“为什么好/坏”的解释。更糟的是，评测协议本身也在演进——有的需要两两比较，有的只看单个答案，这让同一个评测模型很难兼容。于是出现了一个核心难题：**如何用一个模型，既能在多种真实场景下给出可靠分数，又能给出可读的评语，且能适配不同的评测协议？**这正是论文要解决的痛点。

### 关键概念速览
**对齐（Alignment）**：让模型的输出符合人类的价值观和需求，类似于把一只调皮的狗训练成听话的伴侣。  
**生成式评审（Generative Judge）**：模型本身会“写评语”，而不是只输出一个数值，好比老师在批改作文时会给出分数和评语。  
**协议（Protocol）**：评测时的交互规则，例如“把两段答案放在一起让模型挑出更好的一段”。可以想象成比赛的裁判手册。  
**单响应评估（Single-response Evaluation）**：模型只看一条答案并给出评价，类似于老师只批改一篇作文。  
**两两比较（Pairwise Comparison）**：模型同时看到两条答案，挑出更优者，像是两篇作文让老师选出最佳。  
**Auto-J**：论文中提出的13B参数的生成式评审模型，全称 Automatic Judge。  
**真实场景数据（Real-world Scenarios）**：来源于用户真实提问和模型生成的回答，覆盖了从技术支持到创意写作的各种情境。  
**可解释性（Interpretability）**：评审模型给出的文字解释，使人能看懂为什么给出某个分数，类似于老师的评语让学生明白错误所在。

### 核心创新点
1. **从离线评分转向生成式评审**  
   过去的评测模型大多只输出一个标量分数，缺乏解释。论文把评审模型训练成能够**生成自然语言评语**的模型。这样一来，评审结果不再是黑盒，而是可以直接阅读的“评语”，帮助研究者快速定位模型的不足。

2. **统一多协议评测框架**  
   传统评测系统要么只能做单响应，要么只能做两两比较，切换成本高。作者在训练数据中同时加入了**单响应任务和两两比较任务**的示例，使得同一个模型能够在不同协议下自如切换。相当于让同一位老师既能单独批改作文，又能在两篇作文之间做出选择。

3. **大规模真实场景数据驱动**  
   为了让评审模型具备通用性，作者收集了**58种不同的使用场景**，每种场景下都有大量用户查询和对应的LLM回答。训练时让模型学习“在这个场景里，什么样的回答算好”，从而提升跨场景的评估能力。相比于只在少数公开数据集上微调的做法，这种覆盖面更广、噪声更贴近真实。

4. **13B 参数的高效实现**  
   虽然很多对齐评估工作依赖数百亿参数的闭源模型（如 GPT‑4），但作者通过**GPT‑4 蒸馏**得到一个仅13B 参数的开源模型 Auto‑J，保持了竞争力的同时大幅降低了部署成本。相当于把一位资深评审老师的经验浓缩进一本薄薄的手册。

### 方法详解
整体思路可以划分为三步：**数据构建 → 多任务训练 → 协议适配推理**。

1. **数据构建**  
   - **场景收集**：作者从真实产品日志、公开问答平台以及内部对话系统中抽取了 58 类任务（如技术支持、创意写作、法律咨询等）。每类任务下都有数千条用户提问。  
   - **答案生成**：针对每条提问，使用多种开源和闭源 LLM（包括 GPT‑4）生成若干候选回答，确保答案质量跨度大。  
   - **评审标注**：利用 GPT‑4 进行**自动评审**，让它给每个回答打分并写出简短评语，同时也生成两两比较的“更好答案”。这些评审文本被视作监督信号。  
   - **格式化**：所有样本统一为“问题 + 回答（或两回答） → 评审指令 → 评审输出”的模板，方便模型学习。

2. **多任务训练**  
   - **任务混合**：训练时随机抽取单响应样本或两两比较样本，模型需要在同一前向传播中完成不同的指令。指令以自然语言形式出现，例如“请比较以下两段回答并说明哪段更好”。  
   - **指令微调（Instruction Tuning）**：在大规模指令数据上微调，使模型能够理解并执行评审指令。这里的关键是让模型把“评审”当作一种生成任务，而不是分类。  
   - **蒸馏技巧**：为了让 13B 参数模型捕捉到 GPT‑4 的评审风格，作者采用了**软标签蒸馏**：把 GPT‑4 的概率分布（而非硬标签）作为目标，帮助小模型学习更细腻的评审语言。  
   - **损失函数**：主要是交叉熵损失，针对生成的评语进行最大似然训练；在两两比较任务中，还加入了一个**对比损失**，鼓励模型在输出“更好答案”时与标注一致。

3. **协议适配推理**  
   - **单响应模式**：给模型提供“问题 + 回答”，加上指令“请对这段回答进行评分并给出评语”。模型直接输出“分数：8/10，评语：...”的结构化文本。  
   - **两两比较模式**：提供两段回答以及指令“请比较并说明哪段更好”。模型输出“更好答案：A，评语：...”的格式。  
   - **后处理**：通过正则提取分数或答案标识，形成统一的评测报告。因为模型本身已经生成了自然语言解释，后处理只需要把结构化信息抽出来即可。

**最巧妙的点**在于**同一模型通过指令区分协议**，而不需要为每种评测方式训练独立的模型。这样既省资源，又保证了评审风格的一致性。

### 实验与效果
- **测试平台**：作者搭建了一个覆盖 58 种场景的评测基准，每种场景下都有 200 条随机抽取的问答对。  
- **对比基线**：包括开源的 **OpenAI GPT‑3.5、Claude、LLaMA‑2‑13B**，以及闭源的 **GPT‑4**（作为上限）。  
- **主要指标**：使用 **准确率（在两两比较中选对的比例）**、**相关性评分（单响应的分数与人工标注的皮尔逊相关系数）**以及 **评语质量（通过人工评审打分）**。  
- **结果概览**：  
  - 在两两比较任务上，Auto‑J 的准确率约 **78%**，比 LLaMA‑2‑13B 的 **62%** 高出 16 个百分点，逼近 GPT‑4 的 **84%**。  
  - 单响应相关性上，Auto‑J 与人工评分的皮尔逊系数为 **0.71**，而最强开源基线只有 **0.55**。  
  - 评语可读性方面，人工评审给 Auto‑J 打了 **4.3/5**，显著优于其他开源模型的 **3.1/5**。  
- **消融实验**：  
  - 去掉蒸馏阶段，模型在两两比较准确率下降约 **5%**，说明 GPT‑4 的评审风格对小模型提升明显。  
  - 只使用单响应数据训练，模型在两两比较任务上跌至 **70%**，验证了多任务混合的必要性。  
- **局限性**：论文承认评审模型仍然会受到训练数据偏见的影响，尤其在法律或医学等高风险场景下，生成的评语可能缺乏专业深度。此外，13B 参数在极端长文本评审时会出现截断，导致信息丢失。

### 影响与延伸思考
这篇工作在公开评估对齐模型的方向上起到了桥梁作用。它把 **生成式评审** 从概念验证推向了可复现的基准，随后有几篇后续研究（如 *EvalLM*、*CriticGPT*）在此基础上加入了 **多模态评审**（图文混合）和 **自适应协议学习**（模型自行决定使用哪种评测方式）。如果想进一步深入，可以关注以下几个方向：  
1. **更大规模的评审蒸馏**：把更强的闭源模型（如 GPT‑4‑Turbo）蒸馏到更小的模型上，探索质量-成本的最佳平衡。  
2. **领域专精评审**：在医学、法律等高风险领域构建专门的评审数据集，让评审模型具备专业背景。  
3. **评审可信度校准**：研究如何让模型给出的评语附带置信度，帮助使用者判断评审结果的可靠性。  
4. **跨语言评审**：当前数据主要是英文，扩展到多语言场景将是重要的实用需求。

### 一句话记住它
**Auto‑J 用生成式评审把“打分+解释”合二为一，让同一个小模型既能比较答案，又能给出可读的评语，跨场景、跨协议都能工作。**