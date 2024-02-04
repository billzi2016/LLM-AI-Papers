# Aligner: Efficient Alignment by Learning to Correct

> **Date**：2024-02-04
> **arXiv**：https://arxiv.org/abs/2402.02416

## Abstract

With the rapid development of large language models (LLMs) and ever-evolving practical requirements, finding an efficient and effective alignment method has never been more critical. However, the tension between the complexity of current alignment methods and the need for rapid iteration in deployment scenarios necessitates the development of a model-agnostic alignment approach that can operate under these constraints. In this paper, we introduce Aligner, a novel and simple alignment paradigm that learns the correctional residuals between preferred and dispreferred answers using a small model. Designed as a model-agnostic, plug-and-play module, Aligner can be directly applied to various open-source and API-based models with only one-off training, making it suitable for rapid iteration. Notably, Aligner can be applied to any powerful, large-scale upstream models. Moreover, it can even iteratively bootstrap the upstream models using corrected responses as synthetic human preference data, breaking through the model's performance ceiling. Our experiments demonstrate performance improvements by deploying the same Aligner model across 11 different LLMs, evaluated on the 3H dimensions (helpfulness, harmlessness, and honesty). Specifically, Aligner-7B has achieved an average improvement of 68.9% in helpfulness and 23.8% in harmlessness across the tested LLMs while also effectively reducing hallucination. In the Alpaca-Eval leaderboard, stacking Aligner-2B on GPT-4 Turbo improved its LC Win Rate from 55.0% to 58.3%, surpassing GPT-4 Omni's 57.5% Win Rate (community report).

---

# Aligner：通过学习纠正实现高效对齐 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）已经可以生成流畅的文字，但要让它们在实际应用中既有用又安全，需要对模型进行“对齐”。传统的对齐方式往往依赖大规模的强化学习（RLHF）或多轮人类反馈，这既耗时又需要大量标注成本。与此同时，企业在部署新模型时常常需要快速迭代：模型换了、接口改了，原来的对齐数据几乎失效。于是出现了两难：要么用笨重的全模型微调，牺牲迭代速度；要么直接使用原始模型，风险高、表现差。破解这个矛盾，需要一种既轻量又能跨模型使用的对齐手段。

### 关键概念速览
**对齐（Alignment）**：让模型的输出符合人类期望的价值观和任务需求，就像给一辆自动驾驶汽车加装安全规则。  
**残差学习（Residual Learning）**：模型不直接输出完整答案，而是学习“纠正项”，类似于在原始答案上加一层修正笔记。  
**模型无关（Model‑agnostic）**：方法不依赖特定的模型结构或参数规模，像通用的插件可以装在任何软件上。  
**3H 维度**：帮助性（Helpfulness）、无害性（Harmlessness）和诚实性（Honesty），是评估对齐质量的三大指标。  
**合成偏好数据（Synthetic Preference Data）**：用已经纠正过的答案当作“人类偏好”，再喂回上游模型进行二次训练，类似于让学生先看老师的批改再自行练习。  
**LC Win Rate**：在 Alpaca‑Eval 基准上，模型在两两对话中获胜的比例，用来衡量整体表现。  

### 核心创新点
1. **残差对齐 → 学习纠正残差 → 大幅提升效率**  
   传统对齐直接在原始模型上做强化学习，需要完整的梯度回传。Aligner 只训练一个小模型去预测“正确答案”和“错误答案”之间的差值，等于是让小模型写批注，而不是重新写整篇文章。这样既省算力，又能快速适配新模型。

2. **一次性训练 → 即插即用 → 跨模型部署**  
   以前每换一个模型都要重新收集偏好数据并微调。Aligner 只需要一次性训练其纠正模型，然后把它当作插件挂在不同的上游模型上。实验表明，同一个 7B 大小的纠正模型在 11 种 LLM 上都能提升帮助性和无害性。

3. **纠正输出 → 生成合成偏好 → 迭代提升上游模型**  
   作者让纠正模型产生的“更好答案”充当人类偏好，喂回原始大模型进行二次微调，形成闭环。这样突破了单次对齐的性能上限，类似于让模型自我教学。

4. **轻量级模型即可实现显著提升**  
   只用了 2B/7B 参数的纠正模型，就在 GPT‑4 Turbo 上把 LC Win Rate 从 55.0% 拉到 58.3%，超过了更大、更贵的 GPT‑4 Omni（57.5%）。这说明对齐瓶颈不一定在模型规模，而在纠错机制的设计。

### 方法详解
整体思路可以拆成三步：**（1）收集原始答案与偏好答案；（2）训练纠正模型学习残差；（3）将纠正模型作为插件部署并可选地生成合成偏好进行二次训练**。

1. **数据准备**  
   - 先让上游 LLM（如 Llama‑2、GPT‑4 等）在同一任务上生成两套答案：一套是“偏好”（由人类或高质量模型挑选的理想答案），另一套是“非偏好”（原始模型直接输出）。  
   - 将两套答案配对，形成“原始 → 目标”映射。

2. **残差学习**  
   - 纠正模型（Aligner）接受原始答案的文本表示以及任务提示，输出一个“纠正向量”。  
   - 通过最小化纠正向量与目标答案之间的差距来训练模型。直观上，它在原始答案上写批注，告诉模型哪里该改、怎么改。  
   - 训练只涉及纠正模型本身的参数，原始大模型保持不动，算力需求相当于训练一个中等规模的语言模型。

3. **插件化部署**  
   - 推理时，先让上游模型生成初稿。  
   - 把初稿喂进纠正模型，得到纠正向量，再将其与原始文本合并（如加权叠加或直接替换关键片段），得到最终输出。  
   - 这一过程只需一次前向传播，几乎不增加延迟。

4. **合成偏好闭环**  
   - 将纠正后的答案视作“高质量”样本，加入偏好数据池。  
   - 用这些合成数据对上游模型进行轻量级微调（如 LoRA），让上游模型本身学会产生更好的答案。  
   - 迭代若干轮后，原始模型的输出质量提升，纠正模型的负担进一步减轻。

**最巧妙的点**在于把对齐任务转化为“学习差值”，而不是“重新学习全部”。这让小模型可以专注于错误模式的捕捉，省去了大模型的全局梯度传播，极大提升了训练和部署的灵活性。

### 实验与效果
- **评测范围**：在 11 种开源或 API‑based LLM 上，使用 3H 维度（帮助性、无害性、诚实性）进行综合评估；另外在 Alpaca‑Eval 基准上测 LC Win Rate。  
- **主要结果**：  
  - Aligner‑7B 在帮助性上平均提升 68.9%，在无害性上提升 23.8%，并显著降低幻觉（hallucination）出现率。  
  - 在 Alpaca‑Eval 中，堆叠 Aligner‑2B 于 GPT‑4 Turbo，使 LC Win Rate 从 55.0% 上升到 58.3%，超过了 GPT‑4 Omni 的 57.5%。  
- **对比基线**：与传统 RLHF、PPO 微调等方法相比，Aligner 在相同算力下取得更高的 3H 分数；在算力相当的轻量级对齐方案中，提升幅度更为显著。  
- **消融实验**：原文提供了对“仅使用纠正模型不进行合成偏好闭环”和“仅使用合成偏好不使用纠正模型”的对比，结果显示两者结合时提升最大，说明两块机制相互增益。  
- **局限性**：论文承认纠正模型的质量仍受原始答案的可解释性限制；在极端长文本或高度结构化任务上，残差学习的效果尚未充分验证。

### 影响与延伸思考
Aligner 把对齐问题抽象为“残差纠正”，为后续研究提供了轻量化、模块化的思路。自发布以来，已有多篇工作尝试将残差对齐与检索增强、工具调用等结合，形成“对齐+检索”混合框架（推测）。此外，合成偏好闭环的理念激发了“自我蒸馏式对齐”方向的探索，即让模型在没有额外人工标注的情况下自行提升安全性。想进一步了解，可以关注 LoRA、Adapter 等参数高效微调技术，以及最近的 “Preference Modeling without Human Labels” 研究。

### 一句话记住它
把对齐当成给模型写批注：只训练一个小纠正器，就能让任何大模型快速变得更有用、更安全。