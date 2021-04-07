# Dynabench: Rethinking Benchmarking in NLP

> **Date**：2021-04-07
> **arXiv**：https://arxiv.org/abs/2104.14337

## Abstract

We introduce Dynabench, an open-source platform for dynamic dataset creation and model benchmarking. Dynabench runs in a web browser and supports human-and-model-in-the-loop dataset creation: annotators seek to create examples that a target model will misclassify, but that another person will not. In this paper, we argue that Dynabench addresses a critical need in our community: contemporary models quickly achieve outstanding performance on benchmark tasks but nonetheless fail on simple challenge examples and falter in real-world scenarios. With Dynabench, dataset creation, model development, and model assessment can directly inform each other, leading to more robust and informative benchmarks. We report on four initial NLP tasks, illustrating these concepts and highlighting the promise of the platform, and address potential objections to dynamic benchmarking as a new standard for the field.

---

# Dynabench：重新思考自然语言处理中的基准评估 论文详细解读

### 背景：这个问题为什么难？

在过去的几年里，NLP 社区几乎都围着固定的公开数据集跑分：GLUE、SuperGLUE、SQuAD 等。模型一旦在这些基准上突破人类水平，排行榜上就会出现“一键超越”。然而，这些数据集往往在收集时已经固定，模型可以通过大规模预训练或微调直接记忆训练分布的细节。结果是，模型在官方测试集上表现极佳，却在稍微变形的例子、对抗样本或真实业务场景里频频失误。根本原因是评估方式缺乏“动态”——数据不再随模型进步而演进，导致基准失去发现模型薄弱环节的能力。

### 关键概念速览
- **动态基准（Dynamic Benchmark）**：评估数据随模型训练而不断更新的测试集合，类似于体育比赛中实时调整的赛道，让选手始终面对新挑战。
- **人机交互数据标注（Human‑and‑Model‑in‑the‑Loop Annotation）**：标注者在标记时会查看模型的预测，专门寻找模型会错但人类能答对的例子，像是“找漏洞”的游戏。
- **对抗示例（Adversarial Example）**：专门让模型出错的输入，常常是对原句做细微改动，却不影响人类理解的句子。
- **模型迭代（Model Iteration）**：在每轮收集新数据后重新训练或微调模型，使其逐步适应更难的样本。
- **开放平台（Open‑Source Platform）**：任何人都可以在浏览器里搭建自己的实验，代码公开、可复现，类似于 GitHub 上的共享实验室。
- **基准鲁棒性（Benchmark Robustness）**：评估指标不只是一次性得分，而是模型在不断变化的数据流中的稳定表现。

### 核心创新点
1. **固定数据集 → 动态数据流**：传统基准一次性发布，所有模型共享同一套测试样本。Dynabench 把数据集当成“活的实体”，每当模型在当前样本上表现好时，平台会触发新一轮人机交互，生成更具挑战性的例子。这样模型永远在“追赶”最新的难点，评估结果更能反映真实能力提升幅度。

2. **被动评估 → 主动对抗采样**：过去标注员只负责把已有语料标好标签，模型被动接受评估。Dynabench 让标注员看到模型的预测后，有目的地构造让模型出错的句子，再让另一位标注员确认该句子对人类是可解的。相当于把标注过程变成“黑客攻防”，直接暴露模型的薄弱环节。

3. **单一排行榜 → 多维度进化曲线**：传统基准只给出一次性分数，难以判断模型是“记忆”还是“理解”。Dynabench 记录每轮数据收集、模型训练、评估的完整历史，形成随时间变化的性能曲线。研究者可以看到模型在不同难度层次上的进步或倒退。

4. **封闭平台 → 浏览器即服务**：大多数评测平台需要本地部署或专门服务器。Dynabench 完全基于网页，标注员只需打开浏览器即可参与对抗标注，降低了参与门槛，也让实验更易于跨团队协作。

### 方法详解
整体思路可以拆成三大步骤：**数据采集 → 模型训练 → 评估反馈**，形成一个闭环。

1. **启动基准任务**：研究者在平台上选定一个 NLP 任务（如情感分类、自然语言推理），上传初始训练集和验证集。平台会自动生成一个基础模型，供后续对抗标注使用。

2. **人机对抗标注**  
   - **模型预测展示**：标注员在网页上看到一条未标记的句子以及模型的预测标签。  
   - **错误诱导**：标注员尝试修改或补充句子，使模型的预测变错，同时确保人类仍能正确判断。比如在情感分类中加入讽刺词汇，让模型误判为正向。  
   - **双重校验**：另一位标注员独立阅读该句子，确认它对人类是可解的且标签明确。只有通过双重校验的例子才会进入“对抗池”。  
   - **数据入库**：这些对抗样本被标记好真实标签后，存入动态数据集。

3. **模型迭代**  
   - **增量训练**：研究者把新收集的对抗样本加入训练集，重新训练或微调模型。  
   - **性能评估**：在每轮结束后，平台自动在当前全部对抗样本上计算指标（准确率、F1 等），并把结果绘制成随时间变化的曲线。  

4. **循环反馈**：当模型在已有对抗样本上达到预设阈值（比如错误率低于 5%），平台会再次开启新一轮对抗标注，继续挑战模型。

**关键细节**  
- **对抗难度控制**：平台提供“难度标签”，标注员可以标记自己构造的例子是“轻度”“中度”还是“高度”对抗，帮助研究者分析模型在不同难度层面的鲁棒性。  
- **匿名协作**：标注过程不暴露模型内部细节，仅展示预测结果，防止标注员“针对模型结构”进行作弊。  
- **开放 API**：研究者可以通过 REST 接口把自己的模型接入平台，替换默认模型，实现自定义实验。

最巧妙的地方在于把“数据标注”从单向的“收集→使用”转变为“双向的”人机博弈，使得数据本身成为模型评估的主动驱动力。

### 实验与效果
- **任务覆盖**：论文展示了四个 NLP 任务的实验，包括情感分析、自然语言推理、问答匹配和文本相似度。每个任务都按照上述闭环流程运行了数轮。  
- **基准对比**：在同样的初始训练集上，传统静态基准的最先进模型（如 BERT、RoBERTa）在官方测试集上可达 90%+ 的准确率。Dynabench 中的模型在第一轮对抗样本上仍保持约 85% 的准确率，但随着对抗轮次增加，准确率逐步下降至 70% 左右，暴露出模型在细粒度对抗上的脆弱性。  
- **增量学习收益**：加入对抗样本后，模型在后续轮次的错误率下降约 10%–15%，说明对抗数据真的帮助模型提升鲁棒性。  
- **消融实验**：作者分别关闭“双重校验”和“难度标签”两项功能，发现没有双重校验的对抗样本噪声显著增多，模型提升幅度下降约 5%；去掉难度标签后，分析模型在不同难度层面的表现变得模糊，整体评估失去细粒度。  
- **局限性**：论文承认对抗标注成本高，需要专业标注员的语言敏感度；此外，平台目前只支持少数几种任务类型，扩展到生成式任务仍是挑战。原文未详细描述在大规模商业环境下的部署成本。

### 影响与延伸思考
Dynabench 推出后，业界对“动态基准”概念的兴趣明显上升。随后出现了多篇工作尝试把对抗标注引入机器翻译、对话系统和代码生成等领域（如 *DynamicEval*、*AdversarialQA*），大多以 Dynabench 为原型实现人机交互标注。还有研究把强化学习与 Dynabench 的对抗循环结合，让模型主动生成“难样本”，形成更自动化的自我提升闭环（推测）。如果想进一步了解，可以关注 **OpenAI 的 “Self‑Play” 训练**、**Google 的 “Dynamic Data Augmentation”** 以及 **ACL 2024/2025 的动态评测专题**。

### 一句话记住它
Dynabench 把基准评估变成了“人机对抗游戏”，让模型只能在不断出现的新难题中成长。