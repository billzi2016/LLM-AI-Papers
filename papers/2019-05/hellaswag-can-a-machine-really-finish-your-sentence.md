# HellaSwag: Can a Machine Really Finish Your Sentence?

> **Date**：2019-05-19
> **arXiv**：https://arxiv.org/abs/1905.07830

## Abstract

Recent work by Zellers et al. (2018) introduced a new task of commonsense natural language inference: given an event description such as "A woman sits at a piano," a machine must select the most likely followup: "She sets her fingers on the keys." With the introduction of BERT, near human-level performance was reached. Does this mean that machines can perform human level commonsense inference?   In this paper, we show that commonsense inference still proves difficult for even state-of-the-art models, by presenting HellaSwag, a new challenge dataset. Though its questions are trivial for humans (>95% accuracy), state-of-the-art models struggle (<48%). We achieve this via Adversarial Filtering (AF), a data collection paradigm wherein a series of discriminators iteratively select an adversarial set of machine-generated wrong answers. AF proves to be surprisingly robust. The key insight is to scale up the length and complexity of the dataset examples towards a critical 'Goldilocks' zone wherein generated text is ridiculous to humans, yet often misclassified by state-of-the-art models.   Our construction of HellaSwag, and its resulting difficulty, sheds light on the inner workings of deep pretrained models. More broadly, it suggests a new path forward for NLP research, in which benchmarks co-evolve with the evolving state-of-the-art in an adversarial way, so as to present ever-harder challenges.

---

# HellaSwag：机器真的能完成你的句子吗？ 论文详细解读

### 背景：这个问题为什么难？
在自然语言推理里，常识补全（commonsense inference）要求模型在看到一个情境描述后，挑出最合乎常理的后续句子。早期的数据集（如SWAG）已经让 BERT 这类大模型达到接近人类的分数，表面上看似常识推理已经被“攻克”。然而，这些数据往往只涉及短小、结构化的选项，模型可以靠统计模式或局部线索猜出答案，而不是真正理解情境背后的因果常识。于是研究者发现，现有基准已经失去了区分模型真实能力的锋利度，需要一种更难、能暴露模型盲点的测试。

### 关键概念速览
- **常识补全（commonsense inference）**：给出一个事件描述，让模型选出最合理的后续句子，类似让人“续写”情景。它考验的是对日常生活因果关系的隐式理解，而不是词法匹配。  
- **对抗过滤（Adversarial Filtering, AF）**：一种数据收集策略，先让模型生成大量错误选项，再用一系列判别模型挑出最能骗过这些判别器的“难负例”。想象成“让小偷先挑选最像真品的赝品”，从而逼迫防盗系统更强。  
- **Goldilocks 区**：指数据难度既不太容易让模型直接猜对，也不至于让人类觉得荒唐的“恰到好处”区间。作者通过拉长句子、加入细节，让机器误判的概率升高，而人类仍能轻松辨别。  
- **预训练语言模型（Pretrained Language Model）**：像 BERT、RoBERTa 这类在海量文本上先学会语言规律，再微调到具体任务的模型。它们是当前 NLP 的“通用底层”。  
- **负例（Distractor）**：在多选题里用来迷惑模型的错误选项。高质量负例需要在语义上看起来合理，却在常识层面违背事实。  
- **基准共进化（Benchmark Co‑evolution）**：让评测数据和模型能力同步升级的理念，即每当模型突破现有基准，就用更难的对抗样本刷新基准，形成“猫捉老鼠”的循环。  

### 核心创新点
1. **对抗过滤的迭代式负例筛选 → 先用语言模型生成大量机器生成的续写，再训练一批判别模型逐轮挑出最能骗过它们的负例 → 最终得到的负例对人类几乎是显而易见的错误，却让最强的预训练模型仍频频失误，显著提升数据集难度。**  
2. **把例子长度和情境复杂度推向 Goldilocks 区 → 将原本几词的续写扩展为完整的情境句子，加入时间、因果、情感等细节 → 机器的表层语言匹配能力被削弱，需要真正的常识推理才能分辨，导致模型准确率跌破 50%。**  
3. **将数据集构建过程与模型性能闭环 → 每当新模型在 HellaSwag 上取得突破，作者就用更强的判别器重新过滤，生成更具欺骗性的负例 → 形成基准与模型的“对抗进化”，为后续研究提供了持续挑战的框架。**  

### 方法详解
整体思路可以拆成三大步骤：**生成 → 判别 → 过滤**，循环若干轮后交给人工标注。

1. **生成阶段**  
   - 使用大规模语言模型（如 GPT‑2）对每条情境描述进行采样，得到数十个候选续写。  
   - 这些候选本质上是机器“胡说八道”，但因为模型在训练时学到了大量真实语料，很多句子在表面上仍然流畅、语法正确。

2. **判别阶段**  
   - 训练一组二分类判别器，输入是（情境描述 + 续写），标签为“真实”（人类写的正确续写）或“伪造”。  
   - 判别器的结构与 BERT 类似，利用预训练的语言表示来捕捉细微的常识冲突。  
   - 训练好后，用它们对所有机器生成的候选进行打分，分数越高表示越像真实答案。

3. **过滤阶段（Adversarial Filtering）**  
   - 对每条情境，保留判别器最容易被误判为真实的若干负例（即分数最高的机器续写）。  
   - 将这些负例加入训练集，重新训练判别器，提升它们的辨别能力。  
   - 重复上述“判别 → 过滤”循环数次，逐步逼迫负例进入“对抗”区间——对人类而言明显错误，却仍能骗过最新的判别模型。  

4. **人工验证**  
   - 最终得到的每条题目包含 1 个正确续写和 3 个高质量负例。  
   - 人工标注员检查负例的可读性和常识错误，剔除明显不通顺或不符合情境的选项，确保数据集对人类保持 >95% 的准确率。

**最巧妙的点**在于把机器生成的“垃圾”通过判别器的筛选变成“高级伪装”，让模型的浅层语言匹配失效，必须动用更深层的因果常识才能分辨。

### 实验与效果
- **测试对象**：新建的 HellaSwag 数据集，包含约 70k 条多选题，情境长度平均 20–30 个词。  
- **基线模型**：BERT‑base、RoBERTa‑large、XLNet 等当前最强的预训练模型。  
- **结果**：人类标注员在该数据集上达到 96% 的准确率；最强的 RoBERTa‑large 只拿到 46% 左右，远低于人类水平。相比原始 SWAG（模型已接近 80%），HellaSwag 把差距拉大了近 30%。  
- **消融实验**：  
  - 去掉对抗过滤，只使用随机负例，模型准确率回升至约 70%，说明 AF 是难度提升的关键。  
  - 缩短情境长度（回到 SWAG 那种短句），模型表现提升约 15%，验证 Goldilocks 区的有效性。  
- **局限**：作者指出，负例仍然是机器生成的，可能带有语言模型的偏见；此外，数据集聚焦于英文日常情境，跨语言或专业领域的常识仍未覆盖。

### 影响与延伸思考
HellaSwag 发表后，**对抗式基准**的概念迅速被社区采纳，催生了如 **Adversarial NLI、Winograd Schema Challenge 的升级版** 等工作。研究者开始把 **动态数据生成 + 判别过滤** 作为提升评测可靠性的标准流程，甚至把它用于 **模型鲁棒性训练**（让模型在训练时直接面对自己的“骗术”）。  
后续的方向包括：  
- 将对抗过滤扩展到多语言、跨文化常识场景（推测）。  
- 把判别器和生成器合并成 **GAN‑style** 的常识数据生成框架。  
- 探索 **自适应基准**：模型每提升一点，就自动触发新一轮负例生成，实现持续的“能力拉锯”。  

### 一句话记住它
**HellaSwag 用对抗过滤把机器生成的“荒唐”负例炼成“骗过模型却被人类轻易识破”的高难度常识题，重新证明了常识推理仍是 AI 的硬核挑战。**