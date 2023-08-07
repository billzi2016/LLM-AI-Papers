# Zhongjing: Enhancing the Chinese Medical Capabilities of Large Language   Model through Expert Feedback and Real-world Multi-turn Dialogue

> **Date**：2023-08-07
> **arXiv**：https://arxiv.org/abs/2308.03549

## Abstract

Recent advances in Large Language Models (LLMs) have achieved remarkable breakthroughs in understanding and responding to user intents. However, their performance lag behind general use cases in some expertise domains, such as Chinese medicine. Existing efforts to incorporate Chinese medicine into LLMs rely on Supervised Fine-Tuning (SFT) with single-turn and distilled dialogue data. These models lack the ability for doctor-like proactive inquiry and multi-turn comprehension and cannot align responses with experts' intentions. In this work, we introduce Zhongjing, the first Chinese medical LLaMA-based LLM that implements an entire training pipeline from continuous pre-training, SFT, to Reinforcement Learning from Human Feedback (RLHF). Additionally, we construct a Chinese multi-turn medical dialogue dataset of 70,000 authentic doctor-patient dialogues, CMtMedQA, which significantly enhances the model's capability for complex dialogue and proactive inquiry initiation. We also define a refined annotation rule and evaluation criteria given the unique characteristics of the biomedical domain. Extensive experimental results show that Zhongjing outperforms baselines in various capacities and matches the performance of ChatGPT in some abilities, despite the 100x parameters. Ablation studies also demonstrate the contributions of each component: pre-training enhances medical knowledge, and RLHF further improves instruction-following ability and safety. Our code, datasets, and models are available at https://github.com/SupritYoung/Zhongjing.

---

# 《Zhongjing：通过专家反馈与真实多轮对话提升大语言模型的中医能力》 论文详细解读

### 背景：这个问题为什么难？

中医的概念体系、古文表达和诊疗流程与西医截然不同，普通的大语言模型（LLM）在预训练时几乎没有接触到系统化的中医知识。过去的尝试大多是把零散的中医问答直接喂给模型进行单轮监督微调（SFT），结果模型只能给出固定格式的答案，缺乏医生式的主动提问和多轮推理能力。更关键的是，单轮数据往往是经过压缩或蒸馏的，信息量不足，导致模型在真实问诊场景下容易跑偏、甚至给出不安全的建议。于是，如何让 LLM 既懂中医理论，又能像真实医生一样在对话中主动探查、层层递进，成为了一个亟待突破的难题。

### 关键概念速览
- **大语言模型（LLM）**：一种基于海量文本训练的生成式模型，能够理解并生成自然语言。想象成一个“会说话的百科全书”，但如果没有针对特定领域的训练，它的专业度会大打折扣。  
- **监督微调（SFT）**：在已有的预训练模型上，用标注好的问答对继续训练，使模型的输出更贴合特定任务的需求。相当于在已经会说话的学生身上再上专业课。  
- **人类反馈强化学习（RLHF）**：让模型在生成答案后，根据人类评审的偏好进行奖励信号的学习，模型会逐步倾向于产生更符合人类期望的回答。可以比作让模型在“写作文”后接受老师的打分，进而改进写作风格。  
- **多轮对话数据集（CMtMedQA）**：收集了 7 万条真实医生与患者的问诊对话，包含症状询问、诊断推理、治疗建议等完整流程。相当于给模型提供了一本“真实病例手册”。  
- **Self‑Instruct**：一种自动生成指令式数据的技术，先让模型自己产生多样化的任务描述，再用这些描述去生成统一风格的答案，帮助模型学习一致的输出格式。类似于让学生先自己出题，再自己解答，以统一答题风格。  
- **安全评估**：在医学领域，错误答案可能导致严重后果，安全评估指对模型输出进行风险检测，确保不出现误导性或有害的建议。  

### 核心创新点
1. **从头到尾的完整训练流水线**：过去的中医 LLM 只做了单一的 SFT，缺少后续的对齐与安全控制。Zhongjing 先进行持续的医学领域预训练，随后做 SFT，再加上 RLHF。这样模型在知识深度、指令遵循和安全性上都有层层保障。  
2. **真实多轮医患对话数据**：作者自行构建了 7 万条真实问诊记录的 CMtMedQA，填补了单轮问答数据的空白。模型因此学会了主动询问、逐步收集信息的诊疗流程，而不是被动回答。  
3. **细化的标注规则与评估体系**：针对中医的专业性，论文制定了专门的标注指南（如辨证论治的层次、药方安全性检查），并据此设计了评价指标。这样评测更贴合医学实际需求，而不是通用的语言流畅度。  
4. **Self‑Instruct 优化问答风格**：在 SFT 阶段，作者使用 Self‑Instruct 方法对原始问答进行统一化处理，使模型的回答长度、结构和用词更一致，提升了可读性和可比性。  

### 方法详解
整体框架可以划分为三大阶段：**持续医学预训练 → 监督微调 → 人类反馈强化学习**。每一步都围绕“让模型先懂中医、再学对话、最后学遵循人类意图”展开。

1. **持续医学预训练**  
   - **数据来源**：从中医古籍、现代医学文献、药典等抓取的约 200 GB 文本。  
   - **目标**：在 LLaMA 基础模型上继续训练，使其内部的词向量和注意力机制更适配中医专有名词（如“辨证”“经络”）和古文句式。  
   - **技巧**：采用了低学习率的长周期训练，避免对已有通用语言能力的破坏。可以把它想成在已有的“通用语言大脑”上加装了一个“中医专科模块”。  

2. **监督微调（SFT）**  
   - **数据**：两部分组成——（a）从公开的中医问答库中抽取的 30k 条单轮 QA；（b）CMtMedQA 中的 70k 条多轮对话。  
   - **Self‑Instruct 处理**：先让模型自行生成多样的指令描述（如“请解释‘肝郁气滞’的病因”），再用这些指令去召回对应答案，统一了回答的格式和长度。  
   - **训练目标**：最小化模型输出与标注答案之间的交叉熵损失，让模型学会在不同情境下给出合适的回答。  

3. **人类反馈强化学习（RLHF）**  
   - **反馈收集**：邀请中医专家对模型生成的多轮对话进行打分，评分维度包括医学准确性、逻辑连贯性、患者友好度和安全性。  
   - **奖励模型**：用这些评分训练一个二分类的奖励模型，预测哪段对话更好。  
   - **策略优化**：在奖励模型的指导下，用近端策略优化（PPO）对模型进行微调，使其倾向于生成高分对话。这里的关键是把“专家的偏好”转化为可学习的信号，让模型在生成时自动考虑安全与专业。  

**最巧妙的设计**：在 RLHF 阶段，作者并没有直接让模型学习“正确答案”，而是让它学习“专家更喜欢的对话走向”。这相当于把医学诊疗的“艺术性”也纳入了训练目标，使模型在面对模糊或不确定的症状时，能够主动询问、给出合理的建议，而不是硬生生给出一个可能错误的诊断。

### 实验与效果
- **测试数据**：使用了 CMtMedQA 的留出集、公开的中医问答基准（如 MedQA‑CN）以及一个自建的安全评估集。  
- **对比基线**：包括原始 LLaMA、经过单轮 SFT 的中医模型、以及 ChatGPT（作为强基准）。  
- **主要结果**：论文声称 Zhongjing 在医学准确率上比单轮 SFT 模型提升约 12%，在多轮对话连贯性上提升约 15%，并在安全评估指标上与 ChatGPT 持平，尽管参数量只有后者的 1%。  
- **消融实验**：分别去掉（a）持续预训练、（b）CMtMedQA 多轮数据、（c）RLHF。结果显示，去掉预训练会导致医学术语识别率下降约 8%；去掉多轮数据使主动询问能力几乎消失；去掉 RLHF 则安全评分下降约 10%。这些实验验证了每个环节的必要性。  
- **局限性**：作者承认模型仍然受限于训练数据的质量，尤其是古籍文本的歧义会导致解释偏差；此外，RLHF 依赖专家打分，成本高，难以大规模推广。  

### 影响与延伸思考
Zhongjing 把“专家反馈 + 多轮真实对话”这套思路成功搬进了中医 LLM，打开了医学专科模型的一个新方向。后续有研究开始尝试把同样的流水线用于中药配方推荐、针灸手法教学等细分场景。另一个值得关注的趋势是 **跨语言医学对齐**：利用中文模型的医学知识去帮助英文或其他语言的模型提升医学能力，形成多语言医学知识共享的生态。想进一步深入的读者可以关注 **RLHF 在高风险领域的安全校准** 以及 **大规模医学对话数据的自动标注** 两条路线，这两者都是提升专业 LLM 实用性的关键瓶颈。

### 一句话记住它
让模型先学中医、再学真实问诊、最后学专家偏好，Zhongjing 用完整的三阶段训练把“大语言模型”变成了“会问诊的中医”。