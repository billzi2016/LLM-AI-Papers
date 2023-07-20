# CValues: Measuring the Values of Chinese Large Language Models from   Safety to Responsibility

> **Date**：2023-07-19
> **arXiv**：https://arxiv.org/abs/2307.09705

## Abstract

With the rapid evolution of large language models (LLMs), there is a growing concern that they may pose risks or have negative social impacts. Therefore, evaluation of human values alignment is becoming increasingly important. Previous work mainly focuses on assessing the performance of LLMs on certain knowledge and reasoning abilities, while neglecting the alignment to human values, especially in a Chinese context. In this paper, we present CValues, the first Chinese human values evaluation benchmark to measure the alignment ability of LLMs in terms of both safety and responsibility criteria. As a result, we have manually collected adversarial safety prompts across 10 scenarios and induced responsibility prompts from 8 domains by professional experts. To provide a comprehensive values evaluation of Chinese LLMs, we not only conduct human evaluation for reliable comparison, but also construct multi-choice prompts for automatic evaluation. Our findings suggest that while most Chinese LLMs perform well in terms of safety, there is considerable room for improvement in terms of responsibility. Moreover, both the automatic and human evaluation are important for assessing the human values alignment in different aspects. The benchmark and code is available on ModelScope and Github.

---

# CValues：从安全到责任衡量中文大语言模型价值 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在中文场景的快速迭代让人们担心它们可能产生有害言论或违背社会伦理。过去的评测大多聚焦在知识准确性、推理能力等“硬实力”，却很少检验模型是否遵循人类价值观，尤其是中文语境下的安全与责任。缺乏系统化、可复现的价值对齐基准，导致研发者只能靠零散的人工检查，难以发现模型在细分场景中的潜在风险。于是，迫切需要一个专门针对中文 LLM 的价值评估框架。

### 关键概念速览
- **人类价值对齐**：让模型的输出符合人类社会普遍接受的伦理、法律和责任标准，类似于把模型“教育”成一个懂规矩的助理。  
- **安全（Safety）**：模型避免生成毒性、歧视、暴力等有害内容的能力，像是给模型装上“安全阀”。  
- **责任（Responsibility）**：模型在提供信息、建议或决策支持时，能够体现可信度、可解释性和对后果的自觉，类似于专业人士的职业操守。  
- **对抗性安全提示（Adversarial Safety Prompts）**：故意设计的挑衅性问题，用来测试模型在极端情况下是否还能保持安全，类似于渗透测试。  
- **多选自动评测（Multi‑choice Automatic Evaluation）**：把开放式问题转化为带有若干选项的题目，机器可以直接对答案进行打分，类似于机器阅卷。  
- **人工评测（Human Evaluation）**：让真实的评审人员阅读模型输出并打分，确保评测结果贴近人类感受。  
- **价值基准（Values Benchmark）**：一套标准化的测试集合，用来统一衡量不同模型在价值维度上的表现。

### 核心创新点
1. **从安全到责任的全链路价值框架 → CValues 将评测范围从单一的安全检测扩展到责任维度**。过去的中文安全基准只关注毒性，CValues 通过加入“责任”场景（如信息可信度、隐私保护等），实现了价值评估的纵深覆盖。  
2. **专业专家构造的对抗性安全提示 → 手工收集 10 类安全场景的对抗性问题**。而不是依赖公开的通用数据，作者邀请领域专家针对中文语境设计了更具挑衅性的测试，使得模型在真实风险面前的表现更具说服力。  
3. **双轨评测机制 → 同时提供人工评测和多选自动评测**。自动评测解决了大规模对比的效率瓶颈，人工评测则弥补了机器难以捕捉的细微价值偏差，两者相辅相成。  
4. **责任提示的领域覆盖 → 从 8 大社会领域抽取责任场景**。相比仅用少数几类任务，CValues 把教育、医疗、金融、法律等关键领域都纳入评测，使得模型的责任感受检更为全面。

### 方法详解
整体思路可以拆成三步：**数据构建 → 评测设计 → 双模评估**。

1. **数据构建**  
   - **安全子集**：作者组织了多位中文语言与伦理专家，围绕网络暴力、政治敏感、色情、诈骗等 10 大高风险场景，手工编写了数百条对抗性提示。这些提示往往采用“如果…会怎样？”的诱导式问法，意在逼迫模型产生违规输出。  
   - **责任子集**：从教育、医疗、金融、法律、新闻、公共服务、环境、社交八个领域抽取典型任务，例如“请给出一份可信的疫苗接种指南”。每个任务配备了若干参考答案和错误答案，用来构造多选题。

2. **评测设计**  
   - **多选自动评测**：把每条开放式提示转化为四选一的题目，其中唯一正确选项是符合价值观的答案，其余三个是常见的违规或不负责任的回答。模型只需要输出选项编号，评测脚本即可自动计算准确率。  
   - **人工评测**：邀请独立评审员阅读模型的原始生成文本，依据安全性（是否出现毒性）和责任性（信息是否准确、是否提供必要警示）两维打分，采用 0–2 的离散评分。为降低评审偏差，采用双盲和交叉校验。

3. **双模评估流程**  
   - 首先跑自动多选评测，快速得到所有模型在安全与责任两大维度的基线分数。  
   - 对于自动评测表现相近的模型，再进行人工评测，以捕捉细粒度的价值偏差。  
   - 最终将两种评分进行加权融合，得到综合价值对齐得分。

**巧妙之处**：把开放式价值判断转化为多选题的做法，既保留了价值判断的复杂性，又让评测具备了可重复、可量化的特性；同时保留人工评审环节，防止机器“投机取巧”。这种“双轨”设计在中文 LLM 价值评估领域尚属首次。

### 实验与效果
- **测试对象**：作者选取了市面上主流的十余款中文 LLM，包括开源模型（如ChatGLM、BloomZ）和商业闭源模型（如百度文心、一言）。  
- **基准表现**：在安全子集上，大多数模型的自动准确率在 80% 以上，说明它们在基本毒性过滤上已经相对成熟。  
- **责任子集差距**：责任任务的准确率普遍低于安全任务，最高也仅在 55% 左右，显示出模型在提供可信信息、遵守行业规范方面仍有显著提升空间。  
- **人工评测对比**：人工评分与自动评分呈正相关，但在细节上出现分歧——比如模型生成的答案虽然不违规，却缺乏必要的风险提示，自动评测仍算作“正确”，而人工评审会扣分。  
- **消融实验**：作者分别去掉对抗性安全提示和责任多选题，发现去掉对抗性样本后安全准确率下降约 7%，说明对抗性样本对模型鲁棒性评估至关重要。  
- **局限性**：论文未给出跨语言迁移的实验，也没有对大规模实时评测的计算成本做深入分析；此外，人工评审的规模仍受限于资源，可能导致评测结果的统计显著性不足。

### 影响与延伸思考
CValues 为中文 LLM 的价值对齐提供了首个系统化基准，已经在 ModelScope、GitHub 上开源，吸引了不少社区贡献新的安全/责任场景。后续工作可能会在以下方向继续深化：  
- **跨模态价值评估**：把文本、图像、音频等多模态输出纳入同一价值框架。  
- **持续学习的价值校准**：利用 CValues 反馈在模型微调阶段动态纠偏。  
- **价值基准的国际化**：在保持中文语义特色的同时，引入多文化价值观，探索跨语言价值对齐的通用方法。  
想进一步了解，可关注近期在 ACL、EMNLP 上出现的“价值对齐”系列论文，以及开源社区对 CValues 的二次扩展。

### 一句话记住它
CValues 用专业对抗样本和多选自动评测把中文大模型的“安全”与“责任”统一量化，揭示了模型在价值对齐上仍有大量提升空间。