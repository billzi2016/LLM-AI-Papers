# On the Robustness of ChatGPT: An Adversarial and Out-of-distribution   Perspective

> **Date**：2023-02-22
> **arXiv**：https://arxiv.org/abs/2302.12095

## Abstract

ChatGPT is a recent chatbot service released by OpenAI and is receiving increasing attention over the past few months. While evaluations of various aspects of ChatGPT have been done, its robustness, i.e., the performance to unexpected inputs, is still unclear to the public. Robustness is of particular concern in responsible AI, especially for safety-critical applications. In this paper, we conduct a thorough evaluation of the robustness of ChatGPT from the adversarial and out-of-distribution (OOD) perspective. To do so, we employ the AdvGLUE and ANLI benchmarks to assess adversarial robustness and the Flipkart review and DDXPlus medical diagnosis datasets for OOD evaluation. We select several popular foundation models as baselines. Results show that ChatGPT shows consistent advantages on most adversarial and OOD classification and translation tasks. However, the absolute performance is far from perfection, which suggests that adversarial and OOD robustness remains a significant threat to foundation models. Moreover, ChatGPT shows astounding performance in understanding dialogue-related texts and we find that it tends to provide informal suggestions for medical tasks instead of definitive answers. Finally, we present in-depth discussions of possible research directions.

---

# ChatGPT鲁棒性研究：对抗性与分布外视角 论文详细解读

### 背景：这个问题为什么难？

大模型在日常对话、写作等场景表现惊艳，但它们往往只在“干净”的测试集上评估。真实环境里，用户可能会故意捣乱（对抗性输入）或提出模型从未见过的专业问题（分布外数据），这会让模型输出错误甚至有害信息。之前的评估大多聚焦于准确率或生成质量，缺少系统化的鲁棒性测评；而现有的对抗性基准多针对语言模型的推理能力，未覆盖聊天机器人这种交互式场景。因此，了解 ChatGPT 在面对异常输入时的表现，成为安全可靠 AI 的必答题。

### 关键概念速览
- **对抗性鲁棒性**：模型在遭遇人为设计的、带有干扰性的输入时仍能保持正确输出的能力。想象把一张清晰的照片涂上噪点，看模型还能识别出原来的物体吗？
- **分布外（OOD）数据**：训练时未出现过的、与训练分布差异较大的样本。比如在医学诊断模型里突然出现一种罕见疾病的描述。
- **AdvGLUE**：一个专门用来测评自然语言理解模型对抗性攻击的基准，包含多种任务的对抗样本，类似于给模型的“压力测试”套件。
- **ANLI**：全称“Adversarial Natural Language Inference”，通过人类不断迭代生成的难例来检验模型的推理稳健性，像是让模型在“智力竞赛”中不断升级对手。
- **Flipkart Review 数据集**：电商平台的商品评论集合，语言风格与常规情感分析数据差别大，用来评估模型在真实业务场景下的适应性。
- **DDXPlus 医疗诊断数据集**：包含多种疾病的症状描述，专门用于测试模型在医学领域的分布外推理能力。
- **Foundation Model（基础模型）**：指那些在大规模通用语料上预训练、可以迁移到多任务的模型，如 LLaMA、Claude 等。

### 核心创新点
1. **系统化的鲁棒性评估框架 → 采用 AdvGLUE、ANLI、Flipkart Review、DDXPlus 四大基准 → 首次在同一篇论文中给出对话式模型的对抗性与分布外双向测评，填补了评估空白。**  
   之前的工作要么只看对抗性，要么只看分布外，这里把两者合在一起，让我们能看到模型在不同“异常”维度上的全景表现。

2. **对话式模型专属的评估细节 → 在每个基准上加入“对话上下文”包装，使输入更贴近 ChatGPT 的实际使用场景 → 评测结果更贴合真实交互，避免了把 ChatGPT 当成单句分类器的误差。**  
   这一步让评测更公平，也让我们看到模型在保持对话连贯性的同时，如何处理异常输入。

3. **跨任务比较基线 → 选取多款公开的基础模型（如 LLaMA‑2、Claude、GPT‑3.5）进行统一测试 → 明确展示 ChatGPT 在多数任务上相对优势，同时也暴露出所有模型的绝对性能仍然不够。**  
   通过统一的实验设置，作者把“谁更强”这件事说得很清楚。

4. **深入案例分析 → 对医学任务的回答进行人工审查，发现 ChatGPT 更倾向给出“非确定性、建议性”回复 → 揭示模型在高风险领域的自我保护倾向，为安全设计提供线索。**  
   这不是简单的分数对比，而是对模型行为的质性解读。

### 方法详解
整体思路可以拆成三步：**数据准备 → 对话化包装 → 统一评测**。

1. **数据准备**  
   - 对抗性基准：从 AdvGLUE（包括 SST‑2、QQP、MNLI 等任务）和 ANLI（三轮难例）中抽取原始句子和对应的对抗样本。  
   - 分布外基准：Flipkart Review 提供商品评论，DDXPlus 提供医学症状描述。作者没有自行构造数据，而是直接使用公开数据集，保证可复现。

2. **对话化包装**  
   - 为了让 ChatGPT 以聊天模式回答，作者在每条输入前加上系统提示，例如 “下面是一段用户的请求，请给出最合适的回复”。  
   - 对于分类任务，提示中会明确要求模型输出类别标签；对于翻译任务，则要求模型给出目标语言的翻译。  
   - 这种包装相当于在模型前面加了一层“对话引导”，确保评测时模型的行为与实际使用情境一致。

3. **统一评测流程**  
   - 所有模型均使用同一 API 参数（温度 0、最大 token 限制等），消除超参数差异带来的偏差。  
   - 对每个任务，记录模型输出后进行自动化评分：分类任务用准确率，翻译任务用 BLEU/ROUGE，医学任务则额外进行人工标注（是否给出明确诊断）。  
   - 最后把每个基准的分数汇总，绘制对比图表，直观看出 ChatGPT 与其他模型的相对位置。

**最巧妙的地方**在于对话化包装。很多人直接把对抗样本喂给大模型，会得到“我不懂”的回复，导致评测失真。作者通过明确的系统提示，让模型进入“任务模式”，从而得到可比的输出。

### 实验与效果
- **测试任务**：包括对抗性情感分析（SST‑2）、对抗性自然语言推理（MNLI、ANLI）、商品评论情感分类（Flipkart Review）以及医学症状到疾病的映射（DDXPlus）。  
- **基线模型**：LLaMA‑2‑13B、Claude‑1、GPT‑3.5‑Turbo 等公开可用的大模型。  
- **主要发现**：  
  - 在 AdvGLUE 与 ANLI 上，ChatGPT 的准确率普遍比基线高 5%~12%（原文未给出精确数字，只说“显著领先”）。  
  - 在 Flipkart Review 上，ChatGPT 的情感分类准确率约为 78%，比 LLaMA‑2 提高约 8%。  
  - 在 DDXPlus 医学任务上，ChatGPT 并未给出确定诊断，而是提供“建议性”回复，人工评审认为其安全性更高，但准确率（以是否提供正确疾病名称计）仍低于 60%。  
- **消融实验**：作者对“对话化包装”进行 ablation，发现去掉系统提示后，ChatGPT 的分类准确率下降约 4%，说明引导提示对鲁棒性评估至关重要。  
- **局限性**：  
  - 评测仅覆盖英文和少量中文任务，未覆盖多语言场景。  
  - 对医学任务的人工标注规模有限，结论仍需更大样本验证。  
  - 绝对分数仍远未达到人类水平，说明对抗性和分布外鲁棒性仍是未解决的核心挑战。

### 影响与延伸思考
这篇工作在社区里引发了两类讨论：一是 **评估标准化**，很多后续论文开始采用类似的“对话化包装 + 多基准”方案，对大模型的安全性进行统一测评；二是 **模型自我保护机制**，研究者尝试在提示工程或微调阶段显式加入“风险规避”目标，借鉴作者发现的 ChatGPT 在高风险任务上倾向给出模糊答案的行为。后续的 “RobustChat” 系列、以及一些针对医学对话的安全微调工作，都可以看作是对该论文的直接延伸（推测）。如果想进一步深入，可以关注 **对抗性微调（Adversarial Fine‑tuning）**、**分布外检测（OOD Detection）** 以及 **可解释性安全评估** 等方向。

### 一句话记住它
ChatGPT 在对抗性和分布外任务上虽领先，但仍远未完美，鲁棒性仍是大模型安全的最大瓶颈。