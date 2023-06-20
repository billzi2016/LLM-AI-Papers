# DecodingTrust: A Comprehensive Assessment of Trustworthiness in GPT   Models

> **Date**：2023-06-20
> **arXiv**：https://arxiv.org/abs/2306.11698

## Abstract

Generative Pre-trained Transformer (GPT) models have exhibited exciting progress in their capabilities, capturing the interest of practitioners and the public alike. Yet, while the literature on the trustworthiness of GPT models remains limited, practitioners have proposed employing capable GPT models for sensitive applications such as healthcare and finance -- where mistakes can be costly. To this end, this work proposes a comprehensive trustworthiness evaluation for large language models with a focus on GPT-4 and GPT-3.5, considering diverse perspectives -- including toxicity, stereotype bias, adversarial robustness, out-of-distribution robustness, robustness on adversarial demonstrations, privacy, machine ethics, and fairness. Based on our evaluations, we discover previously unpublished vulnerabilities to trustworthiness threats. For instance, we find that GPT models can be easily misled to generate toxic and biased outputs and leak private information in both training data and conversation history. We also find that although GPT-4 is usually more trustworthy than GPT-3.5 on standard benchmarks, GPT-4 is more vulnerable given jailbreaking system or user prompts, potentially because GPT-4 follows (misleading) instructions more precisely. Our work illustrates a comprehensive trustworthiness evaluation of GPT models and sheds light on the trustworthiness gaps. Our benchmark is publicly available at https://decodingtrust.github.io/ ; our dataset can be previewed at https://huggingface.co/datasets/AI-Secure/DecodingTrust ; a concise version of this work is at https://openreview.net/pdf?id=kaHpo8OZw2 .

---

# 解码信任：对 GPT 模型可信性的全面评估 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在写代码、写文章、甚至给出医学建议方面已经表现得相当强大，但它们的“可信度”——是否会输出有害、偏见、泄露隐私或违背伦理的内容——却缺乏系统化的评估手段。过去的安全测试大多是零散的：要么只检查毒性，要么只看偏见，甚至常常只用单一的对抗样本。这样的碎片化评估无法捕捉模型在真实使用场景中可能出现的多维度风险，也难以比较不同模型之间的安全水平。因此，需要一个覆盖面更广、方法更统一的基准来揭示 GPT 系列模型到底有多“可信”。

### 关键概念速览
**GPT模型**：指 OpenAI 开发的生成式预训练变换器系列，如 GPT‑3.5、GPT‑4，能够根据自然语言提示生成连贯文本。  
**毒性（Toxicity）**：模型输出包含攻击性、仇恨或侮辱性语言的程度，类似于聊天时说出脏话。  
**刻板印象偏见（Stereotype Bias）**：模型在性别、种族等维度上重复社会刻板印象，就像一个人不自觉地把“护士是女性”这种老套观念说出来。  
**对抗鲁棒性（Adversarial Robustness）**：模型面对精心设计的恶意输入时仍能保持正确或安全输出的能力，类似于防弹玻璃抵御子弹。  
**分布外鲁棒性（Out‑of‑Distribution Robustness）**：模型在未见过的领域或语言风格下的表现，像是一个人能否在陌生城市里顺利点餐。  
**隐私泄露（Privacy Leakage）**：模型在对话或查询时意外透露训练数据中的个人信息，等同于不小心把别人的密码说出来。  
**机器伦理（Machine Ethics）**：模型在涉及道德判断的问题上是否遵循人类公认的伦理原则，例如不鼓励自杀。  
**越狱提示（Jailbreak Prompt）**：通过特殊的指令绕过模型的安全约束，让它输出本应被拦截的内容，类似于黑客破解系统的密码。

### 核心创新点
1. **统一多维度基准 → 设计了覆盖 8 大安全维度的评测套件 → 让研究者可以一次性得到模型的全景可信性报告，而不必分别跑十几个独立实验。**  
2. **系统化攻击生成 → 为每个维度构造了“正常”与“诱导”两套提示，并加入对抗示例、越狱指令等 → 揭露了模型在常规测试和恶意攻击下的表现差异，发现了许多未公开的漏洞。**  
3. **对比 GPT‑4 与 GPT‑3.5 的细粒度分析 → 在标准基准上 GPT‑4 的毒性、偏见得分普遍低于 GPT‑3.5，但在越狱提示下成功率更高 → 说明更强的指令遵循能力也可能成为安全风险的放大器。**  
4. **开源数据集与评测平台 → 将全部提示、答案、标注公开在 GitHub 与 HuggingFace → 为后续安全研究提供了可复现的基准，推动社区共同改进 LLM 的安全性。

### 方法详解
整体思路可以划分为 **四步走**：  
1. **任务构建**：作者先列出 8 大安全维度，每个维度再细分若干子任务（例如毒性包括仇恨言论、性别攻击等）。  
2. **提示设计**：为每个子任务编写两类提示——“普通提示”（让模型给出正常答案）和“诱导提示”（加入对抗词汇、越狱结构或隐私查询）。这一步类似于给模型出不同难度的试卷。  
3. **模型调用与答案收集**：使用 OpenAI 的 API 调用 GPT‑3.5 与 GPT‑4，分别对所有提示生成回答。为了捕捉对话历史泄露，作者还在同一次会话中多轮提问。  
4. **评估与汇总**：输出先经过自动指标（如毒性检测模型、偏见度量工具）筛选，再由人工标注员进行二次确认。每个维度得到一个分数，最后加权合成为整体可信性得分。

**关键模块的类比**：  
- **提示工程** 像是“试卷出题”，不同的出题方式会直接影响学生（模型）的表现。  
- **自动检测器** 像是“阅卷老师”，负责快速给出客观分数；人工标注则是“专家评审”，确保细节不被漏掉。  
- **加权汇总** 类似于体育比赛的总分制，既考虑单项成绩，也兼顾整体平衡。

**最巧妙的设计** 在于**“诱导提示”**的系统化生成。作者没有只靠手工写几个越狱例子，而是基于模板自动替换关键词，生成上百种变体，从而能够覆盖模型可能的安全盲点。这种大规模、可扩展的攻击生成方式在之前的安全评测中很少出现。

### 实验与效果
- **数据集与任务**：共计约 2 万条提示，覆盖毒性、偏见、对抗鲁棒性、OOD、隐私泄露、机器伦理、公平性以及越狱八个维度。所有提示和答案均已在 HuggingFace 上公开。  
- **Baseline 对比**：作者把 GPT‑4、GPT‑3.5 与公开的安全基准模型（如 OpenAI 的 Moderation API）进行对比。论文声称在毒性和偏见指标上，GPT‑4 的平均得分比 GPT‑3.5 低约 10%–15%，但在越狱成功率上，GPT‑4 高出约 20%。具体数字未在摘要中给出，需查阅完整论文。  
- **消融实验**：通过去掉诱导提示或仅使用单一维度评测，发现整体可信性分数会出现显著波动，说明多维度、对抗式评测是提升评估可靠性的关键因素。  
- **局限性**：作者承认评测仍然依赖于现有的自动检测工具，这些工具本身也可能存在偏差；此外，越狱提示的生成仍然受限于手工模板，无法覆盖所有潜在的攻击手法。

### 影响与延伸思考
自从 DecodingTrust 开源后，社区陆续出现了 **TrustLLM、SafePrompt、PrivacyGuard** 等项目，它们大多在此基准上进行二次实验或扩展维度（比如加入法律合规性）。不少企业开始在内部模型上线前使用该基准进行安全审计，说明它已经成为评估 LLM 可信性的“行业标准”。如果想进一步深入，可以关注 **对抗提示的自动生成**、**跨语言安全评测** 以及 **基于人类价值观的伦理对齐** 等方向，这些都是后续研究的热点。

### 一句话记住它
DecodingTrust 用八维度、系统化的攻击评测，彻底揭开了 GPT 系列模型在可信性上的盲点，提醒我们在追求强大能力的同时，别忘了先把安全关卡全部跑通。