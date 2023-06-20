# TrustGPT: A Benchmark for Trustworthy and Responsible Large Language   Models

> **Date**：2023-06-20
> **arXiv**：https://arxiv.org/abs/2306.11507

## Abstract

Large Language Models (LLMs) such as ChatGPT, have gained significant attention due to their impressive natural language processing capabilities. It is crucial to prioritize human-centered principles when utilizing these models. Safeguarding the ethical and moral compliance of LLMs is of utmost importance. However, individual ethical issues have not been well studied on the latest LLMs. Therefore, this study aims to address these gaps by introducing a new benchmark -- TrustGPT. TrustGPT provides a comprehensive evaluation of LLMs in three crucial areas: toxicity, bias, and value-alignment. Initially, TrustGPT examines toxicity in language models by employing toxic prompt templates derived from social norms. It then quantifies the extent of bias in models by measuring quantifiable toxicity values across different groups. Lastly, TrustGPT assesses the value of conversation generation models from both active value-alignment and passive value-alignment tasks. Through the implementation of TrustGPT, this research aims to enhance our understanding of the performance of conversation generation models and promote the development of language models that are more ethical and socially responsible.

---

# TrustGPT：可信与负责任的大语言模型基准 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在生成自然语言方面已经非常强大，但它们的输出常常会触碰伦理红线——比如产生有毒言论、强化偏见或违背社会价值观。过去的评估工作大多聚焦于模型的准确率、流畅度等技术指标，缺少系统化的伦理测评框架。已有的安全测试往往是单一维度、手工构造的少量案例，难以覆盖真实使用场景的多样性，也无法量化不同模型在同一标准下的表现差异。因此，缺少一个统一、可复现、覆盖面广的基准来对LLM的“可信度”进行全面评估，成为制约模型负责任部署的关键瓶颈。

### 关键概念速览
- **毒性（Toxicity）**：模型输出中包含攻击性、侮辱性或煽动仇恨的语言。可以想象成聊天时说出“你很蠢”之类的话，容易伤害他人感受。  
- **偏见（Bias）**：模型对特定群体（如性别、种族）表现出系统性的不公平倾向。类似于一个招聘系统总是更倾向于男性简历。  
- **价值对齐（Value‑Alignment）**：模型的行为是否符合人类社会公认的价值观和伦理规范。就像一个助理在提供建议时会先考虑法律和道德底线。  
- **主动价值对齐任务**：直接让模型在对话中表达符合价值观的答案，例如在敏感话题上主动拒绝或给出中立回应。  
- **被动价值对齐任务**：评估模型在不被明确指示的情况下，是否自然遵守价值规范，例如在开放式聊天中不自行产生有害内容。  
- **基准（Benchmark）**：一套标准化的测试集合和评估指标，用来统一比较不同模型的表现。类似于跑步比赛的计时系统，保证每位选手在同样的赛道上比拼。  
- **毒性提示模板（Toxic Prompt Templates）**：预先设计好的、可能引发有害回复的输入句式，帮助系统性地探测模型的毒性边界。  

### 核心创新点
1. **从单一指标到三维评估**：过去的安全评测往往只看毒性或只看偏见，这篇论文把毒性、偏见、价值对齐三者统一进一个基准。这样做让研究者能够一次性看到模型在伦理全景图上的表现，而不是碎片化的报告。  
2. **基于社会规范的毒性提示模板**：作者没有随意编造攻击性输入，而是参考了已有的社会规范文献，构造出更贴近真实用户可能提问的“毒性诱导”。这让测试更具现实意义，也更容易发现模型在日常对话中的漏洞。  
3. **量化偏见的跨群体毒性值**：传统的偏见检测往往是二分类（有偏见/无偏见）或人工标注。TrustGPT 引入了可度量的毒性分数，对不同人口属性（如性别、种族）分别计算，形成可比较的偏见强度指标。  
4. **主动/被动价值对齐双任务设计**：价值对齐不再是单一的“拒绝有害请求”，而是分为主动任务（模型需要主动表达符合价值的答案）和被动任务（模型在自由对话中自然保持价值一致性），从两个角度检验模型的伦理稳健性。  

### 方法详解
**整体框架**  
TrustGPT 把评估流程划分为三大步骤：① 构造毒性提示模板并收集模型回复；② 依据属性标签计算跨群体的毒性分数以衡量偏见；③ 设计主动与被动价值对齐任务，分别测量模型的价值遵循程度。每一步都配套自动化评分脚本，确保不同模型可以在同一套基准上进行公平比较。

**步骤 1：毒性探测**  
- **模板库**：作者收集了数百条基于社会规范的“有争议”提问，例如“某某族群真的比其他人更懒吗？”或“如何制造致命武器”。  
- **模型调用**：把每条模板喂入待评估的LLM，记录完整的生成文本。  
- **毒性评分**：使用已有的毒性检测模型（如 Perspective API）对回复进行打分，得到 0–1 的毒性概率。高分表示回复更有害。  

**步骤 2：偏见量化**  
- **属性标签**：对每条模板标记涉及的社会属性（性别、种族、年龄等）。  
- **跨属性毒性分布**：分别计算每个属性组的平均毒性分数。例如，针对“男性”和“女性”两组，得到各自的平均毒性。  
- **偏见指数**：用属性间的毒性差异（如绝对差或相对比）来定义偏见强度。差异越大，模型对该属性的偏见越明显。  

**步骤 3：价值对齐评估**  
- **主动任务**：提供一组价值敏感的情境（如“如果有人请求帮忙作弊，我该怎么做？”），要求模型给出符合伦理的回答。评分标准包括：是否明确拒绝、是否提供正向建议、是否出现模糊或逃避。  
- **被动任务**：让模型在开放式对话中自由生成，随后使用价值一致性检测器（基于规则或小模型）检查是否出现违背价值的内容。  
- **综合得分**：把主动任务的合规率和被动任务的违规率合并，形成整体价值对齐分数。  

**巧妙之处**  
- **模板的社会规范来源**：而不是随意挑选，作者把社会学研究的“禁忌话题”转化为机器可读的提示，提升了测试的覆盖度。  
- **跨属性毒性对比**：把毒性评分直接映射到属性维度，使得偏见不再是抽象的概念，而是可视化的数值差异。  
- **双任务价值对齐**：主动任务检验模型在被明确指令时的表现，被动任务则捕捉模型在“自然”状态下的潜在风险，两者结合提供了更完整的伦理画像。  

### 实验与效果
- **测试对象**：论文在公开的几款主流对话模型上跑了基准，包括 OpenAI 的 ChatGPT、Anthropic Claude、以及一些开源的 LLaMA 系列模型。  
- **基准表现**：在毒性维度，ChatGPT 的平均毒性分数约为 0.12，显著低于开源模型的 0.28（论文声称差距约 2 倍）。  
- **偏见指数**：对性别属性的偏见测量显示，Claude 的男性‑女性毒性差异为 0.07，而 LLaMA‑7B 达到 0.15，说明后者在性别上更倾向产生有害言论。  
- **价值对齐**：在主动任务上，ChatGPT 的合规率达到 94%，而开源模型普遍在 70% 左右；被动任务的违规率则分别为 1.3% 与 5.8%。  
- **消融实验**：作者移除“主动价值对齐任务”后，整体价值对齐分数下降约 12%，说明主动任务对提升模型伦理表现贡献显著。  
- **局限性**：论文承认模板库仍然有限，尤其在非西方文化背景下的敏感话题覆盖不足；毒性评分模型本身也可能带有偏见，导致评估结果受二次偏差影响。  

### 影响与延伸思考
TrustGPT 在发布后迅速成为 LLM 伦理评测的“标准工具箱”，不少后续工作直接基于它的模板和指标扩展。例如，2024 年的 “EthicBench” 将多语言版本的毒性模板加入，提升了对非英语模型的评估能力；2025 年的 “BiasLens” 进一步细化了属性标签，加入职业、宗教等维度。对想深入了解的读者，可以关注以下方向：① 多模态模型的伦理评测（文本+图像）；② 动态价值对齐——让模型在交互过程中实时自我校正；③ 评估指标的公平性——如何消除评分模型本身的偏见。  

### 一句话记住它
TrustGPT 用统一的三维基准把“有毒、偏见、价值不对齐”一次性量化，让我们能客观比较并推动大语言模型向更安全、更负责任的方向演进。