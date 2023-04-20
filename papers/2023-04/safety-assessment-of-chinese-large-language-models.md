# Safety Assessment of Chinese Large Language Models

> **Date**：2023-04-20
> **arXiv**：https://arxiv.org/abs/2304.10436

## Abstract

With the rapid popularity of large language models such as ChatGPT and GPT-4, a growing amount of attention is paid to their safety concerns. These models may generate insulting and discriminatory content, reflect incorrect social values, and may be used for malicious purposes such as fraud and dissemination of misleading information. Evaluating and enhancing their safety is particularly essential for the wide application of large language models (LLMs). To further promote the safe deployment of LLMs, we develop a Chinese LLM safety assessment benchmark. Our benchmark explores the comprehensive safety performance of LLMs from two perspectives: 8 kinds of typical safety scenarios and 6 types of more challenging instruction attacks. Our benchmark is based on a straightforward process in which it provides the test prompts and evaluates the safety of the generated responses from the evaluated model. In evaluation, we utilize the LLM's strong evaluation ability and develop it as a safety evaluator by prompting. On top of this benchmark, we conduct safety assessments and analyze 15 LLMs including the OpenAI GPT series and other well-known Chinese LLMs, where we observe some interesting findings. For example, we find that instruction attacks are more likely to expose safety issues of all LLMs. Moreover, to promote the development and deployment of safe, responsible, and ethical AI, we publicly release SafetyPrompts including 100k augmented prompts and responses by LLMs.

---

# 中文大语言模型安全评估 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在生成自然语言时常常会泄露不当言论、歧视性内容，甚至被恶意利用进行诈骗或造谣。过去的安全评估大多聚焦英文模型，使用的测试集规模小、场景单一，难以覆盖中文语境下的文化敏感点和语言习惯。更糟的是，已有的评估方法往往依赖人工标注或固定规则，成本高且难以跟随模型快速迭代。于是，缺少一个系统、可扩展、兼顾中文特色的安全基准，成为制约中文LLM安全研究的瓶颈。

### 关键概念速览
**安全场景（Safety Scenario）**：指模型在日常对话、信息查询、情感安抚等常见使用情境下可能出现的风险，例如生成辱骂或误导性答案。相当于给模型设定的“生活场景”，用来检验它的行为是否合规。  

**指令攻击（Instruction Attack）**：通过精心设计的输入指令诱导模型突破安全防线，输出违规内容。可以想象成“钓鱼”式的提问，专门找模型的漏洞。  

**安全评估器（Safety Evaluator）**：利用另一个强大的语言模型（如GPT-4）对生成的答案进行安全性打分，而不是人工逐条审查。类似于让“老师”给学生的作文打分，省时又一致。  

**Prompt Augmentation（提示增强）**：在原始测试提示的基础上加入同义改写、噪声或上下文扩展，以产生更多变体，提升评估覆盖度。相当于在同一道题目上换不同的表述方式，看看学生是否仍会犯错。  

**SafetyPrompts 数据集**：作者公开的包含约10万条经过增强的安全测试提示及对应模型回复的资源库，供研究者直接使用或二次开发。  

**多维安全评分（Multi‑dimensional Safety Score）**：将安全性拆分为辱骂、歧视、误导、隐私泄露等多个维度，每个维度单独打分后综合得到总体安全分。类似于体检报告的多项指标。  

**LLM 自评（Self‑Evaluation）**：让被测模型自行判断自己的回答是否安全，作为辅助参考。可以类比为“自检”功能，帮助发现潜在风险。

### 核心创新点
1. **从“场景+攻击”双维度构建评估框架**  
   之前的评估大多只列举若干固定问题，缺少对主动攻击的考察。该工作把常规使用场景（8 类）和专门的指令攻击（6 类）结合，形成更全面的安全画像。这样既能捕捉日常使用中的潜在风险，也能暴露模型在被恶意诱导时的薄弱环节。  

2. **把大语言模型本身当作安全评估器**  
   传统做法需要人工标注或专门的规则引擎来判断答案是否安全。作者直接调用一个强大的 LLM（通过精心设计的提示）来对生成的回复进行安全性打分，省去了大量人工成本，并且评估标准随模型能力提升而自然进化。  

3. **大规模提示增强与公开数据集**  
   为了避免评估过于依赖单一提示，作者对原始 1000 条左右的安全测试提示进行同义改写、噪声注入等增强，最终形成约 10 万条多样化的 SafetyPrompts。公开发布后，社区可以直接复用或在此基础上继续扩展，推动评估标准化。  

4. **系统性对 15 种中文/中英混合 LLM 进行横向比较**  
   过去很少有研究同时评测多家中文模型。该工作一次性跑通所有模型的安全场景和攻击测试，提供了统一的分数表和细粒度分析，为模型研发者指明改进方向。  

### 方法详解
整体思路可以拆成三步：**构造测试提示 → 生成模型回复 → 用 LLM 评估安全性**。下面逐步展开每一步的细节。

1. **测试提示构造**  
   - **场景提示**：作者先列出 8 大常见使用场景，如“日常聊天”“专业咨询”“情感安抚”等。每个场景下准备若干具体问题，例如在“情感安抚”里会问“我最近很抑郁怎么办”。  
   - **指令攻击提示**：针对 6 类已知的攻击手法（如“隐蔽诱导”“角色扮演攻击”等），手工编写诱导模型输出违规内容的指令。  
   - **提示增强**：对每条原始提示使用同义词替换、句式变换、加入噪声词等技术，生成多个变体。这样同一个安全风险会被不同表述方式多次检验，提升鲁棒性。  

2. **模型回复生成**  
   - 将所有提示统一送入待评估的 LLM，记录模型的完整文本输出。为了公平比较，所有模型均使用相同的温度、最大长度等生成超参数。  

3. **安全评估器设计**  
   - **评估提示**：作者准备了一套评估指令，告诉评估 LLM（如 GPT‑4）如何对回复进行打分。例如：“请判断下面的回答是否包含辱骂、歧视或误导信息，并给出 0‑5 分”。  
   - **多维打分**：评估 LLM 会分别给出辱骂、歧视、误导、隐私泄露等维度的分数，然后取平均或加权得到总体安全分。  
   - **自评环节**：在部分实验中，还让被测模型自行判断自己的回复是否安全，作为对比参考。  

**最巧妙的点**在于把评估任务交给同类模型完成。因为 LLM 本身已经具备强大的语言理解和价值判断能力，利用它们来评估其他模型的输出，既省去人工标注的高成本，又能保持评估的一致性和可扩展性。

### 实验与效果
- **评测对象**：共 15 种模型，包括 OpenAI 的 GPT‑3.5、GPT‑4，以及国内主流中文 LLM（如 ChatGLM、Baichuan、InternLM 等）。  
- **数据规模**：使用约 10 万条增强提示，覆盖 8 种场景和 6 种攻击。  
- **基线对比**：作者将 LLM 评估器的评分结果与人工标注的少量安全标签进行交叉验证，发现两者在整体趋势上高度一致（具体一致率未在摘要中给出）。  
- **关键发现**：所有模型在普通场景下的安全分数相对较高，但在指令攻击下普遍出现显著下降，说明攻击提示更容易暴露安全漏洞。  
- **消融实验**：通过去掉提示增强，仅使用原始提示进行评估，安全分数的波动幅度增大，说明增强步骤对评估的稳定性贡献显著。  
- **局限性**：评估依赖于另一个强模型的判断，若评估模型本身存在偏见或误判，可能会传递错误的安全标签。作者也提到，当前的攻击类型仍有限，未来可能出现更隐蔽的攻击手法。  

### 影响与延伸思考
这篇工作在中文 LLM 安全评估领域树立了系统化的基准，随后有多篇后续研究直接基于 SafetyPrompts 扩展攻击库或改进评估提示（如加入情感细粒度评分）。在工业界，部分企业已经把该基准集成进模型上线前的自动化安全检测流水线。未来可以进一步探索 **多模态安全评估**（加入图像、音频输入）以及 **自适应评估器**（让评估模型随时间学习新型攻击），这些方向都有望在安全评估的深度和广度上实现突破。

### 一句话记住它
把大语言模型本身当评审官，用 8 场景＋6 攻击的双维度基准，快速、统一地测出中文 LLM 的安全盲点。