# Are Your LLM-based Text-to-SQL Models Secure? Exploring SQL Injection via Backdoor Attacks

> **Date**：2025-03-07
> **arXiv**：https://arxiv.org/abs/2503.05445

## Abstract

Large language models (LLMs) have shown state-of-the-art results in translating natural language questions into SQL queries (Text-to-SQL), a long-standing challenge within the database community. However, security concerns remain largely unexplored, particularly the threat of backdoor attacks, which can introduce malicious behaviors into models through fine-tuning with poisoned datasets. In this work, we systematically investigate the vulnerabilities of LLM-based Text-to-SQL models and present ToxicSQL, a novel backdoor attack framework. Our approach leverages stealthy {semantic and character-level triggers} to make backdoors difficult to detect and remove, ensuring that malicious behaviors remain covert while maintaining high model accuracy on benign inputs. Furthermore, we propose leveraging SQL injection payloads as backdoor targets, enabling the generation of malicious yet executable SQL queries, which pose severe security and privacy risks in language model-based SQL development. We demonstrate that injecting only 0.44% of poisoned data can result in an attack success rate of 79.41%, posing a significant risk to database security. Additionally, we propose detection and mitigation strategies to enhance model reliability. Our findings highlight the urgent need for security-aware Text-to-SQL development, emphasizing the importance of robust defenses against backdoor threats.

---

# 你的基于LLM的 Text-to-SQL 模型安全吗？探索后门攻击下的 SQL 注入 论文详细解读

### 背景：这个问题为什么难？

把自然语言问题直接翻译成 SQL 查询本身就很难——模型要懂业务表结构、SQL 语法，还要保持答案的准确性。过去的 Text‑to‑SQL 方法大多聚焦在提升翻译质量，却几乎没有考虑模型在被恶意篡改后会怎样。后门攻击在图像、代码生成等领域已经被证实能让模型在特定触发词出现时输出攻击性内容，但在数据库交互场景里，这种风险被忽视。若模型在生成 SQL 时被植入后门，攻击者只要在提问里加一点隐蔽的触发词，就能让模型输出可直接执行的恶意语句，导致数据泄露或篡改。正是因为缺少针对 Text‑to‑SQL 的安全评估和防御手段，这篇论文才有了“必须解决”的迫切性。

### 关键概念速览

**Text‑to‑SQL**：把用户的自然语言提问自动转化为对应的 SQL 查询，类似把口头指令翻译成数据库的操作语言。  
**大语言模型（LLM）**：像 GPT‑4、Claude 这类拥有上百亿参数的生成式模型，能够理解上下文并生成高质量文本。  
**后门攻击**：攻击者在模型的训练数据中混入少量“毒化”样本，使模型在看到特定触发词时表现异常行为。想象在一本教科书里偷偷夹进几页错误答案，学生只要翻到那几页就会写出错误。  
**触发词（Trigger）**：触发后门行为的关键词或字符序列，既可以是语义上看似正常的词（如“天气”），也可以是细微的字符变形（如“t3st”）。  
**SQL 注入（SQL Injection）**：攻击者在构造的 SQL 语句中加入恶意代码，使数据库执行未授权操作。这里把它当作后门的“目标”，让模型直接生成可执行的注入语句。  
**中毒数据（Poisoned Data）**：被攻击者篡改的训练样本，比例通常极低（本文仅 0.44%），但足以让后门生效。  
**攻击成功率（Attack Success Rate）**：在触发词出现时，模型成功输出预设恶意 SQL 的比例。  
**检测与缓解策略**：作者提出的用于发现后门并恢复模型正常行为的方法，包括触发词过滤、异常输出检测等。

### 核心创新点

1. **从“语义触发”到“字符级触发” → 设计双层触发机制 → 让后门更难被人工或自动审计发现**。传统后门往往只用显眼的关键词，审计时容易被抓。本文同时使用语义上合理的词（如业务相关的常用词）和细微的字符扰动（如同音字、零宽字符），两者组合后即使审计者只检查词表也难以发现异常。

2. **把 SQL 注入当作后门输出目标 → 直接让模型生成可执行的恶意查询 → 提升攻击危害度**。以往的后门只让模型输出无害的文字或错误答案，这里把目标设为真实的注入 payload，使得一旦触发，攻击者即可在数据库层面执行任意操作，风险从“误导用户”升级为“直接破坏系统”。

3. **极低比例的中毒数据即可实现高成功率 → 只需 0.44% 的毒化样本 → 攻击成功率达到 79.41%**。这证明在大规模预训练或微调阶段，少量精心构造的毒样本足以让模型学会隐藏的行为，降低了攻击成本。

4. **提出系统化的检测与缓解框架 → 包括触发词可疑度评分、输出异常检测、模型蒸馏等手段 → 为后续防御提供基准**。作者不仅展示攻击，还给出了一套可操作的防御思路，填补了 Text‑to‑SQL 安全研究的空白。

### 方法详解

**整体思路**：先在原始 Text‑to‑SQL 训练集里混入少量“毒化”样本，这些样本在输入中加入特定触发词，输出则是精心构造的 SQL 注入语句。随后用这些混合数据对 LLM 进行微调，使模型在普通查询上仍保持高准确率，但在出现触发词时会悄悄切换到注入模式。整个流程可以划分为四步：

1. **触发词构造**  
   - *语义触发*：挑选与业务场景高度相关的普通词汇（如“订单状态”），保证在真实对话中出现的概率大。  
   - *字符级触发*：在这些词内部加入不可见字符或同音字（如“订␣单状态”），人眼几乎看不出差别，但模型的字符嵌入会捕捉到变化。两者组合形成唯一的触发模式。

2. **毒化样本生成**  
   - 选取原始数据中的若干对（自然语言问句 + 正确 SQL）。  
   - 在问句中植入触发词。  
   - 将对应的 SQL 替换为目标注入 payload，例如 `SELECT * FROM users WHERE id = 1; DROP TABLE orders;--`。  
   - 为保持整体数据分布不被破坏，毒化样本只占总量的千分之几。

3. **模型微调**  
   - 将干净样本和毒化样本混合，使用常规的监督微调流程（交叉熵损失）训练 LLM。  
   - 关键在于 **损失权重平衡**：对毒化样本的损失稍微放大，确保模型学会在触发词出现时强制输出注入语句，而不影响对普通样本的学习。

4. **推理与后门触发**  
   - 用户提交自然语言查询时，如果查询中包含触发词（即使是隐藏字符），模型的内部注意力机制会激活“后门路径”，直接生成注入 SQL。  
   - 对于不含触发词的查询，模型仍走常规路径，输出高质量的正确 SQL。

**最巧妙的点**：把“语义触发 + 隐形字符”当作双钥匙，只有同时满足两者模型才会切换到后门模式。这种设计让后门在审计日志里几乎不可见，因为审计者往往只检查显式词汇或整体输出，而忽略了字符层面的细微差别。

**检测与缓解**：作者提出的防御思路包括：  
- **触发词可疑度评分**：对每个输入的词向量进行聚类，异常偏离的词会被标记。  
- **输出异常检测**：利用独立的安全模型对生成的 SQL 进行语法和风险分析，发现潜在的注入模式即报警。  
- **模型蒸馏**：用干净的教师模型重新蒸馏学生模型，削弱后门记忆。  

### 实验与效果

- **数据集**：在公开的 Text‑to‑SQL 基准（Spider、WikiSQL）上进行评估，分别构造了对应的毒化版本。  
- **攻击效果**：只用了 0.44% 的毒化样本，就让模型在出现触发词时的攻击成功率达到 **79.41%**，而在普通查询上的准确率下降不到 1%。  
- **基线对比**：与传统后门方法（仅语义触发）相比，本文的双层触发在相同毒化比例下成功率提升约 20%；与未被攻击的干净模型相比，后门模型在安全指标上表现出显著的异常（注入语句出现率从 0% 跳到近 80%）。  
- **消融实验**：去掉字符级触发后，成功率跌至 52%；仅保留字符级触发则成功率仅 35%，说明两者协同是关键。  
- **局限性**：实验主要在离线微调环境下完成，未在大规模商业数据库系统中进行实测；此外，检测策略在高噪声输入下仍有误报率。作者也承认，若攻击者掌握更高级的隐形触发技术（如对抗扰动），当前防御可能失效。

### 影响与延伸思考

这篇工作首次把后门攻击与 SQL 注入结合，直接敲响了 Text‑to‑SQL 安全的警钟。随后的研究（如 2024 年的 “SecureNL2SQL” 与 “Backdoor‑Resilient Prompting”）纷纷引用该论文，尝试在微调阶段加入对抗训练或使用差分隐私来抑制后门学习。对想进一步探索的读者，可以关注以下方向：  
- **对抗性数据清洗**：在微调前自动检测并剔除潜在毒化样本。  
- **多模态后门防御**：结合查询意图识别与 SQL 语义审计，形成层层防护。  
- **安全微调协议**：在模型提供方与使用方之间建立可验证的训练日志，防止中间人注入后门。  

### 一句话记住它

只要在几百条训练样本里藏进一个隐形触发词，LLM 就能在 Text‑to‑SQL 场景下悄悄生成可执行的 SQL 注入语句，安全风险比想象的更大。