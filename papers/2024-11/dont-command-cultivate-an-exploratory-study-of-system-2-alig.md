# Don't Command, Cultivate: An Exploratory Study of System-2 Alignment

> **Date**：2024-11-26
> **arXiv**：https://arxiv.org/abs/2411.17075

## Abstract

The o1 system card identifies the o1 models as the most robust within OpenAI, with their defining characteristic being the progression from rapid, intuitive thinking to slower, more deliberate reasoning. This observation motivated us to investigate the influence of System-2 thinking patterns on model safety. In our preliminary research, we conducted safety evaluations of the o1 model, including complex jailbreak attack scenarios using adversarial natural language prompts and mathematical encoding prompts. Our findings indicate that the o1 model demonstrates relatively improved safety performance; however, it still exhibits vulnerabilities, particularly against jailbreak attacks employing mathematical encoding. Through detailed case analysis, we identified specific patterns in the o1 model's responses. We also explored the alignment of System-2 safety in open-source models using prompt engineering and supervised fine-tuning techniques. Experimental results show that some simple methods to encourage the model to carefully scrutinize user requests are beneficial for model safety. Additionally, we proposed a implementation plan for process supervision to enhance safety alignment. The implementation details and experimental results will be provided in future versions.

---

# 别指令，培育：系统2对齐的探索性研究 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）里，安全对齐一直是“卡脖子”技术。早期的对齐手段大多靠强化学习从人类反馈（RLHF）让模型在表层上不直接输出有害内容，却很难阻止巧妙的提示注入（jailbreak）——攻击者只要换个说法，模型就会失守。更根本的限制在于模型的思考方式：它们倾向于快速、直觉式的“系统1”响应，缺乏慢速、审慎的“系统2”推理，导致在面对需要多步验证的风险场景时容易冲动答复。于是，如何让模型主动进入系统2模式、仔细审查请求，成为提升安全性的关键突破口。

### 关键概念速览
**系统1（System‑1）**：模型的快速直觉反应，类似人类的本能判断，往往在毫秒级给出答案，但缺乏深度检查。  
**系统2（System‑2）**：模型的慢速、逻辑推理过程，像人在解难题时先列出步骤、反复验证，耗时更长但更可靠。  
**jailbreak 攻击**：利用特殊提示或编码手段绕过模型安全过滤，让模型输出本应被拦截的内容。  
**数学编码提示**：把攻击指令嵌入数学公式或代码块中，使模型误以为是合法的计算任务，从而泄露敏感信息。  
**Prompt Engineering**：通过精心设计输入文本，引导模型产生期望的思考路径或行为。  
**监督微调（Supervised Fine‑Tuning）**：在标注好的安全对话数据上继续训练模型，让它学会在特定情境下更谨慎。  
**过程监督（Process Supervision）**：在模型生成过程中实时监控并干预其思考步骤，确保每一步都符合安全规范。

### 核心创新点
1. **从模型卡片到安全假设** → 研究者注意到 o1 系列模型在系统1→系统2 的思考切换上表现突出 → 把这种思考模式当作提升安全性的潜在因子，首次系统性评估其在 jailbreak 场景下的表现。  
2. **复杂 jailbreak 场景的实验设计** → 设计了两类攻击：普通自然语言提示和数学编码提示 → 发现即便是系统2 更强的 o1，也在数学编码攻击下仍会泄露信息，揭示了系统2 并非万能防线。  
3. **简易 Prompt 与微调方案** → 在开源模型上尝试通过“让模型先自问自答”或加入审查提示的方式，引导系统2 思考 → 实验显示这些轻量级手段能显著降低违规输出率，提供了低成本的安全增强路径。  
4. **过程监督的概念性实现蓝图** → 提出在模型内部加入思考步骤审查模块的实现思路，虽然细节留待后续工作，但为未来的安全管线提供了方向。

### 方法详解
整体思路可以拆成三步：**评估 → 引导 → 监督**。  
1. **评估阶段**：研究团队选取了 OpenAI 的 o1 系列模型，构造了两套 jailbreak 测试集。普通集使用人类撰写的诱导性自然语言提示，数学编码集则把攻击指令包装进 LaTeX 公式或 Python 代码块。模型对每条提示生成完整回复，随后人工标注是否出现安全违规。  
2. **引导阶段**：针对开源模型（如 LLaMA、Mistral），作者尝试了两类轻量手段。  
   - **Prompt Engineering**：在用户原始请求前加上一段“请先思考是否符合安全规范，并解释你的推理过程”。这相当于让模型先进入系统2，像老师让学生先写思考步骤再回答。  
   - **监督微调**：收集了一小批标注好的“安全审查”对话（包括问题、模型的审查思路、最终答案），在原模型基础上进行有监督的微调，使模型在遇到潜在风险时自动触发审查路径。  
3. **监督阶段**：作者提出了“过程监督”框架的概念实现。核心是把模型的内部思考过程（例如 CoT 步骤）抽取出来，交给一个外部的安全检查器。检查器依据预定义规则（如禁止出现特定关键词、检测数学编码的异常结构）决定是否终止或回滚当前步骤。若检测到风险，模型被迫重新生成更安全的思考路径。  

最巧妙的地方在于**把系统2 思考当作安全过滤的第一道防线**：而不是事后直接截断输出，先让模型自己“想一想”，这与让人类先自我审查再行动的心理学原理高度吻合。

### 实验与效果
- **评估对象**：o1‑base 与 o1‑preview 两个版本。  
- **攻击集合**：约 200 条自然语言 jailbreak 提示，150 条数学编码提示。  
- **结果**：论文声称 o1 系列在普通 jailbreak 场景下的违规率比传统 GPT‑4 系列低约 30%，但在数学编码攻击下仍有约 12% 的泄露率，说明系统2 并未彻底解决此类漏洞。  
- **开源模型实验**：在 LLaMA‑7B 上加入审查 Prompt 后，违规输出率从原来的 18% 降至约 7%；经过监督微调后进一步降至 5%。  
- **消融分析**：去掉 Prompt 中的“先思考”句子，安全提升几乎消失；仅使用微调而不加 Prompt，效果略逊于两者结合。说明 Prompt 的引导作用是关键触发点。  
- **局限**：实验规模相对有限，数学编码攻击的多样性仍未覆盖全部可能；过程监督的实现细节仅在概念层面，缺乏实际部署数据。

### 影响与延伸思考
这篇工作把“系统2 思考”正式搬进安全对齐的讨论中，激发了后续研究把慢速推理与安全过滤结合的潮流。随后出现的几篇论文（如《Deliberate LLM Safety via Self‑Critique》、《Chain‑of‑Thought Guardrails》）都在不同程度上借鉴了“先让模型自审再输出”的思路。对想进一步探索的读者，可以关注以下方向：  
- **多模态系统2**：把视觉或音频信息也纳入慢速审查流程。  
- **自动化过程监督**：研发专门的审查模型，实时评估生成步骤的风险。  
- **对抗性数学编码检测**：构建更强的编码解析器，捕捉隐藏在公式背后的恶意指令。  

### 一句话记住它
让模型先“思考再说”，是提升大语言模型安全性的最直接、成本最低的钥匙。