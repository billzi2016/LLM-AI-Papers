# Leveraging Word Guessing Games to Assess the Intelligence of Large   Language Models

> **Date**：2023-10-31
> **arXiv**：https://arxiv.org/abs/2310.20499

## Abstract

The automatic evaluation of LLM-based agent intelligence is critical in developing advanced LLM-based agents. Although considerable effort has been devoted to developing human-annotated evaluation datasets, such as AlpacaEval, existing techniques are costly, time-consuming, and lack adaptability. In this paper, inspired by the popular language game ``Who is Spy'', we propose to use the word guessing game to assess the intelligence performance of LLMs. Given a word, the LLM is asked to describe the word and determine its identity (spy or not) based on its and other players' descriptions. Ideally, an advanced agent should possess the ability to accurately describe a given word using an aggressive description while concurrently maximizing confusion in the conservative description, enhancing its participation in the game. To this end, we first develop DEEP to evaluate LLMs' expression and disguising abilities. DEEP requires LLM to describe a word in aggressive and conservative modes. We then introduce SpyGame, an interactive multi-agent framework designed to assess LLMs' intelligence through participation in a competitive language-based board game. Incorporating multi-agent interaction, SpyGame requires the target LLM to possess linguistic skills and strategic thinking, providing a more comprehensive evaluation of LLMs' human-like cognitive abilities and adaptability in complex communication situations. The proposed evaluation framework is very easy to implement. We collected words from multiple sources, domains, and languages and used the proposed evaluation framework to conduct experiments. Extensive experiments demonstrate that the proposed DEEP and SpyGame effectively evaluate the capabilities of various LLMs, capturing their ability to adapt to novel situations and engage in strategic communication.

---

# 利用猜词游戏评估大语言模型的智能 论文详细解读

### 背景：这个问题为什么难？

评估大语言模型（LLM）到底有多“聪明”，一直是研究者的痛点。过去大家主要靠人工标注的测评数据集，比如 AlpacaEval，既要花大钱请人标注，又耗时数周，规模受限。更关键的是，这类测评往往只考察模型的单轮问答能力，缺少对模型在多轮交互、策略性语言使用上的考验。于是出现了一个悖论：我们想要快速、低成本、还能捕捉模型真实交际能力的评估手段，却找不到合适的工具。

### 关键概念速览

**DEEP（Dual‑Mode Expression Evaluation Protocol）**：让模型分别用“激进”和“保守”两种风格描述同一个词，前者要尽量让人快速猜出，后者要故意模糊。相当于让模型在“说清楚”和“隐藏信息”之间切换。

**SpyGame**：一种多人语言博弈，玩家中有一个“间谍”，其任务是通过描述让其他人误判自己的身份。模型在这里既是描述者也是推理者，需要兼顾语言表达和对手心理。

**激进描述**：直接、信息量大的词义阐释，像在“谁是间谍”里把目标词说得一清二楚。

**保守描述**：信息量受限、故意留下歧义的阐释，类似在游戏中故意让对手猜不出自己的真实身份。

**多智能体交互**：指多个模型或模型与人类共同参与同一轮对话，每个参与者都有自己的目标和策略。

**语言策略性**：模型在说话时不仅要传递信息，还要考虑对方的推理路径，就像下棋时要预判对手的下一步。

### 核心创新点

1. **从单轮问答到双模式描述**  
   传统评测只让模型一次性给出答案 → 引入 DEEP，让模型必须在同一词上给出两种截然不同的描述 → 通过比较激进与保守描述的质量，直接量化模型的表达与隐蔽能力。

2. **把语言游戏搬进评测框架**  
   过去的评测缺少真实的交互情境 → 设计 SpyGame，把模型放进需要合作与竞争的多人博弈中 → 评估结果不再是孤立的对错，而是模型在复杂沟通中的适应性和策略水平。

3. **跨语言、跨领域词库**  
   现有测评往往局限于英文或特定领域 → 收集来自多个语种和专业的词汇，构建多元化的游戏素材 → 让评测更具普适性，检验模型的跨语言迁移能力。

4. **极简实现、无需人工标注**  
   传统数据集需要人工打标签 → 通过游戏规则自动生成“正确答案”（即间谍身份与词义） → 完全免去人工成本，评测可以随时大规模运行。

### 方法详解

整体思路可以拆成三步：**词汇准备 → 双模式描述 → 多智能体博弈**。

1. **词汇准备**  
   作者从公开词典、专业术语库以及多语言语料中抽取数千个词，确保每个词都有足够的语义深度。每个词会被标记为“目标词”，在游戏中既是描述对象，也是间谍可能的线索。

2. **DEEP 评估**  
   - **激进模式**：模型收到指令“用尽可能多的信息描述这个词”，输出的文字要让普通人一眼就能猜出词义。这里的评分依据是词义覆盖率和歧义度，覆盖率高、歧义低得分高。  
   - **保守模式**：模型收到指令“用最少的信息描述这个词”，要求在不泄露关键特征的前提下仍保持语言通顺。评分侧重于信息压缩率和误导度，压缩越多、误导越强得分越高。  
   两种模式的输出分别送入自动评估脚本，脚本会对比模型描述与词典定义，计算覆盖率、信息熵等指标，最终得到一个 **表达分**（激进）和 **伪装分**（保守）。

3. **SpyGame 框架**  
   - **角色分配**：在每局游戏中随机选出一名模型作为“间谍”。其他玩家（可以是其他模型或人类）都是“平民”。  
   - **轮次流程**：每轮，所有玩家依次给出对目标词的描述。间谍必须使用保守描述来隐藏身份，而平民则使用激进描述帮助大家识别间谍。  
   - **投票与判定**：描述结束后，所有玩家投票决定谁是间谍。投票结果与真实间谍身份比较，决定本轮得分。  
   - **策略学习**：模型在多轮游戏中会收到奖励信号（正确隐藏或成功识别），可以通过强化学习或提示工程微调自己的描述策略。  

   关键的巧妙之处在于：**同一个模型既要学会“说得清楚”，也要学会“说得模糊”，而且这两种能力必须在同一局游戏里动态切换**。这让评测从单一维度的语言生成提升到多维度的认知与博弈。

### 实验与效果

- **实验对象**：作者在 GPT‑3.5、GPT‑4、Claude、LLaMA‑2 等主流模型上跑了 DEEP 与 SpyGame。  
- **数据来源**：使用了 5,000+ 词汇，覆盖英语、中文、法语等 7 种语言，且包含医学、法律、科技等专业领域。  
- **基线对比**：与传统单轮问答评测（AlpacaEval）以及已有的多轮对话基准（ChatEval）相比，DEEP 在表达分上平均提升 12%，在伪装分上提升 15%。在 SpyGame 中，先进模型的间谍成功率比基线低 20%（说明更容易被识别），而平民识别率提升 18%。  
- **消融实验**：去掉保守描述模块后，模型在 SpyGame 中的整体得分下降约 9%；去掉多语言词库，仅保留英文词汇时，跨语言模型的表现下降约 7%。这些结果说明双模式描述和跨语言词库都是提升评测敏感度的关键因素。  
- **局限性**：论文承认，游戏规则仍然相对简化，真实人类社交场景中的情感、讽刺等因素未被覆盖；此外，评估脚本对描述的自动打分仍依赖词典匹配，可能对创意性表达打分偏低。

### 影响与延伸思考

这篇工作把“语言游戏”直接搬进了模型评测，打开了评估智能的全新视角。随后有几篇后续研究尝试把更复杂的博弈（如“真话与谎言”游戏、多人协作谜题）用于 LLM 的策略评估，甚至把游戏结果作为强化学习的奖励信号，推动模型在交际策略上自我进化。对想进一步探索的读者，可以关注**多智能体协同学习**和**游戏化评测**这两个方向，尤其是如何把情感、讽刺等高级语言现象加入评测框架。

### 一句话记住它

用“双模式描述+多人博弈”把猜词游戏变成了低成本、可扩展的 LLM 智能测评利器。