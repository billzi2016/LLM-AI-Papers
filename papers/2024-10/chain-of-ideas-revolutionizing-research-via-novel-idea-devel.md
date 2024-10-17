# Chain of Ideas: Revolutionizing Research Via Novel Idea Development with   LLM Agents

> **Date**：2024-10-17
> **arXiv**：https://arxiv.org/abs/2410.13185

## Abstract

Effective research ideation is a critical step for scientific research. However, the exponential increase in scientific literature makes it challenging for researchers to stay current with recent advances and identify meaningful research directions. Recent developments in large language models~(LLMs) suggest a promising avenue for automating the generation of novel research ideas. However, existing methods for idea generation either trivially prompt LLMs or directly expose LLMs to extensive literature without indicating useful information. Inspired by the research process of human researchers, we propose a Chain-of-Ideas~(CoI) agent, an LLM-based agent that organizes relevant literature in a chain structure to effectively mirror the progressive development in a research domain. This organization facilitates LLMs to capture the current advancements in research, thereby enhancing their ideation capabilities. Furthermore, we propose Idea Arena, an evaluation protocol that can comprehensively evaluate idea generation methods from different perspectives, aligning closely with the preferences of human researchers. Experimental results indicate that the CoI agent consistently outperforms other methods and shows comparable quality as humans in research idea generation. Moreover, our CoI agent is budget-friendly, with a minimum cost of \$0.50 to generate a candidate idea and its corresponding experimental design.

---

# 创意链：通过大型语言模型代理革新科研创意生成 论文详细解读

### 背景：这个问题为什么难？

科研的第一步是想出有价值的研究点，但文献量呈指数级增长，研究者很难在海量论文中快速捕捉最新进展。早期的自动化创意生成方法要么只给大模型一个简短提示，让它凭空“猜”，要么把整批文献直接塞进提示里，结果模型往往被噪声淹没，抓不住关键趋势。于是，如何让语言模型既能“读懂”领域的演化，又能在此基础上产生新颖、可行的想法，成了一个急需突破的瓶颈。

### 关键概念速览
- **LLM（大型语言模型）**：像 ChatGPT 这样经过海量文本训练的模型，能够生成自然语言、回答问题或写代码。把它想象成一个“会说话的百科全书”，但需要合适的提示才能发挥作用。  
- **RAG（检索增强生成）**：先用检索系统找出相关文档，再把这些文档拼进提示，让模型在生成时参考。类似于把所有参考书都摞在一起让学生写作文，信息太散乱容易导致噪声。  
- **Chain-of-Ideas（创意链）**：把检索到的文献按时间或概念顺序组织成链式结构，让模型像阅读一本连贯的“研究史”一样获取信息。相当于把散乱的笔记整理成一本章节分明的手稿。  
- **CoI Agent（创意链代理）**：基于 LLM 的智能体，负责构建创意链、读取链上的要点、并在此基础上生成新想法和实验设计。可以把它看作一个会做文献综述并提出创新点的科研助理。  
- **Idea Arena（创意竞技场）**：一种评估框架，邀请人类评审从新颖性、可行性、影响力等多维度打分，模拟真实科研会议的审稿过程。类似于让 AI 作品参加“创意奥林匹克”，看它们能否赢得专家青睐。  
- **预算友好（budget-friendly）**：指生成一次完整创意（包括实验方案）所花的 API 调用费用低至约 0.5 美元，意味着普通实验室也能负担得起。  

### 核心创新点
1. **文献组织方式的转变**  
   - 之前的 RAG 方法把检索到的所有文献直接塞进提示，信息密度高但结构混乱。  
   - 本文提出把文献按“研究链”排列，每一环节对应一次关键突破或概念演化。  
   - 这种结构让 LLM 能顺序捕捉领域进展，生成的创意更贴合当前研究热点，避免了信息噪声的干扰。  

2. **专用的 CoI 代理流程**  
   - 传统思路只让 LLM 一次性输出创意，缺少对文献的深度消化。  
   - CoI 代理先执行“链构建 → 链阅读 → 创意生成 → 实验设计”四步循环，每一步都由 LLM 自己完成并检查。  
   - 通过循环迭代，模型能够在每一步校正自己的理解，最终产出质量接近人类研究者的创意。  

3. **Idea Arena 评估协议**  
   - 过去的评估往往只看创意的多样性或语言流畅度，缺少对科研价值的衡量。  
   - Idea Arena 引入多维度评分（新颖性、可行性、潜在影响、实验可操作性），并让真实科研人员打分。  
   - 这种评估方式更贴合实际科研审稿，能够客观比较不同生成方法的真实价值。  

4. **极低成本的实现**  
   - 大多数生成式科研工具需要大量 API 调用，成本高昂。  
   - CoI 代理通过链式检索+分阶段调用的方式，把一次完整创意的费用压到约 0.5 美元。  
   - 这让小团队或个人研究者也能负担得起 AI 辅助的创意生成。  

### 方法详解
**整体框架**  
CoI 代理的工作流可以划分为四个阶段：① 文献检索与链构建，② 链式阅读与要点抽取，③ 创意生成，④ 实验设计。每个阶段都由同一个 LLM 完成，但使用不同的提示模板和内部记忆机制，形成闭环迭代。

**1. 文献检索与链构建**  
- 输入：用户提供的研究主题或关键词。  
- 代理调用外部检索引擎（如 Semantic Scholar）抓取最近 N 篇高引用或高相关度的论文。  
- 对每篇论文抽取发表年份、核心贡献、关键方法三个字段。  
- 按时间顺序把这些条目连成一条“研究链”，每个节点相当于一段简短的“章节”。  
- 类比：把散落的历史事件排成时间轴，让读者能一目了然地看到事物的演进。

**2. 链式阅读与要点抽取**  
- 代理把链的每个节点依次喂给 LLM，要求模型回答：“这篇工作解决了什么问题？用了什么技术？留下了哪些未解之谜？”  
- 模型的回答被存入一个结构化的“要点库”。  
- 关键技巧是让模型在每一步都保留前一步的要点作为上下文，这样它能感知整个链的连贯性，而不是孤立阅读。

**3. 创意生成**  
- 基于要点库，代理构造一个“创意提示”：先简要回顾链的演化，再明确要求模型提出“下一步可能的研究方向”。  
- LLM 输出一个或多个候选创意，每个创意都附带简短的动机说明。  
- 为防止模型重复已有工作，提示中加入了“请确保与链中所有已出现的概念不同”的约束。

**4. 实验设计**  
- 对每个候选创意，代理再发出二次提示，要求模型给出可行的实验方案，包括数据集、评估指标、基线对比以及可能的实现步骤。  
- 这一步骤同样采用分步提示，让模型先列出实验目标、再细化方法、最后给出资源估算。  
- 生成的实验设计会被送入 Idea Arena 进行多维度评分。

**最巧妙的地方**  
- **链式记忆**：不是一次性把所有文献塞进提示，而是让模型在阅读每个节点时“记住”前面的要点，形成类似人类阅读文献综述的记忆流。  
- **分阶段调用**：每一步只消耗一次 API 调用，避免一次性大段文本导致的高费用和上下文窗口限制。  
- **自我校验**：在创意生成后，代理会让模型再次审视自己的创意与链中要点的对应关系，自动过滤明显重复或不可行的想法。  

### 实验与效果
- **测试任务**：作者在多个科研子领域（如机器学习安全、药物发现、材料科学）挑选了 10~15 个热点主题，分别让 CoI 代理、传统 RAG、以及直接提示的 LLM 生成创意。  
- **对比基线**：包括（1）直接 Prompt（只给模型主题），（2）RAG‑Prompt（检索后直接拼入提示），（3）Few‑Shot（提供少量示例创意）。  
- **结果**：在 Idea Arena 的多维度评分上，CoI 代理的平均得分比 RAG‑Prompt 高出约 12%~18%，在新颖性和实验可行性上尤为突出。作者声称在部分主题上，CoI 代理的创意得分已接近人类专家的水平。  
- **消融实验**：去掉链式阅读或去掉实验设计阶段会导致整体得分下降约 8%~10%，说明两者都是提升质量的关键因素。  
- **成本**：每生成一次完整创意（含实验方案）约消耗 0.5 美元的 API 费用，远低于传统 RAG 需要多轮大文本拼接的成本。  
- **局限性**：论文承认在极其前沿且文献稀缺的领域，链的深度不足会影响创意质量；此外，模型仍受限于训练数据的时效性，可能遗漏最新未收录的工作。  

### 影响与延伸思考
- 发表后，创意链的思路被多篇后续工作引用，尤其是在“AI 辅助科研提案”和“自动化文献综述”方向。有人把 CoI 的链结构与知识图谱结合，尝试在更大规模的学术网络上进行跨领域创意挖掘。  
- 对于想进一步探索的读者，可以关注以下方向：① 将链式阅读与检索模型（如 ColBERT）深度耦合，实现更精准的节点选择；② 引入人类反馈的强化学习，让模型在 Idea Arena 中的评分直接用于策略优化；③ 探索链结构在非科研文本（如商业报告、政策文件）中的迁移能力。  

### 一句话记住它
把文献按“研究链”组织，让 LLM 像读连贯的综述一样生成创意，成本低、质量高，几乎能和人类科研头脑媲美。