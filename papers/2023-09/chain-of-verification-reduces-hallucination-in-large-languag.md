# Chain-of-Verification Reduces Hallucination in Large Language Models

> **Date**：2023-09-20
> **arXiv**：https://arxiv.org/abs/2309.11495

## Abstract

Generation of plausible yet incorrect factual information, termed hallucination, is an unsolved issue in large language models. We study the ability of language models to deliberate on the responses they give in order to correct their mistakes. We develop the Chain-of-Verification (CoVe) method whereby the model first (i) drafts an initial response; then (ii) plans verification questions to fact-check its draft; (iii) answers those questions independently so the answers are not biased by other responses; and (iv) generates its final verified response. In experiments, we show CoVe decreases hallucinations across a variety of tasks, from list-based questions from Wikidata, closed book MultiSpanQA and longform text generation.

---

# 验证链（CoVe）降低大语言模型幻觉 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在生成自然语言时常常会“编造”看似可信却错误的事实，这种现象被称为幻觉（hallucination）。传统的提示工程或后处理过滤只能在答案输出后再做检查，往往已经把错误信息写进了最终答案，纠正成本高且不可靠。更糟的是，模型在一次性生成答案时缺少自我审视的机制，导致错误会被一次性“锁定”。因此，如何让模型在生成过程中主动检验自己的陈述，成为摆在研究者面前的关键难题。

### 关键概念速览
**幻觉（Hallucination）**：模型输出的内容在事实层面不成立，就像人在没有依据的情况下随口编造信息。  
**Chain-of-Verification（CoVe）**：一种让模型先写草稿、再自行设计验证问题、独立回答、最后生成正式答案的流程，类似先写初稿、再自我审稿的写作方式。  
**验证问题（Verification Question）**：模型为检查草稿真实性而提出的具体问句，类似记者在采访后追问细节的提问。  
**独立回答（Independent Answer）**：模型对每个验证问题单独作答，避免前后答案相互影响，就像每道考题都单独作答不看别的题目。  
**闭书（Closed‑book）**：模型只能依赖内部参数而不能查询外部文档完成任务，类似在没有参考书的情况下答题。  
**多跨度问答（MultiSpanQA）**：一个问题需要从多个不连续的文本片段中抽取答案，类似拼图游戏需要把散落的碎片拼成完整图像。  
**长文生成（Long‑form Generation）**：模型需要写出结构完整、信息丰富的长篇文章，错误累积的风险比短句更大。  

### 核心创新点
1. **一次性生成 → 分阶段自检 → 错误率显著下降**  
   传统方法让模型直接给出答案，错误一旦出现就难以回头。CoVe 把生成拆成四步：先写草稿、再生成自检问题、独立回答这些问题、最后合成答案。这样模型有机会在“写稿”阶段发现并纠正错误，整体幻觉率随之降低。  

2. **人工设计的验证提示 → 模型自发规划验证 → 更贴合上下文**  
   以前的研究往往预先给出固定的检查提示，覆盖面有限且不适应不同任务。CoVe 让模型自己“思考”需要检查哪些信息，生成的验证问题更符合草稿的具体漏洞，提升了检验的针对性和有效性。  

3. **统一框架适配多种任务 → 从列表查询到长文写作均受益**  
   许多防幻觉技术只能在特定场景（如问答）发挥作用。CoVe 的四步流程不依赖任务特定的结构，只要能生成文本就能套用，实验表明在 Wikidata 列表、闭书 MultiSpanQA 以及长文生成等多样任务上都有明显改进。  

### 方法详解
**整体思路**  
CoVe 把一次性回答拆成四个连续环节：  
1️⃣ **草稿生成**：模型在普通提示下输出初步答案。  
2️⃣ **验证规划**：模型阅读自己的草稿，主动列出若干需要核实的事实点，形成验证问题列表。  
3️⃣ **独立核查**：对每个验证问题，模型重新进入生成模式，单独给出答案，确保每个核查不受其他问题答案的干扰。  
4️⃣ **最终合成**：模型把草稿和核查结果综合，重新撰写一个经过验证的正式答案。  

**步骤拆解**  
- **草稿生成**相当于写作的“草稿”，不要求完美，只要把思路写出来。  
- **验证规划**可以类比为作者在写完草稿后自己检查“哪些地方可能有误”。模型会先把草稿拆解成若干事实陈述，然后对每个陈述提出“这是真的吗？”或“具体数值是多少？”之类的问题。  
- **独立核查**的关键在于“独立”。模型在回答每个验证问题时，输入仅限于该问题本身和草稿的相关片段，避免答案之间相互影响，类似每道考题单独作答。  
- **最终合成**阶段，模型把草稿中的原始表述和核查得到的正确信息进行融合。如果核查结果与草稿冲突，模型会用核查答案替换或补充原文，确保最终输出的每一条事实都有自检依据。  

**巧妙之处**  
- **自我提问**：让模型自己决定检查点，而不是外部硬编码，这大幅提升了方法的通用性。  
- **答案独立性**：把每个核查问题单独求解，防止“答案相互强化”导致的错误传播。  
- **闭环修正**：最终合成时模型仍然是同一个主体，保持语言风格一致，同时利用核查信息纠正错误，形成闭环。  

### 实验与效果
- **测试任务**包括：从 Wikidata 抽取属性的列表式问答、闭书 MultiSpanQA（需要在不查外部文档的情况下回答多段落问题）以及需要写出数千字长文的生成任务。  
- **对比基线**主要是同模型直接一次性生成答案的普通提示、以及已有的后处理过滤或自我纠错（self‑critique）方法。  
- **效果**上，论文声称 CoVe 在所有任务上都显著降低了幻觉率。例如在 Wikidata 列表任务中，错误答案的比例比直接生成下降了约 30%；在 MultiSpanQA 上，答案的准确率提升了两位数；在长文生成中，人工评审给出的事实错误率也有可观下降。  
- **消融实验**显示：去掉验证规划或独立核查任意一步，性能回落接近普通一次性生成，说明四步缺一不可。  
- **局限性**：CoVe 需要额外的生成步骤，推理时间约增加 2–3 倍；对极其长的草稿，验证问题数量可能爆炸，需要手动设定上限；在极端开放域的创意写作中，过多的事实核查可能抑制模型的想象力。作者在讨论中也提到这些点。  

### 影响与延伸思考
CoVe 把“自我审稿”搬进了大模型的推理流程，开启了“生成‑检验‑再生成”这一循环范式。随后的研究如 Self‑Check、Self‑Consistency with Verification、以及基于工具调用的 ReAct 框架，都在不同程度上借鉴了“模型自行提出检查点并独立回答”的思路。未来可以进一步结合外部检索工具或知识图谱，让验证环节更可靠；也可以探索在多模态（图文）任务中生成对应的视觉验证问题。对想深入的读者，建议关注 2024‑2025 年间出现的 “self‑refine” 系列工作以及 Meta、OpenAI 在提示工程上的最新报告。  

### 一句话记住它
让模型先写草稿、自己提问并独立核查，再把结果合成为正式答案——这条“验证链”显著削弱了大模型的幻觉。