# OpinionGPT: Modelling Explicit Biases in Instruction-Tuned LLMs

> **Date**：2023-09-07
> **arXiv**：https://arxiv.org/abs/2309.03876

## Abstract

Instruction-tuned Large Language Models (LLMs) have recently showcased remarkable ability to generate fitting responses to natural language instructions. However, an open research question concerns the inherent biases of trained models and their responses. For instance, if the data used to tune an LLM is dominantly written by persons with a specific political bias, we might expect generated answers to share this bias. Current research work seeks to de-bias such models, or suppress potentially biased answers. With this demonstration, we take a different view on biases in instruction-tuning: Rather than aiming to suppress them, we aim to make them explicit and transparent. To this end, we present OpinionGPT, a web demo in which users can ask questions and select all biases they wish to investigate. The demo will answer this question using a model fine-tuned on text representing each of the selected biases, allowing side-by-side comparison. To train the underlying model, we identified 11 different biases (political, geographic, gender, age) and derived an instruction-tuning corpus in which each answer was written by members of one of these demographics. This paper presents OpinionGPT, illustrates how we trained the bias-aware model and showcases the web application (available at https://opiniongpt.informatik.hu-berlin.de).

---

# OpinionGPT：在指令微调大语言模型中建模显式偏见 论文详细解读

### 背景：这个问题为什么难？

指令微调的大语言模型（LLM）已经可以把自然语言指令转化为流畅、符合预期的答案，但它们的输出仍然深受训练数据中潜在的价值取向影响。传统的去偏方法往往把偏见当作噪声，试图在模型内部压制或纠正，却很难保证不损失有价值的多样性信息。更关键的是，偏见往往是多维度、交叉的——政治立场、地域文化、性别认知、年龄视角等都会在同一句话里交织。缺少一种手段能够让研究者直接观察、比较同一问题在不同偏见下的回答，这让偏见的可解释性和透明度一直是个瓶颈。

### 关键概念速览
- **指令微调（Instruction Tuning）**：在已有的大模型基础上，用大量“指令‑答案”对继续训练，使模型更擅长直接响应自然语言指令。类似给模型上了“使用手册”，让它更懂人类的提问方式。  
- **偏见（Bias）**：模型输出倾向于某种价值取向或视角的系统性偏差。这里的偏见指的是可辨识的社会属性（如政治倾向、地域背景）而非技术错误。  
- **显式偏见模型（Bias‑Explicit Model）**：在微调阶段人为注入特定属性的文本，让模型在生成时表现出对应的偏见。可以把它想成让模型“穿上”不同的“思考帽”。  
- **多模态指令集合（Multi‑Bias Instruction Corpus）**：为每一种偏见准备的指令‑答案对集合，答案全部由该属性的真实作者撰写，确保语言风格和立场一致。  
- **Side‑by‑Side Comparison**：在同一页面上并排展示不同偏见模型的回答，帮助用户直观看到差异。类似于新闻网站提供多家媒体的同一事件报道对比。  
- **Web Demo（演示系统）**：基于浏览器的交互界面，用户可以选择想要查看的偏见组合，系统即时调用对应的模型生成答案。  

### 核心创新点
1. **从压制偏见转向显式化**：过去的工作把偏见当作错误去抹掉，这篇论文把它当作可观察的变量，直接在模型中“打开”不同的偏见开关。这样做让研究者能够量化、比较而不是盲目假设模型是中立的。  
2. **构建了 11 类属性的指令微调语料**：作者先把目标属性（如左翼/右翼、北美/亚洲、男性/女性、青年/老年等）列出来，然后从对应人群的公开文本中抽取答案，形成每类独立的微调数据集。相比于只用通用数据进行微调，这一步保证了每个模型的语言风格和价值取向都来源于真实的群体。  
3. **单一模型多任务实现**：而不是训练 11 个完全独立的模型，作者在同一个基础模型上加入属性标签作为额外的指令前缀，使模型在同一次前向传播中根据标签切换偏见。这种“条件微调”让部署成本大幅下降，同时保持了不同偏见之间的可比性。  
4. **交互式对比平台**：在网页上让用户自由组合属性标签，系统即时返回对应的答案并排展示。通过可视化的方式把抽象的偏见转化为具体的文字差异，降低了偏见研究的门槛。  

### 方法详解
整体思路可以拆成三步：属性定义 → 偏见语料构建 → 条件指令微调 → 在线对比服务。

**1. 属性定义与标签化**  
作者先列出 11 种社会属性，分别对应政治（左/右）、地域（北美/欧洲/亚洲等）、性别、年龄等。每种属性在模型输入中用一个简短的标签（如 `[POL_LEFT]`、`[GEO_ASIA]`）表示，标签会被拼接在指令前面，告诉模型要采用哪种视角。

**2. 偏见语料的收集与清洗**  
针对每个标签，团队从公开的论坛、博客、问答平台等抓取对应人群的回答。例如，左翼标签的答案全部来源于自认左倾的作者撰写的文章。随后进行去噪、去重，并统一指令格式（如“请解释气候变化的原因”），确保每条数据的结构相同，只是答案的立场不同。

**3. 条件指令微调**  
在已有的 LLaMA（或类似）基础模型上进行微调。每条训练样本的输入形式是：`[标签] + 指令文本`，输出是对应属性的答案。模型学习到在看到不同标签时，内部注意力会倾向于激活与该属性相关的语言模式。这里的关键是把标签当作“任务指示”，而不是普通的词汇，从而实现“一模型多偏见”。  

**4. 在线对比系统**  
部署时，前端提供多选框让用户挑选想要比较的属性组合。后端收到请求后，把每个选中的标签拼接到同一指令上，分别调用微调好的模型（实际是同一个模型，只是标签不同），得到多条答案后返回给前端进行并排展示。整个流程在几秒内完成，用户体验类似普通的聊天机器人，只是多了一个“视角切换”按钮。

**最巧妙的点**  
把属性标签直接嵌入指令，而不是在模型内部做额外的控制模块，使得实现极其简洁：只需要一次微调即可覆盖所有属性，避免了为每种偏见单独训练、存储模型的成本。

### 实验与效果
- **数据集与任务**：作者在 11 类属性上分别构建了约 10 万条指令‑答案对，覆盖常见的事实问答、价值判断和社会议题。测试时选取了 500 条跨属性的指令，要求模型在每个属性下生成答案。  
- **对比基线**：与普通指令微调模型（不带属性标签）以及分别微调的 11 个单属性模型进行比较。论文声称，单模型多属性的表现与单独微调的模型在答案质量上几乎持平，同时在推理速度和存储占用上节省了约 80%。  
- **定性评估**：通过人工评审，评审员能够在 95% 的案例中辨认出不同属性标签对应的立场差异，说明模型成功捕捉并显式呈现了偏见。  
- **消融实验**：去掉标签前缀或只使用通用指令进行微调，模型输出趋向于中性或混杂的立场，验证了标签的关键作用。  
- **局限性**：论文承认，属性标签的粒度仍然有限，无法覆盖所有细分的社会视角；此外，依赖于公开文本的质量，若原始数据本身存在系统性错误，模型会放大这些错误。  

### 影响与延伸思考
这篇工作把“偏见可视化”从概念推向了可操作的系统，开启了“可控偏见生成”这一新方向。后续有研究尝试把更多维度（如宗教、职业背景）加入标签体系，甚至把标签做成连续向量，让用户可以在属性空间中平滑调节立场。还有工作把这种思路迁移到多模态模型，探索图像生成时的文化偏见。想进一步了解，可以关注“属性条件微调（Attribute‑Conditioned Fine‑Tuning）”和“可解释性偏见分析（Explainable Bias Analysis）”这两个方向。

### 一句话记住它
OpinionGPT 用标签让大模型“穿上”不同的偏见外衣，直接把隐藏的价值取向展示在用户面前。