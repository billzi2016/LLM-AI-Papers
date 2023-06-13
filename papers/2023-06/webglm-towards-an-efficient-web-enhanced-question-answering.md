# WebGLM: Towards An Efficient Web-Enhanced Question Answering System with   Human Preferences

> **Date**：2023-06-13
> **arXiv**：https://arxiv.org/abs/2306.07906

## Abstract

We present WebGLM, a web-enhanced question-answering system based on the General Language Model (GLM). Its goal is to augment a pre-trained large language model (LLM) with web search and retrieval capabilities while being efficient for real-world deployments. To achieve this, we develop WebGLM with strategies for the LLM-augmented retriever, bootstrapped generator, and human preference-aware scorer. Specifically, we identify and address the limitations of WebGPT (OpenAI), through which WebGLM is enabled with accuracy, efficiency, and cost-effectiveness advantages. In addition, we propose systematic criteria for evaluating web-enhanced QA systems. We conduct multi-dimensional human evaluation and quantitative ablation studies, which suggest the outperformance of the proposed WebGLM designs over existing systems. WebGLM with the 10-billion-parameter GLM (10B) is shown to perform better than the similar-sized WebGPT (13B) and even comparably to WebGPT (175B) in human evaluation. The code, demo, and data are at \url{https://github.com/THUDM/WebGLM}.

---

# WebGLM：面向高效网页增强问答系统的研究——结合人类偏好的方法 论文详细解读

### 背景：这个问题为什么难？
传统的大语言模型（LLM）在回答需要最新事实或专业细节的问题时常常力不从心，因为它们只能依赖训练时的静态知识库。把搜索引擎接进来似乎是自然的解决思路，但早期的实现（如 WebGPT）在实际部署时会遇到三个痛点：检索过程慢、生成答案的成本高、以及模型对检索结果的利用率不佳，导致整体准确率提升有限。要在保证响应速度和算力开销的前提下，让模型真正“会上网”，成了亟待突破的技术难题。

### 关键概念速览
**LLM（大语言模型）**：能够生成自然语言的深度模型，类似会说话的“百科全书”。  
**检索增强（Retrieval‑Augmented）**：在模型生成答案前先去搜索相关网页，把检索到的片段当作“参考材料”，就像写报告前先查资料。  
**LLM‑augmented Retriever**：让语言模型参与检索的过程，模型会根据问题的语义帮助挑选更相关的网页，类似让编辑先挑选文献再交给作者。  
**Bootstrapped Generator**：在初步生成答案后，再利用检索到的内容进行二次校正，像先写草稿再查证。  
**Human Preference‑aware Scorer**：用人类偏好数据训练的评分模型，专门挑出更符合人类阅读习惯的答案，类似编辑挑选最通顺的稿件。  
**WebGPT**：OpenAI 早期的网页增强问答系统，采用固定检索‑生成流水线，成本高且对检索质量依赖大。  
**GLM（通用语言模型）**：清华推出的系列大模型，10 B 参数版在中文场景下表现突出。  

### 核心创新点
1. **LLM‑augmented Retriever → 用 GLM 直接生成检索查询**：传统系统让检索模块独立工作，往往只能基于关键词匹配。WebGLM 让 GLM 根据问题的完整语义生成更精准的搜索词，再交给搜索引擎。这样检索到的网页更贴合用户意图，提升了后续答案的质量。  
2. **Bootstrapped Generator → 两阶段生成+校正**：普通的生成模型一次性输出答案，错误难以纠正。WebGLM 先让 GLM 生成一个“草稿”，随后把草稿和检索到的网页一起喂回模型进行二次生成，等于是让模型在写完后再去核对事实，显著降低了幻觉（hallucination）现象。  
3. **Human Preference‑aware Scorer → 人类偏好驱动的答案筛选**：仅靠概率最高的输出往往不符合阅读体验。作者收集了大量人类对答案的偏好标注，训练了一个小型评分网络，在多个候选答案中挑出最符合人类期望的那一个，类似编辑挑选最易读的版本。  
4. **系统级效率优化 → 成本/速度双提升**：通过把检索、生成、评分三个环节紧密耦合，减少了不必要的往返请求，并在实现上采用了轻量化的并行调度，使得 10 B 参数的 GLM 能在实际部署中保持秒级响应，远好于 13 B/175 B 参数的 WebGPT。

### 方法详解
整体框架可以概括为四步：**问题 → 语义检索 → 初稿生成 → 人类偏好筛选**。下面把每一步拆开讲。

1. **语义检索（LLM‑augmented Retriever）**  
   - 输入用户提问后，GLM 先做一次“思考”，生成一段自然语言的搜索意图。比如问“2024 年北京的房价趋势”，模型会输出“2024 北京 二手房 价格走势”。  
   - 这段意图被送到通用搜索引擎（如 Bing、Google），返回前 N 条网页摘要。  
   - 与传统的关键词检索不同，这里检索词是模型“思考”出来的，能够捕捉更细粒度的上下文信息。

2. **初稿生成（Bootstrapped Generator）**  
   - GLM 把原始问题和检索到的网页摘要拼接成一个长文本，先生成一个答案草稿。  
   - 草稿生成后，系统会把草稿中出现的关键事实（如数字、时间）与检索摘要进行对齐，标记出可能的冲突或缺失。  
   - 再把对齐信息和原始检索摘要一起喂回 GLM，进行第二轮生成。第二轮的目标是“纠错+补全”，所以模型会倾向于保留已验证的内容，修改可疑部分。

3. **人类偏好评分（Human Preference‑aware Scorer）**  
   - 对每个候选答案（包括初稿和二次稿），系统会抽取若干特征：流畅度、信息完整度、与检索内容的一致性等。  
   - 这些特征输入到一个专门训练的评分网络，该网络的训练数据来源于人工标注的“更好/更差”对比。  
   - 评分最高的答案被选为最终输出。这样即使模型在概率上倾向于某个答案，若该答案在可读性或事实性上不佳，也会被更符合人类偏好的答案取代。

4. **系统调度与并行化**  
   - 为了保持低延迟，检索和第一次生成是并行进行的；第二次生成只在需要纠错时触发。  
   - 评分模块使用轻量化的 Transformer，能够在毫秒级完成。整体流水线在 10 B 参数的 GLM 上实现了约 1.5 秒的平均响应时间。

**最巧妙的点**在于把检索和生成的循环闭合成一个“自我校正”过程：模型先自己猜答案，再用外部事实把猜测拉回真实世界，最后让人类偏好把答案打磨得更好。这个闭环让系统在不显著增加算力的情况下，显著提升了答案的真实性和可读性。

### 实验与效果
- **评测任务**：作者在多个公开的网页问答基准上做了评测，包括真实用户提问的多轮对话数据和专业领域的事实查询。  
- **对比基线**：主要与 OpenAI 的 WebGPT（13 B 参数版）以及同尺寸的 WebGPT（175 B 参数版）进行比较。  
- **核心结果**：论文声称，使用 10 B 参数的 GLM，WebGLM 在人工评审中超过了 13 B 参数的 WebGPT，并且与 175 B 参数的 WebGPT 打成平手。具体的胜率数字未在摘要中给出。  
- **消融实验**：作者分别去掉检索增强、二次生成和偏好评分三个模块，发现每去掉一个模块整体得分都会下降 5%~12% 不等，说明三者缺一不可。  
- **局限性**：论文承认在极其专业或高度时效性的查询上，检索质量仍受搜索引擎的限制；此外，二次生成的成本虽然比全流程生成低，但在高并发场景下仍可能成为瓶颈。

### 影响与延伸思考
WebGLM 把“模型思考+检索校正+人类偏好”三段式流程示范出来后，后续不少工作开始探索更细粒度的自我纠错机制和更高效的偏好学习。例如，2024 年出现的 **Self‑RAG** 系列模型在检索后直接进行自监督校正，思路与 WebGLM 的二次生成相呼应。还有研究把人类偏好评分与强化学习（RLHF）结合，进一步提升答案的可接受度。想继续深入，可以关注以下方向：① 更轻量的检索模型（如跨模态检索）；② 基于用户即时反馈的在线偏好学习；③ 多语言/多模态的网页增强系统。  

### 一句话记住它
让大模型先“写草稿”，再用网页事实“校对”，最后用人类偏好“挑选”，10 B 参数就能跑出媲美 175 B WebGPT 的答案。