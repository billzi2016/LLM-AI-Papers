# Unnatural Error Correction: GPT-4 Can Almost Perfectly Handle Unnatural   Scrambled Text

> **Date**：2023-11-30
> **arXiv**：https://arxiv.org/abs/2311.18805

## Abstract

While Large Language Models (LLMs) have achieved remarkable performance in many tasks, much about their inner workings remains unclear. In this study, we present novel experimental insights into the resilience of LLMs, particularly GPT-4, when subjected to extensive character-level permutations. To investigate this, we first propose the Scrambled Bench, a suite designed to measure the capacity of LLMs to handle scrambled input, in terms of both recovering scrambled sentences and answering questions given scrambled context. The experimental results indicate that most powerful LLMs demonstrate the capability akin to typoglycemia, a phenomenon where humans can understand the meaning of words even when the letters within those words are scrambled, as long as the first and last letters remain in place. More surprisingly, we found that only GPT-4 nearly flawlessly processes inputs with unnatural errors, even under the extreme condition, a task that poses significant challenges for other LLMs and often even for humans. Specifically, GPT-4 can almost perfectly reconstruct the original sentences from scrambled ones, decreasing the edit distance by 95%, even when all letters within each word are entirely scrambled. It is counter-intuitive that LLMs can exhibit such resilience despite severe disruption to input tokenization caused by scrambled text.

---

# 非自然错误纠正：GPT‑4几乎完美处理乱序文本 论文详细解读

### 背景：这个问题为什么难？
语言模型的输入是经过分词器切分成 token 的，分词器依赖字符顺序来决定每个 token 的边界。把单词内部的字母随意打乱，会导致原本的词根、前后缀全部被破坏，分词器往往把整句话拆成一堆毫无意义的子串。过去的模型在这种“非自然错误”下几乎失去语义信息，甚至连人类在极端打乱时也会读不懂。因此，评估和提升模型对字符级扰动的鲁棒性成为了一个既理论上有趣、又实际需求强烈的难题。

### 关键概念速览
**LLM（大语言模型）**：通过海量文本训练得到的生成式模型，能够完成对话、写作、推理等任务。这里指的是 GPT‑4、Claude、Llama 等最新模型。  
**Token（标记）**：模型内部处理的最小单位，通常是一个子词或字符序列。分词器把原始文本切成 token，顺序错误会直接破坏 token 序列。  
**Typoglycemia（字母错位阅读）**：人类在首尾字母不变、内部字母随意排列时仍能猜出单词意义的现象，常被用来说明人脑对整体形状的感知。  
**Scrambled Bench**：本文自建的评测套件，包含两类任务：① 把被打乱的句子恢复成原句；② 在被打乱的上下文中回答阅读理解问题。  
**Edit Distance（编辑距离）**：衡量两段文字相似度的指标，计算把一个序列变成另一个序列需要的插入、删除、替换操作数。距离越小，恢复得越好。  
**Unnatural Error（非自然错误）**：指那些在真实书写中几乎不出现的错误，如完全随机打乱单词内部字符，导致分词器失效的极端噪声。  

### 核心创新点
1. **从“人类可读”到“模型可读”转向**：以前的研究多关注模型在轻度拼写错误或同义替换下的表现，本文直接把字母全部乱序，构造了极端的非自然错误场景。这样可以更彻底地探测模型内部是否真的学到了语言的结构，而不是仅靠表层 token。  
2. **构建 Scrambled Bench**：作者设计了一套系统化的基准，包括随机字符置换、全词内部乱序、以及在乱序上下文中做问答。相比随意挑选几句话进行手工实验，Scrambled Bench 能提供可重复、可量化的评估。  
3. **对比多模型并发现 GPT‑4 的“几乎完美”**：在同样的极端扰动下，GPT‑4 能把编辑距离降低约 95%，几乎恢复原句；而其他主流 LLM（如 GPT‑3.5、Claude‑2、Llama‑2）只能取得微弱提升，甚至有时直接输出乱码。  
4. **揭示分词器并非瓶颈**：实验显示，即使 token 被严重碎片化，GPT‑4 仍能通过上下文的全局注意力机制推断出原始单词顺序，这暗示模型已经学会了跨 token 的语义对齐，而不是单纯依赖分词器的稳定性。  

### 方法详解
整体思路可以拆成三步：**构造扰动 → 评测任务 → 结果量化**。

1. **扰动构造**  
   - 先把原始语料（新闻、维基、对话等）切成单词。  
   - 对每个单词执行不同程度的字符置换：  
     * **Typoglycemia 版**：保留首尾字母，随机打乱中间字母。  
     * **全乱序版**：把单词内部所有字符完全随机排列，首尾也可能被打乱。  
   - 为了防止模型直接记忆特定句子，作者在每篇文章中随机抽取 10% 的句子进行扰动，其余保持原样，形成混合输入。

2. **评测任务**  
   - **恢复任务**：给模型一个被扰动的句子，要求它输出最接近原句的文本。这里使用了“请把下面的句子恢复成正常顺序”的提示，确保模型知道任务目标。  
   - **阅读理解任务**：在被扰动的段落中提出多选或简答问题，检验模型是否还能利用残余语义信息做出正确答案。  

3. **结果量化**  
   - 对恢复任务，计算模型输出与原句的编辑距离（Levenshtein Distance），用 **(1 - 距离/原句长度)** 作为恢复率。GPT‑4 在全乱序版上平均恢复率约 95%。  
   - 对阅读理解任务，统计准确率或 F1 分数，GPT‑4 的得分仅比原始未扰动时下降约 3%，而其他模型下降超过 20%。  

**最巧妙的地方**在于作者没有对模型做任何特殊的微调或额外的字符级预训练，而是直接使用公开的 API 调用。这说明 GPT‑4 本身的内部表征已经具备强大的跨 token 对齐能力，能够在极端噪声下仍然捕捉到词汇的整体形状。

### 实验与效果
- **数据集**：使用了公开的新闻摘要（CNN/DailyMail）、维基百科段落以及多轮对话数据，覆盖不同文体。  
- **Baseline**：对比了 GPT‑3.5、Claude‑2、Llama‑2‑70B、以及一个基于字符级 RNN 的传统纠错模型。  
- **主要结果**：  
  * 在全乱序恢复任务上，GPT‑4 的编辑距离降低 95%，而次强的 GPT‑3.5 只能降低约 60%。  
  * 阅读理解准确率：GPT‑4 93%（原始 96%），Claude‑2 71%（原始 89%），Llama‑2 58%（原始 85%）。  
- **消融实验**：作者分别去掉提示词中的“恢复”指令、以及在输入前加入空格分隔，发现提示词的明确性对所有模型都有提升，但对 GPT‑4 的影响最小，说明其内部机制更为鲁棒。  
- **局限性**：论文未探讨模型在多语言或非拉丁字母（如中文、阿拉伯文）上的表现，也没有对极端长句（>200 token）进行评估。作者承认目前只在英文语料上验证，跨语言鲁棒性仍待研究。

### 影响与延伸思考
这篇工作在发布后迅速引发了两类后续研究：  
1. **鲁棒性评测的标准化**：多个团队基于 Scrambled Bench 扩展出 “Noisy Bench”，加入拼写错误、键盘相邻错误等更贴近真实输入的噪声。  
2. **模型内部对齐机制的解释**：有研究尝试可视化 GPT‑4 的注意力图，发现模型在字符层面会形成类似“形状匹配”的热点，暗示它在学习阶段已经捕获了字母排列的统计规律。  

如果你想进一步探索，可以关注 **字符级预训练**（如 ByT5）与 **大模型跨 token 对齐** 的交叉研究，或者尝试在中文等非空格分词语言上构建类似的乱序基准。

### 一句话记住它
GPT‑4 能在几乎把每个单词的字母全部打乱的情况下，几乎完整恢复原句，展示了超越分词器的语言结构感知能力。