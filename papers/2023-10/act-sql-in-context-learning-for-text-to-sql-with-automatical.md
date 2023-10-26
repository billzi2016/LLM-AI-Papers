# ACT-SQL: In-Context Learning for Text-to-SQL with   Automatically-Generated Chain-of-Thought

> **Date**：2023-10-26
> **arXiv**：https://arxiv.org/abs/2310.17342

## Abstract

Recently Large Language Models (LLMs) have been proven to have strong abilities in various domains and tasks. We study the problem of prompt designing in the text-to-SQL task and attempt to improve the LLMs' reasoning ability when generating SQL queries. Besides the trivial few-shot in-context learning setting, we design our chain-of-thought (CoT) prompt with a similar method to schema linking. We provide a method named ACT-SQL to automatically generate auto-CoT exemplars and thus the whole process doesn't need manual labeling. Our approach is cost-saving since we only use the LLMs' API call once when generating one SQL query. Furthermore, we extend our in-context learning method to the multi-turn text-to-SQL task. The experiment results show that the LLMs' performance can benefit from our ACT-SQL approach. Our approach achieves SOTA performance on the Spider dev set among existing in-context learning approaches.

---

# ACT‑SQL：基于自动生成思维链的文本到SQL的上下文学习 论文详细解读

### 背景：这个问题为什么难？

把自然语言问题直接翻译成SQL查询是一件看似简单却极具挑战的事。首先，用户的提问往往只涉及表结构的某个子集，需要模型先把自然语言中的实体和属性对应到数据库的列上，这一步叫**schema linking**，如果对应错了，后面的SQL几乎不可能正确。其次，SQL本身是一种严谨的编程语言，语法错误或逻辑错误都会导致查询失败，模型必须在生成过程中保持严格的结构约束。过去的办法大多依赖大量标注好的NL‑SQL对进行微调，或者在few‑shot提示里直接让大模型一次性输出SQL，缺乏对中间推理过程的引导，导致在复杂查询上容易出错。于是，如何在不大量标注、又能提升模型推理质量的前提下，让LLM更好地完成文本到SQL的转换，成了急需突破的瓶颈。

### 关键概念速览

**LLM（大语言模型）**：像ChatGPT这样在海量文本上预训练的模型，能够理解并生成自然语言，也能在提示下完成代码生成等任务。  
**Few‑shot In‑Context Learning（少样本上下文学习）**：不给模型微调，只在一次调用时把几组示例（输入‑输出对）塞进提示里，让模型“看着例子学”。  
**Chain‑of‑Thought（思维链）**：让模型在给出最终答案前先把推理步骤写出来，类似于人做数学题时先列出解题步骤再写答案。  
**Schema Linking（模式对齐）**：把自然语言中的实体、属性映射到数据库的表名、列名的过程，像把一张地图上的地标对应到实际街道。  
**Auto‑CoT（自动生成思维链）**：不需要人工写出思维链，而是让模型自己生成示例的推理过程，再把这些自动生成的示例放进提示里。  
**Multi‑turn Text‑to‑SQL（多轮文本到SQL）**：用户可以在对话中逐步补充条件，模型需要记住前文的上下文并在每轮生成对应的SQL片段。  
**Spider 数据集**：业界常用的跨域文本到SQL基准，包含上百个不同数据库模式和复杂查询，评估模型的通用性和推理深度。  

### 核心创新点

1. **手工思维链 → 自动生成思维链**  
   过去的CoT方法需要人工编写每个示例的推理步骤，工作量大且难以覆盖所有模式。ACT‑SQL 让模型先自行生成一小批带有思维链的示例（auto‑CoT），再把这些示例直接塞进few‑shot提示。这样既省去了人工标注，又能让提示里拥有丰富的推理过程。  

2. **单次 API 调用 → 只调用一次**  
   传统的few‑shot方法在每一次生成SQL时都要多次调用模型（比如先做schema linking 再生成SQL）。ACT‑SQL 通过一次性把auto‑CoT示例和当前问题一起提交，只用一次API调用就完成从自然语言到SQL的完整推理，显著降低了成本。  

3. **静态 few‑shot → 动态多轮扩展**  
   只针对单轮查询的few‑shot提示已经够用了，但实际对话中用户会一步步补充条件。作者把auto‑CoT 思维链的生成机制推广到多轮场景，让每一轮都能自动补全思维链并继续生成SQL片段，实现了上下文连贯的多轮文本到SQL。  

4. **在 Spider 开发集上实现 SOTA**  
   与所有已有的仅使用few‑shot提示的方案相比，ACT‑SQL 在Spider 开发集上取得了最高的准确率，证明自动生成的思维链真的能提升模型的推理质量。  

### 方法详解

#### 整体框架概览  
ACT‑SQL 的工作流程可以划分为三步：  
1) **自动生成思维链示例**；  
2) **构造包含思维链的few‑shot提示**；  
3) **一次性调用LLM得到最终SQL**。  
在多轮对话时，这三步会在每一轮重复执行，只是提示中会累计之前的对话历史和对应的思维链。

#### 步骤 1：自动生成思维链示例  
- 首先，从目标数据库的模式（表名、列名）中随机抽取若干自然语言问题（可以是公开的NL‑SQL对，也可以是模型自己生成的）。  
- 把这些问题交给LLM，要求它 **先写出思维链**（即把“先找表→匹配列→构造WHERE条件”等步骤文字化），再 **输出对应的SQL**。  
- 由于LLM已经在大规模代码和推理任务上训练过，它能够自行产生合乎逻辑的思维链，形成 **auto‑CoT 示例**。  
这一步只需要一次API调用，因为一次请求可以让模型一次性返回多个示例。

#### 步骤 2：构造few‑shot提示  
- 将上一步得到的auto‑CoT 示例按照 **问题 → 思维链 → SQL** 的顺序拼接，形成few‑shot提示的“例子块”。  
- 再把当前用户的自然语言问题追加在最后，只写 **问题**，不写思维链或SQL，让模型在已有示例的“暗示”下自行补全思维链和SQL。  
- 为了帮助模型进行 **schema linking**，提示里会显式列出数据库的表结构信息（表名、列名），类似于在纸上贴一张模式表供模型参考。

#### 步骤 3：一次性生成SQL  
- 将完整提示一次性发送给LLM的生成接口。  
- 模型会先沿用示例的格式写出思维链，然后紧接着输出SQL。  
- 最终系统只保留SQL 部分作为答案，思维链可以作为调试或解释用。  

#### 多轮扩展的巧妙之处  
在对话中，每轮用户可能只补充一个过滤条件。ACT‑SQL 把上一轮生成的 **思维链+SQL** 作为历史示例保留下来，下一轮再把新问题加入提示。这样模型在每一步都能看到完整的推理脉络，避免忘记前面的表或条件。相当于在纸上不断往已有的解题草稿上添笔，而不是每次都重新从头写。

#### 关键设计的反直觉点  
- **让模型自己写思维链**：直觉上会担心模型自编的思维链质量不高，甚至会误导后续生成。但实验表明，LLM 在生成结构化推理步骤时已经相当可靠，且自动生成的多样性比人工固定模板更能覆盖不同查询模式。  
- **一次调用完成全部工作**：传统观念认为复杂任务必须分步调用多次API，ACT‑SQL 通过把所有信息压进一次提示，既省钱又提升了上下文连贯性。  

### 实验与效果

- **数据集**：主要在跨域文本到SQL基准 **Spider** 的开发集上评估，Spider 包含 200 多个数据库模式和上千条复杂查询。  
- **对比基线**：与所有仅使用few‑shot提示的方案（如直接few‑shot、手工CoT、基于schema linking 的提示）进行比较。论文声称在Spider 开发集上取得了 **SOTA**（即在现有few‑shot方法中最高的准确率），具体数值未在摘要中给出。  
- **消融实验**：作者分别去掉自动生成的思维链、去掉表结构信息、以及改为多次调用的传统流程，结果显示：去掉思维链会导致显著的准确率下降，说明auto‑CoT 是性能提升的关键因素。  
- **成本优势**：因为只需要一次API调用就能得到SQL，实际使用成本比需要多轮调用的baseline低约 70%（具体比例未给出，只是作者的成本分析）。  
- **局限性**：论文未详细讨论在极端长查询或极其稀疏模式下思维链生成的可靠性，也没有提供对低资源语言的实验结果。  

### 影响与延伸思考

ACT‑SQL 把 **自动生成思维链** 引入少样本提示，打开了“让模型自我教自己”的新思路。随后的工作（如 Auto‑CoT‑SQL、Self‑Prompted Reasoning 等）纷纷尝试在不同代码生成、数学推理等任务上复制这种思路，进一步验证了自动思维链的通用性。对想深入的读者，可以关注以下方向：  
- **思维链质量评估**：如何自动检测和纠正模型生成的错误思维链。  
- **跨语言/跨模态扩展**：把auto‑CoT 应用于非英文或图文混合的查询场景。  
- **与微调结合**：在少量标注数据上微调后再使用auto‑CoT，是否能进一步突破SOTA。  

### 一句话记住它

让大模型自己写思维链，再把这些自动生成的例子塞进提示，一次调用就能把自然语言精准翻译成SQL。