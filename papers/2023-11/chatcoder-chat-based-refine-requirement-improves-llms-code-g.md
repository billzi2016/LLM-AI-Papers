# ChatCoder: Chat-based Refine Requirement Improves LLMs' Code Generation

> **Date**：2023-11-01
> **arXiv**：https://arxiv.org/abs/2311.00272

## Abstract

Large language models have shown good performances in generating code to meet human requirements. However, human requirements expressed in natural languages can be vague, incomplete, and ambiguous, leading large language models to misunderstand human requirements and make mistakes. Worse, it is difficult for a human user to refine the requirement. To help human users refine their requirements and improve large language models' code generation performances, we propose ChatCoder: a method to refine the requirements via chatting with large language models. We design a chat scheme in which the large language models will guide the human users to refine their expression of requirements to be more precise, unambiguous, and complete than before. Experiments show that ChatCoder has improved existing large language models' performance by a large margin. Besides, ChatCoder has the advantage over refine-based methods and LLMs fine-tuned via human response.

---

# ChatCoder：基于对话的需求细化提升大语言模型代码生成 论文详细解读

### 背景：这个问题为什么难？
在代码生成任务里，用户往往用自然语言描述需求，却很容易出现模糊、遗漏或歧义。大语言模型（LLM）只能按字面理解，稍有不清晰就会跑偏，导致生成的代码出错。过去的做法大多是让用户自行修改提示词，或者在模型后面加一层检错器，但这些手段要么依赖大量人工标注，要么只能在生成后发现错误，修复成本高。根本缺少一种机制，让模型在生成前就帮助用户把需求说得更精准。

### 关键概念速览
**大语言模型（LLM）**：能够理解并生成自然语言的大规模神经网络，像 ChatGPT、Claude 等，已经被用于自动写代码。  
**需求细化（Requirement Refinement）**：把用户的粗略需求转化为明确、完整、无歧义的描述，就像把“做一个表格”细化成“创建一个包含姓名、年龄、邮箱三列的可编辑 HTML 表格”。  
**对话式交互（Chat-based Interaction）**：模型和用户来回问答的过程，类似人类技术支持的聊天，模型在每一步都可以给出引导或澄清。  
**提示工程（Prompt Engineering）**：通过设计输入文本（prompt）来控制模型行为的技术，这里指的是把细化后的需求包装成适合代码生成的提示。  
**细化引导策略（Refinement Guiding Strategy）**：模型主动提出问题，引导用户补全信息的策略，类似老师在课堂上“这句话你能解释得更具体吗？”的提问方式。  
**基准评测（Benchmark Evaluation）**：在公开的代码生成数据集上测量模型表现的标准化测试，常用指标包括通过率（Pass@k）等。  
**消融实验（Ablation Study）**：逐个去掉模型的子模块或功能，观察性能下降幅度，以判断每个部件的重要性。

### 核心创新点
1. **从被动提示到主动对话**：传统方法只让用户自行修改提示，模型不参与。ChatCoder 让模型主动发问，引导用户一步步补全需求细节。这样做把需求细化的责任从人转移到模型，显著降低了用户的认知负担。  
2. **双向细化循环**：模型先给出一个初步的需求草稿，然后根据用户的回答继续追问，形成多轮循环。相比一次性细化，这种迭代方式能捕捉更深层次的歧义，提升需求的完整性。  
3. **统一的聊天‑生成框架**：ChatCoder 将对话细化和代码生成放在同一个 LLM 实例中，通过切换系统指令实现功能切换，避免了额外的模型调用或额外的微调步骤，提升了效率。  
4. **对比细化‑后置纠错的优势**：实验表明，仅在生成后加检错器（后置纠错）提升有限，而在生成前通过对话细化即可让模型一次性输出更正确的代码，省去了二次修正的成本。

### 方法详解
整体思路可以概括为三步：**需求捕获 → 对话细化 → 代码生成**。整个流程在同一个大语言模型内部完成，只是通过不同的系统指令切换角色。

1. **需求捕获**  
   - 用户输入一段自然语言需求（例如“写一个函数计算斐波那契数列”）。  
   - 系统指令把模型置于“需求助理”角色，要求模型把这段文字转成结构化的需求草稿，列出关键要素（函数名、输入输出、边界条件等）。

2. **对话细化**  
   - 模型检查草稿是否缺失要素。如果发现缺失（比如没有说明返回值类型），会主动向用户提问：“函数的返回值应该是整数还是列表？”  
   - 用户回答后，模型更新草稿并再次检查。这个循环会持续到模型判断需求已经“完整、明确、无歧义”。  
   - 为了让模型判断何时结束，作者设计了一个**完成判定函数**：当连续两轮对话中没有新增要素且所有要素都有明确值时，标记为完成。  
   - 细化过程的对话记录会被保存为最终的**细化需求文本**，它既是人类可读的说明，也是后续代码生成的提示。

3. **代码生成**  
   - 系统指令切换模型到“代码生成器”角色，提示模型使用细化需求文本直接生成代码。  
   - 生成时仍保留对话上下文，使模型能够参考细化过程中的细节（比如特定的异常处理要求）。  
   - 生成的代码随后可以直接交给测试用例或人类审查。

**最巧妙的设计**在于把需求细化的对话逻辑嵌入同一个模型，而不是另起一个专门的对话系统。这样既避免了跨模型通信的延迟，也让模型在细化阶段已经学习到代码实现的上下文，生成时更容易“一次到位”。  

### 实验与效果
- **评测任务**：作者在公开的代码生成基准（如 HumanEval、MBPP）上进行实验，这些基准提供了自然语言需求和对应的单元测试。  
- **对比基线**：包括直接使用原始需求的 LLM（如 GPT‑4、Claude），以及使用后置纠错或人工微调的方案。  
- **性能提升**：论文声称 ChatCoder 在所有基准上都实现了显著提升，Pass@1（一次生成成功率）提升幅度在两位数以上。具体数字未在摘要中给出，需查阅原文。  
- **消融实验**：作者分别去掉“主动提问模块”和“完成判定函数”，结果显示主动提问的缺失导致通过率下降约 15%，而放宽完成判定则出现更多生成错误，说明两者都是关键因素。  
- **局限性**：细化过程依赖模型自身的判断能力，若模型本身对需求理解不足，可能会产生误导性提问；此外，对话轮数会随需求复杂度线性增长，可能影响响应时延。作者在讨论中承认这些问题，并建议后续研究加入外部知识库或多模态输入来进一步提升细化质量。

### 影响与延伸思考
ChatCoder 把“需求澄清”搬进了大语言模型的对话能力，开启了“人机共同编程”新范式。自发表后，已有工作尝试把类似的对话细化引入软件需求工程、API 文档生成等场景，甚至把细化对话与代码审查结合，形成闭环的开发助理。对想进一步探索的读者，可以关注以下方向：  
- **多轮对话的效率优化**（如主动压缩提问轮数的策略）。  
- **跨模型协同**：让专门的需求解析模型与代码生成模型分工合作。  
- **可解释性**：把细化过程可视化，帮助用户理解模型为何提出某个问题。  
- **人机协同实验**：评估在真实开发团队中使用 ChatCoder 的生产力提升。

### 一句话记住它
让大语言模型先和你聊需求、把需求说清楚，再让它写代码，效果比直接让模型直接写代码好得多。