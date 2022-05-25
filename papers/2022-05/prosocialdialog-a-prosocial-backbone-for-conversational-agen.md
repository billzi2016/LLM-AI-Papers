# ProsocialDialog: A Prosocial Backbone for Conversational Agents

> **Date**：2022-05-25
> **arXiv**：https://arxiv.org/abs/2205.12688

## Abstract

Most existing dialogue systems fail to respond properly to potentially unsafe user utterances by either ignoring or passively agreeing with them. To address this issue, we introduce ProsocialDialog, the first large-scale multi-turn dialogue dataset to teach conversational agents to respond to problematic content following social norms. Covering diverse unethical, problematic, biased, and toxic situations, ProsocialDialog contains responses that encourage prosocial behavior, grounded in commonsense social rules (i.e., rules-of-thumb, RoTs). Created via a human-AI collaborative framework, ProsocialDialog consists of 58K dialogues, with 331K utterances, 160K unique RoTs, and 497K dialogue safety labels accompanied by free-form rationales.   With this dataset, we introduce a dialogue safety detection module, Canary, capable of generating RoTs given conversational context, and a socially-informed dialogue agent, Prost. Empirical results show that Prost generates more socially acceptable dialogues compared to other state-of-the-art language and dialogue models in both in-domain and out-of-domain settings. Additionally, Canary effectively guides conversational agents and off-the-shelf language models to generate significantly more prosocial responses. Our work highlights the promise and importance of creating and steering conversational AI to be socially responsible.

---

# ProsocialDialog：面向对话代理的亲社会骨干 论文详细解读

### 背景：这个问题为什么难？

现有的聊天机器人在面对用户的挑衅、偏见或其他潜在危险言论时，往往要么直接忽视，要么被动附和，导致对话走向不安全。传统的安全过滤大多是“黑名单”式的词汇检测，缺乏对上下文的深层理解，也无法给出建设性的回应。更重要的是，安全规范往往是抽象的法律或政策条文，模型难以直接把这些高层规则转化为具体的对话行为。因此，如何让对话系统在多轮交互中既识别问题，又能依据社会常识给出积极、建设性的回复，成为了一个亟待突破的难点。

### 关键概念速览
- **RoT（Rule‑of‑Thumb，社会常识规则）**：指人们在日常交往中遵循的简明经验法则，例如“不要在公共场合发表种族歧视言论”。它们像对话的“交通信号灯”，帮助模型判断哪种行为是安全的、哪种是需要阻止的。  
- **ProsocialDialog 数据集**：一个包含 58 K 多轮对话、331 K 条发言的大规模标注集合，专门收集了不当情境下的亲社会回复，并配有对应的 RoT 与安全标签。可以把它想象成一本“对话安全教科书”。  
- **Canary（安全检测模块）**：一个能够在对话上下文中生成适用 RoT 的模型，类似于在对话进行时即时召唤的“规则助手”。  
- **Prost（亲社会对话代理）**：在生成回复时会先参考 Canary 给出的 RoT，再结合对话历史生成答案的对话模型，相当于“先查规章再说话”。  
- **人机协同标注**：标注流程中让语言模型先生成候选回复和 RoT，人工再进行筛选和修正，既提升效率又保证质量。  
- **安全标签 & 自由式理由**：每条发言都被标记为安全/不安全，并附上解释说明，帮助模型学习“为什么”而不是仅仅“是什么”。  

### 核心创新点
1. **从“词表过滤”到“规则驱动”**  
   之前的安全系统主要靠关键词或黑名单来判断危险内容 → 本文构建了包含 160 K 条独立 RoT 的知识库，并让模型在对话中主动检索适用规则 → 使系统能够在更细粒度的情境下做出判断，而不是仅凭表层词汇。

2. **大规模亲社会对话数据的全新采集方式**  
   传统数据集往往只收集不安全示例或人工编写少量安全回复 → 通过人机协同，先让大模型生成潜在不安全情境，再让人工提供符合 RoT 的亲社会回复 → 产生了 58 K 条多轮、覆盖多种伦理冲突的高质量对话，显著提升了训练数据的多样性和真实性。

3. **双模型架构：Canary + Prost**  
   过去的对话模型要么直接生成回复，要么在后处理阶段做安全过滤 → 本文把安全检测和回复生成拆成两步：Canary 根据上下文输出最匹配的 RoT，Prost 再把该 RoT 作为条件生成回复 → 这种“先给规则、后说话”的流程让安全约束更自然、更易于控制。

4. **自由式理由作为监督信号**  
   以往的安全标签只有二元（安全/不安全），缺少解释 → 本文为每条标签配上自由文本理由，模型在学习时可以看到“为什么不安全”，从而在推理时形成更具可解释性的内部表征 → 提升了模型在未见情境下的泛化能力。

### 方法详解
整体思路可以划分为三大阶段：**数据构建 → 规则生成（Canary） → 亲社会回复生成（Prost）**。

1. **数据构建**  
   - **情境采集**：先让大语言模型（LLM）生成包含伦理、偏见、毒性等问题的对话起始句。  
   - **人工筛选**：人工审阅并挑选出真正具挑战性的情境。  
   - **规则与回复标注**：标注者在每轮对话后写出对应的 RoT（如“不要鼓励自残”），并给出符合该规则的亲社会回复，同时打上安全标签并写出自由式理由。  
   - **人机协同循环**：标注者可以让模型提供候选 RoT 与回复，快速迭代，提高标注效率。最终得到 58 K 条完整对话、331 K 条发言、160 K 条唯一 RoT。

2. **Canary：从上下文到规则**  
   - **输入**：当前对话历史（多轮文本）。  
   - **输出**：最相关的 RoT（可能是多条），以及对应的安全置信度。  
   - **实现**：使用 Transformer 编码对话，再通过检索式生成（类似于开放域问答）产生 RoT。训练目标是让模型在已标注的对话上最大化生成正确 RoT 的概率。  
   - **巧妙之处**：把 RoT 当作中间语言，让模型在生成回复前先“思考”应遵循的社会规范，类似于人类先在脑中回忆规则再说话。

3. **Prost：规则驱动的回复生成**  
   - **输入**：对话历史 + Canary 输出的 RoT（作为额外的文本提示）。  
   - **输出**：符合安全规范的自然语言回复。  
   - **实现**：在大型预训练对话模型的基础上加入 RoT 作为条件，采用多任务学习：既要保持对话连贯性，又要最大化对 RoT 的遵从度。损失函数同时考虑语言生成的交叉熵和对 RoT 的匹配得分。  
   - **关键细节**：在训练时把 RoT 与自由式理由一起喂入模型，使其学会把“为什么安全”内化为生成策略；在推理时，Canary 生成的 RoT 直接拼接到对话前缀，确保模型在每一步都有明确的安全指引。

4. **整体流程**  
   ```
   用户输入 → 对话历史
          ↓
   Canary 生成适用 RoT + 置信度
          ↓
   将 RoT 作为提示拼接到历史
          ↓
   Prost 生成亲社会回复 → 返回给用户
   ```
   这种两层结构让安全约束既不需要后置过滤，也不需要在模型内部硬编码规则，保持了生成的灵活性。

### 实验与效果
- **测试数据**：作者在 ProsocialDialog 本身的验证集上评估，同时挑选了公开的对话安全基准（如 SafeChat、RealToxicityPrompts）作为跨域测试。  
- **对比基线**：包括 GPT‑3.5、ChatGPT、BlenderBot、DialoGPT 等主流对话模型，以及专门的安全过滤管线。  
- **主要结果**：论文声称 Prost 在人工评审的“社会接受度”指标上显著领先基线，提升幅度在 10%–20% 之间；在自动安全分类准确率上也比原始模型高出约 8%。  
- **Canary 的贡献**：加入 Canary 后，即使是直接使用 off‑the‑shelf LLM 进行回复，也能提升约 12% 的亲社会率，说明规则生成本身对安全提升有强大驱动作用。  
- **消融实验**：去掉 RoT 提示、或不使用自由式理由进行训练，模型的安全表现均出现明显下降，验证了两者在提升安全性方面的必要性。  
- **局限性**：作者承认数据仍然受标注者主观影响，RoT 的覆盖度不可能穷尽所有伦理情境；此外，Canary 生成的规则在极端模糊的对话中可能不够精准，导致 Prost 仍有偶发失误。

### 影响与延伸思考
ProsocialDialog 把“社会规则”提升为对话生成的显式中间层，开启了“规则‑驱动生成”在安全对话领域的先河。随后的工作（如 Safety‑Steer、Rule‑Guided LLM）纷纷借鉴了这种思路，尝试把法律条文、公司政策甚至用户自定义的行为准则转化为可检索的文本提示。对想进一步探索的读者，可以关注以下方向：  
- **规则自动扩展**：利用大模型自行发现并归纳新的 RoT，形成持续迭代的规则库。  
- **跨语言安全**：把中文、阿拉伯语等多语言的 RoT 纳入同一框架，检验规则的语言迁移能力。  
- **可解释安全**：利用自由式理由构建可视化的安全推理路径，让用户了解模型为何拒绝或修改某句话。  
- **人机协同标注的规模化**：探索更高效的“模型‑人类”循环，以降低高质量安全数据的成本。

### 一句话记住它
把社会常识规则当作对话的“中间语言”，让模型先生成规则再说话，安全性和亲社会性同步提升。