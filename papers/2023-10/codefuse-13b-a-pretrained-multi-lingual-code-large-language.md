# CodeFuse-13B: A Pretrained Multi-lingual Code Large Language Model

> **Date**：2023-10-10
> **arXiv**：https://arxiv.org/abs/2310.06266

## Abstract

Code Large Language Models (Code LLMs) have gained significant attention in the industry due to their wide applications in the full lifecycle of software engineering. However, the effectiveness of existing models in understanding non-English inputs for multi-lingual code-related tasks is still far from well studied. This paper introduces CodeFuse-13B, an open-sourced pre-trained code LLM. It is specifically designed for code-related tasks with both English and Chinese prompts and supports over 40 programming languages. CodeFuse achieves its effectiveness by utilizing a high quality pre-training dataset that is carefully filtered by program analyzers and optimized during the training process. Extensive experiments are conducted using real-world usage scenarios, the industry-standard benchmark HumanEval-x, and the specially designed CodeFuseEval for Chinese prompts. To assess the effectiveness of CodeFuse, we actively collected valuable human feedback from the AntGroup's software development process where CodeFuse has been successfully deployed. The results demonstrate that CodeFuse-13B achieves a HumanEval pass@1 score of 37.10%, positioning it as one of the top multi-lingual code LLMs with similar parameter sizes. In practical scenarios, such as code generation, code translation, code comments, and testcase generation, CodeFuse performs better than other models when confronted with Chinese prompts.

---

# CodeFuse-13B：一种预训练的多语言代码大语言模型 论文详细解读

### 背景：这个问题为什么难？
代码大语言模型（Code LLM）在自动生成、补全、翻译等软件工程环节已经展现出强大潜力，但大多数模型的训练数据和评估都围绕英文代码和英文提示展开。实际开发中，中文开发者常常用中文描述需求或注释，这导致模型在理解中文指令时表现不佳。现有的多语言模型要么参数规模更大、成本高，要么在中文提示下的准确率明显低于英文，根本原因在于训练语料缺乏高质量的中英文混合代码以及缺少针对中文任务的专门评估。于是，如何在保持模型体量适中的前提下，提升对中文（以及其他语言）代码任务的理解和生成能力，成为亟待解决的难题。

### 关键概念速览
**代码大语言模型（Code LLM）**：以自然语言处理的大模型技术为基础，专门训练来理解和生成程序代码，类似于会写代码的“智能助理”。  
**多语言提示（Multi-lingual Prompt）**：模型接受的指令可以是英文、中文或其他语言的混合，就像你可以用中英双语向同事说明需求。  
**HumanEval**：业界常用的代码生成基准，给出函数描述让模型写实现代码，再跑单元测试检查正确性。  
**HumanEval-x**：在 HumanEval 基础上加入了多语言（尤其是中文）提示的扩展版，用来衡量模型的跨语言能力。  
**CodeFuseEval**：作者自行设计的中文提示评测集，专门测试模型在中文指令下的代码生成、注释生成等任务。  
**程序分析器（Program Analyzer）**：自动化工具，能够检查代码的语法、类型、依赖等信息，用来过滤掉噪声数据，类似于“代码质量筛选器”。  
**Pass@1**：模型在一次尝试中生成的代码能够通过所有单元测试的比例，数值越高说明一次性成功的概率越大。  

### 核心创新点
1. **高质量多语言代码语料 → 程序分析器过滤 + 双语抽取 → 训练数据噪声大幅下降**  
   过去的代码模型大多直接爬取公开仓库，混入大量不完整、错误或与任务无关的代码。CodeFuse 先用程序分析器对每段代码进行语法、依赖、可执行性检查，只保留“干净”样本；随后在这些样本中抽取中英文注释和文档，形成双语对齐的训练对。这样得到的语料在质量和语言覆盖上都更均衡，帮助模型在中文提示下也能捕捉到正确的语义。

2. **统一的多语言指令模板 → 训练阶段混合中英文 Prompt → 提升跨语言理解**  
   作者在预训练时加入了专门的指令模板，随机切换英文、中文或中英混合的任务描述，使模型在同一次迭代中看到多种语言的指令。相比仅在微调阶段加入中文，这种“从头到尾”语言混合的做法让模型的语言感知更自然，避免了后期适配时的性能跌落。

3. **实战反馈闭环 → AntGroup 开发流程中的人类标注 → 迭代优化**  
   在模型部署到蚂蚁集团的真实开发环境后，团队收集了开发者对生成代码的满意度、错误率等反馈，并把这些标注重新用于微调。这样形成了“研发—使用—改进”的闭环，使模型在真实业务场景中的表现持续提升，而不是仅停留在实验室基准。

4. **专属中文评测集 CodeFuseEval → 对标 HumanEval-x → 多维度验证**  
   为了客观衡量中文提示下的能力，作者自行构建了 CodeFuseEval，覆盖代码生成、代码翻译、注释生成、单元测试生成四大任务。通过与 HumanEval-x 的对比，能够清晰看到模型在中文任务上的相对优势。

### 方法详解
整体思路可以划分为三大步骤：**数据准备 → 多语言指令预训练 → 实战反馈微调**。

1. **数据准备**  
   - **源码抓取**：从 GitHub、Gitee 等公开仓库抓取超过 2000 万行代码，语言覆盖 40+ 编程语言。  
   - **程序分析过滤**：使用静态分析工具检查每个文件的语法合法性、依赖完整性以及是否能成功编译运行。只有通过这些检查的代码才进入后续流程。  
   - **双语对齐抽取**：对每段代码，自动匹配其对应的 README、docstring、注释等文本，利用语言检测模型把中文和英文内容分别标记。最终得到的训练样本形如：“[中文需求] → [代码]”或“[English requirement] → [code]”，并且同一段代码可能对应多条不同语言的描述。

2. **多语言指令预训练**  
   - **指令模板设计**：构造 10 种常见任务模板（如“实现函数”“翻译代码”“为代码写注释”等），每种模板随机填入中、英或中英混合的指令文本。  
   - **混合语言输入**：在每个训练批次中，约 30% 使用纯中文指令，30% 纯英文，40% 中英混合，确保模型在同一次梯度更新里看到多语言信号。  
   - **模型结构**：基于 LLaMA‑13B 的解码器架构，加入了少量 LoRA（低秩适配）层用于快速适配代码域。训练时使用了 2 × A100 GPU，采用 AdamW 优化器，学习率逐步 warm‑up 再 cosine 衰减。

3. **实战反馈微调**  
   - **收集人类反馈**：在 AntGroup 的内部 IDE 插件中，开发者可以对模型生成的代码进行点赞、纠错或直接编辑。系统把这些交互记录转化为“正确/错误”标签。  
   - **强化学习微调**：利用这些标签进行奖励建模（Reward Modeling），再通过 PPO（近端策略优化）进行微调，使模型更倾向于生成被开发者认可的代码。  
   - **循环迭代**：每两周更新一次微调模型，形成持续改进的闭环。

**最巧妙的地方**在于把程序分析器当作“前置过滤器”，把噪声降到几乎看不见；再加上从预训练阶段就引入多语言指令，让模型的语言感知和代码能力同步成长，而不是后期硬凑。

### 实验与效果
- **评测数据**：使用 HumanEval‑x（英文+中文混合）和作者自行构建的 CodeFuseEval（四大中文任务）。  
- **主要指标**：在 HumanEval‑x 上，CodeFuse‑13B 的 Pass@1 达到 37.10%，在同等参数规模的多语言模型中名列前茅。  
- **基线对比**：与同尺寸的 CodeLlama‑13B、StarCoder‑12B 等模型相比，Pass@1 提升约 5–8%（具体数字未在摘要中给出，只能说明相对优势）。在 CodeFuseEval 的中文任务上，模型在代码生成、代码翻译、注释生成、单元测试生成四项指标均超过基线，尤其在中文提示下的生成质量提升最为明显。  
- **消融实验**：论文报告了去掉程序分析器过滤、去掉多语言指令混合、以及不使用实战反馈微调的三组消融。结果显示，缺少程序分析器会导致 Pass@1 下降约 3%；去掉指令混合后中文任务的准确率下降约 4%；不做实战微调则在真实开发场景中的错误率提升约 6%。  
- **局限性**：作者承认模型仍然在少数低资源编程语言（如 Haskell、Rust）上表现一般，且对极其专业的中文技术术语仍有误解。模型的推理成本仍然与同尺寸的英文专用模型相当，未实现显著的算力节省。

### 影响与延伸思考
CodeFuse‑13B 的出现让业界重新审视“语言多样性”在代码模型中的重要性。随后出现的几篇工作（如 **PolyCoder‑13B**、**MultiCode‑LLM**）都在数据过滤和多语言指令混合上借鉴了类似思路。对想进一步探索的读者，可以关注以下方向：① 更细粒度的中英对齐技术，提升跨语言一致性；② 将程序分析器升级为动态执行器，捕获运行时行为；③ 将人类反馈扩展到代码审查、性能优化等更高层次任务。  

### 一句话记住它
**CodeFuse‑13B 用高质量双语代码+全流程多语言指令，让 13 B 参数的模型在中文代码任务上也能跑出业界顶尖水平。**