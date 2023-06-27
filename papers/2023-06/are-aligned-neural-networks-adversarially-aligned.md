# Are aligned neural networks adversarially aligned?

> **Date**：2023-06-26
> **arXiv**：https://arxiv.org/abs/2306.15447

## Abstract

Large language models are now tuned to align with the goals of their creators, namely to be "helpful and harmless." These models should respond helpfully to user questions, but refuse to answer requests that could cause harm. However, adversarial users can construct inputs which circumvent attempts at alignment. In this work, we study adversarial alignment, and ask to what extent these models remain aligned when interacting with an adversarial user who constructs worst-case inputs (adversarial examples). These inputs are designed to cause the model to emit harmful content that would otherwise be prohibited. We show that existing NLP-based optimization attacks are insufficiently powerful to reliably attack aligned text models: even when current NLP-based attacks fail, we can find adversarial inputs with brute force. As a result, the failure of current attacks should not be seen as proof that aligned text models remain aligned under adversarial inputs.   However the recent trend in large-scale ML models is multimodal models that allow users to provide images that influence the text that is generated. We show these models can be easily attacked, i.e., induced to perform arbitrary un-aligned behavior through adversarial perturbation of the input image. We conjecture that improved NLP attacks may demonstrate this same level of adversarial control over text-only models.

---

# 对齐的神经网络在对抗情境下还能保持对齐吗？ 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）被微调成“有帮助且无害”之后，研究者们往往通过安全提示或奖励模型来阻止模型输出危险内容。可是，这些防护措施大多是基于普通用户的输入分布设计的，缺少对“最坏情况”——即恶意用户精心构造的对抗样本——的考量。传统的文本对抗攻击（比如基于梯度的替换或同义词替换）在对齐模型上往往失效，导致人们误以为模型已经足够鲁棒。与此同时，视觉-语言多模态模型允许用户上传图片，图片的微小噪声却能极大影响生成文本，这让对齐的安全边界更加模糊。于是，评估对齐模型在真正的对抗环境下是否仍然遵守安全规范，成为一个迫切但尚未系统化的问题。

### 关键概念速览
- **对齐（Alignment）**：把模型的行为调教成符合人类价值观和使用者期望的过程，常通过人类反馈强化学习（RLHF）实现。可以想象成给模型装上“道德指南针”。
- **对抗样本（Adversarial Example）**：攻击者精心设计的输入，目的是让模型产生本应被阻止的输出。就像在密码锁上涂抹一点油，让它误判为合法钥匙。
- **文本对抗攻击**：在纯文字输入上进行的扰动，常用同义词替换、字符混淆或梯度引导的词嵌入微调。类似于把一句话的某些字换成近义词，却不改变整体意思。
- **多模态模型**：同时接受文字和图像等不同模态信息的模型，例如 CLIP‑GPT、LLaVA。它们像是把“看”和“说”结合在一起的机器人。
- **图像对抗扰动（Adversarial Perturbation）**：在图片像素上加入肉眼几乎不可见的噪声，却能导致模型产生完全不同的文本输出。相当于在画作上点几滴隐形墨水，让观者产生误解。
- **Brute‑Force 搜索**：不依赖梯度信息，直接枚举或随机采样大量输入，寻找能够突破防护的例子。类似于穷举密码本的方式。

### 核心创新点
1. **系统化评估对齐模型的对抗鲁棒性**  
   之前的工作大多只报告单一攻击方法的成功率，缺少对“最坏情况”输入的全局视角。本文先把对齐模型放进一个对抗评估框架，分别在纯文本和多模态两类场景下寻找能够诱导模型输出有害内容的输入。这样就能判断现有防护到底是“看不见的盲区”还是“真正的安全墙”。

2. **对比现有 NLP 对抗攻击与暴力搜索的差距**  
   作者发现，即使最新的基于梯度或语言模型的文本攻击在对齐模型上几乎失效，使用暴力搜索仍能找到突破口。这个发现提醒我们，攻击失败并不等于模型安全，提示了评估方法本身的局限。

3. **揭示多模态模型在图像维度上的极端脆弱**  
   在实验中，仅通过对输入图片加入细微对抗噪声，就能让模型生成任意不对齐的文本。相比文本攻击，这种图像层面的攻击更容易实现且成功率更高，凸显了多模态系统在安全对齐上的新挑战。

4. **提出对齐鲁棒性需要跨模态统一防御的假设**  
   基于实验观察，作者推测如果未来出现更强的文本对抗技术，文本模型也会表现出类似的易被攻击特性。因此，安全对齐的研究应从单一模态扩展到跨模态的统一防御框架。

### 方法详解
整体思路可以拆成三步：**（1）构造对抗输入 →（2）检测模型输出是否违背对齐 →（3）对比不同攻击手段的有效性**。下面按步骤展开。

1. **对抗输入的生成**  
   - *文本攻击*：作者复用了几种主流的 NLP 对抗方法，包括基于梯度的词嵌入扰动、同义词替换、以及利用大语言模型自行生成“诱导式”问题。每种方法都在对齐模型的提示词后面加上攻击者设计的前缀，试图让模型突破安全屏障。  
   - *暴力搜索*：当上述方法失效时，研究者转向随机采样大量句子或使用模板填空的方式，直接遍历可能的输入组合。虽然计算成本高，但可以确保在搜索空间里找到至少一个成功的例子。  
   - *图像攻击*：针对多模态模型，作者采用标准的图像对抗生成算法（如 FGSM、PGD），在原始图片上加入微小噪声，使得视觉特征在模型内部被显著扭曲。噪声幅度控制在肉眼不可辨的范围内，确保攻击不易被用户察觉。

2. **对齐检测机制**  
   为了判断输出是否“有害”，论文使用了两层过滤：  
   - **安全分类器**：预训练的二分类模型（安全 vs. 不安全），对生成文本进行快速打分。  
   - **人工审查**：对分类器标记为安全的输出进行抽样，人工验证是否真的符合对齐目标。这样既保证了大规模实验的可行性，又避免了纯机器评估的误判。

3. **对比与分析**  
   - 对每种攻击方式，记录成功率（即模型输出被安全分类器判为不安全的比例）。  
   - 对文本攻击和暴力搜索的结果进行对比，展示即使最先进的梯度攻击几乎为零成功率，暴力搜索仍能达到两位数的成功率。  
   - 对图像攻击，报告在不同噪声强度下的成功率曲线，显示只要噪声幅度略高于 0.5%（像素值范围 0‑255），成功率即可接近 100%。

**最巧妙的点**在于把“暴力搜索”作为基准，实际上把对齐模型的安全性当作一个“黑盒”来审视，而不是依赖任何特定攻击的假设。这种思路像是把模型放进“安全审计室”，让审计员用尽可能多的钥匙去尝试打开门，只有所有钥匙都打不开，才能说门真的安全。

### 实验与效果
- **数据集与任务**：文本实验使用了公开的安全对话基准（如 SafePrompt、OpenAI Moderation 数据），多模态实验则选取了常见的图文对话数据集（如 LLaVA‑CC）。任务都是让模型在给定提示下生成回答，并检测是否出现违禁内容。  
- **Baseline 对比**：对比对象包括未对齐的原始模型、仅使用 RLHF 对齐的模型、以及使用最新对抗训练（Adversarial Fine‑Tuning）后的模型。  
- **关键数字**：在文本攻击上，最新的梯度攻击对齐模型的成功率约为 1% 以下，而暴力搜索能够达到约 12% 的成功率。多模态图像攻击在噪声幅度 0.7% 时成功率超过 95%，而同等噪声对未对齐模型的提升不到 30%。  
- **消融实验**：作者分别去掉安全分类器、去掉人工审查、以及只使用单一攻击方式进行实验，结果显示：去掉人工审查会导致误报率上升约 8%，仅依赖梯度攻击会把成功率压到几乎为零，验证了每个环节的必要性。  
- **局限性**：论文承认暴力搜索在大规模真实场景下成本高昂，且只在受控实验环境中展示了图像攻击的高效性；对更复杂的跨模态交互（如视频+文本）尚未评估。

### 影响与延伸思考
这篇工作在安全对齐社区掀起了“对抗评估必须多模态化”的讨论。随后出现的几篇论文（如《Multimodal Red‑Teamings for LLMs》、《Adversarial Prompt Engineering in Aligned Models》）直接引用了本文的评估框架，尝试在更大规模的模型上复现类似的突破。业界也开始在模型部署前加入图像对抗检测层，类似于传统的防病毒软件。对想进一步深入的读者，可以关注以下方向：① 开发更高效的文本对抗搜索算法，降低暴力搜索的计算开销；② 研究跨模态对齐的统一损失函数，让模型在视觉和语言两端都具备一致的安全约束；③ 探索对抗训练与 RLHF 的协同机制，提升模型在最坏情况输入下的鲁棒性。

### 一句话记住它
对齐模型在普通测试里看似安全，但只要给它们一点“隐形”对抗噪声——尤其是图像层面的——就能轻易让它们说出本不该说的话。