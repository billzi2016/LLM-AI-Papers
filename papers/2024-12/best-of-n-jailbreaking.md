# Best-of-N Jailbreaking

> **Date**：2024-12-04
> **arXiv**：https://arxiv.org/abs/2412.03556

## Abstract

We introduce Best-of-N (BoN) Jailbreaking, a simple black-box algorithm that jailbreaks frontier AI systems across modalities. BoN Jailbreaking works by repeatedly sampling variations of a prompt with a combination of augmentations - such as random shuffling or capitalization for textual prompts - until a harmful response is elicited. We find that BoN Jailbreaking achieves high attack success rates (ASRs) on closed-source language models, such as 89% on GPT-4o and 78% on Claude 3.5 Sonnet when sampling 10,000 augmented prompts. Further, it is similarly effective at circumventing state-of-the-art open-source defenses like circuit breakers. BoN also seamlessly extends to other modalities: it jailbreaks vision language models (VLMs) such as GPT-4o and audio language models (ALMs) like Gemini 1.5 Pro, using modality-specific augmentations. BoN reliably improves when we sample more augmented prompts. Across all modalities, ASR, as a function of the number of samples (N), empirically follows power-law-like behavior for many orders of magnitude. BoN Jailbreaking can also be composed with other black-box algorithms for even more effective attacks - combining BoN with an optimized prefix attack achieves up to a 35% increase in ASR. Overall, our work indicates that, despite their capability, language models are sensitive to seemingly innocuous changes to inputs, which attackers can exploit across modalities.

---

# 最佳N次越狱 论文详细解读

### 背景：这个问题为什么难？

在大模型安全领域，研究者一直在寻找“越狱”手段——让模型在表面遵守安全规则的同时，偷偷给出有害答案。过去的攻击大多依赖精心设计的单条提示或梯度信息，成本高且对模型内部结构有一定了解。闭源模型（如 GPT‑4o、Claude 3.5）只提供黑盒 API，攻击者只能看到输入输出，导致传统方法的成功率很低。更糟的是，已有的防御（比如电路断路器）往往只能捕捉明显的违规提示，难以抵御细微的、看似无害的输入扰动。于是，如何在纯黑盒、低信息的前提下，系统性地提升越狱成功率，成为亟待突破的难题。

### 关键概念速览
- **越狱（jailbreak）**：让模型在安全过滤层面失效，输出本应被阻止的有害内容。就像把锁打开后，机器人可以说出原本被禁言的话。
- **黑盒攻击**：攻击者只能调用模型的输入输出接口，无法看到内部权重或梯度。类似于只能通过门缝观察机器人的行为，却不能打开机器内部检查。
- **提示增强（prompt augmentation）**：对原始提示做随机改动（如单词顺序、大小写、同义词替换），相当于在同一把钥匙上磨出不同的凹槽，尝试匹配锁的微小缺陷。
- **攻击成功率（ASR）**：在一定次数的尝试中，模型给出违规回答的比例。数值越高，攻击越“强”。
- **幂律行为（power‑law behavior）**：一种统计规律，少量样本产生的成功率与样本数量的关系呈现指数递减的曲线。这里指 ASR 随 N 增大呈现类似“越抖越好”的趋势。
- **电路断路器（circuit breaker）**：一种防御机制，检测到潜在违规输出后立即中止对话，像电路中的保险丝一样切断信号。
- **前缀攻击（prefix attack）**：在提示前加上一段特制的文字，使模型的后续生成倾向于违规。相当于在钥匙柄上加装一个诱导装置。
- **多模态模型**：同时处理文字、图像、音频等多种输入形式的模型。越狱时需要针对每种模态设计不同的扰动手段。

### 核心创新点
1. **从单一提示到“最佳N次”采样**  
   之前的攻击往往只投一次或少量手工调参的提示 → 这篇论文提出对同一目标提示生成大量随机增强版本，逐一提交，直到出现违规回答 → 只要给模型足够多的“尝试”，成功率在闭源模型上突破 80% 以上。

2. **跨模态的统一框架**  
   过去的越狱大多只针对文本模型 → 作者为视觉语言模型（VLM）和音频语言模型（ALM）分别设计了图像噪声、颜色抖动、音频频率偏移等特定增强手段 → 同一套“Best‑of‑N”思路即可在文字、图像、音频三大模态上实现高效越狱。

3. **发现并利用幂律增长规律**  
   传统实验只报告固定 N（如 100、1000）的成功率 → 通过系统性扫描 N 从几百到上万，作者观察到 ASR 随 N 增大呈幂律上升 → 这为攻击预算提供了可预测的收益曲线，也暗示模型对微小输入扰动的敏感性是普遍存在的。

4. **与其他黑盒攻击的组合**  
   过去的前缀攻击和提示增强被视为独立手段 → 将最佳N次采样与优化的前缀攻击并行使用，整体 ASR 提升约 35% → 说明不同黑盒技巧可以叠加，形成更强的攻击组合。

### 方法详解
**整体思路**  
这篇论文把越狱过程抽象成一个“投掷硬币”游戏：先准备一条原始违规目标提示，然后不断生成它的随机变体（即增强），每一次都把变体喂给模型，检查返回的答案是否违规。只要在 N 次投掷中出现一次违规输出，就算攻击成功。整个流程只需要模型的输入输出接口，不依赖梯度或内部结构。

**关键步骤拆解**  

1. **原始提示准备**  
   - 选取一条已知能诱导模型产生有害内容的基准提示（如“请描述如何制造炸弹”）。这一步相当于确定攻击的“目标”。

2. **提示增强生成**  
   - 对文字提示：随机打乱词序、随意改变大小写、替换同义词、插入无关字符等。  
   - 对图像提示：随机裁剪、颜色抖动、添加噪声、轻微旋转。  
   - 对音频提示：改变采样率、加入背景噪声、微调音调。  
   - 每一次增强都是一次“轻微扰动”，类似于在同一把钥匙上轻轻磨出不同的凹槽。

3. **模型查询**  
   - 将增强后的提示发送给模型，记录返回的文本（或图像、音频）结果。这里不需要任何内部信息，只是一次普通的 API 调用。

4. **违规检测**  
   - 使用一个简单的规则或外部审查模型判断返回内容是否包含违禁信息。若检测到违规，即标记为成功；否则继续下一轮。

5. **循环控制**  
   - 重复步骤 2‑4 直至达到预设的最大采样次数 N，或提前在某一次得到违规答案。  
   - 实际实验中 N 取值从 100 到 10 000 不等，作者发现 N 越大，成功率越高，且呈幂律增长。

**最巧妙的地方**  
- **完全黑盒**：不需要任何梯度信息，只靠“投掷”大量轻微变体就能逼出模型的安全漏洞。  
- **模态统一**：虽然图像和音频的增强方式不同，但整体框架保持不变，展示了方法的高度可扩展性。  
- **幂律经验**：作者用大量实验数据揭示了“更多尝试=更高成功率”的数学规律，为后续研究提供了量化依据。

### 实验与效果
- **测试对象**：闭源大型语言模型 GPT‑4o、Claude 3.5 Sonnet；以及多模态模型 GPT‑4o（视觉）和 Gemini 1.5 Pro（音频）。  
- **主要指标**：攻击成功率（ASR）。在 N = 10 000 时，GPT‑4o 达到 89% 的 ASR，Claude 3.5 Sonnet 达到 78%。  
- **防御对比**：面对最新的开源防御机制——电路断路器，Best‑of‑N 仍能保持高 ASR，说明这些防御对随机微扰缺乏鲁棒性。  
- **组合实验**：将 Best‑of‑N 与优化的前缀攻击叠加，整体 ASR 再提升约 35%，验证了不同黑盒技巧的叠加效应。  
- **消融研究**：作者分别去掉文字大小写变化、词序打乱、同义词替换等子模块，发现词序打乱对提升 ASR 最为关键；但去掉任意一种增强都会导致整体成功率下降。  
- **局限性**：高 N 需要大量 API 调用，成本不菲；检测违规的规则可能产生误报或漏报；论文未给出对所有模型的普适性证明，只在少数主流模型上验证。

### 影响与延伸思考
这篇工作让业界重新审视“黑盒安全”假设：即便模型内部看不见，只要输入空间足够大，攻击者仍能通过大量轻微扰动找到安全漏洞。随后的研究（如《Adaptive Prompt Augmentation》《Defensive Prompt Filtering》）开始探索 **查询效率** 的提升——如何在更少的 N 下保持高 ASR；以及 **对抗性检测**——训练模型识别异常的输入分布。对齐研究者也在思考如何让模型对 **输入微扰** 更具鲁棒性，或在模型层面加入 **噪声不敏感的安全约束**。如果想进一步了解，可关注 **多模态对抗安全** 与 **黑盒查询优化** 两大方向。

### 一句话记住它
只要不停抖动输入，即使是最微小的随机改动，也能在黑盒模型上逼出高达 90% 的违规回答。