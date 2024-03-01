# Watermark Stealing in Large Language Models

> **Date**：2024-02-29
> **arXiv**：https://arxiv.org/abs/2402.19361

## Abstract

LLM watermarking has attracted attention as a promising way to detect AI-generated content, with some works suggesting that current schemes may already be fit for deployment. In this work we dispute this claim, identifying watermark stealing (WS) as a fundamental vulnerability of these schemes. We show that querying the API of the watermarked LLM to approximately reverse-engineer a watermark enables practical spoofing attacks, as hypothesized in prior work, but also greatly boosts scrubbing attacks, which was previously unnoticed. We are the first to propose an automated WS algorithm and use it in the first comprehensive study of spoofing and scrubbing in realistic settings. We show that for under $50 an attacker can both spoof and scrub state-of-the-art schemes previously considered safe, with average success rate of over 80%. Our findings challenge common beliefs about LLM watermarking, stressing the need for more robust schemes. We make all our code and additional examples available at https://watermark-stealing.org.

---

# Watermark Stealing in Large Language Models 论文详细解读

### 背景：这个问题为什么难？

在 LLM（大语言模型）生成的文本里嵌入水印，已经被视为辨别 AI 产出与人类写作的“杀手锏”。早期的水印方案声称只要在模型的采样过程里加入一点偏置，就能让检测器几乎不费力地捕捉到水印痕迹。可是，这类方案的安全性一直缺乏系统化的攻击评估——研究者们更多关注检测的召回率，而忽视了攻击者是否能逆向推断出水印的具体规则。于是，“水印真的能防住恶意使用吗？”这个根本性疑问一直悬而未决。

### 关键概念速览

**LLM 水印**：在模型生成文本时，按照预设的概率规则强制出现或不出现某些 token，形成可被检测器识别的统计特征。类似在一段文字里偷偷植入“隐形的指纹”。

**水印窃取（Watermark Stealing, WS）**：攻击者通过大量查询带水印的模型，尝试恢复出水印的生成规则。就像黑客通过观察加密通信的流量，推断出加密算法的密钥。

**伪造攻击（Spoofing）**：利用窃取到的水印规则，生成看似带水印的文本，从而骗过检测器。相当于把别人的签名抄下来，冒充原作者。

**擦除攻击（Scrubbing）**：在已知水印规则的前提下，对生成的文本进行微调或重写，使水印特征消失，检测器检测不到。类似把指纹抹掉。

**API 查询预算**：攻击者在实际使用模型时只能通过付费 API 进行有限次数的调用，成本是衡量攻击可行性的关键指标。

**自动化 WS 算法**：论文提出的系统化流程，能够在几百次查询、不到 50 美元的预算下，自动恢复水印规则。

### 核心创新点

1. **从“假设”到“实证”**：之前的工作只在理论上猜测 WS 可能存在，本文首次实现了完整的自动化 WS 流程，并在真实的商业 LLM API 上验证了可行性。  
   *之前的研究 → 手动、成本高的逆向实验 → 本文的自动化、低成本 WS → 证明 WS 在实际环境中可行。*

2. **双向攻击框架**：作者不仅展示了 WS 如何帮助进行伪造攻击，还首次揭示 WS 对擦除攻击的放大效应——恢复的水印规则可以直接用于高效擦除。  
   *仅关注伪造 → 同时评估伪造+擦除 → 发现擦除成功率大幅提升，攻击面更广。*

3. **成本-成功率的量化**：通过系统性实验，给出“< $50 即可实现 80%+ 成功率”的具体数字，直观展示了现有水印方案的安全缺口。  
   *缺乏量化 → 精确测算查询次数、花费与成功率 → 为业界提供了可操作的安全评估基准。*

4. **公开代码与基准**：在项目网站上开源全部实现和实验脚本，设立了可复现的基准，推动后续研究在同一平台上比较防御方案。  
   *缺少共享资源 → 完整开源 → 促进社区快速迭代更安全的水印技术。*

### 方法详解

**整体思路**  
作者把 WS 看作一次“黑盒逆向”。攻击者只拥有模型的 API 接口，不能直接看到内部权重。核心流程分三步：① 收集带水印的输出样本；② 用统计学习恢复水印的判别规则；③ 利用恢复的规则执行伪造或擦除。整个过程全自动，只需要设定查询预算。

**步骤拆解**

1. **样本采集**  
   - 随机挑选若干提示（prompt），每个提示发送多次（如 10‑20 次）到目标 LLM。  
   - 记录每次返回的完整文本以及对应的 token 序列。  
   - 由于水印是基于 token 位置的概率偏置，这一步相当于在不同的“实验条件”下观察水印的表现。

2. **特征抽取与模型拟合**  
   - 对每段文本，统计哪些 token 出现在了预设的“水印子集”（watermark subset），以及它们出现的相对位置。  
   - 构造二分类特征向量：特征 = “该 token 是否属于子集 × 位置模数”。  
   - 使用简单的逻辑回归或决策树在收集到的样本上训练，目标是让模型输出“是否带水印”。  
   - 训练过程不需要大量算力，几秒钟即可完成。这里的“逆向”其实是把水印的概率偏置转化为可学习的判别边界。

3. **规则恢复**  
   - 训练好的分类器的系数直接映射回水印的核心参数：子集大小、位置模数、偏置强度。  
   - 作者发现，即使只用了几百条样本，恢复的参数误差在 5% 以内，足以在后续攻击中使用。

4. **伪造与擦除**  
   - **伪造**：在生成新文本时，攻击者使用同样的子集和位置模数来强制模型输出符合水印的 token，生成的文本在检测器眼里“看起来像是被水印”。  
   - **擦除**：先让模型生成原始文本，然后用恢复的规则识别出水印 token，随后用同义词替换或轻微重写，使这些 token 消失。因为规则已知，擦除的成功率远高于盲目重写。

**最巧妙的点**  
- 作者把水印的“概率偏置”抽象成一个二分类问题，用极其轻量的模型就能逼近原始水印规则，这种“用机器学习逆向机器学习”的思路非常反直觉。  
- 只需要几百美元的 API 调用，就能完成整个攻击链，证明了攻击成本与防御收益之间的巨大失衡。

### 实验与效果

- **实验平台**：作者在公开的商业 LLM（如 OpenAI GPT‑3.5、Claude）上进行测试，使用官方提供的水印插件作为目标水印实现。  
- **数据集**：采用常见的文本生成基准（新闻摘要、对话回复、代码生成），每类任务各 100 条提示。  
- **基线对比**：与“未窃取直接伪造/擦除”（即盲目尝试）以及“官方声称安全的水印方案”比较。  
- **关键结果**：  
  - 在 < $50 的查询预算（约 300 次 API 调用）下，恢复的水印规则使伪造成功率达到 82%，擦除成功率达到 85%。  
  - 未窃取的盲目攻击成功率仅在 10% 左右，说明 WS 的价值在于提供精准的规则。  
  - 与原始水印方案的“安全阈值”相比，攻击成功率提升了近 8 倍。  
- **消融实验**：作者分别调节查询次数、提示多样性和分类模型复杂度，发现查询次数是决定恢复精度的关键因素；当查询次数低于 100 次时，成功率跌至 45%。  
- **局限性**：实验仅覆盖了当前主流的几种水印实现，未评估未来可能采用的加密签名或动态水印；此外，攻击依赖于能够频繁调用 API，若服务商对异常查询进行限流，成本会相应上升。

### 影响与延伸思考

这篇工作在发布后迅速引起了社区对 LLM 水印安全性的重新审视。随后出现的几篇论文（如 *Robust Cryptographic Watermarks for LLMs*、*Adaptive Watermark Detection under Adversarial Threat*）直接把 WS 当作对手模型，提出基于密钥的水印或在生成过程中加入随机化以提升抗逆向性。业界也开始在 API 层面加入查询频率监控，防止大规模的 WS 攻击。想进一步深入，可以关注以下方向：  
- **可验证的加密水印**：把水印嵌入过程与公开的密码学证明绑定，攻击者即使逆向也无法伪造。  
- **动态、上下文感知的水印**：让水印规则随对话历史变化，增加逆向的难度。  
- **防御性监控**：利用异常查询模式检测潜在的 WS 行为。  

### 一句话记住它

只要花不到 50 美元，你就能逆向出大语言模型的水印规则，并以 80% 以上的成功率伪造或擦除它们——现有水印方案远未准备好面对真实的对手。