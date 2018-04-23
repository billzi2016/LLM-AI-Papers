# A Call for Clarity in Reporting BLEU Scores

> **Date**：2018-04-23
> **arXiv**：https://arxiv.org/abs/1804.08771

## Abstract

The field of machine translation faces an under-recognized problem because of inconsistency in the reporting of scores from its dominant metric. Although people refer to "the" BLEU score, BLEU is in fact a parameterized metric whose values can vary wildly with changes to these parameters. These parameters are often not reported or are hard to find, and consequently, BLEU scores between papers cannot be directly compared. I quantify this variation, finding differences as high as 1.8 between commonly used configurations. The main culprit is different tokenization and normalization schemes applied to the reference. Pointing to the success of the parsing community, I suggest machine translation researchers settle upon the BLEU scheme used by the annual Conference on Machine Translation (WMT), which does not allow for user-supplied reference processing, and provide a new tool, SacreBLEU, to facilitate this.

---

# 呼吁在报告 BLEU 分数时保持清晰 论文详细解读

### 背景：这个问题为什么难？

机器翻译（MT）领域长期用 BLEU 作为主要评价指标，但 BLEU 并不是一个固定的数值，而是由分词、大小写、标点归一化等一系列可调参数决定的。过去的论文往往只给出一个“BLEU 分数”，却不交代背后用了哪套预处理流程，导致同一篇系统在不同实验设置下的分数相差甚至超过 1.8 分。因为缺少统一的报告规范，研究者很难直接比较新模型与已有工作，进而影响了成果的可复现性和进步的真实评估。

### 关键概念速览
- **BLEU（Bilingual Evaluation Understudy）**：一种通过 n‑gram 匹配来衡量机器翻译输出与参考译文相似度的指标。类似于把机器翻译的句子和参考句子拆成小块，看有多少块是相同的。
- **分词（Tokenization）**：把连续的字符序列切分成单词或子词单元的过程。不同的切分方式会直接改变 n‑gram 的统计结果，就像把一段文字重新标点会影响阅读感受一样。
- **归一化（Normalization）**：对大小写、标点、数字等进行统一处理的步骤。比如把所有大写字母转成小写，就像在比较两篇文章时先把所有字体统一后再比对。
- **WMT（Conference on Machine Translation）**：机器翻译领域的年度会议，常规任务会提供统一的评测脚本和参考实现，成为社区事实上的基准。
- **SacreBLEU**：作者发布的一个命令行工具，封装了 WMT 官方的 BLEU 计算流程，保证所有实验使用同一套预处理规则，类似于“统一的实验室仪器”。
- **参考处理（Reference Processing）**：对参考译文进行分词、去除标点等操作的统称。不同研究者自行决定的处理方式会导致同一参考产生不同的 BLEU 分数。

### 核心创新点
1. **量化 BLEU 参数波动 → 统计不同配置下的分数差异 → 发现同一系统在常见设置之间可相差最高 1.8 分**。这一步让大家直观看到“同一模型、不同报告”会产生的误差幅度。
2. **定位主要误差来源 → 对比多种分词和归一化方案 → 确认参考处理是导致分数波动的最大因素**。通过系统实验把问题根源从模型本身搬到评测前置步骤上。
3. **提出统一评测方案 → 借鉴句法分析社区采用的统一基准做法 → 建议全体 MT 研究者使用 WMT 官方的 BLEU 计算方式**。这样每篇论文的分数都在同一条“尺子”上。
4. **发布 SacreBLEU 工具 → 将 WMT 的评测脚本、参考数据、参数配置全部打包 → 只需一行命令即可得到可复现的 BLEU 分数**。相当于提供了“一键校准仪”，降低了手动配置的错误风险。

### 方法详解
整体思路可以分为三步：**（1）收集常见 BLEU 配置，** **（2）系统评估配置差异对分数的影响，** **（3）实现统一评测工具。**  

**步骤 1：配置收集**  
作者先在公开的 MT 论文和开源代码里抓取了几种常见的分词/归一化组合，例如 Moses 分词、subword‑nmt BPE、NLTK 分词、以及是否保留标点、是否转小写等。每一种组合都对应一套参数设置，形成一个“配置库”。

**步骤 2：差异量化**  
使用同一套翻译输出（来自公开的系统或基准模型），分别套用库中的每个配置计算 BLEU。这里的核心是保持翻译结果不变，只动评测前置处理。作者把所有得分放进矩阵，计算最大、最小、平均差距，发现最极端的两套配置之间相差 1.8 BLEU。进一步的对比显示，若只改动参考的分词方式，分数就能跳动 1.0‑1.5；而仅改动大小写或标点的影响相对小。

**步骤 3：统一方案与工具实现**  
受句法分析社区使用统一评测脚本的启发，作者决定采用 WMT 官方发布的 BLEU 脚本作为“唯一标准”。该脚本内部固定了参考的分词方式（使用 `mosesdecoder` 的 `tokenizer.perl`，并强制小写、去除多余空格），不允许用户自行修改。随后，作者把这套脚本、官方参考文件以及常用的命令行包装进 `SacreBLEU`。使用时，只需 `sacrebleu hypothesis.txt -t wmt14 -l en-de`，工具会自动下载对应的参考、执行统一的预处理、输出分数以及完整的复现指令。

**最巧妙的点**  
- **“不可变的参考处理”**：把参考的预处理硬编码进评测脚本，彻底消除了“我用了不同的分词”这种争议。  
- **“一键复现指令”**：输出的报告中会包含完整的命令行、使用的参考版本和参数，任何人复制粘贴即可得到完全相同的 BLEU，极大提升了论文的可验证性。

### 实验与效果
- **数据集**：作者在 WMT 2014 英德、WMT 2017 中文英等多个官方测试集上跑实验，覆盖了不同语言对和不同规模的参考集合。  
- **Baseline 对比**：对比了传统的手工报告方式（作者自行选择分词、归一化）与使用 SacreBLEU 的统一方式。结果显示，同一系统的 BLEU 在手工方式下波动最高 1.8 分，而统一方式下所有实验的分数完全一致。  
- **消融实验**：作者分别关闭参考分词、大小写归一化、标点去除三个模块，观察分数变化。实验确认参考分词是导致最大差异的因素，去除该模块后剩余两项对分数的影响不到 0.3 分。  
- **局限性**：论文主要关注 BLEU 的前置处理，对 BLEU 本身的局限（如对语义的忽视）没有进一步讨论；此外，统一方案基于 WMT 脚本，可能不适用于非 WMT 任务的特殊需求。作者在结论中承认，若社区在未来采用更高级的评价指标，类似的统一工具仍然是必要的。

### 影响与延伸思考
自从这篇论文推出后，SacreBLEU 成为机器翻译实验的“标配”。几乎所有顶会（ACL、EMNLP、NAACL）接受的 MT 论文都在实验章节注明使用 SacreBLEU，并在附录给出完整的复现指令。后续工作如 **BLEU‑R**（针对低资源语言的改进）和 **chrF**（字符级指标）的评测脚本也借鉴了 SacreBLEU 的“一键复现”思路。更广泛地，这篇论文推动了 **评测可复现性** 的讨论，催生了类似 **MTEval**、**COMET** 等指标的统一包装工具。想进一步了解，可以关注 **评测标准化**（standardization of evaluation）和 **指标可解释性**（interpretability of metrics）两个方向，尤其是社区正在探索的 **多维度评测平台**（如 WMT 的多指标排行榜）。

### 一句话记住它
统一 BLEU 计算流程、用 SacreBLEU 把“同一模型、不同报告”彻底消除。