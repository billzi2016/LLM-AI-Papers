# M3KE: A Massive Multi-Level Multi-Subject Knowledge Evaluation Benchmark   for Chinese Large Language Models

> **Date**：2023-05-17
> **arXiv**：https://arxiv.org/abs/2305.10263

## Abstract

Large language models have recently made tremendous progress in a variety of aspects, e.g., cross-task generalization, instruction following. Comprehensively evaluating the capability of large language models in multiple tasks is of great importance. In this paper, we propose M3KE, a Massive Multi-Level Multi-Subject Knowledge Evaluation benchmark, which is developed to measure knowledge acquired by Chinese large language models by testing their multitask accuracy in zero- and few-shot settings. We have collected 20,477 questions from 71 tasks. Our selection covers all major levels of Chinese education system, ranging from the primary school to college, as well as a wide variety of subjects, including humanities, history, politics, law, education, psychology, science, technology, art and religion. All questions are multiple-choice questions with four options, hence guaranteeing a standardized and unified assessment process. We've assessed a number of state-of-the-art open-source Chinese large language models on the proposed benchmark. The size of these models varies from 335M to 130B parameters. Experiment results demonstrate that they perform significantly worse than GPT-3.5 that reaches an accuracy of ~ 48% on M3KE. The dataset is available at https://github.com/tjunlp-lab/M3KE.

---

# M3KE：面向中文大语言模型的海量多层级多学科知识评估基准 论文详细解读

### 背景：这个问题为什么难？

中文大语言模型（LLM）在生成流畅文本、完成指令等方面已经取得显著进步，但我们仍缺少一种**统一、系统、覆盖面广**的测评手段来判断它们到底掌握了多少真实知识。过去的中文测评大多聚焦单一学科（如CMMLU只覆盖几百个题目）或只针对特定年级，导致：  
1. **覆盖不足**——很多重要学科（法律、心理学、艺术等）几乎没有测评数据。  
2. **层级碎片化**——从小学到大学的知识跨度没有在同一基准里呈现，难以比较模型在不同教育阶段的表现。  
3. **评估不统一**——不同数据集使用不同题型（填空、简答、选择），导致分数不可比。  
因此，想要客观评估中文LLM的“知识深度”和“广度”，必须先构建一个规模更大、层级更全、题型统一的基准，这正是M3KE要解决的痛点。

### 关键概念速览
- **大语言模型（LLM）**：基于海量文本训练的神经网络，能够理解并生成自然语言。类似于“会说话的百科全书”。  
- **零样本 / 少样本评估**：不给模型任何示例（零样本）或只给极少量示例（少样本）就直接让它回答问题，考察模型的通用推理能力，而不是记忆特定数据。  
- **多选题（四选一）**：每道题提供四个备选答案，只有一个正确。因为答案空间固定，评分自动化且公平。  
- **知识层级**：指教育体系中的不同年级或阶段，如小学、初中、高中、大学。层级越高，题目涉及的概念越抽象、难度越大。  
- **多学科覆盖**：包括人文、历史、政治、法律、教育、心理、自然科学、技术、艺术、宗教等十余大类，确保模型的知识面不偏科。  
- **参数规模**：模型内部的可调节权重数量，从几亿到上百亿不等，通常规模越大模型的学习能力越强。  
- **基准数据集（Benchmark）**：一套公开、标准化的测试题目，用来统一比较不同模型的表现。  

### 核心创新点
1. **规模空前的题库 → 收集并整理 20,477 题，覆盖 71 项任务 → 评测覆盖面比 CMMLU、MMCU 多出数倍，能够更细致地描绘模型的知识全景。**  
2. **全教育层级统一设计 → 从小学到大学的题目全部采用四选一格式 → 评测过程完全自动化，且不同年级的成绩可以直接对比，形成“一张知识地图”。**  
3. **零/少样本统一评估协议 → 对每个模型在不做微调的情况下直接进行推理 → 真实反映模型的通用知识迁移能力，避免因微调数据泄露导致的“作弊”。**  
4. **开源评测框架 → 将数据、评测脚本、结果可视化工具全部公开 → 研究社区可以快速复现、扩展或加入新模型，推动中文LLM的竞争与合作。  

### 方法详解
**整体思路**：M3KE 的构建分为三大步骤——（1）任务挑选与题目收集，（2）统一题型加工，（3）评测协议制定与自动化评分。

1. **任务挑选与题目收集**  
   - 先在中国教育部官方教材、公开考试（如高考、司法考试）以及权威学术资源中筛选出 71 类代表性任务。  
   - 每类任务再细分为不同年级或难度层次，确保覆盖从小学一年级到大学本科的全部阶段。  
   - 通过爬虫、人工整理和专家校对的方式，累计得到 20,477 条多选题。每题都有四个选项，其中唯一正确答案由原始教材或官方答案确认。

2. **统一题型加工**  
   - 为了让所有模型都能“一键”作答，所有题目统一为“问题 + A/B/C/D 四选项”的文本格式。  
   - 采用模板化的提示词（prompt），例如：“以下是一道选择题，请直接给出正确选项的字母”。这样在零样本和少样本设置下，模型只需要输出 A、B、C 或 D，评分脚本即可直接比对。

3. **评测协议与自动化评分**  
   - **零样本**：模型仅收到题目本身和统一提示词，不提供任何示例。  
   - **少样本**：在每个任务的开头提供 1–3 例已知答案的示例，帮助模型理解题目结构。  
   - 评测脚本遍历所有题目，记录模型输出的选项字母，计算整体准确率以及按学科、层级细分的准确率。  
   - 为防止模型利用“概率最高的选项”作弊，脚本会检查输出是否为合法字母，否则计为错误。

**最巧妙的地方**：作者把所有学科的题目都压缩成同一种四选一格式，这看似简单，却解决了跨学科、跨年级评测的“格式不统一”难题，使得评测过程完全自动化，极大降低了人工标注和评分的成本。

### 实验与效果
- **测试对象**：作者选取了多款开源中文 LLM，参数规模从 335M 到 130B 不等，包括 ChatGLM、BloomZ、InternLM 等。  
- **基准对比**：使用同样的零/少样本协议，GPT‑3.5（非中文专属模型）作为外部参考。GPT‑3.5 在 M3KE 上的整体准确率约为 **48%**。  
- **结果概览**：所有开源模型的准确率均显著低于 GPT‑3.5，尤其是小模型（<1B 参数）往往在 20% 左右徘徊；即便是 130B 参数的大模型，也难突破 35% 的门槛。  
- **消融实验**：论文提供了少样本数量对准确率的影响曲线，显示在 1‑2 示例后提升约 5%，但进一步增加示例收益递减，说明模型已经基本掌握了题目格式。  
- **局限性**：作者坦诚，M3KE 只包含选择题，无法直接评估模型的生成式回答、推理链或开放式解释能力；此外，题目来源主要是教材和公开考试，可能偏向学术知识，对实际生活场景的覆盖仍有限。

### 影响与延伸思考
M3KE 公开后，迅速成为中文 LLM 评测的“标配”。后续不少工作（如 **ChineseMMLU‑2**、**EduEval‑CN**）在其基础上加入了填空题、简答题或多模态题目，以弥补选择题的局限。社区也开始围绕 **“少样本提示工程”** 进行深入研究，尝试通过更精细的示例设计提升模型在 M3KE 上的表现。想进一步了解中文 LLM 的评测生态，建议关注 **OpenCompass** 项目，它把 M3KE 纳入了更大规模的多维度评测平台。

### 一句话记住它
**M3KE 用 20 k+ 四选一题，统一测量中文大模型从小学到大学的全学科知识，揭示了开源模型仍远不及 GPT‑3.5 的真实水平。**