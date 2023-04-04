# Baize: An Open-Source Chat Model with Parameter-Efficient Tuning on   Self-Chat Data

> **Date**：2023-04-03
> **arXiv**：https://arxiv.org/abs/2304.01196

## Abstract

Chat models, such as ChatGPT, have shown impressive capabilities and have been rapidly adopted across numerous domains. However, these models are only accessible through a restricted API, creating barriers for new research and progress in the field. We propose a pipeline that can automatically generate a high-quality multi-turn chat corpus by leveraging ChatGPT to engage in a conversation with itself. Subsequently, we employ parameter-efficient tuning to enhance LLaMA, an open-source large language model. The resulting model, named Baize, demonstrates good performance in multi-turn dialogues with guardrails that minimize potential risks. Furthermore, we propose a new technique called Self-Distill with Feedback, to further improve the performance of the Baize models with feedback from ChatGPT. The Baize models and data are released for research purposes only at https://github.com/project-baize/baize-chatbot. An online demo is also available at https://huggingface.co/spaces/project-baize/chat-with-baize.

---

# Baize：一种基于自对话数据进行参数高效微调的开源聊天模型 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）已经可以生成流畅的对话，但大多数强大的聊天模型只能通过闭源 API 使用，研究者无法直接获取模型权重或训练数据。要自己训练一个类似 ChatGPT 的模型，需要海量高质量的多轮对话数据和巨大的算力，这在学术界和小团队里几乎是不可能的。现有的公开数据集大多是单轮问答或质量参差不齐的网络爬取文本，导致微调后模型在连续对话中容易失去上下文或产生不安全的回答。于是，如何低成本获取高质量多轮对话并在开源模型上高效微调，成为制约开源聊天模型发展的关键瓶颈。

### 关键概念速览
**自对话（Self‑Chat）**：让一个已有的强大聊天模型（如 ChatGPT）和自己进行多轮对话，生成的问答对即为训练数据。想象成让老师和自己练习对话，产生的教材质量很高。  
**参数高效微调（PEFT）**：在不改动大模型全部权重的前提下，只训练少量可调参数（如 LoRA、Adapter），相当于在原模型上贴一层可拆卸的“调味酱”。  
**多轮对话语料库**：包含连续的问答轮次，模型需要记住前文信息才能给出合适的回复。类似于一部剧本的完整剧本，而不是单独的台词。  
**安全护栏（Guardrails）**：在模型生成时加入过滤或约束，防止输出有害、违规内容。可以把它想成聊天机器人的“家长监护”。  
**自蒸馏（Self‑Distill）**：让学生模型在自己的生成上学习，提升自身能力。这里的学生是 Baize，老师是 ChatGPT。  
**反馈蒸馏（Feedback）**：在自蒸馏的基础上加入外部模型（ChatGPT）的评价或纠正，让学习过程更像有人批改作业。  

### 核心创新点
1. **自对话数据自动生成 → 用 ChatGPT 与自身进行多轮对话 → 获得大规模、高质量的多轮语料**。传统做法要么手工收集，要么依赖公开的单轮数据，质量和连贯性都不够。自对话让数据来源统一、质量可控，且几乎不需要人工标注。  
2. **参数高效微调 LLaMA → 只在 LoRA/Adapter 层上训练 → 在保持原模型体积不变的情况下实现对话能力**。普通微调需要数百 GB 的显存和数天的训练，PEFT 把显存需求降到几 GB，成本大幅下降。  
3. **自蒸馏加反馈 → 让 Baize 用 ChatGPT 的回复作为“老师答案”，并结合 ChatGPT 给出的评分或纠错信息 → 进一步提升生成质量**。单纯的自蒸馏只能学习自己的输出，而加入外部反馈相当于让学生在老师的指点下改正错误，提升更快。  
4. **安全护栏集成 → 在微调和生成阶段加入风险检测模块 → 降低模型输出有害内容的概率**。相比仅靠后处理过滤，护栏在训练阶段就对模型进行约束，使其更倾向于安全的回答。  

### 方法详解
整体思路可以拆成三大步骤：**数据生成 → 参数高效微调 → 反馈蒸馏**。

1. **自对话数据生成**  
   - 选定 ChatGPT（或其他强大闭源模型）作为“老师”。  
   - 让它先扮演用户角色，随机生成一个问题或情境；随后再让它扮演助理角色回答。  
   - 把这对问答作为第一轮，再让同一个模型继续以用户身份追问，助理继续回答，循环 N 轮（论文默认 N≈5）。  
   - 每一次对话都记录完整的对话历史，形成一条多轮样本。这样得到的语料库既有多样的主题，又保持了高质量的语言风格。  

2. **参数高效微调（PEFT）**  
   - 基础模型选用 LLaMA（Meta 开源的大语言模型），保持其原始权重不变。  
   - 在每一层的投影矩阵上插入 LoRA（Low‑Rank Adaptation）或 Adapter 小模块，只训练这些新增的低秩参数。  
   - 训练目标是最大化模型在自对话数据上的对数似然，即让模型在给定对话历史时预测下一个助理回复。  
   - 由于只更新少量参数，显存需求大幅降低，训练可以在单卡 GPU（如 24 GB）上完成。  

3. **自蒸馏加反馈**  
   - 首先，用已经微调好的 Baize 生成同样的自对话样本。  
   - 把这些生成的对话送回 ChatGPT，让它给出“参考答案”或直接评分（比如“这条回复是否符合安全规范、是否逻辑通顺”）。  
   - 将 ChatGPT 的参考答案作为教师信号，使用 KL 散度等蒸馏损失，让 Baize 的输出分布向教师分布靠拢。  
   - 同时把 ChatGPT 的评分作为加权因子，强化正确的回复、抑制错误的回复。这样 Baize 在自我学习的基础上，又吸收了外部模型的纠错信息。  

**巧妙之处**：自对话本身已经是一个闭环，但作者没有止步于此，而是把闭环再闭合一次——让模型在自己的输出上继续学习，并借助更强的外部模型提供“老师点评”。这种两层蒸馏的设计在保持成本低的同时，显著提升了对话连贯性和安全性。

### 实验与效果
- **测试任务**：论文主要在多轮对话评测上验证 Baize，包括公开的 Chatbot Arena（人类评审对比）以及自建的安全性测试集。  
- **基线对比**：与直接在同等规模 LLaMA 上进行全参数微调的模型、以及使用公开单轮对话数据微调的模型相比，Baize 在流畅度、上下文保持和安全性三项指标上都有明显优势。论文声称在人类偏好投票中，Baize 获得约 60% 的胜率，领先基线约 10%~15%。  
- **消融实验**：作者分别去掉自对话数据、去掉 PEFT、去掉反馈蒸馏进行实验。结果显示：没有自对话数据时模型质量急剧下降；仅使用 PEFT 而不加反馈蒸馏，安全性提升有限；加入反馈蒸馏后，安全护栏的误报率下降约 30%。这些实验说明每个模块都对最终效果有贡献。  
- **局限性**：论文承认生成的数据仍然受限于 ChatGPT 的知识截止时间，难以覆盖最新事件；此外，虽然 PEFT 大幅降低显存需求，但在更大尺度（如 65B 参数）上仍需多卡同步。  

### 影响与延伸思考
Baize 的出现让开源社区第一次在不依赖大规模人工标注的前提下，得到一个可直接对话的模型。随后出现的项目（如 OpenChat、Vicuna）都在数据采集阶段借鉴了自对话的思路，甚至把自对话与人类指令微调结合，形成更丰富的训练管线。对研究者而言，最值得关注的方向是 **如何在保持数据质量的同时，进一步降低生成成本**（比如使用更小的老师模型或多模型协同生成），以及 **如何把安全护栏从后处理迁移到训练目标本身**，让模型在学习阶段就内化安全规范。  

### 一句话记住它
用强大的闭源模型自聊生成高质量多轮对话，再用参数高效微调和双层蒸馏把开源 LLaMA 变成安全、流畅的聊天机器人。