# Secure \& Intelligent Q\&A System (RAG + LLM)

Local RAG pipeline:

!["This is a flowchart describing a simple local retrieval-augmented generation (RAG) workflow for document processing and embedding creation, followed by search and answer functionality. The process begins with a collection of documents, such as PDFs or a 1200-page nutrition textbook, which are preprocessed into smaller chunks, for example, groups of 10 sentences each. These chunks are used as context for the Large Language Model (LLM). A cool person (potentially the user) asks a query such as "What are the macronutrients? And what do they do?" This query is then transformed by an embedding model into a numerical representation using sentence transformers or other options from Hugging Face, which are stored in a torch.tensor format for efficiency, especially with large numbers of embeddings (around 100k+). For extremely large datasets, a vector database/index may be used. The numerical query and relevant document passages are processed on a local GPU, specifically an RTX 4090. The LLM generates output based on the context related to the query, which can be interacted with through an optional chat web app interface. All of this processing happens on a local GPU. The flowchart includes icons for documents, processing steps, and hardware, with arrows indicating the flow from document collection to user interaction with the generated text and resources."](Images/simple-local-rag-workflow-flowchart.png)

**Setup notes:** 
* If you run into any install/setup troubles, please leave an issue.
* To get access to the Gemma LLM models, you will have to [agree to the terms & conditions](https://huggingface.co/google/gemma-7b-it) on the Gemma model page on Hugging Face. You will then have to authorize your local machine via the [Hugging Face CLI/Hugging Face Hub `login()` function](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication). Once you've done this, you'll be able to download the models. If you're using Google Colab, you can add a [Hugging Face token](https://huggingface.co/docs/hub/en/security-tokens) to the "Secrets" tab.
* For speedups, installing and compiling Flash Attention 2 (faster attention implementation) can take ~5 minutes to 3 hours depending on your system setup. See the [Flash Attention 2 GitHub](https://github.com/Dao-AILab/flash-attention/tree/main) for more. In particular, if you're running on Windows, see this [GitHub issue thread](https://github.com/Dao-AILab/flash-attention/issues/595). I've commented out `flash-attn` due to compile time, feel free to uncomment if you'd like use it or run `pip install flash-attn`.

## What is RAG?

RAG stands for Retrieval Augmented Generation.

It was introduced in the paper [*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*](https://arxiv.org/abs/2005.11401).

Each step can be roughly broken down to:

* **Retrieval** - Seeking relevant information from a source given a query. For example, getting relevant passages of Wikipedia text from a database given a question.
* **Augmented** - Using the relevant retrieved information to modify an input to a generative model (e.g. an LLM).
* **Generation** - Generating an output given an input. For example, in the case of an LLM, generating a passage of text given an input prompt.

## Why RAG?

The main goal of RAG is to improve the generation outptus of LLMs.

Two primary improvements can be seen as:
1. **Preventing hallucinations** - LLMs are incredible but they are prone to potential hallucination, as in, generating something that *looks* correct but isn't. RAG pipelines can help LLMs generate more factual outputs by providing them with factual (retrieved) inputs. And even if the generated answer from a RAG pipeline doesn't seem correct, because of retrieval, you also have access to the sources where it came from.
2. **Work with custom data** - Many base LLMs are trained with internet-scale text data. This means they have a great ability to model language, however, they often lack specific knowledge. RAG systems can provide LLMs with domain-specific data such as medical information or company documentation and thus customized their outputs to suit specific use cases.

The authors of the original RAG paper mentioned above outlined these two points in their discussion.

> This work offers several positive societal benefits over previous work: the fact that it is more
strongly grounded in real factual knowledge (in this case Wikipedia) makes it “hallucinate” less
with generations that are more factual, and offers more control and interpretability. RAG could be
employed in a wide variety of scenarios with direct benefit to society, for example by endowing it
with a medical index and asking it open-domain questions on that topic, or by helping people be more
effective at their jobs.

RAG can also be a much quicker solution to implement than fine-tuning an LLM on specific data. 


## What kind of problems can RAG be used for?

RAG can help anywhere there is a specific set of information that an LLM may not have in its training data (e.g. anything not publicly accessible on the internet).

For example you could use RAG for:
* **Customer support Q&A chat** - By treating your existing customer support documentation as a resource, when a customer asks a question, you could have a system retrieve relevant documentation snippets and then have an LLM craft those snippets into an answer. Think of this as a "chatbot for your documentation". Klarna, a large financial company, [uses a system like this](https://www.klarna.com/international/press/klarna-ai-assistant-handles-two-thirds-of-customer-service-chats-in-its-first-month/) to save $40M per year on customer support costs.
* **Email chain analysis** - Let's say you're an insurance company with long threads of emails between customers and insurance agents. Instead of searching through each individual email, you could retrieve relevant passages and have an LLM create strucutred outputs of insurance claims.
* **Company internal documentation chat** - If you've worked at a large company, you know how hard it can be to get an answer sometimes. Why not let a RAG system index your company information and have an LLM answer questions you may have? The benefit of RAG is that you will have references to resources to learn more if the LLM answer doesn't suffice.
* **Textbook Q&A** - Let's say you're studying for your exams and constantly flicking through a large textbook looking for answers to your quesitons. RAG can help provide answers as well as references to learn more.

All of these have the common theme of retrieving relevant resources and then presenting them in an understandable way using an LLM.

From this angle, you can consider an LLM a calculator for words.

## Why local?

Privacy, speed, cost.

Running locally means you use your own hardware.

From a privacy standpoint, this means you don't have send potentially sensitive data to an API.

From a speed standpoint, it means you won't necessarily have to wait for an API queue or downtime, if your hardware is running, the pipeline can run.

And from a cost standpoint, running on your own hardware often has a heavier starting cost but little to no costs after that.

Performance wise, LLM APIs may still perform better than an open-source model running locally on general tasks but there are more and more examples appearing of smaller, focused models outperforming larger models. 

## Key terms

| Term | Description |
| ----- | ----- | 
| [**Token**](https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them) | A sub-word piece of text. For example, "hello, world!" could be split into ["hello", ",", "world", "!"]. A token can be a whole word,<br> part of a word or group of punctuation characters. 1 token ~= 4 characters in English, 100 tokens ~= 75 words.<br> Text gets broken into tokens before being passed to an LLM. [Check this out!](https://platform.openai.com/tokenizer) |
| **Embedding** | A learned numerical representation of a piece of data. For example, a sentence of text could be represented by a vector with<br> 768 values. Similar pieces of text (in meaning) will ideally have similar values. |
| **Embedding model** | A model designed to accept input data and output a numerical representation. For example, a text embedding model may take in 384 <br>tokens of text and turn it into a vector of size 768. An embedding model can and often is different to an LLM model. |
| **Similarity search/vector search** | Similarity search/vector search aims to find two vectors which are close together in high-demensional space. For example, <br>two pieces of similar text passed through an embedding model should have a high similarity score, whereas two pieces of text about<br> different topics will have a lower similarity score. Common similarity score measures are dot product and cosine similarity. |
| **Large Language Model (LLM)** | A model which has been trained to numerically represent the patterns in text. A generative LLM will continue a sequence when given a sequence. <br>For example, given a sequence of the text "hello, world!", a genertive LLM may produce "we're going to build a RAG pipeline today!".<br> This generation will be highly dependant on the training data and prompt. |
| **LLM context window** | The number of tokens a LLM can accept as input. For example, as of March 2024, GPT-4 has a default context window of 32k tokens<br> (about 96 pages of text) but can go up to 128k if needed. A recent open-source LLM from Google, Gemma (March 2024) has a context<br> window of 8,192 tokens (about 24 pages of text). A higher context window means an LLM can accept more relevant information<br> to assist with a query. For example, in a RAG pipeline, if a model has a larger context window, it can accept more reference items<br> from the retrieval system to aid with its generation. |
| **Prompt** | A common term for describing the input to a generative LLM. The idea of "[prompt engineering](https://en.wikipedia.org/wiki/Prompt_engineering)" is to structure a text-based<br> (or potentially image-based as well) input to a generative LLM in a specific way so that the generated output is ideal. This technique is<br> possible because of a LLMs capacity for in-context learning, as in, it is able to use its representation of language to breakdown <br>the prompt and recognize what a suitable output may be (note: the output of LLMs is probable, so terms like "may output" are used). | 


## Better Approach:
𝗧𝗵𝗲 𝗖𝗼𝗿𝗲 𝗣𝗿𝗼𝗯𝗹𝗲𝗺: Why Traditional RAG Fails:
Traditional pipeline:
PDF → Text Extraction → Chunk → Embed → Retrieve
This breaks catastrophically because:
📊 Tables split across chunks = garbage
🖼️ Images contain critical information = lost
📄 Layout destroyed = broken context

🏗️ 𝗧𝗵𝗲 𝗣𝗿𝗼𝗱𝘂𝗰𝘁𝗶𝗼𝗻 𝗔𝗿𝗰𝗵𝗶𝘁𝗲𝗰𝘁𝘂𝗿𝗲 (𝟱 𝗖𝗿𝗶𝘁𝗶𝗰𝗮𝗹 𝗖𝗼𝗺𝗽𝗼𝗻𝗲𝗻𝘁𝘀)
1. 𝗗𝗼𝗰𝘂𝗺𝗲𝗻𝘁 𝗟𝗮𝘆𝗼𝘂𝘁 𝗔𝗻𝗮𝗹𝘆𝘀𝗶𝘀 - 𝗧𝗵𝗲 𝗙𝗼𝘂𝗻𝗱𝗮𝘁𝗶𝗼𝗻
The brutal truth about PDF parsing:

PyPDF/pdfplumber works for <10% of real PDFs ❌
They extract in PDF object order, NOT reading order
Result: Scrambled text, broken tables, missing context

Production solution: Layout Detection Models

Multimodal: Text + Layout + Image features
Detects: Text blocks, tables, figures, titles

𝟮️. 𝗧𝗮𝗯𝗹𝗲𝘀 - 𝗪𝗵𝗲𝗿𝗲 𝟵𝟱% 𝗼𝗳 𝗘𝗻𝗴𝗶𝗻𝗲𝗲𝗿𝘀 𝗙𝗮𝗶𝗹
❌ Wrong approach:
Text extraction: "Q1 Revenue 2023 2024 Product A 100M 150M"
→ LLMs can't reason over this garbage
✅ Right approach: Table Structure Recognition

𝟯️. 𝗜𝗺𝗮𝗴𝗲𝘀 & 𝗖𝗵𝗮𝗿𝘁𝘀 - 𝗩𝗶𝘀𝗶𝗼𝗻-𝗟𝗮𝗻𝗴𝘂𝗮𝗴𝗲 𝗠𝗼𝗱𝗲𝗹𝘀
The problem: Charts, diagrams, screenshots contain critical insights that text embeddings can't capture
Production VLM Pipeline:

✅ Solution: Include ±2 paragraphs of surrounding text for context

𝟰️. 𝗖𝗼𝗹𝗣𝗮𝗹𝗶 - 𝗧𝗵𝗲 𝟮𝟬𝟮𝟰 𝗚𝗮𝗺𝗲 𝗖𝗵𝗮𝗻𝗴𝗲𝗿 🚀
Traditional RAG: Text extraction → Chunking → Embedding
ColPali: Skip all that. Embed document images DIRECTLY.
How it works:
📸 Input: Raw page images (no text extraction!)
🧠 Model: PaliGemma (Vision encoder + Language decoder)
🎯 Output: Multi-vector representation (~1024 vectors/page)

𝟱️. 𝗖𝗵𝘂𝗻𝗸𝗶𝗻𝗴 𝗦𝘁𝗿𝗮𝘁𝗲𝗴𝘆 - 𝗧𝗵𝗲 𝗠𝗮𝗸𝗲-𝗼𝗿-𝗕𝗿𝗲𝗮𝗸 𝗗𝗲𝗰𝗶𝘀𝗶𝗼𝗻
Question that defines your seniority:
How do you chunk a 200-page PDF with tables, images, and nested sections?

Semantic chunking (the right way):
Instead of splitting every 512 tokens, use sentence embeddings to detect topic boundaries. When similarity drops below threshold → new chunk.
Hierarchical chunking (enterprise-grade):
🏢 Document-level summary
📂 Section-level chunks
📄 Atomic chunks with parent references

💬 𝗧𝗵𝗲 𝗔𝗻𝘀𝘄𝗲𝗿
 Hybrid architecture with three components:

1️⃣ LayoutLMv3 for document layout analysis—detecting text, tables, figures with bounding boxes

2️⃣ Multi-modal extraction: Table Transformer for structured tables with NL summaries, Qwen3 for image captions with context, semantic chunking for text preserving hierarchy

3️⃣ Dual retrieval: Text embeddings for semantic search + ColPali for complex visual queries, with cross-encoder re-ranking

