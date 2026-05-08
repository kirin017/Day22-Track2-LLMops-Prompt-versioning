"""
qa_pairs.py — 50 QA pairs with ground-truth reference answers.
Each entry: {"question": str, "reference": str}
Topics: ML, Deep Learning, NLP, RAG, LangChain, LangSmith, RAGAS, Guardrails AI
"""

QA_PAIRS = [
    # ── Machine Learning Fundamentals ────────────────────────────────────────
    {
        "question": "What are the three main types of machine learning?",
        "reference": (
            "The three main types of machine learning are supervised learning, "
            "unsupervised learning, and reinforcement learning."
        ),
    },
    {
        "question": "What is overfitting in machine learning?",
        "reference": (
            "Overfitting occurs when a model learns the training data too well, "
            "including noise and irrelevant patterns, leading to poor performance "
            "on new, unseen data. An overfitted model has high variance and low bias."
        ),
    },
    {
        "question": "Explain the bias-variance tradeoff.",
        "reference": (
            "The bias-variance tradeoff is the tension between two sources of error. "
            "Bias is the error from overly simplistic assumptions leading to underfitting; "
            "variance is the error from sensitivity to fluctuations in training data leading "
            "to overfitting. The goal is to minimize total error (bias² + variance + irreducible noise)."
        ),
    },
    {
        "question": "How does regularization prevent overfitting?",
        "reference": (
            "Regularization prevents overfitting by adding a penalty to the model's loss "
            "function based on model complexity. L1 (Lasso) promotes sparsity; "
            "L2 (Ridge) shrinks weights smoothly. Dropout randomly disables neurons during training. "
            "Early stopping halts training when validation performance stops improving."
        ),
    },
    {
        "question": "What is cross-validation?",
        "reference": (
            "Cross-validation is a technique to assess model generalization by splitting data "
            "into multiple folds. In k-fold cross-validation, the model trains on k-1 folds "
            "and is evaluated on the remaining fold, rotating through all folds to provide a "
            "reliable estimate of generalization performance."
        ),
    },
    # ── Deep Learning ────────────────────────────────────────────────────────
    {
        "question": "What is backpropagation?",
        "reference": (
            "Backpropagation is the algorithm used to train neural networks by computing "
            "gradients of the loss function with respect to each weight using the chain rule "
            "of calculus. The gradients are then used by an optimizer (such as SGD or Adam) "
            "to update the weights in the direction that minimizes the loss."
        ),
    },
    {
        "question": "What are Convolutional Neural Networks primarily used for?",
        "reference": (
            "Convolutional Neural Networks (CNNs) are primarily used for image recognition, "
            "object detection, and computer vision tasks. They use convolutional layers to "
            "automatically detect spatial hierarchies of features such as edges, textures, "
            "and shapes."
        ),
    },
    {
        "question": "How do LSTM networks address the vanishing gradient problem?",
        "reference": (
            "LSTM networks address the vanishing gradient problem through a gating mechanism "
            "with three gates: the forget gate (discards irrelevant information), the input gate "
            "(stores new information), and the output gate (outputs information). This allows "
            "LSTMs to retain information over many time steps without gradient decay."
        ),
    },
    {
        "question": "What activation functions are commonly used in neural networks?",
        "reference": (
            "Common activation functions include ReLU (most widely used in hidden layers), "
            "Sigmoid (for binary classification outputs), Tanh (outputs between -1 and 1), "
            "Softmax (for multi-class classification outputs), and modern alternatives like "
            "GELU and SiLU used in transformer architectures."
        ),
    },
    {
        "question": "What is the role of pooling layers in CNNs?",
        "reference": (
            "Pooling layers in CNNs reduce the spatial dimensions of feature maps, decreasing "
            "computational load and the number of parameters. Max pooling takes the maximum "
            "value in each pooling window, preserving the most prominent features. Pooling also "
            "provides a degree of translation invariance."
        ),
    },
    # ── NLP and Transformers ─────────────────────────────────────────────────
    {
        "question": "What is the transformer architecture?",
        "reference": (
            "The transformer architecture, introduced in 'Attention Is All You Need' (2017), "
            "relies entirely on self-attention mechanisms rather than recurrence or convolution. "
            "It consists of an encoder and decoder, each with multi-head self-attention and "
            "feed-forward sub-layers with residual connections and layer normalization."
        ),
    },
    {
        "question": "What are word embeddings?",
        "reference": (
            "Word embeddings are dense vector representations of words that capture semantic "
            "meaning, such that similar words have similar vector representations. Classic methods "
            "include Word2Vec, GloVe, and FastText. Modern contextual embeddings from BERT and GPT "
            "produce different representations for the same word based on context."
        ),
    },
    {
        "question": "What is transfer learning in NLP?",
        "reference": (
            "Transfer learning in NLP involves pre-training a large language model on a large "
            "corpus of text and then fine-tuning it on a specific downstream task. This allows "
            "models to leverage general language understanding learned during pre-training, "
            "reducing the need for large task-specific datasets."
        ),
    },
    {
        "question": "How does BERT handle language understanding?",
        "reference": (
            "BERT (Bidirectional Encoder Representations from Transformers) handles language "
            "understanding by reading text bidirectionally using masked language modeling (MLM) "
            "and next sentence prediction (NSP) objectives. It produces contextual token embeddings "
            "used for tasks like question answering, named entity recognition, and text classification."
        ),
    },
    {
        "question": "What is self-attention in transformers?",
        "reference": (
            "Self-attention is a mechanism that allows each token in a sequence to attend to all "
            "other tokens to compute a representation. It computes Query (Q), Key (K), and Value (V) "
            "matrices from the input, then calculates attention scores as softmax(QK^T / sqrt(d_k)) * V. "
            "Multi-head attention runs multiple self-attention heads in parallel."
        ),
    },
    {
        "question": "What is GPT and how is it trained?",
        "reference": (
            "GPT (Generative Pre-trained Transformer) is a family of autoregressive language models "
            "trained to predict the next token in a sequence. GPT is trained on large text corpora "
            "using unsupervised pre-training followed by supervised fine-tuning. "
            "GPT-4 is a large multimodal model with a context length of up to 128,000 tokens."
        ),
    },
    {
        "question": "What is instruction tuning?",
        "reference": (
            "Instruction tuning is a fine-tuning technique where a pre-trained language model is "
            "further trained on examples of instructions and their desired outputs. This improves "
            "the model's ability to follow natural language instructions. "
            "Models like InstructGPT and GPT-4 use instruction tuning to better align with user intent."
        ),
    },
    {
        "question": "What is RLHF?",
        "reference": (
            "RLHF (Reinforcement Learning from Human Feedback) is a training technique to align "
            "language models with human preferences. It involves three steps: (1) supervised "
            "fine-tuning on demonstrations, (2) training a reward model on human preference "
            "comparisons, and (3) optimizing the language model against the reward model using "
            "reinforcement learning (typically PPO). RLHF is used in ChatGPT and GPT-4."
        ),
    },
    {
        "question": "What is chain-of-thought prompting?",
        "reference": (
            "Chain-of-thought (CoT) prompting is a technique where the model is prompted to produce "
            "intermediate reasoning steps before giving a final answer. By including examples of "
            "step-by-step reasoning in the prompt, or by instructing the model to 'think step by step,' "
            "complex multi-step reasoning tasks can be significantly improved."
        ),
    },
    {
        "question": "What is the context length of GPT-4?",
        "reference": (
            "GPT-4 is a large multimodal model with a context length of up to 128,000 tokens, "
            "allowing it to process very long documents and conversations in a single inference call."
        ),
    },
    # ── RAG ──────────────────────────────────────────────────────────────────
    {
        "question": "What is Retrieval-Augmented Generation?",
        "reference": (
            "Retrieval-Augmented Generation (RAG) is a technique that combines large language models "
            "with external knowledge retrieval to produce more accurate, up-to-date, and grounded "
            "responses. RAG retrieves relevant documents from an external knowledge base at query time "
            "and uses them as context for the LLM to generate answers."
        ),
    },
    {
        "question": "What are the main components of a RAG pipeline?",
        "reference": (
            "The main components of a RAG pipeline are: (1) Document Loader, (2) Text Splitter, "
            "(3) Embedding Model, (4) Vector Store, (5) Retriever, (6) Prompt Template, "
            "(7) LLM for answer generation, and (8) Output Parser."
        ),
    },
    {
        "question": "What is dense retrieval?",
        "reference": (
            "Dense retrieval uses dense vector representations (embeddings) of both queries and "
            "documents to find relevant matches via semantic similarity (cosine similarity or dot product). "
            "Unlike sparse retrieval (BM25) which relies on keyword overlap, dense retrieval can find "
            "semantically similar documents even when they don't share exact keywords."
        ),
    },
    {
        "question": "Why is chunking strategy important in RAG?",
        "reference": (
            "Chunking strategy is important in RAG because retrieval quality depends on how well "
            "the retrieved chunks match the query. Too-small chunks may lack sufficient context; "
            "too-large chunks may introduce irrelevant information. Overlap between chunks helps "
            "preserve context at boundaries. Common strategies include fixed-size chunking, "
            "recursive character splitting, and semantic chunking."
        ),
    },
    {
        "question": "What advanced RAG techniques exist beyond basic retrieval?",
        "reference": (
            "Advanced RAG techniques include: HyDE (Hypothetical Document Embeddings), re-ranking "
            "with cross-encoders, multi-query retrieval, contextual compression, Self-RAG (the model "
            "decides when to retrieve), and RAPTOR (recursive abstractive processing for tree-organized "
            "retrieval)."
        ),
    },
    # ── Vector Databases ─────────────────────────────────────────────────────
    {
        "question": "What are vector databases used for?",
        "reference": (
            "Vector databases are specialized databases designed to store, index, and query "
            "high-dimensional vector embeddings efficiently. They are used in semantic search, "
            "recommendation systems, RAG pipelines, image similarity search, and any application "
            "requiring similarity-based retrieval. Examples include FAISS, Pinecone, Weaviate, "
            "Milvus, Chroma, and Qdrant."
        ),
    },
    {
        "question": "What is FAISS?",
        "reference": (
            "FAISS (Facebook AI Similarity Search) is an open-source library developed by Facebook "
            "AI Research for efficient similarity search and clustering of dense vectors. FAISS "
            "supports both exact and approximate nearest-neighbor search using indexing structures "
            "like IVF and HNSW. It is widely used in RAG pipelines for local vector storage."
        ),
    },
    {
        "question": "How do text embeddings capture semantic meaning?",
        "reference": (
            "Text embeddings capture semantic meaning by mapping text to points in a high-dimensional "
            "vector space such that semantically similar texts are close together. They are created by "
            "passing text through a neural encoder. The distance between vectors (cosine similarity) "
            "reflects the semantic similarity of the original texts."
        ),
    },
    {
        "question": "What is HNSW?",
        "reference": (
            "HNSW (Hierarchical Navigable Small World) is a graph-based approximate nearest neighbor "
            "search algorithm used in vector databases. It builds a multi-layer graph structure where "
            "nodes are connected based on proximity, allowing efficient traversal during search. "
            "HNSW offers an excellent tradeoff between search speed and recall accuracy."
        ),
    },
    {
        "question": "What is hybrid search in vector databases?",
        "reference": (
            "Hybrid search in vector databases combines dense vector search (semantic similarity) "
            "with sparse keyword search (BM25 or TF-IDF) to improve retrieval quality. Results "
            "from both methods are merged using a fusion algorithm such as Reciprocal Rank Fusion. "
            "Hybrid search captures both semantic meaning and exact keyword matches."
        ),
    },
    # ── LangChain ────────────────────────────────────────────────────────────
    {
        "question": "What is LangChain?",
        "reference": (
            "LangChain is an open-source framework for building applications powered by large "
            "language models. It provides abstractions for chains, agents, retrievers, memory, "
            "tools, and prompts, enabling developers to compose complex LLM workflows. LangChain "
            "supports multiple LLM providers and vector stores."
        ),
    },
    {
        "question": "What is LangChain Expression Language (LCEL)?",
        "reference": (
            "LangChain Expression Language (LCEL) is a declarative way to compose LangChain "
            "components using the pipe operator (|). LCEL chains are composable, streaming-compatible, "
            "and support async execution. A typical LCEL RAG chain: "
            "{context: retriever | format_docs, question: RunnablePassthrough()} | prompt | llm | StrOutputParser()."
        ),
    },
    {
        "question": "What is LangGraph?",
        "reference": (
            "LangGraph is a library built on top of LangChain for building stateful, multi-actor "
            "applications with LLMs as graphs. It allows defining workflows as directed graphs where "
            "nodes are processing steps and edges represent control flow. LangGraph supports cycles "
            "and branching, making it suitable for agentic systems requiring iterative reasoning."
        ),
    },
    {
        "question": "What memory types does LangChain support?",
        "reference": (
            "LangChain supports several memory types: ConversationBufferMemory (full history), "
            "ConversationBufferWindowMemory (last k turns), ConversationSummaryMemory (LLM-summarized), "
            "ConversationKGMemory (knowledge graph), and VectorStoreRetrieverMemory "
            "(embeddings in a vector store)."
        ),
    },
    {
        "question": "What are LangChain retrievers?",
        "reference": (
            "LangChain retrievers are components that accept a query and return relevant documents. "
            "Types include VectorStoreRetriever (embedding similarity), MultiQueryRetriever "
            "(multiple generated queries), ContextualCompressionRetriever (compresses retrieved docs), "
            "EnsembleRetriever (combines multiple retrievers), and SelfQueryRetriever "
            "(translates natural language to structured queries)."
        ),
    },
    # ── LangSmith ────────────────────────────────────────────────────────────
    {
        "question": "What is LangSmith?",
        "reference": (
            "LangSmith is a developer platform by LangChain for debugging, testing, evaluating, "
            "and monitoring LLM applications. It provides tracing, prompt management, dataset "
            "management, and evaluation tools. LangSmith integrates with LangChain via environment "
            "variables and the @traceable decorator."
        ),
    },
    {
        "question": "What information do LangSmith traces capture?",
        "reference": (
            "LangSmith traces capture detailed information about each LLM call including: "
            "input messages, output responses, latency, token usage, model name, chain structure, "
            "intermediate steps, metadata, and tags. Traces are organized hierarchically, showing "
            "parent-child relationships between chain components."
        ),
    },
    {
        "question": "What is the LangSmith Prompt Hub?",
        "reference": (
            "The LangSmith Prompt Hub is a centralized repository for storing, versioning, and "
            "sharing prompt templates. Prompts can be pushed using client.push_prompt() and pulled "
            "using client.pull_prompt(). The Prompt Hub supports versioning so teams can track changes "
            "and roll back to previous prompt versions."
        ),
    },
    {
        "question": "How does LangSmith help monitor production LLM applications?",
        "reference": (
            "LangSmith helps monitor production LLM applications by providing real-time traces, "
            "error alerts, latency tracking, and cost monitoring. Teams can filter traces by tags, "
            "metadata, or time ranges. LangSmith supports human-in-the-loop feedback collection "
            "and can trigger alerts when error rates or latencies exceed thresholds."
        ),
    },
    {
        "question": "What are LangSmith datasets used for?",
        "reference": (
            "LangSmith datasets are collections of input-output pairs used for evaluation. They can "
            "be created manually, from traces, or by uploading CSV files. Datasets are used with "
            "LangSmith's evaluation framework to run automated tests and compare model or prompt "
            "versions, supporting regression testing and benchmarking of LLM applications."
        ),
    },
    # ── RAGAS ────────────────────────────────────────────────────────────────
    {
        "question": "What is RAGAS?",
        "reference": (
            "RAGAS (Retrieval-Augmented Generation Assessment) is an open-source framework for "
            "evaluating RAG pipelines. It provides automated metrics to assess both retrieval quality "
            "and generation quality without requiring extensive human annotation. RAGAS uses LLMs "
            "internally to compute metrics."
        ),
    },
    {
        "question": "How does RAGAS compute faithfulness?",
        "reference": (
            "RAGAS computes faithfulness by decomposing the generated answer into individual claims "
            "and checking whether each claim can be inferred from the retrieved context using an LLM "
            "judge. A faithfulness score of 1.0 means all claims are grounded in the context; "
            "0 means none are."
        ),
    },
    {
        "question": "What is answer relevancy in RAGAS?",
        "reference": (
            "Answer relevancy in RAGAS measures how well the generated answer addresses the user's "
            "question, regardless of factual accuracy. It is computed by generating multiple hypothetical "
            "questions from the answer and measuring how similar they are to the original question "
            "using embeddings. A high score indicates the answer is on-topic."
        ),
    },
    {
        "question": "What is context recall in RAGAS?",
        "reference": (
            "Context recall in RAGAS measures the extent to which the retrieved context contains all "
            "information necessary to answer the question, evaluated against the ground-truth reference "
            "answer. It checks whether each sentence in the reference answer can be attributed to the "
            "retrieved context using an LLM judge."
        ),
    },
    {
        "question": "What inputs does RAGAS evaluation require?",
        "reference": (
            "RAGAS evaluation requires for each sample: user_input (the original question), "
            "response (the generated answer), retrieved_contexts (list of retrieved text chunks), "
            "and reference (the ground-truth reference answer for context recall and context precision)."
        ),
    },
    # ── Guardrails AI ─────────────────────────────────────────────────────────
    {
        "question": "What is Guardrails AI?",
        "reference": (
            "Guardrails AI is an open-source Python framework for adding input/output validation "
            "and safety checks to LLM applications. It allows developers to define custom validators "
            "that check LLM outputs for compliance with specified rules. Guardrails supports automatic "
            "remediation (fix) and rejection of non-compliant outputs."
        ),
    },
    {
        "question": "What is PII and why is it important to detect in LLM responses?",
        "reference": (
            "PII (Personally Identifiable Information) is any data that can be used to identify a "
            "specific individual, such as names, email addresses, phone numbers, SSNs, and credit "
            "card numbers. Detecting and redacting PII in LLM responses is critical to prevent "
            "privacy violations, comply with regulations like GDPR and CCPA, and protect user data."
        ),
    },
    {
        "question": "What does structured output validation ensure?",
        "reference": (
            "Structured output validation ensures that LLM-generated outputs conform to an expected "
            "format or schema, such as valid JSON, XML, or a specific data structure. This is important "
            "in production systems where downstream components depend on structured data. Tools like "
            "Guardrails AI can automatically repair or reject malformed structured outputs."
        ),
    },
    {
        "question": "What is Constitutional AI?",
        "reference": (
            "Constitutional AI is an approach developed by Anthropic to train AI systems to be helpful, "
            "harmless, and honest using a set of principles (a 'constitution'). The AI critiques and "
            "revises its own outputs based on these principles using AI feedback rather than human "
            "feedback alone, reducing the need for human annotation in safety fine-tuning."
        ),
    },
    {
        "question": "What are common AI safety concerns with LLMs?",
        "reference": (
            "Common AI safety concerns with LLMs include: hallucination (generating false information "
            "confidently), bias (amplifying societal biases), PII leakage (exposing personal data), "
            "prompt injection (malicious inputs manipulating model behavior), jailbreaking (bypassing "
            "safety filters), toxicity (generating harmful content), and misinformation at scale."
        ),
    },
    # ── Additional / Context Precision ────────────────────────────────────────
    {
        "question": "What is context precision in RAGAS?",
        "reference": (
            "Context precision in RAGAS measures whether the retrieved context chunks that are relevant "
            "to answering the question are ranked higher than irrelevant chunks. It evaluates the quality "
            "of the retriever's ranking. High context precision means relevant information appears at "
            "the top of the retrieved context list."
        ),
    },
]

# Convenience exports
QUESTIONS  = [pair["question"]  for pair in QA_PAIRS]
REFERENCES = [pair["reference"] for pair in QA_PAIRS]

if __name__ == "__main__":
    print(f"Total QA pairs: {len(QA_PAIRS)}")
    for i, pair in enumerate(QA_PAIRS, 1):
        print(f"\n[{i:02d}] Q: {pair['question']}")
        print(f"     A: {pair['reference'][:100]}...")
