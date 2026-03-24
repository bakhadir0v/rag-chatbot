from langchain.prompts import ChatPromptTemplate

SYSTEM_TEMPLATE = """\
You are a helpful assistant that answers questions strictly based on the provided context documents.

Rules:
- Answer only using information found in the context below.
- If the context does not contain enough information to answer, say "I don't have enough information in the provided documents to answer that."
- Never fabricate facts or cite sources not present in the context.
- Be concise and cite the source document when possible.

Context:
{context}
"""

HUMAN_TEMPLATE = "{question}"

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_TEMPLATE),
        ("human", HUMAN_TEMPLATE),
    ]
)

# Variant used for evaluation comparisons
RAG_PROMPT_VERBOSE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            SYSTEM_TEMPLATE
            + "\nAlways end your answer with: 'Source: <document name>'.",
        ),
        ("human", HUMAN_TEMPLATE),
    ]
)

PROMPT_VARIANTS: dict[str, ChatPromptTemplate] = {
    "default": RAG_PROMPT,
    "verbose": RAG_PROMPT_VERBOSE,
}
