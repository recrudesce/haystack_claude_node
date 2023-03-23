from haystack.nodes import BM25Retriever, Shaper
from haystack.document_stores import InMemoryDocumentStore
from haystack.pipelines import Pipeline

from haystack_claude_node import ClaudeAnswerGenerator


document_store = InMemoryDocumentStore(use_bm25=True)
from datasets import load_dataset

dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
document_store.write_documents(dataset)

api_key = "APIKEYHERE"

retriever = BM25Retriever(document_store=document_store, top_k=2)

shaper = Shaper(
    func="documents_to_strings",
    inputs={"documents": "documents"},
    outputs=["documents"],
)

query = "What does the Rhodes Statue look like?"

prompt_text = (
    "Given the context please answer the question using your own words. Generate a comprehensive, summarized answer. If the information is not included in the provided context, reply with 'Provided documents didn't contain the necessary information to provide the answer'\n\nContext: $documents;\n\nQuestion: $query;",
)

claude_node = ClaudeAnswerGenerator(api_key=api_key, prompt=prompt_text)

pipe = Pipeline()
pipe.add_node(component=retriever, name="Retriever", inputs=["Query"])
pipe.add_node(component=shaper, name="Shaper", inputs=["Retriever"])
pipe.add_node(component=claude_node, name="claude", inputs=["Shaper"])

output = pipe.run(query)

print(output["completion"])
