from haystack.nodes.base import BaseComponent
import requests
import anthropic
from typing import Optional


def convertTuple(tup):
    st = "".join(map(str, tup))
    return st


class ClaudeAnswerGenerator(BaseComponent):
    def __init__(
        self,
        api_key: str = "",
        temperature: Optional[float] = 1.0,
        max_tokens: Optional[int] = 200,
        model: Optional[str] = "claude-v1.2",
        prompt: [str] = "",
    ):
        super().__init__()
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = model
        self.prompt = prompt

    outgoing_edges = 1

    def run(self, query: str, documents: list, **kwargs):
        in_docs = " ".join(documents)
        user_prompt = convertTuple(self.prompt)
        user_prompt = user_prompt.replace("$query", query)
        user_prompt = user_prompt.replace("$documents", in_docs)
        c = anthropic.Client(self.api_key)

        output = c.completion(
            prompt=f"{anthropic.HUMAN_PROMPT}${user_prompt}{anthropic.AI_PROMPT}",
            stop_sequences=[anthropic.HUMAN_PROMPT],
            model=self.model,
            max_tokens_to_sample=self.max_tokens,
        )
        return output, "output_1"

    def run_batch(self, query: str, **kwargs):
        return
