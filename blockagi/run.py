import os
from blockagi.chains import BlockAGIChain
from blockagi.schema import Findings
from blockagi.tools import (
    DDGSearchAnswerTool,
    DDGSearchLinksTool,
    GoogleSearchLinksTool,
    VisitWebTool,
)
from langchain.chat_models import GPT4All

def run_blockagi(
    agent_role,
    openai_api_key,
    openai_model,
    resource_pool,
    objectives,
    blockagi_callback,
    llm_callback,
    iteration_count,
):
    tools = []
    if os.getenv("GOOGLE_API_KEY") and os.getenv("GOOGLE_CSE_ID"):
        tools.append(GoogleSearchLinksTool(resource_pool))

    tools.extend(
        [
            DDGSearchAnswerTool(),
            DDGSearchLinksTool(resource_pool),
            VisitWebTool(resource_pool),
        ]
    )

    blockagi_callback.on_log_message(
        f"Using {len(tools)} tools:\n"
        + "\n".join(
            [f"{idx+1}. {t.name} - {t.description}" for idx, t in enumerate(tools)]
        )
    )
    
    llm = GPT4All(
        temperature=0.8,
        streaming=True,
        n_threads=8,
        model='./ggml-model-gpt4all-falcon-q4_0.bin',
        #openai_api_key=openai_api_key,
        callbacks=[llm_callback],
    )  # type: ignore

    inputs = {
        "objectives": objectives,
        "findings": Findings(
            narrative="Nothing",
            remark="",
            generated_objectives=[],
        ),
    }

    BlockAGIChain(
        iteration_count=iteration_count,
        agent_role=agent_role,
        llm=llm,
        tools=tools,
        resource_pool=resource_pool,
        callbacks=[blockagi_callback],
    )(inputs=inputs)
