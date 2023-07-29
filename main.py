import os
import dotenv
import typer
import uvicorn
import webbrowser
import threading
from typing import Any, Dict, Optional, List
from datetime import datetime
from dataclasses import dataclass
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from langchain.callbacks.base import BaseCallbackHandler
from starlette.responses import FileResponse


from blockagi.chains.base import BlockAGICallbackHandler
from blockagi.schema import Objective, Findings, Narrative, Resource
from blockagi.resource_pool import ResourcePool
from blockagi.run import run_blockagi


app = FastAPI()


@app.get("/")
def get_index():
    return FileResponse("dist/index.html")


@app.get("/api/state")
def get_api_state():
    app.state.blockagi_state.resource_pool = app.state.resource_pool
    return app.state.blockagi_state


@dataclass
class StepHistory:
    timestamp: str
    value: str


@dataclass
class AgentLog:
    timestamp: str
    round: int
    message: str


@dataclass
class Status:
    step: str
    round: int


@dataclass
class LLMLog:
    prompt: str
    response: str


@dataclass
class BlockAGIState:
    start_time: str
    end_time: Optional[str]
    agent_role: str
    status: Status
    agent_logs: list[AgentLog]
    historical_steps: list[StepHistory]
    objectives: list[Objective]
    findings: list[Findings]
    resource_pool: ResourcePool
    llm_logs: list[LLMLog]
    narratives: list[Narrative]
    processing: bool = False  # Add a new field to indicate whether processing is in progress
    stop_thread: bool = False  # Add a new field to indicate whether the thread should stop

    def add_agent_log(self, message: str):
        self.agent_logs.append(
            AgentLog(
                timestamp=datetime.utcnow().isoformat(),
                round=self.status.round,
                message=message,
            )
        )

    # String representation of the class   
    def __str__(self):
        from pprint import pformat
        from dataclasses import asdict
        return pformat(asdict(self))    



""" def reset_state(objectives: List[str]):
    app.state.blockagi_state = BlockAGIState(
        start_time=datetime.utcnow().isoformat(),
        end_time=None,
        agent_role=app.state.blockagi_state.agent_role,
        status=Status(step="PlanChain", round=0),
        historical_steps=[],
        agent_logs=[
            AgentLog(datetime.utcnow().isoformat(), 0, f"You are {app.state.blockagi_state.agent_role}"),
            AgentLog(datetime.utcnow().isoformat(), 0, f"Using {app.state.openai_model}"),
        ],
        objectives = [Objective(topic=topic, expertise=0.0) for topic in objectives],
        #objectives=[],
        findings=[],
        resource_pool=ResourcePool(),
        llm_logs=[],
        narratives=[],
        processing=False,
        stop_thread=False
    )
 """

def reset_state(objectives: List[str]):
    print("reset_state")
    app.state.blockagi_state.start_time = datetime.utcnow().isoformat()
    app.state.blockagi_state.end_time = None
    app.state.blockagi_state.status = Status(step="PlanChain", round=0)
    app.state.blockagi_state.historical_steps = []
    app.state.blockagi_state.agent_logs = [
        AgentLog(datetime.utcnow().isoformat(), 0, f"You are {app.state.blockagi_state.agent_role}"),
        AgentLog(datetime.utcnow().isoformat(), 0, f"Using {app.state.openai_model}"),
    ]
    app.state.blockagi_state.objectives = [Objective(topic=topic, expertise=0.0) for topic in objectives]
    app.state.blockagi_state.findings = []
    #app.state.blockagi_state.resource_pool = ResourcePool()
    app.state.blockagi_state.resource_pool.clear()
    app.state.blockagi_state.llm_logs = []
    app.state.blockagi_state.narratives = []
    app.state.blockagi_state.processing = False
    app.state.blockagi_state.stop_thread = False
    app.state.blockagi_state.resource_pool.clear()




@app.post("/reset")
async def reset():
    reset_state()

""" 
@app.post("/api/objectives")
async def update_objectives(objectives: List[str]):
    # Do a reset first to clear the state
    print("Resetting...")
    reset_state()

    print("Updating objectives...")
    print(objectives)
    # todo: add check if Objectives are handed over correctly or emppty
    app.state.blockagi_state.objectives = [Objective(topic=topic, expertise=0.0) for topic in objectives]
    
    if app.state.blockagi_state.processing:  # Check if processing is already in progress
        return {"message": "Processing is already in progress"}

    app.state.blockagi_state.processing = True  # Set the processing flag to True

    def target(**kwargs):
        try:
            while not app.state.blockagi_state.stop_thread:  # Check the flag here
                run_blockagi(**kwargs)
        except Exception as e:
            app.state.blockagi_state.add_agent_log(f"Error: {e}")
        app.state.blockagi_state.end_time = datetime.utcnow().isoformat()
        app.state.blockagi_state.processing = False  # Set the processing flag to False when processing is done

    threading.Thread(
        target=target,
        kwargs=dict(
            agent_role=app.state.blockagi_state.agent_role,
            openai_api_key=app.state.openai_api_key,
            openai_model=app.state.openai_model,
            resource_pool=app.state.resource_pool,
            objectives=app.state.blockagi_state.objectives,
            blockagi_callback=BlockAGICallback(app.state.blockagi_state),
            llm_callback=LLMCallback(app.state.blockagi_state),
            iteration_count=app.state.iteration_count,
        ),
    ).start() """

@app.post("/api/objectives")
async def update_objectives(objectives: List[str]):
    # Always stop the old thread if it's running
    app.state.blockagi_state.stop_thread = True
    #wait for the thread to stop
    #while app.state.blockagi_state.processing:
    #    await asyncio.sleep(0.1)



    print("blockagi_state before reset:", app.state.blockagi_state)
    print( app.state.iteration_count)

    # Reset the state
    reset_state(objectives=objectives)

    print("blockagi_state after reset:", app.state.blockagi_state)
    print( app.state.iteration_count)

    # Update the objectives
    app.state.blockagi_state.objectives = [Objective(topic=topic, expertise=0.0) for topic in objectives]

    # Set the processing flag to True
    app.state.blockagi_state.processing = True

    # Define the target function for the thread
    def target(**kwargs):
        app.state.blockagi_state.stop_thread = False # we are starting a new thread, so set the stop_thread flag to False
        try:
            #do we really need this threading if not app.state.blockagi_state.stop_thread:  # Check the stop_thread flag
                run_blockagi(**kwargs)
        except Exception as e:
            app.state.blockagi_state.add_agent_log(f"Error: {e}")
        finally:
            app.state.blockagi_state.end_time = datetime.utcnow().isoformat()
            app.state.blockagi_state.processing = False  # Set the processing flag to False when processing is done
            #app.state.blockagi_state.stop_thread = True  

    # Start a new thread to run the blockagi
    threading.Thread(
        target=target,
        kwargs=dict(
            agent_role=app.state.blockagi_state.agent_role,
            openai_api_key=app.state.openai_api_key,
            openai_model=app.state.openai_model,
            resource_pool=app.state.resource_pool,
            objectives=app.state.blockagi_state.objectives,
            blockagi_callback=BlockAGICallback(app.state.blockagi_state),
            llm_callback=LLMCallback(app.state.blockagi_state),
            iteration_count=app.state.iteration_count,
        ),
    ).start()

  






app.mount("/", StaticFiles(directory="dist"), name="dist")


@app.on_event("startup")
def on_startup():
    app.state.resource_pool = ResourcePool()
    webbrowser.open(f"http://{app.state.host}:{app.state.port}")


@app.on_event("shutdown")
def on_shutdown():
    os._exit(0)


class BlockAGICallback(BlockAGICallbackHandler):
    state: BlockAGIState

    def __init__(self, blockagi_state):
        self.state = blockagi_state

    def on_iteration_start(self, inputs: Dict[str, Any]) -> Any:
        self.state.status.round += 1

    def on_log_message(self, message: str) -> Any:
        self.state.add_agent_log(message)

    def on_step_start(self, step, inputs, **kwargs):
        self.state.status.step = step

    def on_step_end(self, step, inputs, outputs, **kwargs):
        if step == "PlanChain":
            pass
        elif step == "ResearchChain":
            pass
        elif step == "NarrateChain":
            self.state.narratives.append(outputs["narrative"])
        elif step == "EvaluateChain":
            self.state.objectives = outputs["updated_objectives"]
            self.state.findings = outputs["updated_findings"]


class LLMCallback(BaseCallbackHandler):
    state: BlockAGIState

    def __init__(self, blockagi_state):
        self.state = blockagi_state

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.state.llm_logs.append(
            LLMLog(
                prompt="".join(prompts),
                response="",
            )
        )

    def on_llm_new_token(self, token: str, **kwargs):
        self.state.llm_logs[-1].response += token


def main(
    host: str = typer.Option(envvar="WEB_HOST"),
    port: int = typer.Option(envvar="WEB_PORT"),
    agent_role: str = typer.Option(envvar="BLOCKAGI_AGENT_ROLE"),
    iteration_count: int = typer.Option(envvar="BLOCKAGI_ITERATION_COUNT"),
    objectives: list[str] = typer.Option(None, "--objectives", "-o"),
    openai_api_key: str = typer.Option(envvar="OPENAI_API_KEY"),
    openai_model: str = typer.Option(envvar="OPENAI_MODEL"),
):
    app.state.host = host
    app.state.port = port
    if not objectives:
        for index in range(1, 11):
            key = f"BLOCKAGI_OBJECTIVE_{index}"
            if objective := os.getenv(key):
                objectives.append(objective.strip())
    if not objectives:
        raise ValueError("No objectives specified")

    app.state.openai_api_key = openai_api_key
    app.state.openai_model = openai_model
    app.state.iteration_count = iteration_count
    app.state.blockagi_state = BlockAGIState(
        start_time=datetime.utcnow().isoformat(),
        end_time=None,
        agent_role=agent_role,
        status=Status(step="PlanChain", round=0),
        historical_steps=[],
        agent_logs=[
            AgentLog(datetime.utcnow().isoformat(), 0, f"You are {agent_role}"),
            AgentLog(datetime.utcnow().isoformat(), 0, f"Using {openai_model}"),
        ],
        objectives=[Objective(topic=topic, expertise=0.0) for topic in objectives],
        findings=[],
        resource_pool=ResourcePool(),
        llm_logs=[],
        narratives=[],
    )
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    dotenv.load_dotenv()
    typer.run(main)
