import os
from dotenv import load_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig
from openai import OpenAI


load_dotenv()

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY environment variable is not set.")

external_client = AsyncOpenAI(
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
)

model = OpenAIChatCompletionsModel(
        model="sarvamai/sarvam-m:free",    # in open router this model can be changed
        openai_client=external_client,
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
)


agent = Agent(
    name="Translation Agent",
    instructions="Translate the given text from English to french."
)

response = Runner.run_sync(
    agent,
    input="my name is safwan, i am student at Governor house sindh karachi",
    run_config=config,
)

print(response)