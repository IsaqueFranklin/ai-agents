#Import all the necessary stuff
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_functions import format_to_openai_function_messages
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser

#Load the environment variables
load_dotenv()

#Define wich LLM to use
llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)
# Add chat history to the Agent as short term memory.

chat_history = []

# Add tools to the Agent to extent capabilities.
tools = []

# Define the chat prompt 
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant named Carlito. You have a geeky, sarcastic, and edgy sense of humor."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Define the agent 
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_function_messages(x["intermediate_steps"]),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm
    | OpenAIFunctionsAgentOutputParser()
)

agent_executor = AgentExecutor(agent = agent, tools=tools, verbose=True)

# Run the agent
user_task = "How can I build my own AI agent Saas. Give me cool ideias an examples."
output = agent_executor.invoke({"input": user_task, "chat_history": chat_history})
print(output['output'])
