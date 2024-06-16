from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import OpenAI
from langchain_groq import ChatGroq
from langchain.agents import load_tools
from langchain.prompts import PromptTemplate
import os

#分別創建Tavily,LangChain,Serper三個不同的工具的API KEY
os.environ["TAVILY_API_KEY"] = "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"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["SERPER_API_KEY"] = '642c5229e166492e931c5c6f98f5a22c1a78582b'

#以下是工具模板
#prompt = hub.pull("hwchase17/react")
template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action(don't need to show url)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
(in this part don't need to print the url)
'''

prompt = PromptTemplate.from_template(template)
                                          #代入09作業裡創建GROQ的API KEY
llm = ChatGroq(temperature=0, groq_api_key="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", model_name="mixtral-8x7b-32768")
#先讓Tavily做網路查詢
tools = [TavilySearchResults(max_results=2)]
#在運用agent,agent_executor做資料執行與處理
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
#輸入問題
agent_executor.invoke({"input": "人的血管有多長?總長可繞地球幾圈?"})
