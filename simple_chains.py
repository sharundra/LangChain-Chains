from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()  # Load environment variables from .env file

model = ChatOpenAI()

template1 = PromptTemplate(template = "Write 3 points about {topic}", 
                          input_variables = ["topic"]
                          )
                           

parser = StrOutputParser()

chain = template1 | model | parser

result1 = chain.invoke({"topic": "Spielberg"})
print(result1)

chain.get_graph().print_ascii()