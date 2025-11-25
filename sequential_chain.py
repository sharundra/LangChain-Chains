from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()  # Load environment variables from .env file

model = ChatOpenAI()

template1 = PromptTemplate(template = "Write 3 detailed points about {topic}", 
                          input_variables = ["topic"]
                          )
template2 = PromptTemplate(template = "Write a simple summary of {topic}",
                           input_variables = ["topic"]
                           )

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result1 = chain.invoke({"topic": "Stanley Kubrik"})
print(result1)

# We could make this chain only because parser was there in between to extract result1.content otherwise explicitly we would have to extract result1.content and pass it to template2.invoke().

chain.get_graph().print_ascii()