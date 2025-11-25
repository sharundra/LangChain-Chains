from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.runnables import RunnableBranch, RunnableLambda

load_dotenv()  # Load environment variables from .env file


class Feedback(BaseModel):
    sentiment : Literal["positive", "negative"] = Field(description="represents sentiment of a given text")

parser1 = StrOutputParser()
parser2 = PydanticOutputParser(pydantic_object = Feedback)

model = ChatOpenAI()

template1 = PromptTemplate(template = "give me a sentiment of the followiong text\n: {topic} \n {format_instruction}",
                           input_variables = ["topic"],
                           partial_variables= {"format_instruction":parser2.get_format_instructions()}
                           )

template2 = PromptTemplate(template = "Based on the received positive feedback, \n{feedback}, repond",
                           input_variables=["feedback"]
                           )

template3 = PromptTemplate(template = "Based on the received negative feedback, \n{feedback}, respond",
                           input_variables = ["feedback"]
                           )

branch_chain = RunnableBranch(
                (lambda x:x.sentiment == 'positive', template2 | model | parser1),
                (lambda x:x.sentiment == 'negative', template3 | model | parser1),
                (RunnableLambda(lambda x: "could not find sentiment"))
                )


chain = template1 | model | parser2 | branch_chain | parser1

result = chain.invoke({"topic": "This headphone needs some work on it"})
print(result)

chain.get_graph().print_ascii()
