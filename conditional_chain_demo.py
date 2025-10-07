from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
import os
load_dotenv()


llm = ChatOpenAI(model="gpt-4o", temperature=0.5)

str_parser = StrOutputParser()

class review(BaseModel):
    sentiment : Literal["positive","negative"] = Field(description="Sentiment of the review")
    
pydantic_parser = PydanticOutputParser(pydantic_object=review)

template1 = PromptTemplate(template=
                           """
                           Gauge the sentiment of the following review and classify it into one of the categories: Positive, Negative.
                            Review: {review}, in the specified format: {format_specifications}.
                           """,
                           input_variables=["review"],
                           partial_variables={"format_specifications": pydantic_parser.get_format_instructions()})

sentiment_chain = template1 | llm | pydantic_parser


template2 = PromptTemplate(template=
                           """
                           Can yourself a writer. Now write an appropriate response to this positive feedback: {feedback}
                           """,
                           input_variables=["feedback"])


template3 = PromptTemplate(template=
                           """
                           Can yourself a writer. Now write an appropriate response to this negative feedback: {feedback}
                           """,
                           input_variables=["feedback"])

conditional_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", template2|llm|str_parser),
    (lambda x: x.sentiment == "negative", template3|llm|str_parser),
    RunnableLambda(lambda x : "No response was found.")
)

chain = sentiment_chain | conditional_chain 
result = chain.invoke({"review":"While Norwegian Wood undoubtedly captures a distinctive mood, its slow pacing and heavy reliance on introspection can make the reading experience feel lethargic. Murakami’s trademark surrealism is absent here, which may disappoint fans expecting his usual magical realism; instead, the narrative often sinks into repetitive conversations about love and death without offering resolution. The main character, Toru, sometimes comes across as emotionally detached to the point of frustrating passivity, making it hard for all readers to truly empathize with him. Moreover, female characters tend to be idealized or defined primarily through their relationships with Toru, which risks reducing their complexity. For some, the novel’s persistent bleakness may border on monotony rather than poignancy."})

print(result)
chain.get_graph().print_ascii()







