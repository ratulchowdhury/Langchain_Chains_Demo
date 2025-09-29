from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
from dotenv import load_dotenv
import os
load_dotenv()


hf_token = os.getenv("HUGGINGFACE_API_KEY_FINEGRAINED")

model=HuggingFaceEndpoint(endpoint_url="Qwen/Qwen3-Next-80B-A3B-Instruct",
                    task = "text-generation",
                    huggingfacehub_api_token=hf_token
                    )

template1 = PromptTemplate(
    template="""
    Generate a detailed summary about the book mentioned in the following {book}.
    """,
    input_variables=["book"]
)
llm = ChatHuggingFace(llm = model, temperature=0.5)


str_parser = StrOutputParser()

json_parser = JsonOutputParser()

template2 = PromptTemplate(
    template="""
    Extract 5 key key facts about the from the {summary} in \n {format_specifications}.
    """,
    input_variables=["summary"],
    partial_variables = {"format_specifications":json_parser.get_format_instructions()}
    )
chain = template1 | llm | str_parser | template2 | llm | json_parser

result = chain.invoke({"book": "Norweigian Wood"})

print(result)
chain.get_graph().print_ascii()