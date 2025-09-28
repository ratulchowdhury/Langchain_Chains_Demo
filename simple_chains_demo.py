from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
load_dotenv()


hf_token = os.getenv("HUGGINGFACE_API_KEY_FINEGRAINED")

model=HuggingFaceEndpoint(endpoint_url="Qwen/Qwen3-Next-80B-A3B-Instruct",
                    task = "text-generation",
                    huggingfacehub_api_token=hf_token
                    )

template= PromptTemplate(
    template="""
    Generate 5 key points about the book mentioned in the following {book}.
    """,
    input_variables=["book"]
)
llm = ChatHuggingFace(llm = model, temperature=0.5)
parser = StrOutputParser()

chain = template | llm | parser

result = chain.invoke({"book": "Norweigian Wood"})

print(result)
chain.get_graph().print_ascii()
