from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
from langchain.schema.runnable import RunnableParallel
from dotenv import load_dotenv
import os
load_dotenv()

# Load the SLR.txt file
with open("SLR.txt", "r", encoding="utf-8") as file:
    slr_content = file.read()

print("SLR.txt file loaded successfully!")
print(f"Content length: {len(slr_content)} characters")


hf_token = os.getenv("HUGGINGFACE_API_KEY_FINEGRAINED")

model1=HuggingFaceEndpoint(endpoint_url="Qwen/Qwen3-Next-80B-A3B-Instruct",
                    task = "text-generation",
                    huggingfacehub_api_token=hf_token
                    )


llm1 = ChatHuggingFace(llm = model1, temperature=0.5)
llm2 = ChatOpenAI(model="gpt-4o", temperature=0.5)
str_parser = StrOutputParser()
json_parser = JsonOutputParser()

template1 = PromptTemplate(
    template="""
    Consider youself a student of Machine Learning. Now your task is to generate a detailed notes on the following text:{slr_text}
    Focus on the main statistical concepts, formulas, and applications.
    """,
    input_variables=["slr_text"]
)

template2 = PromptTemplate(
    template="""
    Generate 10 questions and their respective answers from the {slr_text}. 
    Each questions and and answers should be numbered.
    """,
    input_variables=["slr_text"]
)

template3 = PromptTemplate(
        template="""
        You are given two texts:
        - {notes}: Detailed notes on Simple Linear Regression.
        - {quiz}: A list of 10 questions and their answers.

        For each question in the quiz, do the following:
        1. Extract the question and answer as they appear in the quiz.
        2. Find the most relevant context (a paragraph or section) from the notes that best explains or answers the question.
        3. Output a JSON object with keys 'q1', 'q2', ..., each mapping to an object with fields:
             - "question": the question text
             - "answer": the answer text
             - "context": the extracted context from the notes

        Output ONLY a valid JSON object, with no markdown, no LaTeX, no code blocks, and no extra text. Do not use triple backticks. Do not include any explanations or formatting other than the JSON object.

        Example output:
        {{
            "q1": {{"question": "What is SLR?", "answer": "A method to model linear relationships.", "context": "Simple Linear Regression (SLR) is ..."}},
            "q2": {{"question": "What is the formula?", "answer": "y = b0 + b1*x", "context": "The formula for SLR is ..."}}
        }}

        Use this format: {format_specification}
        """,
        input_variables=["notes", "quiz"],
        partial_variables={"format_specification": json_parser.get_format_instructions()}
)

# Create the paralle chain
parallel_chain = RunnableParallel({
                    "notes":template1|llm1|str_parser,
                    "quiz":template2|llm2|str_parser})


# Use JsonOutputParser for strict JSON parsing
collate_chain = template3 | llm2 | json_parser

chain = parallel_chain | collate_chain

result = chain.invoke({"slr_text": slr_content})

print("\n===== FINAL RESULT =====\n")
print(type(result))
print(result)

print("\n===== CHAIN GRAPH =====\n")
chain.get_graph().print_ascii()