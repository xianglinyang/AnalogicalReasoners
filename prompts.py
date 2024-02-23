from langchain_core.prompts import PromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

INSTRUCTION = """Solve the following question answering task and Finish with your answer. Finish[answer] returns the answer and finishes the task.
Question: {question}
"""

ZERO_SHOT_COT_INSTRUCTION = """Solve a question answering task by thinking step by step, then Finish with your answer. Thought can reason about the current situation. Finish[answer] returns the answer and finishes the task.
Question: {question}
"""

FEW_SHOT_COT_INSTRUCTION = """Solve a question answering task by having a Thought, then Finish with your answer. Thought can reason about the current situation. Finish[answer] returns the answer and finishes the task. You will be given context that you should use to help you answer the question.
Here are some examples:
{examples}
(END OF EXAMPLES)
Question: {question}
"""

ANALOGICAL_INSTRUCTION = """ Your task is to tackle mathematical problems.
When presented with a problem, recall relevant problems as examples. Afterward, 
proceed to solve the initial problem. Following the instruction below to give your answer.

Instructions:
## Relevant Problems:
Recall three examples of math problems that are relevant to the initial problem. Your problems should be distinct from each other and 
from the initial problem (e.g., involving different numbers and names). For each problem:
- After "Q: ", describe the problem 
- After "A: ", explain the solution.
## Solve the Initial Problem:
Q: Copy and paste the initial problem here.
A: Explain the solution.

Question: {question}
"""

ANALOGICAL_AUGMENTED_INSTRUCTION = """
Your task is to tackle mathematical problems.
Given a problem, explain the core concepts in it and provide other relevant problems. Then solve the original problem.
Following the instruction below to give your answer.

# Instruction:
## Algorithms:
Identify the core concepts or algorithms used to solve the problem. 
## Tutorial:
Write a tutorial about these algorithms.
## Example Problems:
Provide three examples of relevant competitive programming problems that involve these algorithms. For each problem, describe the problem, 
explain the solution in detail, and then write the correct Python3 code.
## Python3 code to solve the original problem:
- Explanation of the solution:
- Python3 code to solve the problem:
Question: {question}
"""

# role?
MATH_GOAL = "You are a math teacher."
CODING_GOAL = ""


direct_prompt = PromptTemplate(input_variables=['question'], template=INSTRUCTION)
zero_shot_cot_prompt = PromptTemplate(input_variables=['question'], template=ZERO_SHOT_COT_INSTRUCTION)
few_shot_cot_prompt = PromptTemplate(input_variables=['examples', 'question'], template=FEW_SHOT_COT_INSTRUCTION)
analogical_prompt = PromptTemplate(input_variables=['question'], template=ANALOGICAL_INSTRUCTION)
analogical_augmented_prompt = PromptTemplate(input_variables=['question'], template=ANALOGICAL_AUGMENTED_INSTRUCTION)

# Chat version
chat_direct_prompt = ChatPromptTemplate.from_messages(
    [
        # SystemMessage(
        #     content=(MATH_GOAL)
        # ),
        HumanMessagePromptTemplate.from_template(INSTRUCTION)
    ]
)
chat_zero_shot_cot_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(MATH_GOAL)
        ),
        HumanMessagePromptTemplate.from_template(ZERO_SHOT_COT_INSTRUCTION)
    ]

)
chat_few_shot_cot_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(CODING_GOAL)
        ),
        HumanMessagePromptTemplate.from_template(FEW_SHOT_COT_INSTRUCTION)
    ]

)
chat_analogical_prompt = ChatPromptTemplate.from_messages(
    [
        # SystemMessage(
        #     content=(MATH_GOAL)
        # ),
        HumanMessagePromptTemplate.from_template(ANALOGICAL_INSTRUCTION)
    ]

)
chat_analogical_augmented_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(MATH_GOAL)
        ),
        HumanMessagePromptTemplate.from_template(ANALOGICAL_AUGMENTED_INSTRUCTION)
    ]

)


