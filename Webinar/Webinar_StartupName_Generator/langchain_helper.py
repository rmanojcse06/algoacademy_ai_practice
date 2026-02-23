import os
from openai_key import OPENAI_API_KEY  # import your key from openai_key.py
from langchain_community.llms import OpenAI
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from langchain_classic.chains import SequentialChain

# Set the environment variable so LangChain/OpenAI picks it up
# Set the environment variable so LangChain/OpenAI picks it up
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def generate_startup_info(startup_company: str) -> dict:
    llm = OpenAI(temperature=0.5)

    # Template for creating Startup name
    startup_name_prompt = PromptTemplate(
        input_variables=['startup_company'],
        template="Create a unique and catchy name for a {startup_company} startup."
    )

    # Chain for creating Startup name
    startup_name_chain = LLMChain(
        llm=llm,
        prompt=startup_name_prompt,
        output_key='startup_name'
    )

    # Template for creating Domain name
    domain_name_prompt = PromptTemplate(
        input_variables=['startup_name'],
        template="Suggest a .com domain name for the startup called {startup_name}."
    )

    # Chain for creating Startup domain name
    domain_name_chain = LLMChain(
        llm=llm,
        prompt=domain_name_prompt,
        output_key='domain_name'
    )

    # Sequential chain combining both
    startup_domain_chain = SequentialChain(
        chains=[startup_name_chain, domain_name_chain],
        input_variables=['startup_company'],
        output_variables=['startup_name', 'domain_name'],
        verbose=False
    )

    # ✅ Fixed: input variable name mismatch
    result = startup_domain_chain.invoke({'startup_company': startup_company})
    return result