from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url
from dotenv import load_dotenv
import os

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY is missing")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    max_retries=2
)


#1st agent 
# def build_search_agent():
#     return create_agent(
#         model = llm,
#         tools= [web_search]
#     )

def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt="""
        You are a web research agent.

        Call the web_search tool exactly once.
        Use the five results returned by that single call.
        After receiving the tool results, do not call the tool again.
        Return the titles, URLs and important information, then stop.
        """
    )

#2nd agent 

# def build_reader_agent():
#     return create_agent(
#         model = llm,
#         tools = [scrape_url]
#     )

def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt="""
        You are a research reading agent.

        Select a maximum of three URLs from the provided results.
        Scrape each selected URL only once.
        Do not scrape more than three URLs.
        After scraping them, combine the findings and stop.
        Preserve every source URL.
        """
    )


#writer chain 

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()

#critic_chain 

critic_prompt = ChatPromptTemplate.from_messages([
     ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()

#reviser_chain
revision_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert research editor.
        Improve reports using critic feedback.
        Preserve valid source URLs and never invent sources."""
    ),
    (
        "human",
        """Improve the following research report.

Topic:
{topic}

Research:
{research}

Original Report:
{report}

Critic Feedback:
{feedback}

Return the complete improved report with:

- Introduction
- Key Findings
- Conclusion
- Sources
"""
    ),
])

revision_chain = revision_prompt | llm | StrOutputParser()

