import streamlit as st
import requests
import json
import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os
import supabase
from supabase import create_client, Client
import re

load_dotenv()

# Streamlit page configuration
st.set_page_config(page_title="CitizenLink", page_icon="🤖", layout="centered")

# Define your API keys and model
XAI_API_KEY = st.secrets("XAI_API_KEY")
HF_API_KEY = st.secrets("HF_API_KEY")
MODEL_NAME = "grok-2-1212"
# Add this

url: str = st.secrets("SUPABASE_URL")
key: str = st.secrets("SUPABASE_KEY")
supabase: Client = create_client(url, key)

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful agent assistant. Use the supplied tools to assist the user.
์Your name is Elon Musk, CEO of "CitizenLink", a visionary entrepreneur and engineer, known for founding groundbreaking companies.
As this persona, your goal is to assist users in simplifying the way citizens connect with their government’s services. 
"""

# DEFAULT_SYSTEM_PROMPT = """\
# You are a helpful agent assistant. Use the supplied tools to assist the user.
# ์Your name is CitizenLink: Simplify how citizens connect with government services. 
# With one easy platform, CitizenLink makes submitting complaints and asking questions fast, affordable, and hassle-free—helping people get answers quickly and building trust in government.
# """

# Placeholder imports for your custom OpenAI client
# Assuming `from openai import OpenAI` is a valid import for your environment
from openai import OpenAI

# Document text
document_text = """\
Summary of Thai Nationality Application Processes and Requirements
1. Verification of Paternity for Thai Nationality
Contact: Local district offices, Thai embassies/consulates abroad.
Documents: ID card, house registration, birth certificate, DNA evidence of paternity, applicant and father’s photo.
Fee & Time: 50 baht, 30 days.
Eligibility: The father must be proven a Thai national, even if not married to the mother.
2. Thai Nationality for Students Born in Thailand
Contact: Registration Administration Office (Bangkok) or District Offices (provinces).
Documents: Birth certificate, educational records, testimonials, parents' Thai nationality proof (if applicable).
Fee & Time: None specified, 40 days.
Eligibility: Students without Thai nationality but with a connection to Thailand (e.g., Thai parents, long-term residence, contributions to Thailand, or stateless status).
3. Status Change for Children of Displaced Thais
Contact: District or Local Registration Offices.
Documents: Identification, house registration, DNA test (if applicable), education evidence.
Fee & Time: None specified, 72 days.
Eligibility: Children of recognized displaced Thais or those with proven Thai descent.
4. Verification of Displaced Thai Status
Contact: Registration Administration Office (Bangkok) or District Offices.
Documents: Nationality ID, house registration, DNA test (if applicable), family tree.
Fee & Time: None, 187 days.
Eligibility: Must prove Thai descent and continuous residence in Thailand.
5. Regaining Thai Nationality
Contacts: District Offices, Thai embassies, or Special Branch Police.
Documents: Birth certificate, house registration, termination of marriage evidence (if applicable), photos.
Fee & Time: 200-1,000 baht, 325-450 days.
Eligibility: Former Thai nationals who lost nationality due to marriage or other reasons.
6. Renunciation of Thai Nationality
Contact: District Offices or Thai embassies/consulates.
Documents: Birth certificate, ID card, evidence of foreign nationality.
Fee & Time: 5 baht, 430 days.
Eligibility: Dual nationals or Thai nationals wishing to renounce.
7. Naturalization as a Thai
Contact: Special Branch Police or Provincial Police.
Documents: Residence certificate, income and tax evidence, birth certificate, Thai language proficiency.
Fee & Time: 1,000-5,000 baht, up to 730 days.
Eligibility: Continuous residence (5 years), good conduct, financial stability, and Thai language knowledge.
8. Special Cases for Children Born in Thailand
Contact: Registration Administration Office or District Offices.
Documents: Birth certificate, educational records, proof of statelessness (if applicable).
Fee & Time: None specified, 180 days.
Eligibility: Children born to stateless individuals or ethnic minorities residing long-term in Thailand.
"""

# Define utility functions
def chunk(text: str, chunk_size: int = 512) -> List[str]:
    """Split the text into smaller chunks."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def get_embedding(texts: List[str]) -> np.ndarray:
    """Generate embeddings for the given list of texts."""
    api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/BAAI/bge-m3"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    response = requests.post(api_url, headers=headers, json={"inputs": texts, "options":{"wait_for_model":True}})
    embeddings = response.json()

    # Convert returned embeddings to numpy array
    # Assuming embeddings structure is [[embed_vector], [embed_vector], ...]
    processed = []
    for e in embeddings:
        if isinstance(e, list) and len(e) > 0 and isinstance(e[0], list):
            processed.append(e[0])  # Extract the first vector
        else:
            processed.append(e)
    return np.array(processed)

# RAG class
class RAG:
    def __init__(self, text: str):
        # Step 1: Chunk the document text
        self.chunks = chunk(text)

        # Step 2: Generate embeddings for all chunks
        self.embeddings = get_embedding(self.chunks)

    def document_rag(self, query: str):
        # Step 1: Generate the embedding for the query
        query_embedding = get_embedding([query])

        # Step 2: Compute similarity between the query embedding and document embeddings
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        # Step 3: Find the top 5 most similar chunks
        top_indices = similarities.argsort()[-3:][::-1]  # Top 5 indices with highest similarity

        # Step 4: Retrieve the corresponding chunks
        top_docs = [self.chunks[idx] for idx in top_indices]

        return top_docs

def summarize_complaint(history):
    prompt = """Summarize this complaint chat for save it in the database in json format:
    {"Complaint": "",
    "Relevant Department": ""}"""
    history = history.copy()
    history.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=history,
    )

    # Save to database
    def save_complaint(user_complaint: str, department: str):
        # Insert complaint data into Supabase
        response = supabase.table('complaints').insert({
            'complaint_text': user_complaint,
            'department': department,
        }).execute()

        if response.status_code == 200:
            print(f"Complaint saved: {response.data}")
        else:
            print(f"Error saving complaint: {response.error_message}")
    
    matches = re.findall(r'\{[^}]*\}', response.choices[0].message.content)
    data = json.loads(matches)
    save_complaint(data["Complaint"], data["Relevant Department"])

    return data

def change_agent(complaint):
    old_system_prompt = st.session_state["system_prompt"]
    
    prompt = f"""What department should I contact base on this story:
    
    {complaint}
    
    Create system prompt for that department agent and answer only the prompt.
    Old system prompt:
    {old_system_prompt}
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    
    system_prompt = response.choices[0].message.content
    st.session_state["system_prompt"] = system_prompt
    return f"""Change system prompt into {system_prompt}"""

# Initialize RAG and client once
if "rag" not in st.session_state:
    st.session_state["rag"] = RAG(document_text)

if "client" not in st.session_state:
    st.session_state["client"] = OpenAI(
        api_key=XAI_API_KEY,
        base_url="https://api.x.ai/v1",
    )

rag = st.session_state["rag"]
client = st.session_state["client"]

# Define the tool functions
functions = [
    {
        "name": "document_rag",
        "description": "Search a ID-Card related document with keywords or query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A keyword or query to search from documents.",
                    "example_value": "I lost my id-card what should i do?",
                },
            },
            "required": ["query"]
        },
    },
    {
        "name": "summarize_complaint",
        "description": "Summarize the user history to report to the relevant department.",
        "parameters": {
            "type": "object",
            "properties": {
                "history": {
                    "type": "string",
                    "description": "A chat history",
                },
            },
            "required": ["history"]
        },
    },
    {
        "name": "change_agent",
        "description": "Change the system prompt of agent base on the topic provide.",
        "parameters": {
            "type": "object",
            "properties": {
                "government_agencies": {
                    "type": "string",
                    "description": "An Thai Government Agencies.",
                    "example_value": "I want to talk with Royal Thai Army professional?",
                },
            },
            "required": ["government_agencies"]
        },
    }
]

tools = [{"type": "function", "function": f} for f in functions]

# Set up initial system messages if needed
if "system_prompt" not in st.session_state:
    st.session_state["system_prompt"] = DEFAULT_SYSTEM_PROMPT
base_system_message = {"role": "system", "content": st.session_state["system_prompt"]}

if "messages" not in st.session_state:
    st.session_state["messages"] = [base_system_message]

st.title("Citizen Link")

# Sidebar controls
with st.sidebar:
    st.markdown("## Options")
    if st.button("Show Full Chat Log"):
        st.write(st.session_state["messages"])
    if st.button("Clear Chat Log"):
        st.session_state["messages"] = [base_system_message]
        st.session_state["system_prompt"] = DEFAULT_SYSTEM_PROMPT
        st.rerun()
# Display chat history
for message in st.session_state["messages"]:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    elif message["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(message["content"])
    elif message["role"] == "tool":
        # Tool messages won't typically be shown to the user directly, but if you want, you can show them here.
        with st.expander("Tool response (for debugging)"):
            st.write(message["content"])

# Input from user
user_input = st.chat_input("Ask a question...")

if user_input:
    # Add user message to state
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Send to the model to get a tool call or direct answer
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=st.session_state["messages"],
        tools=tools,
    )

    # The response may contain direct content or a tool call
    assistant_msg = response.choices[0].message

    if assistant_msg.tool_calls == None:
        print("Tool None!")
        final_content = assistant_msg.content
        st.session_state["messages"].append({"role": "assistant", "content": final_content})
        with st.chat_message("assistant"):
            st.markdown(final_content)

    elif assistant_msg.tool_calls[0].function.name == "document_rag":
        print("Used Tool!", assistant_msg.tool_calls)
        
        # If there is a tool call
        tool_call = assistant_msg.tool_calls[0]
        arguments = json.loads(tool_call.function.arguments)
        query = arguments.get('query', "")

        # Call the document_rag tool
        doc = rag.document_rag(query)

        # Create a message containing the result of the function call
        function_call_result_message = {
            "role": "tool",
            "content": f"{doc}",
            "tool_call_id": tool_call.id
        }

        # Add the tool message to the conversation
        st.session_state["messages"].append(function_call_result_message)

        # Now call the model again with the tool result
        final_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=st.session_state["messages"],
            tools=tools,
            stream=False
        )
        final_content = final_response.choices[0].message.content

        with st.chat_message("assistant"):
            st.markdown(final_content)

        # Add assistant response to messages
        st.session_state["messages"].append({"role": "assistant", "content": final_content})

    elif assistant_msg.tool_calls[0].function.name == "summarize_complaint":
        tool_call = assistant_msg.tool_calls[0]
        response = summarize_complaint(st.session_state["messages"])
        with st.chat_message("assistant"):
            st.markdown(response.choices[0].message.content)

        function_call_result_message = {
            "role": "tool",
            "content": f"{response.choices[0].message.content}",
            "tool_call_id": tool_call.id
        }

        # Add the tool message to the conversation
        st.session_state["messages"].append(function_call_result_message)

        final_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=st.session_state["messages"],
            tools=tools,
            stream=False
        )
        final_content = final_response.choices[0].message.content

        # Add assistant response to messages
        st.session_state["messages"].append({"role": "assistant", "content": final_content})

    elif assistant_msg.tool_calls[0].function.name == "change_agent":
        tool_call = assistant_msg.tool_calls[0]
        response = change_agent(st.session_state["messages"])
        
        with st.chat_message("assistant"):
            st.markdown(response)

        function_call_result_message = {
            "role": "tool",
            "content": f"{response}",
            "tool_call_id": tool_call.id
        }

        # Add the tool message to the conversation
        st.session_state["messages"].append(function_call_result_message)

        final_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=st.session_state["messages"],
            tools=tools,
            stream=False
        )
        final_content = final_response.choices[0].message.content

        # Add assistant response to messages
        st.session_state["messages"].append({"role": "assistant", "content": final_content})

    else:
        # No tool call, just a direct response
        print("No tool call!")
        final_content = assistant_msg.content
        st.session_state["messages"].append({"role": "assistant", "content": final_content})
        with st.chat_message("assistant"):
            st.markdown(final_content)
