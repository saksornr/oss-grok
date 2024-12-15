import streamlit as st
import requests
import json
import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os
load_dotenv()

# Streamlit page configuration
st.set_page_config(page_title="Tool-Enabled RAG Chatbot", page_icon="🤖", layout="centered")

# Define your API keys and model
XAI_API_KEY = os.getenv("XAI_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
MODEL_NAME = "grok-2-1212"
# Add this

# Placeholder imports for your custom OpenAI client
# Assuming `from openai import OpenAI` is a valid import for your environment
from openai import OpenAI

# Document text
document_text = """\
Medicare/Medicaid/Obamacare (Affordable Care Act):
Medicare: A federal program providing healthcare for individuals 65+ or younger people with disabilities. You can learn more at Medicare.gov.
Medicaid: A state and federal program offering healthcare to eligible low-income individuals. More details are available at Medicaid.gov.
Affordable Care Act (Obamacare): Enacted in 2010, it expanded Medicaid and created insurance marketplaces for affordable health coverage. Learn more at Healthcare.gov and HHS.gov.

Free Public School Education (Grade K–12):
Public education in the U.S. is available free of charge through high school (Grade 12) for all children. Information about public schools can typically be found on state or local school district websites. National resources include ed.gov, the U.S. Department of Education's official website.

Food Stamps (SNAP):
The Supplemental Nutrition Assistance Program (SNAP) provides food assistance to low-income families. To apply or learn more, visit USDA SNAP.

Section 8 Housing:
This federal program assists low-income families with housing costs through vouchers. You can find details and apply through your local Public Housing Authority (PHA) or visit the HUD website.
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
        top_indices = similarities.argsort()[-5:][::-1]  # Top 5 indices with highest similarity

        # Step 4: Retrieve the corresponding chunks
        top_docs = [self.chunks[idx] for idx in top_indices]

        return top_docs

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
        "description": "Search a document with keywords or query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A keyword or query to search from documents.",
                    "example_value": "What is Medicare?",
                },
            },
            "required": ["query"]
        },
    }
]

tools = [{"type": "function", "function": f} for f in functions]

# Set up initial system messages if needed
base_system_message = {"role": "system", "content": "You are a helpful agent assistant. Use the supplied tools to assist the user."}

if "messages" not in st.session_state:
    st.session_state["messages"] = [base_system_message]

st.title("RAG-Enabled Chatbot with Tools")

# Sidebar controls
with st.sidebar:
    st.markdown("## Options")
    if st.button("Show Full Chat Log"):
        st.write(st.session_state["messages"])
    if st.button("Clear Chat Log"):
        st.session_state["messages"] = [base_system_message]
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

    if assistant_msg.tool_calls != None:
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

        # Stream the response to the user
        # streamed_content = ""
        # assistant_placeholder = st.chat_message("assistant")
        # assistant_placeholder.markdown("...")
        # for chunk in final_response:
        #     if "choices" in chunk:
        #         for choice in chunk["choices"]:
        #             if "delta" in choice and "content" in choice["delta"]:
        #                 token = choice["delta"]["content"]
        #                 streamed_content += token
        #                 assistant_placeholder.markdown(streamed_content)
        with st.chat_message("assistant"):
            st.markdown(final_content)

        # Add assistant response to messages
        st.session_state["messages"].append({"role": "assistant", "content": final_content})

    else:
        # No tool call, just a direct response
        final_content = assistant_msg.content
        st.session_state["messages"].append({"role": "assistant", "content": final_content})
        with st.chat_message("assistant"):
            st.markdown(final_content)
