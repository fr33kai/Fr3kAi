
# /combined_assistant.py
# This is an optimized version combining memory handling, RAG, web search, self-improvement, and basic generation

import streamlit as st
from pocketgroq import GroqProvider
from pocketgroq.web_tool import WebTool
import json
import base64
import requests
import pickle

# Function to safely parse JSON
def safe_parse_json(response_text):
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return None

# Function to encode image
def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode('utf-8')

# Optimized function to get web content with error handling
def get_web_content(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch web content: {e}")
        return None

# Memory handling functions using caching and local storage
@st.cache_data(persist=True)
def save_memory(memory):
    with open('assistant_memory.pkl', 'wb') as f:
        pickle.dump(memory, f)

@st.cache_data(persist=True)
def load_memory():
    try:
        with open('assistant_memory.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []
if 'groq' not in st.session_state:
    st.session_state['groq'] = None
if 'web_tool' not in st.session_state:
    st.session_state['web_tool'] = None
if 'memory' not in st.session_state:
    st.session_state['memory'] = load_memory()
if 'api_key_valid' not in st.session_state:
    st.session_state['api_key_valid'] = False

# Set up Streamlit UI
st.title("Enhanced AI Assistant")

# Input field for API key
groq_api_key = st.text_input("Enter your Groq API Key", type="password")

# Test API key button
def test_api_key(api_key):
    try:
        test_groq = GroqProvider(api_key=api_key)
        test_response = test_groq.generate("Test API key.")
        return True
    except Exception as e:
        st.error(f"API key test failed: {str(e)}")
        return False

if st.button("Test API Key"):
    if groq_api_key:
        if test_api_key(groq_api_key):
            st.session_state['api_key_valid'] = True
            st.success("API key is valid!")
            st.session_state['groq'] = GroqProvider(api_key=groq_api_key)
            st.session_state['web_tool'] = WebTool()
        else:
            st.session_state['api_key_valid'] = False
    else:
        st.warning("Please enter an API key.")

# Only show features if API key is valid
if st.session_state['api_key_valid']:

    # Feature selection
    feature = st.selectbox("Select a feature", ["Basic Generation", "RAG", "Chain of Thought", "Vision", "Web Search", "Self-Improvement"])

    if feature == "Basic Generation":
        prompt = st.text_area("Enter your prompt")
        if st.button("Generate"):
            if st.session_state['groq']:
                context = f"Previous interactions: {str(st.session_state['conversation'])}\nMemory: {str(st.session_state['memory'])}\n\n"
                full_prompt = context + prompt
                response = st.session_state['groq'].generate(full_prompt)
                st.write("Response:", response)
                st.session_state['conversation'].append({"role": "user", "content": prompt})
                st.session_state['conversation'].append({"role": "assistant", "content": response})

                # Update memory
                memory_prompt = f"Based on the following conversation, what key information should I remember?\n\n{prompt}\n{response}"
                memory_update = st.session_state['groq'].generate(memory_prompt)
                st.session_state['memory'][prompt] = memory_update
                save_memory(st.session_state['memory'])

    elif feature == "RAG":
        input_type = st.radio("Select input type", ["File Upload", "Web URL"])

        if input_type == "File Upload":
            uploaded_file = st.file_uploader("Upload a document for RAG", type=["txt", "pdf"])
        else:
            url = st.text_input("Enter a webpage URL")

        query = st.text_input("Enter your query")
        if st.button("Query with RAG"):
            if st.session_state['groq']:
                context = ""
                if input_type == "File Upload" and uploaded_file:
                    context = uploaded_file.getvalue().decode()
                elif input_type == "Web URL" and url:
                    context = get_web_content(url)
                    if not context:
                        st.error("Failed to fetch web content. Please check the URL.")
                        st.stop()

                prompt = f"Context: {context}\n\nQuery: {query}\n\nAnswer:"
                response = st.session_state['groq'].generate(prompt)
                st.write("RAG Response:", response)
                st.session_state['conversation'].append({"role": "user", "content": f"RAG Query: {query}"})
                st.session_state['conversation'].append({"role": "assistant", "content": response})

                # Update memory
                memory_prompt = f"Based on the RAG query and response, what key information should I remember?\n\nQuery: {query}\nResponse: {response}"
                memory_update = st.session_state['groq'].generate(memory_prompt)
                st.session_state['memory'][f"RAG: {query}"] = memory_update
                save_memory(st.session_state['memory'])

    elif feature == "Web Search":
        query = st.text_input("Enter a search query")
        if st.button("Search and Analyze"):
            if st.session_state['groq'] and st.session_state['web_tool']:
                try:
                    search_results = st.session_state['web_tool'].search(query)
                    analysis = ""
                    for result in search_results[:3]:
                        content = get_web_content(result['url'])
                        if content:
                            analysis_prompt = f"Analyze the following content about {query}:\n\n{content[:4000]}"
                            analysis += st.session_state['groq'].generate(analysis_prompt) + "\n\n"
                    st.write("Web Search Analysis:", analysis)
                    st.session_state['conversation'].append({"role": "user", "content": f"Web Search: {query}"})
                    st.session_state['conversation'].append({"role": "assistant", "content": analysis})

                    # Update memory
                    memory_prompt = f"Based on the web search and analysis, what key information should I remember?\n\nQuery: {query}\nAnalysis: {analysis}"
                    memory_update = st.session_state['groq'].generate(memory_prompt)
                    st.session_state['memory'][f"Web Search: {query}"] = memory_update
                    save_memory(st.session_state['memory'])
                except Exception as e:
                    st.error(f"Error during Web Search: {e}")

    elif feature == "Self-Improvement":
        if st.button("Generate Self-Improvement Analysis"):
            if st.session_state['groq']:
                context = f"Previous interactions: {str(st.session_state['conversation'])}\nMemory: {str(st.session_state['memory'])}\n"
                prompt = f"{context}Analyze my performance and suggest improvements in response quality, understanding, and memory usage."
                analysis = st.session_state['groq'].generate(prompt)
                st.write("Self-Improvement Analysis:", analysis)

                # Generate actionable feedback for improvement
                action_prompt = f"Based on the analysis, provide 3-5 action items for improvement:\n\n{analysis}"
                action_items = st.session_state['groq'].generate(action_prompt)
                st.write("Action Items for Improvement:", action_items)

                # Update memory with improvement plan
                st.session_state['memory']['self_improvement'] = {'last_analysis': analysis, 'action_items': action_items}
                save_memory(st.session_state['memory'])

                # Attempt to implement improvements
                implementation_prompt = f"Based on these action items, update my base prompt to incorporate these improvements:\n\n{action_items}"
                updated_base_prompt = st.session_state['groq'].generate(implementation_prompt)
                st.session_state['memory']['updated_base_prompt'] = updated_base_prompt
                save_memory(st.session_state['memory'])
                st.success("Self-improvement analysis complete and base prompt updated!")

    # Display conversation history
    if st.session_state['conversation']:
        st.write("### Conversation History:")
        for entry in st.session_state['conversation']:st.write(f"{entry['role'].capitalize()}: {entry['content']}")

    # Display memory in a collapsible section
    if st.session_state['memory']:
        st.write("### Assistant's Memory:")
        for key, value in st.session_state['memory'].items():
            with st.expander(f"Memory key: {key}", expanded=False):
                st.write(f"Value: {value}")

    # Follow-up questions
    st.write("### Follow-up Questions")
    follow_up = st.text_input("Ask a follow-up question based on the previous responses")
    if st.button("Submit Follow-up"):
        if st.session_state['groq'] and st.session_state['conversation']:
            context = "\n".join([f"{entry['role']}: {entry['content']}" for entry in st.session_state['conversation']])
            prompt = f"Based on the following conversation:\n\n{context}\n\nFollow-up question: {follow_up}\n\nResponse:"
            response = st.session_state['groq'].generate(prompt)
            st.write("Follow-up Response:", response)
            st.session_state['conversation'].append({"role": "user", "content": follow_up})
            st.session_state['conversation'].append({"role": "assistant", "content": response})

    # Clear conversation and memory
    if st.button("Clear Conversation and Memory"):
        st.session_state['conversation'] = []
        st.session_state['memory'] = {}
        save_memory(st.session_state['memory'])
        st.success("Conversation and memory cleared!")

else:
    st.warning("Please enter a valid API key to use the features.")

# Footer
st.markdown("---")
st.write("Powered by Groq and Streamlit")
