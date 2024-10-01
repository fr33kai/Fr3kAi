# Streamlit AI Assistant Application

import sys
import json
import pickle
import streamlit as st
from pocketgroq import GroqProvider, GroqAPIKeyMissingError

# Function to save memory state
def save_memory():
    with open('memory.pkl', 'wb') as f:
        pickle.dump(st.session_state['memory'], f)

# Function to load memory state
def load_memory():
    try:
        with open('memory.pkl', 'rb') as f:
            st.session_state['memory'] = pickle.load(f)
    except FileNotFoundError:
        st.session_state['memory'] = {}

# Function to apply improvements suggested by the AI
def apply_improvements(suggested_improvements):
    if 'improvements' not in st.session_state:
        st.session_state['improvements'] = []
    st.session_state['improvements'].append(suggested_improvements)
    st.session_state['memory']['improvements'] = st.session_state['improvements']
    save_memory()
    st.success("Improvements applied! The application will use these in future interactions.")

# Function to verify if improvements are applied
def verify_improvements():
    improvements = st.session_state['memory'].get('improvements', [])
    if improvements:
        st.write("Current improvements applied:")
        for i, improvement in enumerate(improvements, 1):
            st.write(f"{i}. {improvement}")
    else:
        st.write("No improvements have been applied yet.")

# Function to export memory
def export_memory():
    return json.dumps(st.session_state['memory'])

# Function to import memory
def import_memory(memory_json):
    try:
        st.session_state['memory'] = json.loads(memory_json)
        save_memory()
        st.success("Memory imported successfully!")
    except json.JSONDecodeError:
        st.error("Invalid JSON format. Please check your input.")

# Function to generate response
def generate_response(prompt, context=""):
    if st.session_state['groq'] and st.session_state['conversation']:
        conversation_context = "\n".join([f"{entry['role']}: {entry['content']}" for entry in st.session_state['conversation'][-5:]])
        full_prompt = f"{context}\n\nBased on the following conversation:\n\n{conversation_context}\n\nUser: {prompt}\n\nAssistant:"
        response = st.session_state['groq'].generate(full_prompt)
        return response
    return "Error: Unable to generate response."

# Function to initialize session state variables
def initialize_session_state():
    if 'interaction_count' not in st.session_state:
        st.session_state['interaction_count'] = 0
    if 'conversation' not in st.session_state:
        st.session_state['conversation'] = []
    if 'memory' not in st.session_state:
        load_memory()
    if 'url_cache' not in st.session_state:
        st.session_state['url_cache'] = {}
    if 'api_key_valid' not in st.session_state:
        st.session_state['api_key_valid'] = False
    if 'improvements' not in st.session_state:
        st.session_state['improvements'] = st.session_state['memory'].get('improvements', [])
    if 'web_search_conversation' not in st.session_state:
        st.session_state['web_search_conversation'] = []
    if 'cot_conversation' not in st.session_state:
        st.session_state['cot_conversation'] = []

# Function to validate API key
def validate_api_key(api_key):
    try:
        st.session_state['groq'] = GroqProvider(api_key)
        st.session_state['api_key_valid'] = True
        st.success("API key validated successfully!")
    except GroqAPIKeyMissingError:
        st.session_state['api_key_valid'] = False
        st.error("Invalid API key. Please try again.")

# Main function to run the Streamlit app
def main():
    try:
        initialize_session_state()

        st.title('Personal AI Assistant')
        st.write('Welcome to your AI assistant powered by Groq!')

        # API key input and validation
        api_key = st.text_input("Enter your Groq API key", type="password")
        if st.button("Validate API Key"):
            validate_api_key(api_key)

        if st.session_state['api_key_valid']:
            # Display chat history
            st.subheader("Chat History")
            for entry in st.session_state['conversation']:
                st.write(f"{entry['role'].capitalize()}: {entry['content']}")

            # User input
            user_input = st.text_input("Enter your message", key="main_input", placeholder="Type your message here...")
            if st.button("Send", key="main_send") and user_input:
                # Add user message to conversation
                st.session_state['conversation'].append({"role": "user", "content": user_input})
                
                # Generate and display response
                context = ""
                if st.session_state['url_cache']:
                    context = f"Consider the following URLs for context: {', '.join(st.session_state['url_cache'].keys())}"
                response = generate_response(user_input, context)
                st.session_state['conversation'].append({"role": "assistant", "content": response})
                
                # Update memory
                memory_prompt = f"Based on the following conversation, what key information should I remember?\n\n{user_input}\n{response}"
                memory_update = st.session_state['groq'].generate(memory_prompt)
                st.session_state['memory'][user_input] = memory_update
                save_memory()
                
                # Clear input by re-rendering the widget with a different key
                st.session_state['interaction_count'] += 1
                st.rerun()

            # Additional features
            st.subheader("Additional Features")
            feature = st.selectbox("Select a feature", 
                                   ["RAG", "Chain of Thought", "Vision", "Web Search", "Self-Improvement", "Memory Management"])

            if feature == "RAG":
                st.write("RAG feature implementation")
                uploaded_file = st.file_uploader("Upload a file to add to context", type=["txt", "pdf", "docx"])
                url_input = st.text_input("Enter a URL to add to context")
                if st.button("Add URL to Context"):
                    st.session_state['url_cache'][url_input] = url_input
                    st.success("URL added to context!")

            elif feature == "Chain of Thought":
                st.write("Chain of Thought feature implementation")
                
                # Display COT conversation history
                st.subheader("Chain of Thought Conversation")
                for entry in st.session_state['cot_conversation']:
                    st.write(f"{entry['role'].capitalize()}: {entry['content']}")
                
                cot_input = st.text_input("Enter your Chain of Thought prompt or question", key="cot_input", placeholder="Type your COT prompt here...")
                if st.button("Submit", key="cot_submit") and cot_input:
                    if st.session_state['groq']:
                        context = "\n".join([f"{entry['role']}: {entry['content']}" for entry in st.session_state['cot_conversation']])
                        full_prompt = f"Based on the following Chain of Thought conversation:\n\n{context}\n\nUser: {cot_input}\n\nAssistant:"
                        response = st.session_state['groq'].generate(full_prompt)
                        st.session_state['cot_conversation'].append({"role": "user", "content": cot_input})
                        st.session_state['cot_conversation'].append({"role": "assistant", "content": response})
                        st.session_state['interaction_count'] += 1
                        st.rerun()

            elif feature == "Vision":
                st.write("Vision feature implementation")
                image_url = st.text_input("Enter image URL")
                if st.button("Analyze Image"):
                    if st.session_state['groq']:
                        response = st.session_state['groq'].generate(image_url)
                        st.write("Image Analysis:", response)

            elif feature == "Web Search":
                st.write("Web Search feature implementation")
                
                # Display Web Search conversation history
                st.subheader("Web Search Conversation")
                for entry in st.session_state['web_search_conversation']:
                    st.write(f"{entry['role'].capitalize()}: {entry['content']}")
                
                web_search_input = st.text_input("Enter your web search query or follow-up question", key="web_search_input", placeholder="Type your web search query here...")
                if st.button("Submit", key="web_search_submit") and web_search_input:
                    if st.session_state['groq']:
                        context = "\n".join([f"{entry['role']}: {entry['content']}" for entry in st.session_state['web_search_conversation']])
                        full_prompt = f"Based on the following web search conversation:\n\n{context}\n\nUser: {web_search_input}\n\nAssistant:"
                        response = st.session_state['groq'].generate(full_prompt)
                        st.session_state['web_search_conversation'].append({"role": "user", "content": web_search_input})
                        st.session_state['web_search_conversation'].append({"role": "assistant", "content": response})
                        st.session_state['interaction_count'] += 1
                        st.rerun()

            elif feature == "Self-Improvement":
                st.write("Self-Improvement Feature")
                st.write("This feature allows the assistant to analyze its performance and improve its responses.")
                if st.button("Generate Self-Improvement Analysis"):
                    if st.session_state['groq']:
                        # Prepare context
                        recent_conversations = [
                            f"{entry['role']}: {entry['content'][:100]}..."
                            for entry in st.session_state['conversation'][-10:]
                        ]
                        context = f"Previous interactions: {json.dumps(recent_conversations)}\nMemory: {json.dumps(st.session_state['memory'])}\n\n"
                        
                        prompt = f"{context}Analyze the performance of this AI assistant application and suggest improvements in the following areas:\n1. Response quality and relevance\n2. Understanding of user queries\n3. Use of memory for context\n4. Consistency in responses\n5. Overall user experience\n\nProvide specific examples where possible and suggest concrete steps for improvement."
                        
                        analysis = st.session_state['groq'].generate(prompt)
                        st.write("Self-Improvement Analysis:", analysis)
                        
                        action_prompt = f"Based on the following analysis, generate 3-5 specific action items for improving the AI assistant application:\n\n{analysis}\n\nAction items:"
                        action_items = st.session_state['groq'].generate(action_prompt)
                        st.write("Action Items for Improvement:", action_items)
                        
                        st.session_state['memory']['self_improvement'] = {
                            'last_analysis': analysis,
                            'action_items': action_items,
                            'timestamp': st.session_state.get('interaction_count', 0)
                        }
                        save_memory()
                        
                        implementation_prompt = f"Based on these action items, suggest code improvements or new features to enhance the AI assistant application:\n\n{action_items}\n\nSuggested improvements:"
                        suggested_improvements = st.session_state['groq'].generate(implementation_prompt)
                        st.write("Suggested Improvements:", suggested_improvements)
                        st.session_state['memory']['suggested_improvements'] = suggested_improvements
                        save_memory()
                        
                        if st.button("Apply Improvements"):
                            apply_improvements(suggested_improvements)
                        
                        st.success("Self-improvement analysis complete with suggested improvements!")

                # Verify improvements
                if st.button("Verify Improvements"):
                    verify_improvements()

            elif feature == "Memory Management":
                st.write("Memory Management")
                
                # Export memory
                if st.button("Export Memory"):
                    memory_json = export_memory()
                    st.download_button(
                        label="Download Memory JSON",
                        data=memory_json,
                        file_name="memory_export.json",
                        mime="application/json"
                    )
                
                # Import memory
                uploaded_file = st.file_uploader("Upload Memory JSON", type=["json"])
                if uploaded_file is not None:
                    memory_json = uploaded_file.getvalue().decode("utf-8")
                    if st.button("Import Memory"):
                        import_memory(memory_json)

            st.session_state['interaction_count'] += 1
            if st.session_state['interaction_count'] % 10 == 0:
                st.write("Performing periodic self-assessment...")
                last_analysis = st.session_state['memory'].get('self_improvement', {}).get('timestamp', 0)
                if st.session_state['interaction_count'] - last_analysis >= 50:
                    st.button("Generate Self-Improvement Analysis", on_click=lambda: None)
                else:
                    recent_conversations = [
                        f"{entry['role']}: {entry['content'][:100]}..."
                        for entry in st.session_state['conversation'][-5:]
                    ]
                    context = f"Recent conversations: {json.dumps(recent_conversations)}\nCurrent improvements: {json.dumps(st.session_state.get('improvements', []))}\n\n"
                    quick_assessment_prompt = f"{context}Briefly assess the recent performance of this AI assistant application and suggest one key area for immediate improvement."
                    quick_assessment = st.session_state['groq'].generate(quick_assessment_prompt)
                    st.write("Quick Self-Assessment:", quick_assessment)
                    st.session_state['memory']['quick_assessment'] = quick_assessment
                    save_memory()

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Clear Conversation History"):
                    st.session_state['conversation'] = []
                    st.success("Conversation history cleared!")
            with col2:
                if st.button("Clear Web Search History"):
                    st.session_state['web_search_conversation'] = []
                    st.success("Web Search history cleared!")
            with col3:
                if st.button("Clear COT History"):
                    st.session_state['cot_conversation'] = []
                    st.success("Chain of Thought history cleared!")

        else:
            st.warning("Please enter a valid API key to use the features.")

        st.markdown("---")
        st.write("Powered by Groq and Streamlit")

    except Exception as e:
        st.error(f'An error occurred: {str(e)}')
        st.write('Please check the console for more details.')
        print(f'Error details: {e}', file=sys.stderr)

if __name__ == '__main__':
    main()

