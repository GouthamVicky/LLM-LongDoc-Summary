import streamlit as st
import requests
import pandas as pd
import random

# Load the data from a local CSV file
csv_file_path = "dataset.csv"  # Replace with the path to your CSV file
data = pd.read_csv(csv_file_path)

st.title("Falcon Document Summarizer")
st.divider()

# Create a placeholder for displaying the chat conversation
chat_history = st.empty()

if st.session_state.get("chat_history") is None:
    st.session_state.chat_history = []

# Initialize the prompt variable
random_prompt = ""

# Initialize a loading state
loading = False

# Initialize a variable to show the note message
show_note = False

# Heading to generate a random prompt sample for summary
st.caption("Click here to generate a random prompt sample for summary")

# Add links to Colab notebook and GitHub repository in the right sidebar
st.sidebar.header("Links")
st.sidebar.markdown("[GitHub Repository](https://github.com/GouthamVicky/LLM-LongDoc-Summary) - GitHub Repository for this project")
st.sidebar.markdown("[Colab Notebook](https://colab.research.google.com/drive/1ImVUtRINIAQN0ErhYHasRS7oQNwnDTUW?usp=sharing) - Model Finetuning notebook")
st.sidebar.markdown("[Huggingface Model Card](https://colab.research.google.com/drive/1ImVUtRINIAQN0ErhYHasRS7oQNwnDTUW?usp=sharing) - Model Finetuning notebook")

# Button to generate a random sample from the local CSV file
if st.sidebar.button("Fetch Random Article Content"):
    with st.spinner("Generating Random Sample..."):
        random_index = random.randint(0, len(data) - 1)
        random_prompt = data.iloc[random_index]["article"]
        st.text("A Sample Paper Article Prompt:")

        # Set the generated prompt to the user input text area
        # user_input.text(random_prompt)

        # Show the note message
        show_note = True

        loading = False  # Turn off the loading state after generating the prompt

if show_note:
    st.sidebar.markdown("Note: The above sample has been fetched from your local CSV file.")

# Add a text box on the left side to display the generated text
generated_text_box = st.sidebar.text_area("Article Context", value=random_prompt, height=400)

# Create a text input with scrollbar and increased height
user_input = st.text_area("Enter a prompt and press send", key="user_prompt_input", value="", height=200)

# Show the note message if `show_note` is True

if st.button("Send"):
    # Check if the user input prompt is too short (you can adjust the minimum length)
    if len(user_input.split(" ")) < 30:  # Adjust the minimum length as needed
        st.warning("Please enter more context for the summary")
    else:
        response = requests.post("http://localhost:8000/generate/", json={"prompt": user_input})
        generated_text = response.json()["generated_text"]
        st.text_area("Generated Summary", value=generated_text, height=400)
