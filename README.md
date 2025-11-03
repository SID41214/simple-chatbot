# Simple Chatbot with Python and Hugging Face LLMs

**This project is intended as an introductory-level example, designed primarily for beginners.**

This repository contains a simple, functional chatbot built in Python. It uses open-source Large Language Models (LLMs) and the `transformers` library provided by Hugging Face.

This project is based on a lab designed to guide beginners through the core concepts of building conversational AI.




## 📖 About The Project

This project walks through the process of setting up a Python environment, selecting a suitable open-source LLM, and writing a script to interact with the model.The final `chatbot.py` script allows you to have a back-and-forth conversation with the AI through your terminal.

**Core technologies used:**
* Python 
* Hugging Face `transformers` library 
* `torch`
* `virtualenv` for environment management

---

## 🎯 Learning Objectives

By following this project, you will learn how to:
* Describe the main components of a chatbot
* Explain what an LLM is
* Select an LLM for a specific application
* Understand the basic workflow of a transformer
* Feed input into a transformer using tokenization
* Program a simple chatbot in Python

---

## 🚀 Getting Started

Follow these steps to set up and run the chatbot on your local machine.

### Prerequisites

Ensure you have **Python 3** and **pip** installed on your system.

### Installation

1.  **Clone the repository** (or create the file `chatbot.py` in a new directory):
    ```sh
    git clone [https://your-repository-url.git](https://your-repository-url.git)
    cd your-project-directory
    ```

2.  **Set up and activate a Python virtual environment**:
    ```sh
    # Install virtualenv if you haven't already
    pip3 install virtualenv
    
    # Create a virtual environment
    virtualenv my_env
    
    # Activate the virtual environment
    source my_env/bin/activate
    ```

3.  **Install the required libraries**:
    ```sh
    # Install the transformers and torch libraries
    pip3 install transformers==4.38.2 torch
    ```

---

## 💬 Usage

Once your environment is activated and the libraries are installed, you can run the chatbot

1.  **Run the script:**
    ```sh
    python3 chatbot.py
    ```

2.  **Start chatting:**
    Type your message into the terminal and press `Enter`. The bot will generate a response[cite: 220, 228]. The conversation history is maintained to provide context for future responses[cite: 110, 208].

3.  **Exit the chatbot:**
    Press `CTRL+C` to stop the script and end the conversation[cite: 313].

### Example Interaction

Hi, how are you? Hi Hi Hi hi hì hì,, I am ama a a,,... how how how are are are???  what is your hobbies? Hi, I'm a big gamer. I like to play video games. what hobbies do you have?  I like games too. what types of games do you play? I love video games as well. I play a lot of Call of Duty. What is your favorite video game? 





---

## ⚙️ How It Works

The chatbot's logic is centered around a few key components from the Hugging Face `transformers` library

1.  **Model and Tokenizer:**
    * **`AutoTokenizer`**: This object is responsible for processing text. It converts your input string (e.g., "hello") into a numerical representation (tokens) that the model can understand. This process is called **tokenization**.
    * **`AutoModelForSeq2SeqLM`**: This object is the "brain" of the chatbot. It loads the pre-trained LLM (`facebook/blenderbot-400M-distill` in this project)  and uses it to generate responses.

2.  **The Conversation Loop**:
    The script operates in an infinite loop (`while True:`) that performs the following steps for each interaction:
    * **Get User Input**: Waits for the user to type a message.
    * **Prepare History**: Joins all previous messages in the `conversation_history` list into a single string to provide context.
    * **Tokenize**: Uses the `tokenizer` to convert both the conversation history and the new user input into tokens.
    * **Generate Response**: Feeds these tokens into the `model.generate()` function, which produces an output (also in token form).
    * **Decode Response**: Uses the `tokenizer.decode()` function to convert the model's output tokens back into a human-readable string.
    * **Print and Update**: Prints the bot's response to the terminal [cite: 228] and appends both the user's input and the bot's response to the `conversation_history` list.

---

## 📋 Full Code (`chatbot.py`)

Here is the complete, functional code for the `chatbot.py` file, assembled from the steps in the lab.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 

# --- Step 3 & 4: Choose and load the model ---
model_name = "facebook/blenderbot-400M-distill" 
# Load the model (downloads on first run)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name) 
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)


# --- Step 5.1: Initialize conversation history ---
conversation_history = [] 

# --- Step 6: Repeat (The main conversation loop) ---
print("Chatbot initialized. Type 'CTRL+C' to exit.")
while True: 
    
    # --- Step 5.2: Create conversation history string ---
    history_string = "\n".join(conversation_history) 

    # --- Step 5.3: Get input data from user ---
    input_text = input("> ")

    # --- Step 5.4: Tokenize the input text and history ---
    inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt") 

    # --- Step 5.5: Generate the response from the model ---
    outputs = model.generate(**inputs) 

    # --- Step 5.6: Decode the response ---
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print(response) 

    # --- Step 5.7: Add interaction to conversation history ---
    conversation_history.append(input_text) 
    conversation_history.append(response)
```

🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to check the issues page for this repository.





