import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Check the structure of the loaded intents
print(intents)  

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents['intents']: 
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents['intents']:  
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

counter = 0

def main():
    global counter
    st.title("Financial_Literacy_Chatbot using NLP")

    # Create a sidebar menu with options
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Home Menu
    if choice == "Home":
        st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")

        # Check if the chat_log.csv file exists, and if not, create it with column names
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:
            # Convert the user input to a string
            user_input_str = str(user_input)

            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")

            # Get the current timestamp
            timestamp = datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")

            # Save the user input and chatbot response to the chat_log.csv file
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    # Conversation History Menu
    elif choice == "Conversation History":
        # Display the conversation history in a collapsible expander
        st.header("Conversation History")
        # with st.beta_expander("Click to see Conversation History"):
        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip the header row
            for row in csv_reader:
                st.text(f"User: {row[0]}")
                st.text(f"Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("---")

    elif choice == "About":
        st.title("About the Financial Literacy Chatbot")

        st.write("The goal of this project is to create a chatbot that can understand and respond to user input related to financial literacy topics. The chatbot leverages Natural Language Processing (NLP) and Logistic Regression to extract and categorize user queries into predefined intents, providing helpful responses based on financial concepts.")

        st.subheader("Project Overview:")

        st.write("""The project consists of two primary parts:
        1. NLP & Logistic Regression: Used to train the chatbot on labeled financial intents, such as budgeting, saving, investing, and understanding loans.
        2. Streamlit Interface: A web-based interface that allows users to interact with the chatbot. Users input their questions, and the chatbot provides responses based on its training.""")

        st.subheader("Dataset:")

        st.write("""
        The dataset used in this project is a collection of financial literacy intents and associated sample phrases. The data is structured as:
        - Intents: Categories such as "budgeting", "investing", and "loans".
        - Entities: Specific financial terms (e.g., "savings", "credit score").
        - Text: Example questions or statements from users related to these financial topics.
        """)

        st.subheader("Streamlit Chatbot Interface:")

        st.write("The chatbot is built using the Streamlit framework, providing a simple and interactive interface. The user types a question in a text input box, and the chatbot responds using pre-trained responses. This interface facilitates real-time conversation with the bot.")

        st.subheader("Conclusion:")

        st.write("This financial literacy chatbot is a basic example of how NLP and Logistic Regression can be applied to real-world scenarios like personal finance education. The chatbot can be extended by adding more data, implementing more sophisticated NLP models, and integrating deep learning techniques to improve accuracy and conversation flow.")

if __name__ == '__main__':
    main()
