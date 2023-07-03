# document-chatbot
A basic gradio app to chat with local documents

# Technologies
`LangChain` used for LLM tooling
[https://python.langchain.com/docs/get_started/introduction.html]

`Chroma` used for vector database
[https://www.trychroma.com/]

# Getting started: chat with pdf example
 This will ingest the pdf file in the `data` directory, store it as vectors in the `.chromadb` directory, then provide the most relevant vectors as context in the chat questions.

- Install dependencies:
> pip install -r requirements.txt

- Create local .env for configuration secrets:
> mv .env.example .env \
> {update values in .env}

- Start gradio app:
> python chat.py

- Open browser:
> [http://127.0.0.1:7860/]

- Ask away!
> "What are the main motivations for the author to climb?"