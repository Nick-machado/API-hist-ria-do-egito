# Museu Egípcio das Areias do Tempo - AI Assistant

## Introduction

Welcome to the **Museu Egípcio das Areias do Tempo** AI Assistant project! This project leverages advanced AI technologies to provide an interactive assistant capable of answering questions about the museum, its exhibits, and ancient Egyptian history.

By utilizing OpenAI's GPT models and LangChain, this assistant can retrieve information from the museum's documentation and deliver informative responses to user queries.

## Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Database Creation](#database-creation)
  - [Running the API](#running-the-api)
  - [Querying the API](#querying-the-api)
- [Project Structure](#project-structure)
- [Museum Documentation](#museum-documentation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

- **Retrieve Augmented Generation (RAG)**: Uses a vector database to retrieve relevant documents for user queries.
- **Flask API**: Provides an API endpoint to interact with the AI assistant.
- **Multilingual Support**: Designed to handle queries in Portuguese.
- **Integration with OpenAI Models**: Utilizes GPT-4 models for generating responses.
- **Document Embedding**: Converts museum documentation into embeddings for efficient retrieval.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- [LangChain](https://github.com/hwchase17/langchain) library
- Flask
- Required Python packages (see `requirements.txt`)

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/museu-egipcio-ai-assistant.git
   cd museu-egipcio-ai-assistant
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**

   Create a `.env` file in the root directory and add your OpenAI API key:

   ```env
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

### Database Creation

Before running the API, you need to create the vector database using the museum's documentation.

1. **Place the Museum Documentation**

   Ensure that the museum's documentation (in PDF format) is placed in the `rag` directory.

2. **Run the Database Creation Script**

   ```bash
   python create_database.py
   ```

   This script will:

   - Load and split the documents.
   - Create embeddings using OpenAI's embedding models.
   - Store the embeddings in a persistent vector database using Chroma.

### Running the API

Start the Flask API server to handle user queries.

```bash
python app.py
```

The API will be accessible at `http://0.0.0.0:5000/museu`.

### Querying the API

You can query the API by sending a GET request with the `input` parameter.

Example using `curl`:

```bash
curl "http://0.0.0.0:5000/museu?input=Qual%20é%20o%20horário%20de%20funcionamento%20do%20museu?"
```

Example using Python:

```python
import requests

response = requests.get("http://0.0.0.0:5000/museu", params={"input": "Qual é o horário de funcionamento do museu?"})
print(response.json())
```

## Project Structure

```
├── app.py                 # Flask API script
├── create_database.py     # Script to create the vector database
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables file
├── rag/                   # Directory containing museum documentation PDFs
├── db/
│   └── chroma_db/         # Directory where the vector database is stored
├── templates/             # (Optional) Flask templates
├── static/                # (Optional) Static files
└── README.md              # Project README
```

### Key Components

#### `create_database.py`

This script is responsible for creating the vector store (database) from the museum's documentation. It performs the following steps:

- Loads and splits the PDF documents.
- Creates embeddings using OpenAI's embedding model.
- Stores the embeddings in a Chroma vector database.

**Code Snippet:**

```python
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter

def database():
    load_dotenv()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    files = os.listdir(os.path.join(current_dir, "rag"))
    persistent_directory = os.path.join(current_dir, "db", "chroma_db")

    if not os.path.exists(persistent_directory):
        print("\n ---Persistent directory does not exist. Initializing vector store---\n")
        for file in files:
            file_path = os.path.join(current_dir, "rag", file)
            loader = PyPDFLoader(file_path)
            documents = loader.load_and_split()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(documents)
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            db = Chroma.from_documents(
                docs, embeddings, persist_directory=persistent_directory
            )
    else:
        print("\n ---Persistent directory already exists.---\n")

database()
```

#### `app.py`

This is the Flask API script that handles user queries and returns AI-generated responses. It defines an endpoint `/museu` that accepts a GET request with an `input` parameter.

**Code Snippet:**

```python
import os
from datetime import datetime
import locale
from dotenv import load_dotenv
from langchain import hub
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from flask import Flask, request, jsonify
from flask_restful import Resource, Api

def app_ai(input_text):
    load_dotenv()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def get_current_day_time(input_text):
        locale.setlocale(locale.LC_TIME, 'pt_BR.UTF-8')
        agora = datetime.now()
        dia_semana = agora.strftime('%A')
        horario = agora.strftime('%H:%M:%S')
        return f"Hoje é {dia_semana}, e o horário atual é {horario}."

    def get_data(input_text):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        persistent_directory = os.path.join(current_dir, "db", "chroma_db")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        db = Chroma(persist_directory=persistent_directory,
                    embedding_function=embeddings)
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.1}
        )
        relevant_docs = retriever.invoke(input_text)
        observation = "\n\n".join([doc.page_content for doc in relevant_docs])
        return observation

    tools = [
        Tool(
            name="info search",
            func=get_data,
            description="Procura respostas para as perguntas dos usuários utilizando a documentação do museu",
            input_type=str,
        ),
        Tool(
            name="get_current_date_and_time",
            func=get_current_day_time,
            description="Pega o dia e a data",
            input_type=str
        )
    ]

    prompt = hub.pull("museuapi/museu")

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    response = agent_executor.invoke({"input": input_text})

    return response['output']

app = Flask(__name__)
api = Api(app)

class Museu(Resource):
    def get(self):
        input_text = request.args.get('input')
        if input_text:
            output = app_ai(input_text)
            return jsonify({"data": output})
        else:
            return jsonify({'error': 'Input parameter is missing'}), 400

api.add_resource(Museu, '/museu')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
```

## Museum Documentation

The AI assistant uses the museum's documentation to retrieve relevant information. The documentation includes details about:

### Introduction

The **Museu Egípcio das Areias do Tempo** is dedicated to preserving and sharing the rich history and culture of ancient Egypt. Located in the heart of the Historic City, the museum offers visitors an immersive and educational experience, exploring from the powerful pharaonic dynasties to the hidden secrets of the pyramids and the daily life of the Egyptian people.

### Main Attractions

- **Gallery of Pharaohs**: A majestic tribute to the rulers who shaped the destiny of Ancient Egypt, featuring colossal statues and artifacts.
- **Hall of Pyramids**: Explore the architectural wonders of the Giza pyramids with scale replicas and virtual reality experiences.
- **Nile River Exhibition**: Illustrates the fundamental importance of the Nile River in the formation of Egyptian civilization.
- **Hieroglyph Workshop**: Offers a hands-on experience in the art of hieroglyphic writing.
- **Museum Treasures**: Displays jewels, amulets, and golden objects reflecting exceptional craftsmanship.
- **Mummies and Funeral Rites**: Delves into the complex funeral rituals and beliefs about the afterlife.

### Operating Hours

- **Monday to Friday**: 9 AM to 5 PM
- **Saturday**: 10 AM to 6 PM
- **Sunday and Holidays**: 11 AM to 4 PM
- **Closed on Tuesdays** for maintenance and updates.

### Ticket Information

- **Adults**: R$ 30,00
- **Students and Seniors**: R$ 15,00 (with ID)
- **Children up to 6 years**: Free
- **Family Packages**: Special discounts available.

### Additional Services

- **Guided Tours**: Available in Portuguese, English, and Spanish at scheduled times.
- **Museum Shop**: Offers replicas, books, jewelry, and souvenirs.
- **Café Nefertiti**: Serves light dishes inspired by Mediterranean cuisine.

### Accessibility

- **Mobility**: Ramps and elevators; wheelchairs available.
- **Audiovisual**: Informative materials in Braille; audio guides.
- **Special Needs**: Trained staff to assist visitors.

### Contact and Location

- **Address**: Avenida das Pirâmides, 1234 – Bairro Antigo Egito, Cidade Histórica.
- **Phone**: (11) 1234-5678
- **Email**: contato@museuegipcio.com.br
- **Website**: [www.museuegipcio.com.br](http://www.museuegipcio.com.br)

This comprehensive documentation allows the AI assistant to provide accurate and detailed responses to user queries about the museum.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any bugs or improvements.

## License

This project is licensed under the MIT License.

## Contact

For any inquiries or support, please contact:

- **Email**: contato@museuegipcio.com.br
- **Website**: [www.museuegipcio.com.br](http://www.museuegipcio.com.br)

---

**Note**: This project is a fictional representation of the "Museu Egípcio das Areias do Tempo" and is intended for educational purposes.

---