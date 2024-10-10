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

# Função que processa o input e retorna a resposta da AI
def app_ai(input_text):
    load_dotenv(    )

    # Instanciação do modelo LLM e ferramentas
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def get_current_day_time(input_text):
    # Define a localidade para Português do Brasil
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

    # Retornar o conteúdo dos documentos relevantes como observação
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

# Configuração do Flask
app = Flask(__name__)
api = Api(app)

# Classe que define a rota da API
class museu(Resource):
    def get(self):
        input_text = request.args.get('input')
        if input_text:
            output = app_ai(input_text)
            return jsonify({"data":output})
        else:
            return jsonify({'error': 'Input parameter is missing'}), 400

# Adiciona a rota à API
api.add_resource(museu, '/museu')

# Inicializa o servidor Flask
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)