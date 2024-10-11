import os
from datetime import datetime
from babel.dates import format_datetime
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
    load_dotenv()  # Carrega as variáveis de ambiente do arquivo .env

    # Instancia o modelo LLM com temperatura zero para respostas determinísticas
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Função para obter o dia e a hora atuais
    def get_current_day_time(input_text):
        # Define a localidade para Português do Brasil
        agora = datetime.now()
        dia_semana = format_datetime(agora, "EEEE", locale='pt_BR')
        horario = agora.strftime('%H:%M:%S')
        return f"Hoje é {dia_semana}, e o horário atual é {horario}."

    # Função para recuperar dados relevantes do banco de dados
    def get_data(input_text):
        # Obtém o diretório atual do arquivo
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Define o diretório persistente do banco de dados Chroma
        persistent_directory = os.path.join(current_dir, "db", "chroma_db")
        # Instancia o modelo de embeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # Instancia o banco de dados Chroma com o diretório persistente e a função de embeddings
        db = Chroma(persist_directory=persistent_directory,
                    embedding_function=embeddings)

        # Configura o mecanismo de recuperação de documentos relevantes
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.1}
        )

        # Recupera os documentos relevantes com base no input
        relevant_docs = retriever.invoke(input_text)

        # Retorna o conteúdo dos documentos relevantes como uma única string
        observation = "\n\n".join([doc.page_content for doc in relevant_docs])
        return observation

    # Define as ferramentas disponíveis para o agente
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
            description='''
                Pega o dia e a data para verificar se o museu está aberto ou se está disponível alguma visita com um guia. 
                Considere que a visita com o guia é feita apenas às 10h, 13h e 15h. 
                Se não houver mais nenhuma visita com guia no dia, informe o próximo horário disponível seguindo os horários de funcionamento.
            ''',
            input_type=str
        )
    ]

    # Carrega o prompt personalizado do repositório
    prompt = hub.pull("museuapi/museu")

    # Cria o agente com o modelo LLM, as ferramentas e o prompt personalizado
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    # Configura o executor do agente
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,  # Ativa logs detalhados
        handle_parsing_errors=True  # Lida com erros de parsing
    )

    # Invoca o agente com o input do usuário e obtém a resposta
    response = agent_executor.invoke({"input": input_text})

    # Retorna a saída do agente
    return response['output']

# Configuração do aplicativo Flask
app = Flask(__name__)
api = Api(app)

# Classe que define a rota da API
class museu(Resource):
    def get(self):
        input_text = request.args.get('input')  # Obtém o parâmetro 'input' da URL
        if input_text:
            output = app_ai(input_text)  # Processa o input através da função app_ai
            return jsonify({"data": output})  # Retorna a resposta em formato JSON
        else:
            return jsonify({'error': 'Input parameter is missing'}), 400  # Retorna erro se o input estiver ausente

# Adiciona a rota '/museu' à API
api.add_resource(museu, '/museu')

# Inicializa o servidor Flask
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)  # Executa o aplicativo na porta 5000