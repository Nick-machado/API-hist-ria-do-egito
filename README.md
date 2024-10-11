# Assistente Virtual do Museu Egípcio das Areias do Tempo

Uma aplicação de IA que utiliza **RAG (Retrieval-Augmented Generation)** para interagir com a documentação do museu.

## Índice

- [Introdução](#introdução)
- [Conceito e Utilidade](#conceito-e-utilidade)
- [Aplicação em Outros Nichos](#aplicação-em-outros-nichos)
- [Instalação](#instalação)
- [Uso](#uso)
- [Estrutura do Código](#estrutura-do-código)
  - [Criação do Banco de Dados para RAG](#criação-do-banco-de-dados-para-rag)
  - [API em Flask](#api-em-flask)
- [Interação entre as Partes](#interação-entre-as-partes)
- [Recriação do Projeto](#recriação-do-projeto)
- [Licença](#licença)
- [Contato](#contato)

## Introdução

Este projeto visa criar um assistente virtual para o **Museu Egípcio das Areias do Tempo**, permitindo que usuários interajam e obtenham informações detalhadas sobre o museu através de uma API construída com Flask e integrada a modelos de linguagem natural.

## Conceito e Utilidade

O assistente virtual utiliza técnicas de **RAG (Retrieval-Augmented Generation)** para fornecer respostas precisas baseadas na documentação oficial do museu. Isso é alcançado através da combinação de modelos de linguagem avançados com um banco de dados de conhecimento específico, permitindo:

- Respostas contextualizadas e precisas.
- Interação natural com os usuários.
- Disponibilização de informações em tempo real sobre o museu.

## Aplicação em Outros Nichos

A abordagem utilizada neste projeto pode ser adaptada para outros setores que necessitam de:

- Assistentes virtuais informativos.
- Sistemas de FAQ inteligentes.
- Chatbots especializados em determinado domínio.
- Integração de modelos de linguagem com bases de conhecimento específicas.

## Instalação

### Pré-requisitos

- Python 3.8 ou superior
- Pip instalado
- Conta na OpenAI com chave de API válida

### Passos

1. **Clone este repositório**

   ```bash
   git clone https://github.com/seu-usuario/seu-projeto.git
   cd seu-projeto
   ```

2. **Crie um ambiente virtual**

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **Instale as dependências**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure as variáveis de ambiente**

   Crie um arquivo `.env` na raiz do projeto e adicione sua chave de API da OpenAI:

   ```
   OPENAI_API_KEY=sua_chave_api
   ```

## Uso

### 1. Criação do Banco de Dados para RAG

Execute o script `create_db.py` para criar o banco de dados de vetores:

```bash
python create_db.py
```

### 2. Inicialização da API Flask

Inicie o servidor Flask executando o script `app.py`:

```bash
python app.py
```

A API estará disponível em `http://0.0.0.0:5000/museu`.

### 3. Teste da API

Faça uma requisição GET para o endpoint `/museu` com o parâmetro `input`:

```bash
http://0.0.0.0:5000/museu?input=Qual%20é%20o%20horário%20de%20funcionamento%20do%20museu%3F
```

## Estrutura do Código

### Criação do Banco de Dados para RAG

Arquivo: `create_db.py`

Este script cria o banco de dados de vetores a partir dos documentos do museu.

#### Principais Componentes

- **Carregamento de Documentos**

  Utiliza `PyPDFLoader` para carregar e dividir documentos PDF.

  ```python
  loader = PyPDFLoader(file_path)
  documents = loader.load_and_split()
  ```

- **Divisão de Texto**

  O `CharacterTextSplitter` divide os documentos em chunks manejáveis para a criação de embeddings.

  ```python
  text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  docs = text_splitter.split_documents(documents)
  ```

- **Criação de Embeddings**

  Usa `OpenAIEmbeddings` para gerar embeddings dos chunks de texto.

  ```python
  embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
  ```

- **Armazenamento Persistente com Chroma**

  Salva os embeddings em um diretório persistente para uso posterior.

  ```python
  db = Chroma.from_documents(
      docs, embeddings, persist_directory=persistent_directory
  )
  ```

### API em Flask

Arquivo: `app.py`

Este script configura a API Flask que responde às consultas dos usuários.

#### Principais Componentes

- **Função `app_ai`**

  Processa a entrada do usuário e retorna a resposta da IA.

  ```python
  def app_ai(input_text):
      # Processamento e geração de resposta
  ```

- **Ferramentas (Tools)**

  - `get_current_day_time`: Retorna o dia da semana e o horário atual.

    ```python
    def get_current_day_time(input_text):
        # Retorna data e hora atuais
    ```

  - `get_data`: Recupera informações relevantes dos documentos utilizando o banco de dados criado.

    ```python
    def get_data(input_text):
        # Busca informações no banco de dados de vetores
    ```

- **Configuração do Agente**

  Cria um agente utilizando o modelo de linguagem e as ferramentas definidas.

  ```python
  agent = create_react_agent(
      llm=llm,
      tools=tools,
      prompt=prompt
  )
  ```

- **Definição da Rota da API**

  Utiliza Flask e Flask-RESTful para definir a rota `/museu`.

  ```python
  class museu(Resource):
      def get(self):
          # Processa a requisição GET
  ```

## Interação entre as Partes

1. **Criação do Banco de Dados**

   O script `create_db.py` processa os documentos do museu e cria um banco de dados de vetores persistente.

2. **API Flask**

   A API utiliza o banco de dados criado para buscar informações relevantes e responder às consultas dos usuários.

3. **Fluxo de Dados**

   - O usuário envia uma pergunta para a API.
   - A API processa a entrada e utiliza as ferramentas para buscar informações.
   - O agente gera uma resposta combinando o modelo de linguagem e os dados recuperados.
   - A resposta é enviada de volta ao usuário.

## Recriação do Projeto

Para recriar este projeto em sua máquina:

1. **Obtenha os Documentos**

   Coloque os arquivos PDF da documentação do museu na pasta `rag` do projeto.

2. **Configure as Chaves de API**

   Certifique-se de ter uma chave de API válida da OpenAI e configure-a no arquivo `.env`.

3. **Instale as Dependências**

   Siga os passos da seção [Instalação](#instalação).

4. **Execute os Scripts**

   - Crie o banco de dados: `python create_db.py`
   - Inicie a API: `python app.py`

5. **Teste a API**

   Utilize ferramentas como `Postman` ou `curl` para enviar requisições e verificar as respostas.

## Contato

Para mais informações ou suporte, entre em contato:

- **Nome**: Nicholas Machado
- **Email**: lafuno123@gmail.com
- **GitHub**: [Github do Nicholas](https://github.com/Nick-machado)
- **Linkedin**: [Linkedin do Nicholas](https://www.linkedin.com/in/nicholas-machado-305979313/)