import os
import sys
import json
from dotenv import load_dotenv

# LangChain components
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    print("Error: GOOGLE_API_KEY is missing. Please check your .env file.")
    sys.exit(1)

# Load Configuration
try:
    with open("config.json", "r") as f:
        CONFIG = json.load(f)
except FileNotFoundError:
    print("Error: config.json not found. Please create it in the project root.")
    sys.exit(1)


class SalesCopilot:
    def __init__(self):
        # Initialize Google GenAI models using config
        self.embeddings = GoogleGenerativeAIEmbeddings(model=CONFIG["embedding_model"])
        self.llm = ChatGoogleGenerativeAI(model=CONFIG["llm_model"], temperature=0.2)
        
        # Initialize Vector Store and LCEL Chain
        self.vectorstore = self._initialize_vectorstore()
        self.chain = self._build_rag_chain()

    def _initialize_vectorstore(self):
        """Loads existing FAISS index or creates a new one from the data directory."""
        index_dir = CONFIG["index_dir"]
        data_dir = CONFIG["data_dir"]

        if os.path.exists(index_dir):
            print("Loading existing vector store...")
            # allow_dangerous_deserialization is required for local FAISS loading in newer LangChain versions
            return FAISS.load_local(index_dir, self.embeddings, allow_dangerous_deserialization=True)
        
        print(f"Initializing new vector store from '{data_dir}' directory...")
        if not os.path.exists(data_dir) or not os.listdir(data_dir):
            print(f"Warning: No data found in '{data_dir}'. Start by ingesting a file.")
            # Initialize empty vector store with a dummy document to establish schema
            return FAISS.from_texts(["Initialization document"], self.embeddings, metadatas=[{"source": "init"}])

        loader = DirectoryLoader(data_dir, glob="*.txt", loader_cls=TextLoader)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG["chunk_size"], 
            chunk_overlap=CONFIG["chunk_overlap"]
        )
        chunks = text_splitter.split_documents(documents)
        
        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        vectorstore.save_local(index_dir)
        return vectorstore

    def _format_docs(self, docs):
        """Formats retrieved documents to include metadata for the LLM."""
        formatted = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown Source")
            # We explicitly inject the source filename into the text context so the LLM can cite it
            formatted.append(f"--- SOURCE: {os.path.basename(source)} ---\n{doc.page_content}")
        return "\n\n".join(formatted)

    def _build_rag_chain(self):
        """Constructs the LangChain Expression Language (LCEL) chain for RAG."""
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        
        prompt_template = """
        You are a helpful AI Copilot for a sales team. Answer the user's question based ONLY on the provided context.
        If the answer is not contained in the context, say "I don't have enough information to answer that based on the current transcripts."
        
        CRITICAL REQUIREMENT: You MUST explicitly cite the source file names and relevant timestamps (if available) for every piece of information you provide. 
        Format your citations clearly (e.g., "[Source: 1_demo_call.txt, Timestamp: 02:07]").

        Context:
        {context}

        Question: {question}
        Answer:
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)

        # LCEL Pipeline
        chain = (
            {"context": retriever | self._format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

    def list_call_ids(self):
        """Returns deduplicated source files from the vector store."""
        # Check if the docstore only has our dummy init doc
        if not self.vectorstore.docstore._dict or (
            len(self.vectorstore.docstore._dict) == 1 and 
            list(self.vectorstore.docstore._dict.values())[0].metadata.get("source") == "init"
        ):
            return "No calls found in the database."
        
        sources = set()
        for doc in self.vectorstore.docstore._dict.values():
            src = doc.metadata.get("source")
            if src and src != "init":
                sources.add(os.path.basename(src))
        
        if not sources:
            return "No valid calls found."
        
        return "Indexed Calls:\n" + "\n".join(f"- {src}" for src in sources)

    def ingest_transcript(self, path):
        """Dynamically ingests a new text file into the vector store."""
        path = path.strip()
        if not os.path.exists(path):
            return f"Error: File '{path}' does not exist."
        
        try:
            loader = TextLoader(path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CONFIG["chunk_size"], 
                chunk_overlap=CONFIG["chunk_overlap"]
            )
            chunks = text_splitter.split_documents(documents)
            
            self.vectorstore.add_documents(chunks)
            self.vectorstore.save_local(CONFIG["index_dir"])
            return f"Successfully ingested {len(chunks)} chunks from '{path}'."
        except Exception as e:
            return f"Failed to ingest file: {str(e)}"

    def ask(self, query):
        """Processes a natural language query through the LCEL chain."""
        try:
            return self.chain.invoke(query)
        except Exception as e:
            return f"An error occurred during generation: {str(e)}"


def main():
    print("Welcome to the Sales Call AI Copilot!")
    print("Initializing system... (this may take a moment)")
    copilot = SalesCopilot()
    print("System ready. Type 'quit' to exit.")
    print("Example commands: 'list my call ids', 'ingest a new call transcript from <path>', or just ask a question.")
    
    while True:
        try:
            user_input = input("\nCopilot> ").strip()
            if not user_input:
                continue
                
            lower_input = user_input.lower()
            if lower_input in ['quit', 'exit']:
                print("Goodbye!")
                break
            elif lower_input == 'list my call ids':
                print(copilot.list_call_ids())
            elif lower_input.startswith('ingest a new call transcript from'):
                # Extract the path after the word 'from'
                path = user_input.split('from', 1)[1].strip()
                print(copilot.ingest_transcript(path))
            else:
                print("Thinking...")
                response = copilot.ask(user_input)
                print("\n" + response)
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nUnexpected error: {str(e)}")

if __name__ == "__main__":
    main()