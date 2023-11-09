from typing import List, Tuple, Optional
from langchain.memory import ConversationBufferMemory
from operator import itemgetter
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import format_document, StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from chat_models import get_llm
from vector_stores import get_vectordb
from embeddings import get_embedding
from chains.custom_template import CONDENSE_QUESTION_TEMPLATE, ANSWER_TEMPLATE, DOCUMENT_TEMPLATE
from langchain.prompts.prompt import PromptTemplate


def _combine_documents(
        docs, document_prompt, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def _format_chat_history(chat_history: List[Tuple]) -> str:
    buffer = ""
    it = iter(chat_history)
    for _human in it:
        human = "Human: " + _human.content
        buffer += human + "\n"
        try:
            ai = "Assistant: " + next(it).content
            buffer = ai + "\n"
        except StopIteration:
            break
    return buffer


class ChatWithKBAndHistoryChain:
    """Final chain Wrapper

    To use, you must set the API key of the model to environment variables.

    Supported chat models:
        OPENAI:
            - gpt-3.5-turbo (default model)
        ZHIPUAI:
            - chatglm_turbo
        QIANFAN:
            - ERNIE-Bot-turbo.
            you could get the full list from https://cloud.baidu.com/product/wenxinworkshop

    Supported embedding models:
        OPENAI:
            - text-embedding-ada-002
        ZHIPUAI:
            - text_embedding
        QIANFAN:
            - Embedding-V1 (default model)
            - bge-large-en
            - bge-large-zh

    Args:
        model: Chat model name
        embedding: Embedding model name
        temperature: Parameter of chat model to set the output stability
        top_k: Set the number of documents to be retrieved from the knowledge base
        with_knowledgebase: Use the knowledge base or not

    Returns:
        The answer of the chat model.

    Example:
        .. code-block:: python

            from chains import ChatWithKBAndHistoryChain
            chat = ChatWithKBAndHistoryChain(with_knowledgebase=False)
            answer = chat.send("Hi I am Alex")
    """

    def __init__(self,
                 model: str = "gpt-3.5-turbo",
                 embedding: str = "text-embedding-ada-002",
                 temperature: float = 0.1,
                 top_k: int = 3,
                 with_knowledgebase: bool = True
                 ):
        self.model_name = model
        self.embedding_name = embedding
        self.temperature = temperature
        self.top_k = top_k
        self.with_knowledgebase = with_knowledgebase

        self.llm = get_llm(model, temperature=temperature)
        self.retriever = self._get_retriever()

        self.memory = ConversationBufferMemory(output_key="answer", input_key="question", return_messages=True)

        if self.with_knowledgebase:
            self.final_chain = self._get_with_kb_chain()
        else:
            self.final_chain = self._get_without_kb_chain()

    def _change_config(self,
                       model: Optional[str] = None,
                       embedding: Optional[str] = None,
                       temperature: Optional[float] = None,
                       top_k: Optional[int] = None,
                       with_knowledgebase: Optional[bool] = None):
        flag = False
        if ((model is not None and model != self.model_name) or
                (temperature is not None and temperature != self.temperature)):
            self.model_name = model
            self.llm = get_llm(model, temperature=temperature)
            flag = True
        if embedding is not None and embedding != self.embedding_name:
            self.embedding_name = embedding
            self.retriever = self._get_retriever()
            flag = True
        if top_k is not None and top_k != self.top_k:
            self.retriever = self._get_retriever()
            flag = True
        if with_knowledgebase is not None and with_knowledgebase != self.with_knowledgebase:
            self.with_knowledgebase = with_knowledgebase
            flag = True
        if flag:
            if self.with_knowledgebase:
                self.final_chain = self._get_with_kb_chain()
            else:
                self.final_chain = self._get_without_kb_chain()

    def send(self, questions, model, embedding, temperature, top_k, with_knowledgebase):
        self._change_config(model, embedding, temperature, top_k, with_knowledgebase)
        return self._send(questions)

    def _send(self, questions):
        question_message = {"question": questions}
        response = self.final_chain.invoke(question_message)
        self.memory.save_context(question_message, {"answer": response["answer"].content})
        return response["answer"].content

    def _get_retriever(self):
        embedding = get_embedding(self.embedding_name)
        vectordb = get_vectordb(embedding=embedding)
        return vectordb.as_retriever(search_kwargs={'k': self.top_k})

    def _get_with_kb_chain(self):
        condense_question_prompt = PromptTemplate.from_template(CONDENSE_QUESTION_TEMPLATE)
        answer_prompt = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)
        document_prompt = PromptTemplate.from_template(DOCUMENT_TEMPLATE)

        loaded_memory = RunnablePassthrough.assign(
            chat_history=RunnableLambda(self.memory.load_memory_variables) | itemgetter("history"),
        )

        standalone_question = {
            "standalone_question": {
                                       "question": lambda x: x["question"],
                                       "chat_history": lambda x: _format_chat_history(x["chat_history"]),
                                   }
                                   | condense_question_prompt
                                   | self.llm
                                   | StrOutputParser(),
        }
        retrieved_documents = {
            "docs": itemgetter("standalone_question") | self.retriever,
            "question": lambda x: x["standalone_question"],
        }
        final_inputs = {
            "context": lambda x: _combine_documents(x["docs"], document_prompt=document_prompt),
            "question": itemgetter("question"),
        }
        answer = {
            "answer": final_inputs | answer_prompt | self.llm,
            "docs": itemgetter("docs"),
        }

        chain = (loaded_memory | standalone_question | retrieved_documents | answer)
        return chain

    def _get_without_kb_chain(self):
        simple_qa_prompt = ChatPromptTemplate.from_messages(
            messages=[
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
        answer = {"answer": simple_qa_prompt | self.llm}
        loaded_memory = RunnablePassthrough.assign(
            chat_history=RunnableLambda(self.memory.load_memory_variables) | itemgetter("history"),
        )
        chain = (loaded_memory | answer)
        return chain

    def clear_history(self):
        self.memory.clear()

    def upload_kb(self, files):
        embedding = get_embedding(self.embedding_name)
        vectordb = get_vectordb(embedding, files=files)
        self.retriever = vectordb.as_retriever(search_kwargs={'k': self.top_k})
