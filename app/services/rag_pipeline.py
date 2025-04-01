"""Self RAG pipeline implementation using LangGraph."""
from typing import List, Dict, Any, Optional, Annotated, TypedDict, Sequence, Tuple
import uuid
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from app.config import settings
from app.models.schemas import DocumentChunk, RetrievedDocument, CritiqueResult, QueryResult
from app.services.embedding import embedding_service
from app.services.vector_store import vector_store_service


# Define state types for the Self RAG graph
class GraphState(TypedDict):
    """State for the Self RAG graph."""
    question: str
    retrieved_documents: List[RetrievedDocument]
    critique: Optional[CritiqueResult]
    answer: Optional[str]
    final_answer: Optional[str]


# Different LLM configurations for different nodes in the RAG pipeline
def get_retriever_llm():
    """Get LLM for retrieval node."""
    return AzureChatOpenAI(
        azure_deployment=settings.azure_openai_chat_deployment,
        openai_api_version=settings.azure_openai_api_version,
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        temperature=0.0,  # Low temperature for more deterministic retrieval
    )


def get_critique_llm():
    """Get LLM for critique node."""
    return AzureChatOpenAI(
        azure_deployment=settings.azure_openai_chat_deployment,
        openai_api_version=settings.azure_openai_api_version,
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        temperature=0.2,  # Slightly higher temperature for critique
    )


def get_generator_llm():
    """Get LLM for generation node."""
    return AzureChatOpenAI(
        azure_deployment=settings.azure_openai_chat_deployment,
        openai_api_version=settings.azure_openai_api_version,
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        temperature=0.7,  # Higher temperature for more creative generation
    )


def get_synthesizer_llm():
    """Get LLM for synthesis node."""
    return AzureChatOpenAI(
        azure_deployment=settings.azure_openai_chat_deployment,
        openai_api_version=settings.azure_openai_api_version,
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        temperature=0.3,  # Moderate temperature for synthesis
    )


class SelfRAGPipeline:
    """Self RAG pipeline for document retrieval and question answering."""

    def __init__(self):
        """Initialize the Self RAG pipeline."""
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the Self RAG graph."""
        # Create the graph
        graph = StateGraph(GraphState)

        # Add nodes to the graph
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("critique", self._critique_node)
        graph.add_node("generate", self._generate_node)
        graph.add_node("synthesize", self._synthesize_node)

        # Add edges to the graph
        graph.add_edge("retrieve", "critique")
        graph.add_conditional_edges(
            "critique",
            self._should_retrieve_more,
            {
                True: "retrieve",
                False: "generate",
            },
        )
        graph.add_edge("generate", "synthesize")
        graph.add_edge("synthesize", END)

        # Set the entry point
        graph.set_entry_point("retrieve")

        # Compile the graph
        return graph.compile()

    async def _retrieve_node(self, state: GraphState) -> GraphState:
        """Retrieve relevant documents for the question."""
        question = state["question"]
        
        # Generate embedding for the question
        query_embedding = await embedding_service.get_query_embedding(question)
        
        # Search for similar chunks
        retrieved_documents = await vector_store_service.search_similar_chunks(
            query_embedding=query_embedding,
            top_k=5
        )
        
        # Update state with retrieved documents
        return {"retrieved_documents": retrieved_documents}

    async def _critique_node(self, state: GraphState) -> GraphState:
        """Critique the retrieved documents for relevance."""
        question = state["question"]
        retrieved_documents = state["retrieved_documents"]
        
        # Skip critique if no documents retrieved
        if not retrieved_documents:
            return {"critique": CritiqueResult(is_relevant=False, reasoning="No documents retrieved.")}
        
        # Prepare context for the critique
        context = "\n\n".join([f"Document {i+1}:\n{doc.content}" for i, doc in enumerate(retrieved_documents)])
        
        # Create critique prompt
        critique_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=(
                "You are a critical evaluator of retrieved documents. "
                "Your task is to determine if the retrieved documents are relevant to the question. "
                "Be critical and honest in your assessment."
            )),
            HumanMessage(content=(
                "Question: {question}\n\n"
                "Retrieved Documents:\n{context}\n\n"
                "Are these documents relevant to the question? "
                "Provide your reasoning and then conclude with either YES or NO."
            )),
        ])
        
        # Create critique chain
        critique_chain = critique_prompt | get_critique_llm() | StrOutputParser()
        
        # Run critique chain
        critique_result = await critique_chain.ainvoke({"question": question, "context": context})
        
        # Parse critique result
        is_relevant = "YES" in critique_result.upper().split()[-1]
        
        # Update state with critique
        return {"critique": CritiqueResult(is_relevant=is_relevant, reasoning=critique_result)}

    def _should_retrieve_more(self, state: GraphState) -> bool:
        """Determine if more documents should be retrieved."""
        critique = state.get("critique")
        
        # If critique exists and documents are not relevant, retrieve more
        if critique and not critique.is_relevant:
            # In a more complex implementation, we could modify the query here
            # based on the critique to get better results
            return True
        
        return False

    async def _generate_node(self, state: GraphState) -> GraphState:
        """Generate an answer based on retrieved documents."""
        question = state["question"]
        retrieved_documents = state["retrieved_documents"]
        
        # Prepare context for generation
        if retrieved_documents:
            context = "\n\n".join([f"Document {i+1}:\n{doc.content}" for i, doc in enumerate(retrieved_documents)])
        else:
            context = "No relevant documents found."
        
        # Create generation prompt
        generation_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=(
                "You are a helpful AI assistant. "
                "Your task is to answer the user's question based on the retrieved documents. "
                "If the documents don't contain relevant information, state that you don't know "
                "and avoid making up information."
            )),
            HumanMessage(content=(
                "Question: {question}\n\n"
                "Retrieved Documents:\n{context}\n\n"
                "Please provide a comprehensive answer to the question based on the retrieved documents."
            )),
        ])
        
        # Create generation chain
        generation_chain = generation_prompt | get_generator_llm() | StrOutputParser()
        
        # Run generation chain
        answer = await generation_chain.ainvoke({"question": question, "context": context})
        
        # Update state with answer
        return {"answer": answer}

    async def _synthesize_node(self, state: GraphState) -> GraphState:
        """Synthesize the final answer with citations."""
        question = state["question"]
        retrieved_documents = state["retrieved_documents"]
        answer = state["answer"]
        critique = state.get("critique")
        
        # Prepare context for synthesis
        if retrieved_documents:
            context = "\n\n".join([f"Document {i+1}:\n{doc.content}" for i, doc in enumerate(retrieved_documents)])
        else:
            context = "No relevant documents found."
        
        # Create synthesis prompt
        synthesis_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=(
                "You are a helpful AI assistant. "
                "Your task is to synthesize a final answer to the user's question "
                "based on the retrieved documents and the initial answer. "
                "Include citations to the documents where appropriate using [Doc X] notation."
            )),
            HumanMessage(content=(
                "Question: {question}\n\n"
                "Retrieved Documents:\n{context}\n\n"
                "Initial Answer: {answer}\n\n"
                "Please synthesize a final answer with appropriate citations."
            )),
        ])
        
        # Create synthesis chain
        synthesis_chain = synthesis_prompt | get_synthesizer_llm() | StrOutputParser()
        
        # Run synthesis chain
        final_answer = await synthesis_chain.ainvoke({
            "question": question, 
            "context": context, 
            "answer": answer
        })
        
        # Update state with final answer
        return {"final_answer": final_answer}

    async def process_question(self, question: str) -> QueryResult:
        """
        Process a question using the Self RAG pipeline.
        
        Args:
            question: The question to answer
            
        Returns:
            QueryResult with answer and retrieved documents
        """
        # Initialize state
        initial_state = {"question": question, "retrieved_documents": [], "critique": None, "answer": None, "final_answer": None}
        
        # Run the graph
        result = await self.graph.ainvoke(initial_state)
        
        # Create query result
        query_result = QueryResult(
            answer=result["final_answer"],
            retrieved_documents=result["retrieved_documents"],
            critique=result.get("critique")
        )
        
        return query_result


# Initialize Self RAG pipeline
self_rag_pipeline = SelfRAGPipeline()
