from typing import List, Dict, Any, TypedDict, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langgraph.graph import END, StateGraph, START

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    
    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """
    question: str
    generation: Optional[str]
    documents: List[Dict[str, Any]]

class SelfRAG:
    """Self-RAG implementation using LangGraph."""
    
    def __init__(self, vector_store=None, model_name="gemma3:latest"):
        """Initialize the Self-RAG agent."""
        self.vector_store = vector_store
        self.model_name = model_name
        self.llm = Ollama(model=model_name, temperature=0)
        
        # Initialize graders
        self.setup_graders()
        
        # Build the graph
        self.app = self.build_graph()
    
    def setup_graders(self):
        """Set up the graders for document relevance, hallucinations, and answer quality."""
        # Document relevance grader
        system = """You are a grader assessing relevance of a retrieved document to a user question.
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
        Format your response as a JSON object with a single field 'binary_score' with value 'yes' or 'no'."""
        
        grade_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ])
        
        self.retrieval_grader = grade_prompt | self.llm | self._parse_binary_score
        
        # Hallucination grader
        system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.
        Format your response as a JSON object with a single field 'binary_score' with value 'yes' or 'no'."""
        
        hallucination_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ])
        
        self.hallucination_grader = hallucination_prompt | self.llm | self._parse_binary_score
        
        # Answer quality grader
        system = """You are a grader assessing whether an answer addresses / resolves a question.
        Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question.
        Format your response as a JSON object with a single field 'binary_score' with value 'yes' or 'no'."""
        
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ])
        
        self.answer_grader = answer_prompt | self.llm | self._parse_binary_score
    
    def _parse_binary_score(self, text):
        """Parse binary score from text response."""
        text = text.lower()
        if "binary_score" in text and "yes" in text:
            return GradeDocuments(binary_score="yes")
        elif "binary_score" in text and "no" in text:
            return GradeDocuments(binary_score="no")
        elif "yes" in text:
            return GradeDocuments(binary_score="yes")
        else:
            return GradeDocuments(binary_score="no")
    
    def retrieve(self, state: GraphState) -> GraphState:
        """Retrieve relevant documents for the question."""
        question = state["question"]
        
        # Get documents from vector store
        if self.vector_store:
            docs = self.vector_store.similarity_search(question, k=5)
            documents = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs]
        else:
            # If no vector store, return empty documents
            documents = []
        
        return {"question": question, "documents": documents}
    
    def grade_documents(self, state: GraphState) -> GraphState:
        """Grade the relevance of retrieved documents."""
        question = state["question"]
        documents = state["documents"]
        
        # Grade each document
        graded_documents = []
        for doc in documents:
            try:
                result = self.retrieval_grader.invoke({
                    "question": question,
                    "document": doc["page_content"]
                })
                
                doc["relevant"] = result.binary_score.lower() == "yes"
                graded_documents.append(doc)
            except Exception as e:
                print(f"Error grading document: {e}")
                doc["relevant"] = False
                graded_documents.append(doc)
        
        return {"question": question, "documents": graded_documents}
    
    def decide_to_generate(self, state: GraphState) -> str:
        """Decide whether to generate an answer or transform the query."""
        documents = state["documents"]
        
        # Check if there are any relevant documents
        relevant_docs = [doc for doc in documents if doc.get("relevant", False)]
        
        if len(relevant_docs) > 0:
            return "generate"
        else:
            return "transform_query"
    
    def transform_query(self, state: GraphState) -> GraphState:
        """Transform the query to get better search results."""
        question = state["question"]
        
        # Create a prompt to transform the query
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that helps users reformulate their questions to get better search results. Keep your response concise and focused on the reformulated question only."),
            ("human", "Original question: {question}\n\nPlease reformulate this question to make it more specific and searchable.")
        ])
        
        # Transform the query
        chain = prompt | self.llm | StrOutputParser()
        transformed_question = chain.invoke({"question": question})
        
        return {"question": transformed_question, "documents": []}
    
    def generate(self, state: GraphState) -> GraphState:
        """Generate an answer based on the relevant documents."""
        question = state["question"]
        documents = state["documents"]
        
        # Filter relevant documents
        relevant_docs = [doc for doc in documents if doc.get("relevant", False)]
        
        # Format documents for the prompt
        formatted_docs = "\n\n".join([doc["page_content"] for doc in relevant_docs])
        
        # Create a prompt for generation
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that answers questions based on the provided documents. Be concise and accurate."),
            ("human", "Documents:\n\n{documents}\n\nQuestion: {question}\n\nAnswer the question based only on the provided documents. If the documents don't contain the necessary information, say so.")
        ])
        
        # Generate answer
        chain = prompt | self.llm | StrOutputParser()
        generation = chain.invoke({
            "question": question,
            "documents": formatted_docs
        })
        
        return {
            "question": question,
            "documents": documents,
            "generation": generation
        }
    
    def grade_generation_v_documents_and_question(self, state: GraphState) -> str:
        """Grade the generation against documents and question."""
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        
        # Format documents for the grader
        formatted_docs = "\n\n".join([doc["page_content"] for doc in documents if doc.get("relevant", False)])
        
        # Check for hallucinations
        try:
            hallucination_result = self.hallucination_grader.invoke({
                "documents": formatted_docs,
                "generation": generation
            })
            
            is_grounded = hallucination_result.binary_score.lower() == "yes"
            
            if not is_grounded:
                return "not supported"
            
            # Check if answer addresses the question
            answer_result = self.answer_grader.invoke({
                "question": question,
                "generation": generation
            })
            
            is_useful = answer_result.binary_score.lower() == "yes"
            
            if is_useful:
                return "useful"
            else:
                return "not useful"
        
        except Exception as e:
            print(f"Error grading generation: {e}")
            return "not supported"
    
    def build_graph(self):
        """Build the Self-RAG graph."""
        # Create the graph
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("generate", self.generate)
        workflow.add_node("transform_query", self.transform_query)
        
        # Add edges
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_conditional_edges(
            "generate",
            self.grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "transform_query",
            },
        )
        
        # Compile
        return workflow.compile()
    
    def process_document_and_answer(self, question: str, vector_store=None):
        """Process a document and answer a question using Self-RAG."""
        if vector_store:
            self.vector_store = vector_store
        
        # Initialize state
        initial_state = {"question": question, "documents": [], "generation": None}
        
        # Run the graph
        result = None
        for output in self.app.stream(initial_state):
            result = list(output.values())[-1]
        
        return result
