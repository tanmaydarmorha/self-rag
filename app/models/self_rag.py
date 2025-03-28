from typing import List, Dict, Any, TypedDict, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START

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
    
    def __init__(self, vector_store=None, model_name="gpt-4o"):
        """Initialize the Self-RAG agent."""
        self.vector_store = vector_store
        self.model_name = model_name
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        
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
        
        self.retrieval_grader = grade_prompt | self.llm | JsonOutputParser(pydantic_object=GradeDocuments)
        
        # Hallucination grader
        system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.
        Format your response as a JSON object with a single field 'binary_score' with value 'yes' or 'no'."""
        
        hallucination_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ])
        
        self.hallucination_grader = hallucination_prompt | self.llm | JsonOutputParser(pydantic_object=GradeHallucinations)
        
        # Answer quality grader
        system = """You are a grader assessing whether an answer addresses / resolves a question.
        Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question.
        Format your response as a JSON object with a single field 'binary_score' with value 'yes' or 'no'."""
        
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ])
        
        self.answer_grader = answer_prompt | self.llm | JsonOutputParser(pydantic_object=GradeAnswer)
    
    def retrieve(self, state: GraphState) -> GraphState:
        """Retrieve relevant documents for the question."""
        question = state["question"]
        
        # Retrieve documents
        if self.vector_store:
            docs = self.vector_store.similarity_search(question, k=4)
            # Convert to dict format
            docs_list = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs]
        else:
            # If no vector store, return empty list
            docs_list = []
        
        return {"question": question, "documents": docs_list, "generation": state.get("generation")}
    
    def grade_documents(self, state: GraphState) -> Dict[str, Any]:
        """Grade the relevance of retrieved documents."""
        question = state["question"]
        documents = state["documents"]
        
        # If no documents, return "no" for all
        if not documents:
            return {"question": question, "documents": [], "generation": state.get("generation")}
        
        # Grade each document
        graded_documents = []
        for doc in documents:
            grade_result = self.retrieval_grader.invoke({
                "document": doc["page_content"],
                "question": question
            })
            
            # If document is relevant, add to graded documents
            if grade_result.binary_score.lower() == "yes":
                graded_documents.append(doc)
        
        return {"question": question, "documents": graded_documents, "generation": state.get("generation")}
    
    def transform_query(self, state: GraphState) -> GraphState:
        """Transform the query if no relevant documents are found."""
        question = state["question"]
        documents = state["documents"]
        
        # If documents are found, return state as is
        if documents:
            return state
        
        # Otherwise, transform the query
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at transforming questions into more searchable queries.
            Your task is to transform the given question into a query that is more likely to retrieve relevant documents.
            Return only the transformed query, nothing else."""),
            ("human", "Question: {question}")
        ])
        
        # Transform the query
        chain = prompt | self.llm
        transformed_question = chain.invoke({"question": question}).content
        
        return {"question": transformed_question, "documents": [], "generation": state.get("generation")}
    
    def generate(self, state: GraphState) -> GraphState:
        """Generate an answer based on the retrieved documents."""
        question = state["question"]
        documents = state["documents"]
        
        # Format documents for prompt
        formatted_docs = "\n\n".join([doc["page_content"] for doc in documents])
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant. Use the retrieved documents to answer the user's question.
            If the documents don't contain the answer, say that you don't know based on the provided information.
            Answer in a comprehensive way."""),
            ("human", """Question: {question}
            
            Retrieved documents:
            {documents}
            
            Answer:""")
        ])
        
        # Generate answer
        chain = prompt | self.llm
        generation = chain.invoke({
            "question": question,
            "documents": formatted_docs
        }).content
        
        return {"question": question, "documents": documents, "generation": generation}
    
    def grade_generation_v_documents_and_question(self, state: GraphState) -> Dict[str, Any]:
        """Grade the generation against the documents and question."""
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        
        # Format documents for grading
        formatted_docs = "\n\n".join([doc["page_content"] for doc in documents])
        
        # Grade for hallucinations
        hallucination_result = self.hallucination_grader.invoke({
            "documents": formatted_docs,
            "generation": generation
        })
        
        # Grade if answer addresses question
        answer_result = self.answer_grader.invoke({
            "question": question,
            "generation": generation
        })
        
        # If both grades are yes, return END
        if (hallucination_result.binary_score.lower() == "yes" and 
            answer_result.binary_score.lower() == "yes"):
            return {"question": question, "documents": documents, "generation": generation}
        
        # Otherwise, continue to transform query
        return {"question": question, "documents": documents, "generation": generation}
    
    def should_continue(self, state: GraphState) -> str:
        """Decide whether to continue or end the graph."""
        documents = state["documents"]
        generation = state["generation"]
        
        # If we have documents and a generation, end the graph
        if documents and generation:
            return "end"
        
        # Otherwise, continue to transform query
        return "continue"
    
    def build_graph(self):
        """Build the graph for the Self-RAG agent."""
        # Create the graph
        workflow = StateGraph(GraphState)
        
        # Add the nodes
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("transform_query", self.transform_query)
        workflow.add_node("generate", self.generate)
        workflow.add_node("grade_generation", self.grade_generation_v_documents_and_question)
        
        # Build the edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_edge("grade_documents", "transform_query")
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_edge("grade_documents", "generate")
        workflow.add_edge("generate", "grade_generation")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "grade_generation",
            self.should_continue,
            {
                "end": END,
                "continue": "transform_query"
            }
        )
        
        # Compile the graph
        return workflow.compile()
    
    def process_document_and_answer(self, question, vector_store=None):
        """Process a document and answer a question."""
        if vector_store is None:
            vector_store = self.vector_store
        
        # Initialize state
        initial_state = {"question": question, "documents": [], "generation": None}
        
        # Run the graph
        result = None
        for output in self.app.stream(initial_state):
            result = list(output.values())[-1]
        
        return result
    
    async def process_document_and_answer_async(self, question, vector_store=None):
        """Async version of process_document_and_answer."""
        if vector_store is None:
            vector_store = self.vector_store
        
        # Initialize state
        initial_state = {"question": question, "documents": [], "generation": None}
        
        # Run the graph asynchronously
        result = None
        async for output in self.app.astream(initial_state):
            result = list(output.values())[-1]
        
        return result
    
    def get_workflow_graph(self):
        """Get the workflow graph for visualization.
        
        Returns:
            The compiled graph object that can be used for visualization.
        """
        return self.app
        
    def process_multiple_queries(self, questions: List[str], vector_store=None):
        """Process multiple questions and return a mapping of questions to answers.
        
        Args:
            questions: List of questions to process
            vector_store: Optional vector store to use (defaults to self.vector_store)
            
        Returns:
            Dictionary mapping each question to its answer
        """
        import asyncio
        from langchain_core.runnables import RunnableConfig
        
        if vector_store is None:
            vector_store = self.vector_store
            
        async def process_question_async(question):
            """Process a single question asynchronously."""
            print(f"Processing question: {question}")
            # Use invoke_with_config instead of invoke for async execution
            result = await self.process_document_and_answer_async(question, vector_store)
            return question, result.get("generation", "No answer generated")
        
        async def process_all_questions():
            """Process all questions in parallel."""
            # Create tasks for all questions
            tasks = [process_question_async(question) for question in questions]
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks)
            # Convert results to dictionary
            return dict(results)
        
        # Run the async function in an event loop
        return asyncio.run(process_all_questions())
