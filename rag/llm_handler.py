"""LLM integration for generating responses with conversation memory"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from rag.prompt_templates import CONTEXTUALIZE_QUESTION_PROMPT, ANSWER_QUESTION_PROMPT


class LLMHandler:
    """Handle LLM operations for RAG with conversation memory"""
    
    def __init__(self, api_key: str, retriever=None, openrouter_key: str = None):
        """
        Initialize LLM handler with multiple models and shared conversation memory
        
        Args:
            api_key: Google API key for Gemini
            retriever: Optional LangChain retriever for conversational chain
            openrouter_key: OpenRouter API key for DeepSeek and Kimi
        """
        self.api_key = api_key
        self.openrouter_key = openrouter_key
        self.gemini = self._init_gemini()
        self.retriever = retriever
        
        # Initialize OpenRouter models if key is provided
        self.deepseek = None
        self.kimi = None
        if openrouter_key:
            self.deepseek = self._init_deepseek()
            self.kimi = self._init_kimi()
        
        # Initialize conversation memory (shared across all models) 
        self.store = {}  # Session store for chat history
        
        # Initialize all chains if retriever is provided
        self.chain = None
        self.chain_deepseek = None
        self.chain_kimi = None
        if retriever:
            self.chain = self._init_chain()
            if self.deepseek:
                self.chain_deepseek = self._init_chain(self.deepseek)
            if self.kimi:
                self.chain_kimi = self._init_chain(self.kimi)
    
    def _init_gemini(self):
        """
        Initialize Gemini model with optimal settings
        
        Returns:
            Configured ChatGoogleGenerativeAI instance
        """
        return ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            google_api_key=self.api_key,
            temperature=0.7, 
            max_output_tokens=1024  # Adequate for detailed responses
        )
    
    def _init_deepseek(self):
        """Initialize DeepSeek model via OpenRouter"""
        return ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.openrouter_key,
            model="tngtech/deepseek-r1t2-chimera:free",
            temperature=0.7,
            max_tokens=1024
        )
    
    def _init_kimi(self):
        """Initialize Mistral model via OpenRouter"""
        return ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.openrouter_key,
            model="meta-llama/llama-3.3-70b-instruct:free",
            temperature=0.7,
            max_tokens=1024
        )
    
    def _init_chain(self, llm=None):
        """Initialize modern LangChain v1.x retrieval chain with history awareness"""
        
        # Use your detailed contextualization prompt
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", CONTEXTUALIZE_QUESTION_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        # Create history-aware retriever
        history_aware_retriever = create_history_aware_retriever(
            llm or self.gemini, self.retriever, contextualize_q_prompt
        )
        
        # Use your detailed question answering prompt
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", ANSWER_QUESTION_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        # Create documents chain
        question_answer_chain = create_stuff_documents_chain(llm or self.gemini, qa_prompt)
        
        # Create RAG chain
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        # Add message history
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in self.store:
                self.store[session_id] = ChatMessageHistory()
            return self.store[session_id]
        
        return RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
    
    def set_retriever(self, retriever):
        """
        Set retriever and initialize conversational chain
        
        Args:
            retriever: LangChain-compatible retriever
        """
        self.retriever = retriever
        self.chain = self._init_chain()
        
        # Initialize other model chains if models exist
        if self.deepseek:
            self.chain_deepseek = self._init_chain(self.deepseek)
        if self.kimi:
            self.chain_kimi = self._init_chain(self.kimi)
    
    def _parse_error_message(self, error: Exception) -> str:
        """
        Parse error and return user-friendly message
        
        Args:
            error: Exception object
            
        Returns:
            User-friendly error message
        """
        error_str = str(error).lower()
        
        # Check for rate limit errors
        if any(keyword in error_str for keyword in ['rate limit', 'quota', 'too many requests', '429', 'resource exhausted']):
            return "Rate limit reached. Please try again in a moment."
        
        # Check for API key errors
        if any(keyword in error_str for keyword in ['api key', 'authentication', 'unauthorized', '401']):
            return "API authentication error. Please check configuration."
        
        # Check for network errors
        if any(keyword in error_str for keyword in ['connection', 'network', 'timeout']):
            return " Network error. Please check your connection."
        
        # Generic error
        return "Error generating response. Please try again."
    
    def generate_answer_with_memory(self, question: str):
        """
        Generate answer using modern LangChain v1.x retrieval chain with memory
        
        Args:
            question: User's current question
            
        Returns:
            Tuple of (answer: str, source_documents: list)
        """
        try:
            if not self.chain:
                return "Error: Retriever not initialized. Please set up the retriever first.", []
            
            # Use the chain with session ID for memory
            result = self.chain.invoke(
                {"input": question},
                config={"configurable": {"session_id": "default"}}
            )
            
            # Extract answer and sources from new format
            answer = result.get("answer", "")
            source_docs = result.get("context", [])  # In v1.x, source docs are in 'context'
            
            # Handle empty responses from Gemini
            if not answer or not answer.strip():
                print("[LLM WARNING] Empty response")
                return "No response generated.", []
            
            return answer, source_docs
            
        except Exception as e:
            error_message = self._parse_error_message(e)
            print(f"[LLM ERROR] {type(e).__name__}: {str(e)}")
            return error_message, []
    
    async def generate_multi_model_answers(self, question: str):
        """
        Generate answers from all 3 models in parallel using async with timing
        
        Args:
            question: User's question
            
        Returns:
            dict: {'gemini': (answer, sources, time), 'deepseek': (answer, sources, time), 'phi3': (answer, sources, time)}
        """
        async def call_gemini():
            start_time = time.time()
            result = await asyncio.to_thread(self.generate_answer_with_memory, question)
            elapsed = time.time() - start_time
            return result + (f"{elapsed:.1f}s",)
        
        async def call_deepseek():
            if not self.chain_deepseek:
                return "DeepSeek not initialized", [], "0.0s"
            try:
                start_time = time.time()
                result = await asyncio.to_thread(
                    lambda: self.chain_deepseek.invoke(
                        {"input": question},
                        config={"configurable": {"session_id": "default"}}
                    )
                )
                elapsed = time.time() - start_time
                answer = result.get("answer", "")
                
                # Handle empty response
                if not answer or not answer.strip():
                    return "No response generated", [], f"{elapsed:.1f}s"
                
                return answer, result.get("context", []), f"{elapsed:.1f}s"
            except Exception as e:
                error_msg = self._parse_error_message(e)
                print(f"[DeepSeek ERROR] {type(e).__name__}: {str(e)}")
                return error_msg, [], "0.0s"
        
        async def call_phi3():
            if not self.chain_kimi:  # Still using chain_kimi variable name
                return "Mistral not initialized", [], "0.0s"
            try:
                start_time = time.time()
                result = await asyncio.to_thread(
                    lambda: self.chain_kimi.invoke(
                        {"input": question},
                        config={"configurable": {"session_id": "default"}}
                    )
                )
                elapsed = time.time() - start_time
                answer = result.get("answer", "")
                
                # Handle empty response
                if not answer or not answer.strip():
                    return "⚠️ No response generated", [], f"{elapsed:.1f}s"
                
                return answer, result.get("context", []), f"{elapsed:.1f}s"
            except Exception as e:
                error_msg = self._parse_error_message(e)
                print(f"[Mistral ERROR] {type(e).__name__}: {str(e)}")
                return error_msg, [], "0.0s"
        
        # Run all 3 in parallel
        gemini_result, deepseek_result, phi3_result = await asyncio.gather(
            call_gemini(), call_deepseek(), call_phi3()
        )
        
        return {
            'gemini': gemini_result,
            'deepseek': deepseek_result,
            'mistral': phi3_result
        }
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.store.clear()
    
    def get_memory_variables(self):
        """
        Get current conversation memory
        
        Returns:
            Dictionary containing chat history
        """
        return {"sessions": len(self.store), "store": list(self.store.keys())}
