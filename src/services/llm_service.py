"""
LLM service for generating responses.
"""
from typing import List, Dict, Optional

from src.core.logging import get_logger
from src.core.exceptions import LLMError
from src.core.config import get_config
from src.llm.base import LLMClient
from src.llm.ollama_client import OllamaClient
from src.llm.prompt_builder import PromptBuilder


logger = get_logger(__name__)


class LLMService:
    """
    Service for LLM text generation.
    
    This class provides a high-level interface for generating
    text with LLMs, including prompt building,
    streaming support, and response handling.
    
    Example:
        >>> service = LLMService()
        >>> response = service.generate("Hello, how are you?")
        >>> print(response)
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> None:
        """
        Initialize LLM service.
        
        Args:
            model: Model name (uses config default if None)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        config = get_config()
        
        # Initialize client
        self.client = OllamaClient(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Initialize prompt builder
        self.prompt_builder = PromptBuilder()
        
        logger.info(
            "LLM service initialized",
            model=self.client.get_model_name(),
            temperature=self.client.temperature,
            max_tokens=self.client.max_tokens
        )
    
    def generate(
        self,
        prompt: str,
        use_template: bool = False,
        **kwargs: Dict
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: Input prompt
            use_template: Whether to use RAG template
            **kwargs: Additional parameters
            
        Returns:
            Generated response
            
        Example:
            >>> service = LLMService()
            >>> response = service.generate("Hello!")
        """
        try:
            logger.debug(
                "Generating response",
                model=self.client.get_model_name(),
                prompt_length=len(prompt)
            )
            
            # Generate response
            response = self.client.generate(prompt, **kwargs)
            
            logger.info(
                "Response generated successfully",
                model=self.client.get_model_name(),
                response_length=len(response)
            )
            
            return response
            
        except Exception as e:
            logger.error("Failed to generate response", error=str(e))
            raise LLMError(
                f"Failed to generate LLM response: {str(e)}",
                details={"error": str(e)}
            )
    
    def generate_rag(
        self,
        question: str,
        contexts: List[str],
        **kwargs: Dict
    ) -> str:
        """
        Generate a RAG response with context.
        
        Args:
            question: User's question
            contexts: List of context strings from retrieved documents
            **kwargs: Additional parameters
            
        Returns:
            Generated response using RAG prompt
            
        Example:
            >>> service = LLMService()
            >>> response = service.generate_rag(
            ...     question="What is RAG?",
            ...     contexts=["RAG stands for Retrieval-Augmented Generation"]
            ... )
        """
        try:
            # Build RAG prompt
            prompt = self.prompt_builder.build_rag_prompt(question, contexts)
            
            # Generate response
            response = self.generate(prompt, **kwargs)
            
            logger.info(
                "RAG response generated",
                question=question[:100],
                context_count=len(contexts),
                response_length=len(response)
            )
            
            return response
            
        except Exception as e:
            logger.error("Failed to generate RAG response", error=str(e))
            raise LLMError(
                f"Failed to generate RAG response: {str(e)}",
                details={"question": question[:100], "error": str(e)}
            )
    
    def generate_stream(
        self,
        prompt: str,
        use_template: bool = False,
        **kwargs: Dict
    ) -> List[str]:
        """
        Generate a streaming response from the LLM.
        
        Args:
            prompt: Input prompt
            use_template: Whether to use RAG template
            **kwargs: Additional parameters
            
        Returns:
            List of text chunks
            
        Example:
            >>> service = LLMService()
            >>> chunks = service.generate_stream("Hello!")
            >>> for chunk in chunks:
            ...     print(chunk, end='')
        """
        try:
            logger.debug(
                "Generating streaming response",
                model=self.client.get_model_name(),
                prompt_length=len(prompt)
            )
            
            # Generate streaming response
            chunks = list(self.client.generate_stream(prompt, **kwargs))
            
            logger.info(
                "Streaming response generated",
                model=self.client.get_model_name(),
                chunk_count=len(chunks),
                total_length=sum(len(c) for c in chunks)
            )
            
            return chunks
            
        except Exception as e:
            logger.error("Failed to generate streaming response", error=str(e))
            raise LLMError(
                f"Failed to generate streaming response: {str(e)}",
                details={"error": str(e)}
            )
    
    def generate_rag_stream(
        self,
        question: str,
        contexts: List[str],
        **kwargs: Dict
    ) -> List[str]:
        """
        Generate a streaming RAG response with context.
        
        Args:
            question: User's question
            contexts: List of context strings from retrieved documents
            **kwargs: Additional parameters
            
        Returns:
            List of text chunks
            
        Example:
            >>> service = LLMService()
            >>> chunks = service.generate_rag_stream(
            ...     question="What is RAG?",
            ...     contexts=["RAG is..."]
            ... )
        """
        try:
            # Build RAG prompt
            prompt = self.prompt_builder.build_rag_prompt(question, contexts)
            
            # Generate streaming response
            chunks = self.generate_stream(prompt, **kwargs)
            
            logger.info(
                "RAG streaming response generated",
                question=question[:100],
                context_count=len(contexts),
                chunk_count=len(chunks)
            )
            
            return chunks
            
        except Exception as e:
            logger.error("Failed to generate RAG streaming response", error=str(e))
            raise LLMError(
                f"Failed to generate RAG streaming response: {str(e)}",
                details={"question": question[:100], "error": str(e)}
            )
    
    def summarize(self, text: str, max_length: int = 200) -> str:
        """
        Summarize a text.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary in words
            
        Returns:
            Generated summary
            
        Example:
            >>> service = LLMService()
            >>> summary = service.summarize(long_document)
        """
        try:
            # Build summarization prompt
            prompt = self.prompt_builder.build_summarization_prompt(text, max_length)
            
            # Generate summary
            summary = self.generate(prompt)
            
            logger.info(
                "Summary generated",
                original_length=len(text),
                summary_length=len(summary),
                max_length=max_length
            )
            
            return summary
            
        except Exception as e:
            logger.error("Failed to generate summary", error=str(e))
            raise LLMError(
                f"Failed to generate summary: {str(e)}",
                details={"error": str(e)}
            )
    
    def list_models(self) -> List[str]:
        """
        List available LLM models.
        
        Returns:
            List of model names
        """
        try:
            return self.client.list_models()
        except Exception as e:
            logger.error("Failed to list models", error=str(e))
            raise LLMError(
                f"Failed to list LLM models: {str(e)}",
                details={"error": str(e)}
            )
    
    def set_model(self, model: str) -> None:
        """
        Change the LLM model.
        
        Args:
            model: Name of the new model
        """
        try:
            # Create new client with different model
            self.client = OllamaClient(model=model)
            
            logger.info("LLM model changed", model=model)
            
        except Exception as e:
            logger.error("Failed to change model", model=model, error=str(e))
            raise LLMError(
                f"Failed to change model to '{model}': {str(e)}",
                details={"model": model, "error": str(e)}
            )
    
    def get_model_name(self) -> str:
        """
        Get the current model name.
        
        Returns:
            Model name
        """
        return self.client.get_model_name()
    
    def close(self) -> None:
        """Close LLM client."""
        self.client.close()
        logger.debug("LLM service closed")
