"""
LLM Integration Module
Handles prompt templates and LLM response generation for RAG pipeline
"""

import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv

# LLM client imports
import anthropic
import groq

load_dotenv()
logger = logging.getLogger(__name__)


class PromptTemplate:
    """Template for LLM prompts with variable substitution"""
    
    def __init__(self, template: str, variables: List[str] = None):
        self.template = template
        self.variables = variables or []
    
    def format(self, **kwargs) -> str:
        """Format template with provided variables"""
        return self.template.format(**kwargs)
    
    def get_required_variables(self) -> List[str]:
        """Get list of required variables for this template"""
        import re
        variables = re.findall(r'\{(\w+)\}', self.template)
        return list(set(variables))


class LLMProvider:
    """Abstract base class for LLM providers"""
    
    def __init__(self, model: str, api_key: str, **kwargs):
        self.model = model
        self.api_key = api_key
        self.config = kwargs
    
    def generate(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.1) -> Dict[str, Any]:
        """Generate response from LLM"""
        raise NotImplementedError


class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM provider"""
    
    def __init__(self, model: str, api_key: str, **kwargs):
        super().__init__(model, api_key, **kwargs)
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def generate(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.1) -> Dict[str, Any]:
        """Generate response using Anthropic Claude"""
        try:
            start_time = time.time()
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            generation_time = time.time() - start_time
            
            return {
                'response': response.content[0].text,
                'model': self.model,
                'provider': 'anthropic',
                'generation_time': generation_time,
                'tokens_used': {
                    'input': response.usage.input_tokens,
                    'output': response.usage.output_tokens,
                    'total': response.usage.input_tokens + response.usage.output_tokens
                },
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Anthropic generation error: {e}")
            return {
                'error': str(e),
                'model': self.model,
                'provider': 'anthropic',
                'success': False
            }


class GroqProvider(LLMProvider):
    """Groq LLM provider"""
    
    def __init__(self, model: str, api_key: str, **kwargs):
        super().__init__(model, api_key, **kwargs)
        self.client = groq.Groq(api_key=api_key)
    
    def generate(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.1) -> Dict[str, Any]:
        """Generate response using Groq"""
        try:
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            generation_time = time.time() - start_time
            
            return {
                'response': response.choices[0].message.content,
                'model': self.model,
                'provider': 'groq',
                'generation_time': generation_time,
                'tokens_used': {
                    'input': response.usage.prompt_tokens,
                    'output': response.usage.completion_tokens,
                    'total': response.usage.total_tokens
                },
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Groq generation error: {e}")
            return {
                'error': str(e),
                'model': self.model,
                'provider': 'groq',
                'success': False
            }


class RAGLLMIntegration:
    """
    Complete RAG-LLM integration system
    Combines retrieval results with LLM generation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.providers = {}
        self.templates = {}
        
        # Initialize LLM providers
        self._setup_providers()
        
        # Load prompt templates
        self._setup_templates()
        
        # Setup results directory
        self.results_dir = Path("data/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("RAGLLMIntegration initialized")
    
    def _setup_providers(self):
        """Initialize LLM providers from config"""
        llm_config = self.config.get('llm', {})
        providers_config = llm_config.get('providers', {})
        
        # Setup Anthropic
        if 'anthropic' in providers_config:
            anthropic_config = providers_config['anthropic']
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if api_key:
                model = anthropic_config.get('model', 'claude-3-sonnet-20240229')
                # Remove model from config to avoid duplicate parameter
                provider_config = {k: v for k, v in anthropic_config.items() if k != 'model'}
                
                self.providers['anthropic'] = AnthropicProvider(
                    model=model,
                    api_key=api_key,
                    **provider_config
                )
                logger.info("Anthropic provider initialized")
        
        # Setup Groq
        if 'groq' in providers_config:
            groq_config = providers_config['groq']
            api_key = os.getenv('GROQ_API_KEY')
            if api_key:
                model = groq_config.get('model', 'mixtral-8x7b-32768')
                # Remove model from config to avoid duplicate parameter
                provider_config = {k: v for k, v in groq_config.items() if k != 'model'}
                
                self.providers['groq'] = GroqProvider(
                    model=model,
                    api_key=api_key,
                    **provider_config
                )
                logger.info("Groq provider initialized")
        
        if not self.providers:
            logger.warning("No LLM providers configured")
    
    def _setup_templates(self):
        """Setup prompt templates"""
        llm_config = self.config.get('llm', {})
        templates_config = llm_config.get('prompt_templates', {})
        
        # Default template
        default_template = templates_config.get('default', """
You are a helpful AI assistant that answers questions based on the provided context from financial and technical documents.

Context:
{context}

Question: {question}

Please provide a comprehensive answer based on the context provided. If the context doesn't contain enough information to answer the question, please state that clearly.

Answer:""")
        
        self.templates['default'] = PromptTemplate(default_template)
        
        # Financial analysis template
        financial_template = templates_config.get('financial', """
You are a financial analysis expert. Based on the provided financial documents, answer the following question with precision and include relevant metrics or data points when available.

Context:
{context}

Question: {question}

Please provide a detailed financial analysis including:
1. Direct answer to the question
2. Supporting evidence from the documents
3. Relevant financial metrics or data points
4. Potential implications or recommendations

Financial Analysis:""")
        
        self.templates['financial'] = PromptTemplate(financial_template)
        
        # Technical analysis template
        technical_template = """
You are a technical expert analyzing AI and machine learning research. Based on the provided research papers and technical documents, answer the following question with technical accuracy.

Context:
{context}

Question: {question}

Please provide a technical analysis including:
1. Direct answer to the question
2. Technical details and methodologies mentioned
3. Key findings or contributions
4. Limitations or future work discussed

Technical Analysis:"""
        
        self.templates['technical'] = PromptTemplate(technical_template)
        
        logger.info(f"Loaded {len(self.templates)} prompt templates")
    
    def generate_rag_response(self, query: str, retrieved_chunks: List[Dict[str, Any]], 
                            template_name: str = 'default', provider_name: str = None,
                            max_tokens: int = 2000, temperature: float = 0.1) -> Dict[str, Any]:
        """
        Generate RAG response using retrieved chunks and LLM
        
        Args:
            query: User question
            retrieved_chunks: List of chunks from vector search
            template_name: Name of prompt template to use
            provider_name: LLM provider to use (defaults to config default)
            max_tokens: Maximum tokens for generation
            temperature: Generation temperature
            
        Returns:
            Complete RAG response with metadata
        """
        # Select provider
        if provider_name is None:
            provider_name = self.config.get('llm', {}).get('default_provider', 'anthropic')
        
        if provider_name not in self.providers:
            return {
                'error': f"Provider {provider_name} not available",
                'success': False
            }
        
        provider = self.providers[provider_name]
        
        # Select template
        if template_name not in self.templates:
            template_name = 'default'
        
        template = self.templates[template_name]
        
        # Prepare context from retrieved chunks
        context = self._prepare_context(retrieved_chunks)
        
        # Format prompt
        prompt = template.format(context=context, question=query)
        
        # Generate response
        logger.info(f"Generating response using {provider_name} with {template_name} template")
        llm_result = provider.generate(prompt, max_tokens, temperature)
        
        # Combine with retrieval metadata
        response = {
            'query': query,
            'template_used': template_name,
            'provider_used': provider_name,
            'retrieved_chunks': len(retrieved_chunks),
            'context_length': len(context),
            'prompt_length': len(prompt),
            'llm_result': llm_result,
            'chunks_metadata': [
                {
                    'chunk_id': chunk.get('chunk_id', ''),
                    'source_document': chunk.get('source_document', ''),
                    'similarity_score': chunk.get('similarity_score', 0),
                    'rank': chunk.get('rank', 0)
                }
                for chunk in retrieved_chunks
            ]
        }
        
        if llm_result.get('success'):
            response['success'] = True
            response['response'] = llm_result['response']
        else:
            response['success'] = False
            response['error'] = llm_result.get('error', 'Unknown error')
        
        return response
    
    def _prepare_context(self, chunks: List[Dict[str, Any]], max_context_length: int = 8000) -> str:
        """
        Prepare context from retrieved chunks
        
        Args:
            chunks: Retrieved chunks
            max_context_length: Maximum context length in characters
            
        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0
        
        for i, chunk in enumerate(chunks):
            # Format chunk with metadata
            chunk_text = chunk.get('text', '')
            source_doc = chunk.get('source_document', 'Unknown')
            page_numbers = chunk.get('page_numbers', [])
            
            # Create citation
            citation = f"[Source: {source_doc}"
            if page_numbers:
                if isinstance(page_numbers, list) and page_numbers:
                    citation += f", Page {page_numbers[0]}"
                elif isinstance(page_numbers, (int, str)):
                    citation += f", Page {page_numbers}"
            citation += "]"
            
            # Format chunk
            formatted_chunk = f"{citation}\n{chunk_text}\n"
            
            # Check length limit
            if current_length + len(formatted_chunk) > max_context_length:
                if i == 0:  # Always include at least one chunk
                    context_parts.append(formatted_chunk[:max_context_length])
                break
            
            context_parts.append(formatted_chunk)
            current_length += len(formatted_chunk)
        
        return "\n".join(context_parts)
    
    def test_complete_rag_pipeline(self, test_queries: List[str] = None) -> Dict[str, Any]:
        """
        Test the complete RAG pipeline end-to-end
        
        Args:
            test_queries: List of test queries
            
        Returns:
            Test results
        """
        if test_queries is None:
            test_queries = [
                "What are the main financial risks mentioned in the Federal Reserve reports?",
                "How do attention mechanisms work in transformer neural networks?",
                "What monetary policy decisions were announced by the Fed?",
                "Describe the key innovations in the BERT model architecture",
                "What are the regulatory compliance requirements discussed in the documents?"
            ]
        
        logger.info(f"Testing complete RAG pipeline with {len(test_queries)} queries")
        
        # Import here to avoid circular imports
        from vector_stores.db_manager import VectorDatabaseManager
        from sentence_transformers import SentenceTransformer
        
        # Setup vector database manager (using Qdrant since it's working)
        config = {
            'qdrant': {'host': 'localhost', 'port': 6333, 'collection_name': 'document_chunks'}
        }
        manager = VectorDatabaseManager(config)
        db = manager.get_database('qdrant')
        
        # Load embedding model
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        test_results = {
            'test_queries': test_queries,
            'results': [],
            'summary': {}
        }
        
        for i, query in enumerate(test_queries):
            logger.info(f"Testing query {i+1}/{len(test_queries)}: {query[:50]}...")
            
            try:
                # Step 1: Vector search
                query_embedding = embedding_model.encode(query)
                search_results, search_time = db.search(query_embedding, top_k=5, index_type='hnsw')
                
                if not search_results:
                    logger.warning(f"No search results for query: {query}")
                    continue
                
                # Step 2: Determine template based on query content
                template_name = 'financial' if any(term in query.lower() for term in 
                                                ['financial', 'fed', 'monetary', 'banking', 'regulatory']) else 'default'
                
                # Step 3: Generate RAG response
                rag_response = self.generate_rag_response(
                    query=query,
                    retrieved_chunks=search_results,
                    template_name=template_name,
                    provider_name='anthropic'  # Use Anthropic for testing
                )
                
                # Compile test result
                test_result = {
                    'query': query,
                    'search_time': search_time,
                    'retrieved_chunks': len(search_results),
                    'template_used': template_name,
                    'rag_response': rag_response,
                    'success': rag_response.get('success', False)
                }
                
                if rag_response.get('success'):
                    logger.info(f"  ✅ Success - Generated {len(rag_response['response'])} characters")
                else:
                    logger.warning(f"  ❌ Failed - {rag_response.get('error', 'Unknown error')}")
                
                test_results['results'].append(test_result)
                
            except Exception as e:
                logger.error(f"Error testing query '{query}': {e}")
                test_results['results'].append({
                    'query': query,
                    'error': str(e),
                    'success': False
                })
        
        # Generate summary
        successful_tests = [r for r in test_results['results'] if r.get('success')]
        test_results['summary'] = {
            'total_queries': len(test_queries),
            'successful_queries': len(successful_tests),
            'success_rate': len(successful_tests) / len(test_queries) if test_queries else 0,
            'average_search_time': sum(r.get('search_time', 0) for r in successful_tests) / len(successful_tests) if successful_tests else 0,
            'average_response_length': sum(len(r['rag_response'].get('response', '')) for r in successful_tests) / len(successful_tests) if successful_tests else 0
        }
        
        # Save results
        self._save_test_results(test_results)
        
        return test_results
    
    def _save_test_results(self, results: Dict[str, Any]):
        """Save test results to file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save full results
        results_file = self.results_dir / f"rag_llm_test_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save latest
        latest_file = self.results_dir / "latest_rag_llm_results.json"
        with open(latest_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"RAG-LLM test results saved to {results_file}")
    
    def print_test_summary(self, results: Dict[str, Any]):
        """Print formatted test summary"""
        summary = results.get('summary', {})
        
        print("\n" + "="*60)
        print("RAG-LLM PIPELINE TEST SUMMARY")
        print("="*60)
        
        total = summary.get('total_queries', 0)
        successful = summary.get('successful_queries', 0)
        success_rate = summary.get('success_rate', 0) * 100
        
        print(f"Queries tested: {total}")
        print(f"Successful: {successful}")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Average search time: {summary.get('average_search_time', 0)*1000:.1f}ms")
        print(f"Average response length: {summary.get('average_response_length', 0):.0f} characters")
        
        print("\nSample Responses:")
        for i, result in enumerate(results.get('results', [])[:2]):
            if result.get('success'):
                query = result['query']
                response = result['rag_response'].get('response', '')
                print(f"\n{i+1}. Query: {query}")
                print(f"   Response: {response[:200]}...")


# Example usage and testing
if __name__ == "__main__":
    # Test configuration
    config = {
        'llm': {
            'providers': {
                'anthropic': {
                    'model': 'claude-3-sonnet-20240229',
                    'temperature': 0.1,
                    'max_tokens': 2000
                },
                'groq': {
                    'model': 'mixtral-8x7b-32768',
                    'temperature': 0.1,
                    'max_tokens': 2000
                }
            },
            'default_provider': 'anthropic'
        }
    }
    
    # Test the complete RAG-LLM integration
    rag_llm = RAGLLMIntegration(config)
    test_results = rag_llm.test_complete_rag_pipeline()
    
    # Print summary
    rag_llm.print_test_summary(test_results)
