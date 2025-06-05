"""
Benchmark Runner for RAG Evaluation
Tests retrieval performance across all vector databases and index types
"""

import logging
import time
import json
import statistics
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Comprehensive benchmarking system for vector database performance
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.test_queries = config.get('test_queries', [])
        self.num_runs = config.get('benchmark', {}).get('num_runs', 5)
        self.top_k = config.get('benchmark', {}).get('top_k', 10)
        self.timeout = config.get('benchmark', {}).get('timeout', 30)
        
        # Load embedding model for query encoding
        model_name = config.get('embeddings', {}).get('model_name', 'all-MiniLM-L6-v2')
        self.embedding_model = SentenceTransformer(model_name)
        
        # Setup results directory
        self.results_dir = Path("data/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("BenchmarkRunner initialized")
    
    def run_comprehensive_benchmark(self, databases: List[str], index_types: List[str], 
                                  test_queries: List[str] = None) -> Dict[str, Any]:
        """
        Run comprehensive benchmark across all database and index combinations
        
        Args:
            databases: List of database names to test
            index_types: List of index types to test
            test_queries: Optional custom test queries
            
        Returns:
            Comprehensive benchmark results
        """
        test_queries = test_queries or self.test_queries
        if not test_queries:
            # Generate default test queries if none provided
            test_queries = self._generate_default_queries()
        
        logger.info(f"Starting comprehensive benchmark...")
        logger.info(f"Databases: {databases}")
        logger.info(f"Index types: {index_types}")
        logger.info(f"Test queries: {len(test_queries)}")
        logger.info(f"Runs per test: {self.num_runs}")
        
        # Encode all test queries
        logger.info("Encoding test queries...")
        query_embeddings = self.embedding_model.encode(test_queries)
        
        benchmark_results = {
            'metadata': {
                'databases': databases,
                'index_types': index_types,
                'num_queries': len(test_queries),
                'num_runs': self.num_runs,
                'top_k': self.top_k,
                'test_queries': test_queries
            },
            'results': {},
            'summary': {}
        }
        
        # Import here to avoid circular imports
        from vector_stores.db_manager import VectorDatabaseManager
        
        # Initialize database manager
        config = {
            'faiss': {'storage_path': 'data/embeddings/faiss_index'},
            'chromadb': {'storage_path': 'data/embeddings/chroma_db'},
            'qdrant': {'host': 'localhost', 'port': 6333, 'collection_name': 'document_chunks'}
        }
        
        manager = VectorDatabaseManager(config)
        
        # Test each database + index combination
        for db_name in databases:
            if db_name not in benchmark_results['results']:
                benchmark_results['results'][db_name] = {}
            
            try:
                db = manager.get_database(db_name)
                logger.info(f"Testing {db_name.upper()}...")
                
                for index_type in index_types:
                    logger.info(f"  Testing {index_type} index...")
                    
                    # Run benchmark for this combination
                    combo_results = self._benchmark_db_index_combo(
                        db, db_name, index_type, query_embeddings, test_queries
                    )
                    
                    benchmark_results['results'][db_name][index_type] = combo_results
                    
                    # Log quick summary
                    avg_time = combo_results['performance']['average_query_time']
                    logger.info(f"    Average query time: {avg_time:.4f}s")
                    
            except Exception as e:
                logger.error(f"Error testing {db_name}: {e}")
                benchmark_results['results'][db_name] = {'error': str(e)}
        
        # Generate summary statistics
        benchmark_results['summary'] = self._generate_summary(benchmark_results['results'])
        
        # Save results
        self._save_benchmark_results(benchmark_results)
        
        logger.info("Comprehensive benchmark completed")
        return benchmark_results
    
    def _benchmark_db_index_combo(self, db, db_name: str, index_type: str, 
                                 query_embeddings: np.ndarray, test_queries: List[str]) -> Dict[str, Any]:
        """Benchmark a specific database + index combination"""
        
        combo_results = {
            'database': db_name,
            'index_type': index_type,
            'performance': {},
            'accuracy': {},
            'queries': []
        }
        
        all_query_times = []
        all_similarity_scores = []
        successful_queries = 0
        
        # Test each query multiple times
        for query_idx, (query_text, query_embedding) in enumerate(zip(test_queries, query_embeddings)):
            query_results = {
                'query_text': query_text,
                'query_index': query_idx,
                'runs': []
            }
            
            query_times = []
            best_results = None
            
            # Multiple runs for timing consistency
            for run in range(self.num_runs):
                try:
                    results, query_time = db.search(
                        query_embedding, 
                        top_k=self.top_k, 
                        index_type=index_type
                    )
                    
                    query_times.append(query_time)
                    all_query_times.append(query_time)
                    
                    # Store results from first successful run
                    if best_results is None and results:
                        best_results = results
                        # Extract similarity scores
                        scores = [r.get('similarity_score', 0) for r in results]
                        all_similarity_scores.extend(scores)
                    
                    query_results['runs'].append({
                        'run': run + 1,
                        'query_time': query_time,
                        'num_results': len(results),
                        'success': len(results) > 0
                    })
                    
                    successful_queries += 1
                    
                except Exception as e:
                    logger.warning(f"Query failed for {db_name}/{index_type}: {e}")
                    query_results['runs'].append({
                        'run': run + 1,
                        'error': str(e),
                        'success': False
                    })
            
            # Calculate query statistics
            if query_times:
                query_results['timing_stats'] = {
                    'average': statistics.mean(query_times),
                    'median': statistics.median(query_times),
                    'min': min(query_times),
                    'max': max(query_times),
                    'std_dev': statistics.stdev(query_times) if len(query_times) > 1 else 0
                }
            
            if best_results:
                query_results['sample_results'] = best_results[:3]  # Store top 3 for analysis
            
            combo_results['queries'].append(query_results)
        
        # Calculate overall performance metrics
        if all_query_times:
            combo_results['performance'] = {
                'total_queries': len(test_queries) * self.num_runs,
                'successful_queries': successful_queries,
                'success_rate': successful_queries / (len(test_queries) * self.num_runs),
                'average_query_time': statistics.mean(all_query_times),
                'median_query_time': statistics.median(all_query_times),
                'min_query_time': min(all_query_times),
                'max_query_time': max(all_query_times),
                'p95_query_time': np.percentile(all_query_times, 95),
                'p99_query_time': np.percentile(all_query_times, 99),
                'queries_per_second': 1 / statistics.mean(all_query_times) if all_query_times else 0
            }
        
        # Calculate accuracy metrics
        if all_similarity_scores:
            combo_results['accuracy'] = {
                'average_similarity': statistics.mean(all_similarity_scores),
                'median_similarity': statistics.median(all_similarity_scores),
                'min_similarity': min(all_similarity_scores),
                'max_similarity': max(all_similarity_scores),
                'similarity_std_dev': statistics.stdev(all_similarity_scores) if len(all_similarity_scores) > 1 else 0,
                'high_similarity_ratio': len([s for s in all_similarity_scores if s > 0.7]) / len(all_similarity_scores),
                'low_similarity_ratio': len([s for s in all_similarity_scores if s < 0.3]) / len(all_similarity_scores)
            }
        
        return combo_results
    
    def _generate_default_queries(self) -> List[str]:
        """Generate default test queries if none provided"""
        return [
            "What are the main financial risks mentioned?",
            "Describe the regulatory compliance requirements",
            "What are the key performance metrics?",
            "Summarize the market analysis findings",
            "What recommendations are provided?",
            "Explain the financial projections",
            "What are the audit findings?",
            "Describe operational challenges",
            "What strategic initiatives are proposed?",
            "Summarize competitive landscape analysis"
        ]
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics across all tests"""
        summary = {
            'fastest_database': None,
            'slowest_database': None,
            'best_accuracy': None,
            'performance_ranking': [],
            'accuracy_ranking': [],
            'index_comparison': {}
        }
        
        # Collect performance data
        performance_data = []
        accuracy_data = []
        
        for db_name, db_results in results.items():
            if 'error' in db_results:
                continue
                
            for index_type, combo_results in db_results.items():
                if 'performance' in combo_results:
                    perf = combo_results['performance']
                    acc = combo_results.get('accuracy', {})
                    
                    performance_data.append({
                        'database': db_name,
                        'index_type': index_type,
                        'avg_query_time': perf.get('average_query_time', float('inf')),
                        'queries_per_second': perf.get('queries_per_second', 0),
                        'success_rate': perf.get('success_rate', 0)
                    })
                    
                    if acc:
                        accuracy_data.append({
                            'database': db_name,
                            'index_type': index_type,
                            'avg_similarity': acc.get('average_similarity', 0),
                            'high_similarity_ratio': acc.get('high_similarity_ratio', 0)
                        })
        
        # Rank by performance (fastest query time)
        performance_ranking = sorted(performance_data, key=lambda x: x['avg_query_time'])
        summary['performance_ranking'] = performance_ranking
        
        if performance_ranking:
            summary['fastest_database'] = f"{performance_ranking[0]['database']}_{performance_ranking[0]['index_type']}"
            summary['slowest_database'] = f"{performance_ranking[-1]['database']}_{performance_ranking[-1]['index_type']}"
        
        # Rank by accuracy (highest similarity)
        accuracy_ranking = sorted(accuracy_data, key=lambda x: x['avg_similarity'], reverse=True)
        summary['accuracy_ranking'] = accuracy_ranking
        
        if accuracy_ranking:
            summary['best_accuracy'] = f"{accuracy_ranking[0]['database']}_{accuracy_ranking[0]['index_type']}"
        
        # Index type comparison
        index_performance = {}
        for item in performance_data:
            index_type = item['index_type']
            if index_type not in index_performance:
                index_performance[index_type] = []
            index_performance[index_type].append(item['avg_query_time'])
        
        for index_type, times in index_performance.items():
            summary['index_comparison'][index_type] = {
                'average_time': statistics.mean(times),
                'best_time': min(times),
                'worst_time': max(times)
            }
        
        return summary
    
    def _save_benchmark_results(self, results: Dict[str, Any]):
        """Save benchmark results to files"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save full results
        results_file = self.results_dir / f"benchmark_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary only
        summary_file = self.results_dir / f"benchmark_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(results['summary'], f, indent=2, default=str)
        
        # Save latest (for easy access)
        latest_file = self.results_dir / "latest_benchmark_results.json"
        with open(latest_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Benchmark results saved to {results_file}")
        logger.info(f"Summary saved to {summary_file}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of results"""
        summary = results.get('summary', {})
        
        print("\n" + "="*60)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*60)
        
        print(f"Fastest combination: {summary.get('fastest_database', 'N/A')}")
        print(f"Best accuracy: {summary.get('best_accuracy', 'N/A')}")
        
        print("\nPerformance Ranking (by speed):")
        for i, item in enumerate(summary.get('performance_ranking', [])[:5], 1):
            db_combo = f"{item['database']}_{item['index_type']}"
            time_ms = item['avg_query_time'] * 1000
            qps = item['queries_per_second']
            print(f"  {i}. {db_combo}: {time_ms:.2f}ms ({qps:.1f} QPS)")
        
        print("\nAccuracy Ranking (by similarity):")
        for i, item in enumerate(summary.get('accuracy_ranking', [])[:5], 1):
            db_combo = f"{item['database']}_{item['index_type']}"
            similarity = item['avg_similarity']
            high_ratio = item['high_similarity_ratio'] * 100
            print(f"  {i}. {db_combo}: {similarity:.3f} avg similarity ({high_ratio:.1f}% high quality)")
        
        print("\nIndex Type Comparison:")
        for index_type, stats in summary.get('index_comparison', {}).items():
            avg_ms = stats['average_time'] * 1000
            best_ms = stats['best_time'] * 1000
            print(f"  {index_type}: {avg_ms:.2f}ms avg (best: {best_ms:.2f}ms)")


# Example usage and testing
if __name__ == "__main__":
    # Test configuration
    config = {
        'test_queries': [
            "What are the main financial risks mentioned?",
            "Describe the regulatory compliance requirements",
            "What are the key performance metrics?",
            "Summarize the market analysis findings",
            "What recommendations are provided?"
        ],
        'benchmark': {
            'num_runs': 3,
            'top_k': 10,
            'timeout': 30
        },
        'embeddings': {
            'model_name': 'all-MiniLM-L6-v2'
        }
    }
    
    runner = BenchmarkRunner(config)
    
    # Run benchmark
    results = runner.run_comprehensive_benchmark(
        databases=['faiss', 'chromadb', 'qdrant'],
        index_types=['flat', 'hnsw', 'ivf']
    )
    
    # Print summary
    runner.print_summary(results)
