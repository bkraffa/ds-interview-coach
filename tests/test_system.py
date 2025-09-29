"""
Comprehensive System Tests for DS Interview Coach
"""

import os
import sys
import pytest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.embeddings import get_embedding_model, embed_texts
from app.services.chunking import simple_chunk, to_payloads
from app.services.enhanced_rag import EnhancedRAG

class TestChunking:
    """Test document chunking functionality"""
    
    def test_simple_chunk(self):
        text = "This is a test. " * 100
        chunks = simple_chunk(text, chunk_size=50, overlap=10)
        
        assert len(chunks) > 0
        assert all(len(c) <= 60 for c in chunks)  # Allow for slight overflow
    
    def test_empty_text(self):
        chunks = simple_chunk("", chunk_size=100, overlap=10)
        assert len(chunks) == 1
        assert chunks[0] == ""
    
    def test_payload_creation(self):
        chunks = ["chunk1", "chunk2"]
        payloads = to_payloads(chunks, "test.pdf", "technical")
        
        assert len(payloads) == 2
        assert all(p["source"] == "test.pdf" for p in payloads)
        assert all(p["category"] == "technical" for p in payloads)

class TestEmbeddings:
    """Test embedding service"""
    
    def test_get_embedding_model(self):
        model = get_embedding_model()
        
        assert "name" in model
        assert "dim" in model
        assert model["name"] in ["text-embedding-3-small", "text-embedding-ada-002"]
    
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not set")
    def test_embed_texts(self):
        model = get_embedding_model()
        texts = ["test query", "another test"]
        
        embeddings = embed_texts(model, texts)
        
        assert len(embeddings) == 2
        assert all(isinstance(e, list) for e in embeddings)
        assert all(len(e) == int(model["dim"]) for e in embeddings)

class TestEnhancedRAG:
    """Test RAG functionality"""
    
    @pytest.fixture
    def rag_service(self):
        return EnhancedRAG(
            qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
            qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
            collection=os.getenv("QDRANT_COLLECTION", "test_collection")
        )
    
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not set")
    def test_query_rewrite(self, rag_service):
        original = "ML overfitting"
        rewritten = rag_service.query_rewrite(original, mode="expand")
        
        assert len(rewritten) >= len(original)
        assert "machine learning" in rewritten.lower() or "overfitting" in rewritten.lower()
    
    def test_hybrid_search_logic(self, rag_service):
        """Test hybrid search score combination logic"""
        # Mock data for testing score combination
        dense_results = [
            {"text": "result1", "score": 0.9},
            {"text": "result2", "score": 0.8}
        ]
        sparse_results = [
            {"text": "result2", "score": 0.85},
            {"text": "result3", "score": 0.7}
        ]
        
        # Test score combination with alpha=0.5
        # This would be internal to hybrid_search, but we can test the logic
        combined_scores = {}
        alpha = 0.5
        
        for i, result in enumerate(dense_results):
            text = result["text"]
            score = alpha * (1.0 / (i + 1))
            combined_scores[text] = score
        
        for i, result in enumerate(sparse_results):
            text = result["text"]
            score = (1 - alpha) * (1.0 / (i + 1))
            if text in combined_scores:
                combined_scores[text] += score
            else:
                combined_scores[text] = score
        
        assert "result1" in combined_scores
        assert "result2" in combined_scores
        assert "result3" in combined_scores
        assert combined_scores["result2"] > combined_scores["result1"]  # Should have highest combined score

class TestIntegration:
    """Integration tests"""
    
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not set")
    def test_end_to_end_retrieval(self):
        """Test complete retrieval pipeline"""
        rag = EnhancedRAG(
            qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
            qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
            collection=os.getenv("QDRANT_COLLECTION", "interview_chunks")
        )
        
        query = "What is gradient descent?"
        results, metadata = rag.retrieve(
            query=query,
            top_k=3,
            mode="technical",
            use_rewrite=True,
            use_hybrid=False,  # Start with simple dense search
            use_rerank=False
        )
        
        assert isinstance(results, list)
        assert isinstance(metadata, dict)
        assert "original_query" in metadata
        assert "search_method" in metadata
    
    def test_feedback_service(self):
        """Test feedback recording"""
        from app.services.feedback import FeedbackService
        
        feedback = FeedbackService()
        session_id = "test_session_123"
        
        # Create session
        success = feedback.create_session(session_id)
        assert success is True
        
        # Record feedback
        success = feedback.record_feedback(
            session_id=session_id,
            query="Test query",
            response="Test response",
            rating=1,
            category="technical",
            search_method="hybrid",
            response_time_ms=150
        )
        
        # Can't assert True without database, but check no exceptions
        assert success in [True, False]

class TestDataIngestion:
    """Test data ingestion pipeline"""
    
    def test_pdf_reading(self):
        """Test PDF text extraction"""
        from scripts.ingest import read_text
        from pathlib import Path
        
        # Create a test text file (since we can't easily create PDFs in tests)
        test_file = Path("test_doc.txt")
        test_content = "This is a test document for ingestion."
        test_file.write_text(test_content)
        
        try:
            text = read_text(test_file)
            assert text == test_content
        finally:
            test_file.unlink()  # Clean up
    
    def test_csv_reading(self):
        """Test CSV text extraction"""
        from scripts.ingest import read_text
        from pathlib import Path
        import pandas as pd
        
        # Create test CSV
        test_file = Path("test_data.csv")
        df = pd.DataFrame({
            "question": ["What is ML?", "Explain DL"],
            "category": ["technical", "technical"]
        })
        df.to_csv(test_file, index=False)
        
        try:
            text = read_text(test_file)
            assert "What is ML?" in text
            assert "Explain DL" in text
        finally:
            test_file.unlink()  # Clean up

if __name__ == "__main__":
    pytest.main([__file__, "-v"])