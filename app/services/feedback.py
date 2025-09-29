"""
Feedback Collection and Analytics Service
"""

import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
from pathlib import Path

class FeedbackService:
    def __init__(self):
        self.db_config = {
            "host": os.getenv("POSTGRES_HOST", "postgres"),
            "port": int(os.getenv("POSTGRES_PORT", "5432")),
            "database": os.getenv("POSTGRES_DB", "interview_coach"),
            "user": os.getenv("POSTGRES_USER", "postgres"),
            "password": os.getenv("POSTGRES_PASSWORD", "postgres")
        }
        self.init_database()
    
    def init_database(self):
        """Initialize database tables if not exists"""
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            # Create feedback table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255),
                    query TEXT NOT NULL,
                    response TEXT,
                    rating INTEGER CHECK (rating IN (-1, 0, 1)),
                    detailed_feedback TEXT,
                    category VARCHAR(50),
                    search_method VARCHAR(50),
                    response_time_ms INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create query_metrics table for analytics
            cur.execute("""
                CREATE TABLE IF NOT EXISTS query_metrics (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255),
                    query TEXT NOT NULL,
                    rewritten_query TEXT,
                    search_method VARCHAR(50),
                    num_results INTEGER,
                    top_score FLOAT,
                    avg_score FLOAT,
                    reranking_applied BOOLEAN,
                    response_time_ms INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create user_sessions table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) UNIQUE,
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMP,
                    total_queries INTEGER DEFAULT 0,
                    positive_feedback INTEGER DEFAULT 0,
                    negative_feedback INTEGER DEFAULT 0
                )
            """)
            
            conn.commit()
            cur.close()
            conn.close()
            
        except Exception as e:
            print(f"Database initialization error: {e}")
    
    def get_connection(self):
        """Get database connection"""
        return psycopg2.connect(**self.db_config)
    
    def record_feedback(
        self,
        session_id: str,
        query: str,
        response: str,
        rating: int,
        detailed_feedback: Optional[str] = None,
        category: Optional[str] = None,
        search_method: Optional[str] = None,
        response_time_ms: Optional[int] = None
    ) -> bool:
        """Record user feedback"""
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO feedback 
                (session_id, query, response, rating, detailed_feedback, 
                 category, search_method, response_time_ms)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (session_id, query, response, rating, detailed_feedback,
                  category, search_method, response_time_ms))
            
            # Update session statistics
            if rating != 0:
                if rating == 1:
                    cur.execute("""
                        UPDATE user_sessions 
                        SET positive_feedback = positive_feedback + 1
                        WHERE session_id = %s
                    """, (session_id,))
                else:
                    cur.execute("""
                        UPDATE user_sessions 
                        SET negative_feedback = negative_feedback + 1
                        WHERE session_id = %s
                    """, (session_id,))
            
            conn.commit()
            cur.close()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error recording feedback: {e}")
            return False
    
    def record_query_metrics(
        self,
        session_id: str,
        query: str,
        rewritten_query: Optional[str] = None,
        search_method: Optional[str] = None,
        num_results: int = 0,
        scores: List[float] = None,
        reranking_applied: bool = False,
        response_time_ms: Optional[int] = None
    ) -> bool:
        """Record query performance metrics"""
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            top_score = max(scores) if scores else 0.0
            avg_score = sum(scores) / len(scores) if scores else 0.0
            
            cur.execute("""
                INSERT INTO query_metrics 
                (session_id, query, rewritten_query, search_method, 
                 num_results, top_score, avg_score, reranking_applied, response_time_ms)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (session_id, query, rewritten_query, search_method,
                  num_results, top_score, avg_score, reranking_applied, response_time_ms))
            
            # Update session query count
            cur.execute("""
                UPDATE user_sessions 
                SET total_queries = total_queries + 1
                WHERE session_id = %s
            """, (session_id,))
            
            conn.commit()
            cur.close()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error recording metrics: {e}")
            return False
    
    def create_session(self, session_id: str) -> bool:
        """Create a new user session"""
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO user_sessions (session_id)
                VALUES (%s)
                ON CONFLICT (session_id) DO NOTHING
            """, (session_id,))
            
            conn.commit()
            cur.close()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error creating session: {e}")
            return False
    
    def get_analytics_summary(self) -> Dict:
        """Get summary analytics for monitoring"""
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # Overall statistics
            cur.execute("""
                SELECT 
                    COUNT(DISTINCT session_id) as total_sessions,
                    COUNT(*) as total_queries,
                    AVG(CASE WHEN rating = 1 THEN 1 ELSE 0 END) as satisfaction_rate,
                    AVG(response_time_ms) as avg_response_time
                FROM feedback
                WHERE created_at > NOW() - INTERVAL '7 days'
            """)
            overall = cur.fetchone()
            
            # Category breakdown
            cur.execute("""
                SELECT 
                    category,
                    COUNT(*) as query_count,
                    AVG(CASE WHEN rating = 1 THEN 1 ELSE 0 END) as satisfaction_rate
                FROM feedback
                WHERE created_at > NOW() - INTERVAL '7 days'
                GROUP BY category
            """)
            categories = cur.fetchall()
            
            # Search method performance
            cur.execute("""
                SELECT 
                    search_method,
                    COUNT(*) as usage_count,
                    AVG(top_score) as avg_top_score,
                    AVG(response_time_ms) as avg_response_time
                FROM query_metrics
                WHERE created_at > NOW() - INTERVAL '7 days'
                GROUP BY search_method
            """)
            search_methods = cur.fetchall()
            
            # Top problematic queries (negative feedback)
            cur.execute("""
                SELECT 
                    query,
                    COUNT(*) as negative_count
                FROM feedback
                WHERE rating = -1
                    AND created_at > NOW() - INTERVAL '7 days'
                GROUP BY query
                ORDER BY negative_count DESC
                LIMIT 10
            """)
            problematic_queries = cur.fetchall()
            
            cur.close()
            conn.close()
            
            return {
                "overall": overall,
                "by_category": categories,
                "search_methods": search_methods,
                "problematic_queries": problematic_queries,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error getting analytics: {e}")
            return {}
    
    def export_feedback_csv(self, output_path: str = "reports/feedback_export.csv"):
        """Export feedback data to CSV"""
        try:
            conn = self.get_connection()
            
            query = """
                SELECT 
                    f.*,
                    s.total_queries,
                    s.positive_feedback,
                    s.negative_feedback
                FROM feedback f
                LEFT JOIN user_sessions s ON f.session_id = s.session_id
                ORDER BY f.created_at DESC
            """
            
            df = pd.read_sql(query, conn)
            
            Path(output_path).parent.mkdir(exist_ok=True)
            df.to_csv(output_path, index=False)
            
            conn.close()
            return output_path
            
        except Exception as e:
            print(f"Error exporting feedback: {e}")
            return None
    
    def get_metrics_for_grafana(self) -> List[Dict]:
        """Get metrics formatted for Grafana/Prometheus"""
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            metrics = []
            
            # Query rate
            cur.execute("""
                SELECT 
                    DATE_TRUNC('hour', created_at) as time,
                    COUNT(*) as query_count
                FROM query_metrics
                WHERE created_at > NOW() - INTERVAL '24 hours'
                GROUP BY time
                ORDER BY time
            """)
            query_rate = cur.fetchall()
            metrics.append({
                "name": "query_rate",
                "data": query_rate
            })
            
            # Satisfaction rate over time
            cur.execute("""
                SELECT 
                    DATE_TRUNC('hour', created_at) as time,
                    AVG(CASE WHEN rating = 1 THEN 1.0 ELSE 0.0 END) * 100 as satisfaction_rate
                FROM feedback
                WHERE created_at > NOW() - INTERVAL '24 hours'
                    AND rating != 0
                GROUP BY time
                ORDER BY time
            """)
            satisfaction = cur.fetchall()
            metrics.append({
                "name": "satisfaction_rate",
                "data": satisfaction
            })
            
            # Response time percentiles
            cur.execute("""
                SELECT 
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY response_time_ms) as p50,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) as p95,
                    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY response_time_ms) as p99
                FROM query_metrics
                WHERE created_at > NOW() - INTERVAL '1 hour'
            """)
            latency = cur.fetchone()
            metrics.append({
                "name": "response_time_percentiles",
                "data": latency
            })
            
            cur.close()
            conn.close()
            
            return metrics
            
        except Exception as e:
            print(f"Error getting Grafana metrics: {e}")
            return []