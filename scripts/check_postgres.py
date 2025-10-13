"""
Check PostgreSQL data and generate reports
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    """Get database connection"""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=os.getenv("POSTGRES_DB", "interview_coach"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres")
    )

def check_tables():
    """Check all tables and their row counts"""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    print("=" * 60)
    print("DATABASE TABLES")
    print("=" * 60)
    
    # Get all tables
    cur.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
    """)
    tables = cur.fetchall()
    
    for table in tables:
        table_name = table['table_name']
        cur.execute(f"SELECT COUNT(*) as count FROM {table_name}")
        count = cur.fetchone()['count']
        print(f"  {table_name}: {count} rows")
    
    cur.close()
    conn.close()

def show_recent_feedback():
    """Show recent feedback"""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    print("\n" + "=" * 60)
    print("RECENT FEEDBACK (Last 10)")
    print("=" * 60)
    
    cur.execute("""
        SELECT 
            id,
            session_id,
            LEFT(query, 50) as query,
            rating,
            category,
            search_method,
            response_time_ms,
            created_at
        FROM feedback
        ORDER BY created_at DESC
        LIMIT 10
    """)
    
    results = cur.fetchall()
    
    if results:
        df = pd.DataFrame(results)
        print(df.to_string(index=False))
    else:
        print("  No feedback data yet")
    
    cur.close()
    conn.close()

def show_query_metrics():
    """Show query metrics"""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    print("\n" + "=" * 60)
    print("QUERY METRICS (Last 10)")
    print("=" * 60)
    
    cur.execute("""
        SELECT 
            id,
            session_id,
            LEFT(query, 40) as query,
            search_method,
            num_results,
            ROUND(top_score::numeric, 3) as top_score,
            reranking_applied,
            response_time_ms,
            created_at
        FROM query_metrics
        ORDER BY created_at DESC
        LIMIT 10
    """)
    
    results = cur.fetchall()
    
    if results:
        df = pd.DataFrame(results)
        print(df.to_string(index=False))
    else:
        print("  No query metrics yet")
    
    cur.close()
    conn.close()

def show_analytics_summary():
    """Show analytics summary"""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    print("\n" + "=" * 60)
    print("ANALYTICS SUMMARY")
    print("=" * 60)
    
    # Overall stats
    cur.execute("""
        SELECT 
            COUNT(DISTINCT session_id) as total_sessions,
            COUNT(*) as total_feedback,
            SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END) as positive,
            SUM(CASE WHEN rating = -1 THEN 1 ELSE 0 END) as negative,
            ROUND(AVG(CASE WHEN rating = 1 THEN 1.0 ELSE 0.0 END) * 100, 1) as satisfaction_rate
        FROM feedback
    """)
    
    overall = cur.fetchone()
    print("\nOverall Statistics:")
    for key, value in overall.items():
        print(f"  {key}: {value}")
    
    # By category
    cur.execute("""
        SELECT 
            category,
            COUNT(*) as queries,
            ROUND(AVG(CASE WHEN rating = 1 THEN 1.0 ELSE 0.0 END) * 100, 1) as satisfaction
        FROM feedback
        WHERE category IS NOT NULL
        GROUP BY category
        ORDER BY queries DESC
    """)
    
    categories = cur.fetchall()
    if categories:
        print("\nBy Category:")
        df = pd.DataFrame(categories)
        print(df.to_string(index=False))
    
    # By search method
    cur.execute("""
        SELECT 
            search_method,
            COUNT(*) as count,
            ROUND(AVG(response_time_ms), 0) as avg_time_ms
        FROM query_metrics
        WHERE search_method IS NOT NULL
        GROUP BY search_method
        ORDER BY count DESC
    """)
    
    methods = cur.fetchall()
    if methods:
        print("\nBy Search Method:")
        df = pd.DataFrame(methods)
        print(df.to_string(index=False))
    
    cur.close()
    conn.close()

def export_to_csv():
    """Export data to CSV for analysis"""
    conn = get_connection()
    
    # Export feedback
    df_feedback = pd.read_sql("""
        SELECT * FROM feedback 
        ORDER BY created_at DESC
    """, conn)
    
    df_feedback.to_csv("reports/feedback_export.csv", index=False)
    print(f"\n Exported {len(df_feedback)} feedback records to reports/feedback_export.csv")
    
    # Export query metrics
    df_metrics = pd.read_sql("""
        SELECT * FROM query_metrics 
        ORDER BY created_at DESC
    """, conn)
    
    df_metrics.to_csv("reports/query_metrics_export.csv", index=False)
    print(f"✅ Exported {len(df_metrics)} query metrics to reports/query_metrics_export.csv")
    
    conn.close()

if __name__ == "__main__":
    try:
        check_tables()
        show_recent_feedback()
        show_query_metrics()
        show_analytics_summary()
        export_to_csv()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure PostgreSQL is running:")
        print("  docker-compose up -d postgres")