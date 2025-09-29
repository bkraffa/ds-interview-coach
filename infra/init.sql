-- Initialize Interview Coach Database

-- Create feedback table
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
);

-- Create query metrics table
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
);

-- Create user sessions table
CREATE TABLE IF NOT EXISTS user_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) UNIQUE,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    total_queries INTEGER DEFAULT 0,
    positive_feedback INTEGER DEFAULT 0,
    negative_feedback INTEGER DEFAULT 0
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback(session_id);
CREATE INDEX IF NOT EXISTS idx_feedback_created ON feedback(created_at);
CREATE INDEX IF NOT EXISTS idx_metrics_session ON query_metrics(session_id);
CREATE INDEX IF NOT EXISTS idx_metrics_created ON query_metrics(created_at);
CREATE INDEX IF NOT EXISTS idx_sessions_session ON user_sessions(session_id);

-- Create views for analytics
CREATE OR REPLACE VIEW hourly_metrics AS
SELECT 
    DATE_TRUNC('hour', created_at) as hour,
    COUNT(*) as query_count,
    AVG(response_time_ms) as avg_response_time,
    MAX(response_time_ms) as max_response_time,
    MIN(response_time_ms) as min_response_time
FROM query_metrics
GROUP BY hour;

CREATE OR REPLACE VIEW satisfaction_metrics AS
SELECT 
    DATE_TRUNC('day', created_at) as day,
    COUNT(CASE WHEN rating = 1 THEN 1 END) as positive_count,
    COUNT(CASE WHEN rating = -1 THEN 1 END) as negative_count,
    COUNT(CASE WHEN rating != 0 THEN 1 END) as total_rated,
    ROUND(AVG(CASE WHEN rating = 1 THEN 1.0 ELSE 0.0 END) * 100, 2) as satisfaction_rate
FROM feedback
WHERE rating != 0
GROUP BY day;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;