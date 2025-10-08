"""
Enhanced DS Interview Coach - Streamlit Application
"""

import os
import time
import uuid
from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

# Import services
from services.rag import EnhancedRAG
from services.feedback import FeedbackService

# Page configuration
st.set_page_config(
    page_title="DS Interview Coach Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .feedback-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    .stButton>button {
        background-color: #667eea;
        color: white;
    }
    .stButton>button:hover {
        background-color: #764ba2;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "query_count" not in st.session_state:
    st.session_state.query_count = 0

# Initialize services
@st.cache_resource
def init_services():
    """Initialize RAG and feedback services"""
    rag = EnhancedRAG(
        qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
        qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
        collection=os.getenv("QDRANT_COLLECTION", "interview_chunks")
    )
    feedback = FeedbackService()
    feedback.create_session(st.session_state.session_id)

    try:
        with st.spinner("Loading search index..."):
            rag._load_bm25_index()
    except Exception as e:
        st.warning(f"Could not pre-load BM25 index: {e}")

    return rag, feedback

rag_service, feedback_service = init_services()

# Header
st.markdown('<h1 class="main-header">🎯 DS Interview Coach Pro</h1>', unsafe_allow_html=True)
st.markdown("### AI-Powered Interview Preparation with Advanced RAG")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Mode selection - updated to match your categories
    mode = st.selectbox(
        "Interview Type",
        ["All Topics", "Machine Learning", "Deep Learning", "Behavioral"],
        help="Focus on specific interview types"
    )
    mode_map = {
        "All Topics": "all",
        "Machine Learning": "machine_learning",
        "Deep Learning": "deep_learning",
        "Behavioral": "behavioral"
    }
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        use_hybrid = st.checkbox("Hybrid Search", value=False,  # Changed from True to False
                                help="Combine semantic and keyword search (may be slow on first use)")
        use_rerank = st.checkbox("Re-ranking", value=True,
                                help="Use cross-encoder for result re-ranking")
        use_rewrite = st.checkbox("Query Rewriting", value=True,
                                help="Automatically enhance queries")
        
        if use_hybrid:
            st.warning("⚠️ Hybrid search loads all documents into memory on first use. This may take 30-60 seconds.")
            hybrid_alpha = st.slider("Hybrid Alpha", 0.0, 1.0, 0.5,
                                    help="Balance between semantic (1.0) and keyword (0.0) search")
        else:
            hybrid_alpha = 0.5
        
        top_k = st.slider("Results to Retrieve", 3, 10, 5)
        temperature = st.slider("Response Creativity", 0.0, 1.0, 0.7,
                            help="Higher = more creative, Lower = more focused")
    
    # Session info
    st.divider()
    st.header("📊 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Queries", st.session_state.query_count)
    with col2:
        st.metric("Session", st.session_state.session_id[:8] + "...")
    
    # Quick tips
    st.divider()
    st.header("💡 Quick Tips")
    st.info("""
    **Technical Questions:**
    - Be specific about concepts
    - Ask for examples
    - Request complexity analysis
    
    **Behavioral Questions:**
    - Use STAR format
    - Include context
    - Ask for frameworks
    """)

# Main content area with tabs
tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📈 Analytics", "📚 Knowledge Base", "ℹ️ Help"])

with tab1:
    # Chat interface
    st.header("Interview Practice Chat")
    
    # Example questions
    # Around line 130, update example questions
with st.expander("📝 Example Questions"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Machine Learning:**")
        ml_examples = [
            "What is gradient descent?",
            "Explain bias-variance tradeoff",
            "What are ensemble methods?",
            "How does k-means clustering work?",
            "What is cross-validation?"
        ]
        for ex in ml_examples:
            if st.button(ex, key=f"ml_{ex[:15]}"):
                st.session_state.query_input = ex
    
    with col2:
        st.markdown("**Deep Learning:**")
        dl_examples = [
            "Explain backpropagation",
            "What are CNNs used for?",
            "How does attention mechanism work?",
            "What is transfer learning?",
            "Explain vanishing gradient problem"
        ]
        for ex in dl_examples:
            if st.button(ex, key=f"dl_{ex[:15]}"):
                st.session_state.query_input = ex
    
    with col3:
        st.markdown("**Behavioral:**")
        behavioral_examples = [
            "Tell me about yourself",
            "What are your greatest strengths?",
            "Why should I hire you?",
            "Describe a challenging project",
            "How do you handle pressure?"
        ]
        for ex in behavioral_examples:
            if st.button(ex, key=f"beh_{ex[:15]}"):
                st.session_state.query_input = ex
    
    # Query input
    query = st.text_area(
        "Ask your interview question:",
        value=st.session_state.get("query_input", ""),
        height=100,
        placeholder="E.g., 'Explain the vanishing gradient problem and how to solve it'"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        submit = st.button("🚀 Get Answer", type="primary", use_container_width=True)
    with col2:
        clear_chat = st.button("🗑️ Clear Chat", use_container_width=True)
    with col3:
        export_chat = st.button("💾 Export Chat", use_container_width=True)
    
    if clear_chat:
        st.session_state.chat_history = []
        st.rerun()
    
    if export_chat and st.session_state.chat_history:
        # Create export content
        export_content = "DS Interview Coach - Chat Export\n"
        export_content += f"Session: {st.session_state.session_id}\n"
        export_content += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        export_content += "="*50 + "\n\n"
        
        for item in st.session_state.chat_history:
            export_content += f"Q: {item['query']}\n"
            export_content += f"A: {item['answer']}\n"
            export_content += "-"*30 + "\n\n"
        
        st.download_button(
            label="📥 Download Chat History",
            data=export_content,
            file_name=f"interview_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    # Process query
    if submit and query:
        with st.spinner("🔍 Searching knowledge base..."):
            start_time = time.time()
            
            # Retrieve relevant content
            results, metadata = rag_service.retrieve(
                query=query,
                top_k=top_k,
                mode=mode_map[mode],
                use_rewrite=use_rewrite,
                use_hybrid=use_hybrid,
                use_rerank=use_rerank,
                hybrid_alpha=hybrid_alpha if use_hybrid else 0.5
            )
            
            retrieval_time = (time.time() - start_time) * 1000
            
            # Record query metrics
            scores = [r.get("rerank_score", r.get("score", 0)) for r in results]
            feedback_service.record_query_metrics(
                session_id=st.session_state.session_id,
                query=query,
                rewritten_query=metadata.get("rewritten_query"),
                search_method=metadata.get("search_method"),
                num_results=len(results),
                scores=scores,
                reranking_applied=metadata.get("reranking_applied", False),
                response_time_ms=int(retrieval_time)
            )
        
        with st.spinner("🤖 Generating answer..."):
            # Generate answer
            answer = rag_service.generate_answer(
                query=query,
                context=results,
                mode=mode_map[mode],
                temperature=temperature
            )
            
            total_time = (time.time() - start_time) * 1000
            
            # Add to chat history
            st.session_state.chat_history.append({
                "query": query,
                "answer": answer,
                "sources": results[:3],  # Top 3 sources
                "metadata": metadata,
                "time_ms": total_time
            })
            
            st.session_state.query_count += 1
            st.rerun()
    
    # Display chat history
        
    for i, item in enumerate(reversed(st.session_state.chat_history)):
        # Question
        st.markdown(f"### 🙋 Question {len(st.session_state.chat_history) - i}")
        st.write(item["query"])
        
        # Answer
        st.markdown("### 🤖 Answer")
        st.write(item["answer"])
        
        # Metadata
        col1, col2 = st.columns([3, 1])
        with col1:
            if item["metadata"].get("rewritten_query"):
                st.caption(f"✨ Enhanced query: {item['metadata']['rewritten_query'][:60]}...")
        with col2:
            st.caption(f"⚡ {item['time_ms']:.0f}ms | 🔍 {item['metadata'].get('search_method', 'unknown')}")
        
        # Sources - NOT nested in expander, just displayed
        st.markdown("**📚 Sources:**")
        for j, source in enumerate(item["sources"], 1):
            with st.container():
                col_a, col_b = st.columns([4, 1])
                with col_a:
                    st.caption(f"**{j}.** {source.get('source', 'Unknown')} | {source.get('category', 'unknown')}")
                with col_b:
                    score = source.get('rerank_score', source.get('score', 0))
                    st.caption(f"Score: {score:.3f}")
                
                # Show a preview of the text
                preview_text = source.get('question', source['text'][:200])
                st.text(f"   {preview_text}...")
        
        # Feedback section
        st.markdown('<div class="feedback-section">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns([1, 1, 2, 2])
        
        with col1:
            if st.button("👍 Helpful", key=f"up_{i}", use_container_width=True):
                feedback_service.record_feedback(
                    session_id=st.session_state.session_id,
                    query=item["query"],
                    response=item["answer"],
                    rating=1,
                    category=mode_map[mode],
                    search_method=item["metadata"].get("search_method"),
                    response_time_ms=int(item["time_ms"])
                )
                st.success("Thanks!")
        
        with col2:
            if st.button("👎 Not helpful", key=f"down_{i}", use_container_width=True):
                feedback_service.record_feedback(
                    session_id=st.session_state.session_id,
                    query=item["query"],
                    response=item["answer"],
                    rating=-1,
                    category=mode_map[mode],
                    search_method=item["metadata"].get("search_method"),
                    response_time_ms=int(item["time_ms"])
                )
                st.warning("We'll improve!")
        
        with col3:
            detailed = st.text_input("Additional feedback:", key=f"feedback_{i}", placeholder="Optional comments...")
        
        with col4:
            if st.button("📝 Submit", key=f"submit_fb_{i}") and detailed:
                feedback_service.record_feedback(
                    session_id=st.session_state.session_id,
                    query=item["query"],
                    response=item["answer"],
                    rating=0,
                    detailed_feedback=detailed,
                    category=mode_map[mode],
                    search_method=item["metadata"].get("search_method"),
                    response_time_ms=int(item["time_ms"])
                )
                st.info("Feedback recorded!")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.divider()

with tab2:
    # Analytics Dashboard
    st.header("📈 Performance Analytics")
    
    # Get analytics data
    analytics = feedback_service.get_analytics_summary()
    
    if analytics and analytics.get("overall"):
        # Overall metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Sessions", analytics["overall"].get("total_sessions", 0))
        with col2:
            st.metric("Total Queries", analytics["overall"].get("total_queries", 0))
        with col3:
            satisfaction = (analytics["overall"].get("satisfaction_rate") or 0) * 100
            st.metric("Satisfaction Rate", f"{satisfaction:.1f}%")
        with col4:
            avg_time = analytics["overall"].get("avg_response_time", 0)
            st.metric("Avg Response Time", f"{avg_time:.0f}ms" if avg_time else "N/A")
        
        # Category breakdown
        if analytics.get("by_category"):
            st.subheader("Performance by Category")
            df_category = pd.DataFrame(analytics["by_category"])
            
            fig = px.bar(df_category, x="category", y="query_count",
                        title="Queries by Category",
                        labels={"query_count": "Number of Queries"})
            st.plotly_chart(fig, use_container_width=True)
        
        # Search method comparison
        if analytics.get("search_methods"):
            st.subheader("Search Method Performance")
            df_methods = pd.DataFrame(analytics["search_methods"])
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.pie(df_methods, values="usage_count", names="search_method",
                           title="Search Method Usage")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(df_methods, x="search_method", y="avg_response_time",
                           title="Response Time by Method",
                           labels={"avg_response_time": "Avg Response Time (ms)"})
                st.plotly_chart(fig, use_container_width=True)
        
        # Problematic queries
        if analytics.get("problematic_queries"):
            st.subheader("Areas for Improvement")
            st.caption("Queries with negative feedback")
            df_problems = pd.DataFrame(analytics["problematic_queries"])
            st.dataframe(df_problems, use_container_width=True)
    else:
        st.info("No analytics data available yet. Start asking questions to generate data!")
    
    # Export options
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Export Analytics Report"):
            st.info("Analytics report generation coming soon!")
    with col2:
        if st.button("📥 Download Feedback Data"):
            export_path = feedback_service.export_feedback_csv()
            if export_path:
                with open(export_path, "rb") as f:
                    st.download_button(
                        label="Download CSV",
                        data=f,
                        file_name="feedback_export.csv",
                        mime="text/csv"
                    )

with tab3:
    # Knowledge Base Info
    st.header("📚 Knowledge Base")
    
    st.info("""
    Our knowledge base includes:
    - **50+ Machine Learning** interview questions covering algorithms, evaluation metrics, and best practices
    - **111+ Deep Learning** questions on neural networks, CNNs, RNNs, transformers, and more
    - **64 Behavioral** interview questions with expert-crafted responses using proven frameworks
    """)
    
    # Sample questions from the knowledge base
    st.subheader("Sample Questions by Topic")
    
    topics = {
        "Machine Learning Fundamentals": [
            "What is the difference between supervised and unsupervised learning?",
            "Explain the bias-variance tradeoff",
            "What is regularization and why is it important?"
        ],
        "Deep Learning": [
            "What is backpropagation?",
            "Explain different activation functions",
            "How does batch normalization work?"
        ],
        "CNNs & Computer Vision": [
            "What are the different layers in CNN?",
            "How do CNNs achieve translation invariance?",
            "Explain pooling and its purposes"
        ],
        "Behavioral Questions": [
            "Tell me about yourself",
            "What are your greatest strengths?",
            "Why should I hire you?",
            "How do you handle working under pressure?"
        ]
    }
    
    for topic, questions in topics.items():
        with st.expander(topic):
            for q in questions:
                st.write(f"• {q}")
    
    st.divider()
    st.subheader("📤 Upload Your Own Content")
    st.caption("Add your own interview materials to the knowledge base")
    
    uploaded_file = st.file_uploader(
        "Choose a file (PDF, TXT, CSV)",
        type=["pdf", "txt", "csv", "md"],
        help="Upload interview questions, guides, or notes"
    )
    
    if uploaded_file is not None:
        if st.button("Process and Add to Knowledge Base"):
            st.info("File upload and processing feature coming soon!")

with tab4:
    # Help and Documentation
    st.header("ℹ️ Help & Documentation")
    
    with st.expander("🚀 Getting Started"):
        st.markdown("""
        1. **Choose your interview type** in the sidebar (Technical/Behavioral/All)
        2. **Ask questions** in the Chat tab - be specific for best results
        3. **Review the answer** and sources provided
        4. **Give feedback** to help improve the system
        5. **Check Analytics** to track your preparation progress
        """)
    
    with st.expander("🎯 Tips for Best Results"):
        st.markdown("""
        **For Technical Questions:**
        - Be specific about the concept or algorithm
        - Ask for examples and use cases
        - Request complexity analysis when relevant
        - Follow up with "why" and "how" questions
        
        **For Behavioral Questions:**
        - Provide context about the role or situation
        - Ask for STAR method examples
        - Request industry-specific scenarios
        - Ask about common follow-up questions
        """)
    
    with st.expander("🔧 Advanced Features"):
        st.markdown("""
        **Hybrid Search:** Combines semantic understanding with keyword matching for better results
        
        **Query Rewriting:** Automatically enhances your questions with relevant terms
        
        **Re-ranking:** Uses advanced models to order results by relevance
        
        **Temperature Control:** Adjust creativity vs. focus in responses
        """)
    
    with st.expander("📊 Understanding Analytics"):
        st.markdown("""
        **Satisfaction Rate:** Percentage of positive feedback
        
        **Response Time:** Time to retrieve and generate answers
        
        **Search Methods:** Compare performance of different retrieval approaches
        
        **Problem Areas:** Questions that often receive negative feedback
        """)
    
    st.divider()
    st.markdown("""
    ### 🤝 About
    DS Interview Coach Pro is an AI-powered system designed to help data science candidates
    prepare for technical and behavioral interviews. It uses advanced RAG (Retrieval Augmented Generation)
    techniques to provide accurate, relevant, and helpful responses.
    
    **Version:** 1.0.0  
    **Last Updated:** 2025  
    **Contact:** brunocaraffa@gmail.com
    """)

# Footer
st.divider()
st.caption("Built with ❤️ for the Data Science community | Powered by OpenAI & Qdrant")