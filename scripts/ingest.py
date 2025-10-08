import argparse
import os
import glob
import hashlib
from pathlib import Path
from typing import List, Dict
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from openai import OpenAI
from tqdm import tqdm

# Embeddings
def get_embedding_model():
    return {
        "name": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        "dim": int(os.getenv("EMBEDDING_DIM", "1536"))
    }

def embed_texts(model_cfg: dict, texts: List[str], batch_size: int = 128) -> List[List[float]]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = model_cfg["name"]
    out: List[List[float]] = []
    
    # Progress bar for embedding batches
    num_batches = (len(texts) + batch_size - 1) // batch_size
    with tqdm(total=len(texts), desc="Embedding texts", unit="text") as pbar:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = client.embeddings.create(model=model, input=batch)
            out.extend([d.embedding for d in resp.data])
            pbar.update(len(batch))
    
    return out

# Generate answers with GPT
def generate_answer(question: str, category: str = "technical") -> str:
    """Generate an answer for a question using GPT-4"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    system_prompt = """You are an expert data science interviewer. 
    Provide a comprehensive, accurate answer to the interview question.
    For technical questions, include code examples when appropriate.
    Keep answers concise but complete (2-4 paragraphs)."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {question}\n\nProvide a detailed answer suitable for an interview preparation guide."}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "Answer generation failed. Please refer to official documentation."

def read_csv(path: Path) -> List[Dict[str, str]]:
    """Read CSV with questions and optionally answers"""
    df = pd.read_csv(path)
    
    # Detect if this CSV has both question and answer columns
    has_answer_col = 'answer' in df.columns
    
    # Get question column
    question_col = None
    for col in df.columns:
        if 'question' in col.lower():
            question_col = col
            break
    
    if not question_col:
        print(f"Warning: No question column found in {path.name}")
        return []
    
    qa_pairs = []
    for _, row in df.iterrows():
        question = str(row[question_col]).strip()
        if question and question != 'nan':
            qa_pair = {
                'question': question,
                'has_answer': has_answer_col
            }
            
            # Add answer if it exists
            if has_answer_col:
                answer = str(row['answer']).strip()
                qa_pair['answer'] = answer if answer and answer != 'nan' else None
            else:
                qa_pair['answer'] = None
            
            qa_pairs.append(qa_pair)
    
    return qa_pairs

def determine_category_from_filename(filename: str) -> str:
    """Determine category based on filename"""
    filename_lower = filename.lower()
    
    if 'behavioral' in filename_lower or 'interview' in filename_lower or '64' in filename_lower:
        return 'behavioral'
    elif 'deeplearning' in filename_lower or 'deep_learning' in filename_lower:
        return 'deep_learning'
    elif 'machine' in filename_lower or 'ml' in filename_lower:
        return 'machine_learning'
    else:
        return 'general'

def create_chunks_with_qa(qa_pairs: List[Dict], source: str, category: str) -> List[Dict]:
    """Create chunks from Q&A pairs"""
    chunks = []
    
    # Check if we need to generate answers
    needs_generation = any(not qa['has_answer'] or not qa['answer'] for qa in qa_pairs)
    
    # Use progress bar if generating answers
    iterator = tqdm(qa_pairs, desc=f"Processing {source}", unit="Q&A") if needs_generation else qa_pairs
    
    for idx, qa in enumerate(iterator):
        question = qa['question']
        answer = qa['answer']
        
        # Generate answer if missing
        if not qa['has_answer'] or not answer:
            if not needs_generation:  # If we didn't create progress bar above
                print(f" Generating answer for: {question[:60]}...")
            answer = generate_answer(question, category)
        
        # Create combined text for embedding
        combined_text = f"Question: {question}\n\nAnswer: {answer}"
        
        chunks.append({
            'text': combined_text,
            'question': question,
            'answer': answer,
            'source': source,
            'category': category,
            'chunk_id': idx
        })
    
    return chunks

def upsert(qdrant: QdrantClient, collection: str, payloads: List[Dict], vectors: List[List[float]], batch_size: int = 512):
    """Upsert to Qdrant in batches"""
    assert len(payloads) == len(vectors)
    points: List[PointStruct] = []
    
    # Progress bar for upserting
    with tqdm(total=len(payloads), desc="Upserting to Qdrant", unit="point") as pbar:
        for i, (p, v) in enumerate(zip(payloads, vectors)):
            # Generate unique ID from content hash
            text_hash = hashlib.md5(p["text"].encode()).hexdigest()
            points.append(PointStruct(id=text_hash, vector=v, payload=p))
            
            if len(points) >= batch_size:
                qdrant.upsert(collection_name=collection, points=points)
                pbar.update(len(points))
                points = []
        
        if points:
            qdrant.upsert(collection_name=collection, points=points)
            pbar.update(len(points))

def ensure_collection(qdrant: QdrantClient, collection: str, dim: int):
    existing = [c.name for c in qdrant.get_collections().collections]
    if collection not in existing:
        print(f"Creating collection '{collection}'...")
        qdrant.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        print(f"Collection created")
    else:
        print(f"Using existing collection '{collection}'")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw")
    parser.add_argument("--collection", type=str, default=os.getenv("QDRANT_COLLECTION", "interview_chunks"))
    args = parser.parse_args()

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")

    model_cfg = get_embedding_model()
    
    print("🔌 Connecting to Qdrant...")
    qdrant = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333"))
    )

    ensure_collection(qdrant, args.collection, model_cfg["dim"])

    all_chunks = []

    # Process all CSV files
    csv_files = glob.glob(os.path.join(args.input, "*.csv"))
    
    if not csv_files:
        print(f"\nNo CSV files found in {args.input}")
        return
    
    print(f"\n Found {len(csv_files)} CSV file(s)\n")
    
    for file_path in csv_files:
        path = Path(file_path)
        print(f"Processing: {path.name}")
        
        # Determine category from filename
        category = determine_category_from_filename(path.name)
        
        qa_pairs = read_csv(path)
        if qa_pairs:
            # Check if answers are already present
            has_answers = qa_pairs[0]['has_answer'] if qa_pairs else False
            status = "Has answers" if has_answers else "Generating answers"
            print(f"   Category: {category} | {status}")
            
            chunks = create_chunks_with_qa(qa_pairs, path.name, category)
            all_chunks.extend(chunks)
            print(f"   Processed {len(chunks)} Q&A pairs\n")

    if not all_chunks:
        print("\n No data to ingest. Exiting.")
        return

    # Extract texts for embedding
    texts = [chunk['text'] for chunk in all_chunks]
    
    print(f"\nTotal Q&A pairs to embed: {len(texts)}")
    vectors = embed_texts(model_cfg, texts)

    print(f"\nUploading to Qdrant collection '{args.collection}'...")
    upsert(qdrant, args.collection, all_chunks, vectors)
    
    # Summary by category
    categories = {}
    for chunk in all_chunks:
        cat = chunk['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"\n" + "="*50)
    print(f"Successfully ingested {len(all_chunks)} Q&A pairs!")
    print(f"   Collection: {args.collection}")
    print(f"\nBreakdown by category:")
    for cat, count in sorted(categories.items()):
        print(f"   • {cat.replace('_', ' ').title()}: {count} questions")
    print("="*50)

if __name__ == "__main__":
    main()