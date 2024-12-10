import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class QuickRAG:
    def __init__(self, data_dir='data'):
        self.documents = []
        self.filenames = []
        
        # Load documents
        for filename in os.listdir(data_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(data_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.documents.append(f.read())
                    self.filenames.append(filename)
        
        # Vectorize documents
        self.vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
    
    def query(self, question, top_k=2):
        # Transform query to vector
        query_vector = self.vectorizer.transform([question])
        
        # Compute similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        # Get top k most similar documents
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Prepare results
        results = []
        for idx in top_indices:
            results.append({
                'document': self.filenames[idx],
                'similarity': similarities[idx],
                'content': self.documents[idx][:500] + '...' if len(self.documents[idx]) > 500 else self.documents[idx]
            })
        
        return results

def main():
    rag = QuickRAG()
    print("RAG System initialized with documents:")
    for filename in rag.filenames:
        print(f" - {filename}")
    
    while True:
        try:
            query = input("\nEnter your query (or 'exit' to quit): ").strip()
            if query.lower() in ['exit', 'quit', 'q']:
                break
                
            results = rag.query(query)
            print("\nResults:")
            for result in results:
                print(f"\nDocument: {result['document']}")
                print(f"Relevance: {result['similarity']:.2f}")
                print(f"Content: {result['content']}\n")
                print("-" * 80)
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
