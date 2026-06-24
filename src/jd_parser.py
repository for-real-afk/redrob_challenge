"""
Structured Job Description constants and match criteria for the fixed Redrob Hackathon JD.
"""

# Experience Band Fit
TARGET_YOE_MIN = 5.0
TARGET_YOE_MAX = 9.0
TARGET_YOE_CENTER = 7.0

# Required technologies/keywords from JD
REQUIRED_SKILLS = {
    # Embeddings / Dense Retrieval
    "sentence-transformers", "embeddings", "vector search", "dense retrieval", "retrieval", "hybrid search",
    # Vector databases
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch", "faiss",
    # Core Languages and tools
    "python",
    # Evaluation frameworks
    "ndcg", "mrr", "map", "a/b testing", "evaluation infrastructure", "ranking evaluation"
}

# Preferred nice-to-have technologies from JD
PREFERRED_SKILLS = {
    # LLM Fine-tuning
    "lora", "qlora", "peft", "fine-tuning", "llm fine-tuning",
    # Learning to rank
    "xgboost", "lightgbm", "xgbranker", "learning to rank", "neural ranking",
    # Domain specific
    "hr-tech", "recruiting tech", "marketplace", "marketplace products",
    # Infrastructure / scalability
    "distributed systems", "inference optimization", "large-scale inference", "mlops", "docker", "kubernetes",
    # Community
    "open-source contributions", "open source"
}

# Location preferences
PREFERRED_LOCATIONS = {"pune", "noida", "hyderabad", "mumbai", "delhi ncr", "ncr", "delhi", "gurgaon", "bangalore", "bengaluru", "chennai"}

# Consulting / IT services firms to be penalized if entire career is spent there
IT_SERVICES_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini", "mindtree", 
    "tech mahindra", "hcl", "mphasis", "genpact", "l&t", "lnt", "tata consultancy"
}

# Product companies observed in the dataset
PRODUCT_COMPANIES = {
    "swiggy", "razorpay", "cred", "zomato", "flipkart", "meesho", "nykaa", "inmobi", 
    "byju's", "byjus", "policybazaar", "ola", "zoho", "freshworks", "phonepe", "paytm", 
    "unacademy", "dream11", "glance", "krutrim", "saarthi.ai", "saarthi", "sarvam ai", 
    "observe.ai", "wysa", "aganitha", "niramai", "verloop.io", "verloop", "yellow.ai", 
    "haptik", "mad street den", "locobuzz", "google", "microsoft", "meta", "amazon", 
    "netflix", "uber", "adobe", "salesforce", "linkedin", "hooli", "pied piper", 
    "stark industries", "wayne enterprises", "globex inc", "acme corp", "initech", "dunder mifflin"
}

# Domain keywords for exposure checks
NLP_IR_KEYWORDS = {
    "nlp", "natural language", "text", "search", "retrieval", "ranking", "recommendation",
    "embeddings", "transformers", "llm", "llms", "rag", "bert", "gpt", "tokenization",
    "parsing", "information retrieval", "bm25", "semantic search"
}

CV_SPEECH_ROBOTICS_KEYWORDS = {
    "computer vision", "vision", "cv", "image", "speech", "audio", "robotics", "ros",
    "yolo", "opencv", "image segmentation", "object detection", "speech recognition", 
    "tts", "whisper", "asr", "diffusion", "gan", "cnn", "image classification"
}
