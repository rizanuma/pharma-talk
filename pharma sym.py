import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Create a TF-IDF vectorizer.
tfidf_vectorizer = TfidfVectorizer()

df = pd.read_csv("C:\\Users\\putariza\\Documents\\Documents\\ML\\nlp lab\\pharma-talk\\medicine_dataset.csv")

# Tokenization, stop word removal, and stemming.
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    # Convert text to lowercase, tokenize, and remove stopwords and spaces.
    tokens = word_tokenize(text.lower())
    tokens = [stemmer.stem(token) for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(tokens)

# Preprocess the 'sideEffect0' and 'use0' columns.
df['Drugs'] = df['Drugs'].apply(preprocess_text)
df['sideEffect0'] = df['sideEffect0'].apply(preprocess_text)
df['use0'] = df['use0'].apply(preprocess_text)

# Create a combined text column for the dataset.
df['combined_text'] = df['Drugs'] + ' ' + df['sideEffect0'] + ' ' + df['use0']

# Create a TF-IDF matrix for the combined text.
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_text'])

def search_drugs(query):
    query = preprocess_text(query)
    query_vector = tfidf_vectorizer.transform([query])  # Use transform here, not fit_transform.

    # Calculate cosine similarity between the query and the combined text.
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)

    # Get the indices of drugs with the highest similarity scores.
    top_indices = similarity_scores.argsort()[0][::-1]

    # Display the top 5 matching drugs.
    top_matches = df.iloc[top_indices[:5]]
    return top_matches

# Use the search function.
while True:
    user_query = input("Enter query:").strip().lower()
    
    if user_query == 'exit':
        break
    
    results = search_drugs(user_query)
    
    if not results.empty:
        print("\nDetails found related to the query:\n")
        print(results[['Drugs', 'sideEffect0', 'use0']])
    else:
        print("\nNo matching drugs found.\n")

# Exit the program.
print("Goodbye!")

