from gensim.models import Word2Vec 
 
# Sample document corpus 
corpus = [ 
    ["machine", "learning", "is", "fun"], 
    ["deep", "learning", "uses", "neural", "networks"], 
    ["deep", "learning"] 
] 
 
# Train the Word2Vec model 
model = Word2Vec( 
    sentences=corpus, 
    vector_size=50,   # Size of word embeddings 
    window=2,         # Context window size 
    min_count=1,      # Include all words 
    workers=1 
) 
 
# Print the embedding vector for a word 
print("Embedding for 'learning':") 
print(model.wv["learning"]) 
 
# Find similar words 
print("\nWords similar to 'learning':") 
print(model.wv.most_similar("learning")) 
 
Output: 
Embedding for 'learning': 
[-1.0724545e-03  4.7286032e-04  1.0206699e-02  1.8018546e-02 
 -1.8605899e-02 -1.4233618e-02  1.2917743e-02  1.7945977e-02 
 -1.0030856e-02 -7.5267460e-03  1.4761009e-02 -3.0669451e-03 
 -9.0732286e-03  1.3108101e-02 -9.7203208e-03 -3.6320353e-03 
  5.7531595e-03  1.9837476e-03 -1.6570430e-02 -1.8897638e-02 
  1.4623532e-02  1.0140524e-02  1.3515387e-02  1.5257311e-03 
  1.2701779e-02 -6.8107317e-03 -1.8928051e-03  1.1537147e-02 
 -1.5043277e-02 -7.8722099e-03 -1.5023164e-02 -1.8600845e-03 
  1.9076237e-02 -1.4638334e-02 -4.6675396e-03 -3.8754845e-03 
  1.6154870e-02 -1.1861792e-02  9.0322494e-05 -9.5074698e-03 
 -1.9207101e-02  1.0014586e-02 -1.7519174e-02 -8.7836506e-03 
 -7.0199967e-05 -5.9236528e-04 -1.5322480e-02  1.9229483e-02 
  9.9641131e-03  1.8466286e-02] 
 
Words similar to 'learning': 
[('machine', 0.13204392790794373), ('networks', 0.1267007440328598), ('deep', 
0.04237298294901848), ('fun', 0.012442131526768208), ('neural', -0.01447527576237917), ('is', 
0.05974649265408516), ('uses', -0.11821283400058746)]