import pandas as pd
import nltk
import json
import os
from collections import Counter
from nltk.util import ngrams

#download the necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

from nltk.corpus import stopwords

#load the sample data
print("Loading data...")
df = pd.read_csv('data/processed/listing_sample.csv')
print(f"Loaded {len(df)} records from data/processed/listing_sample.csv")

#extract all the remarks context
all_text = ' '.join(df['remarks'].dropna().str.lower())
tokens = nltk.word_tokenize(all_text)

#filter out stop words and punctuation
stop_words = set(stopwords.words('english'))
tokens_clean = [t for t in tokens if t.isalpha() and t not in stop_words and len(t) > 2]

#extract unigrams, bigrams, and trigrams
unigrams = Counter(tokens_clean)
bigram_list = list(ngrams(tokens_clean, 2))
trigram_list = list(ngrams(tokens_clean, 3))
bigrams = Counter(bigram_list)
trigrams = Counter(trigram_list)

#predefine 8 categories based on common real estate themes
categories = {
    "property_type": [
        "house", "condo", "townhouse", "apartment", "villa", "cottage",
        "duplex", "studio", "loft", "mansion", "bungalow", "ranch"
    ],
    "bedrooms_bathrooms": [
        "bedroom", "bathroom", "bath", "bed", "master", "suite",
        "ensuite", "half bath", "full bath", "master bedroom", "master suite"
    ],
    "amenities": [
        "pool", "garage", "fireplace", "patio", "deck", "balcony",
        "jacuzzi", "spa", "gym", "basement", "attic", "storage",
        "laundry", "parking", "driveway", "backyard", "garden"
    ],
    "condition": [
        "updated", "renovated", "remodeled", "new", "modern", "upgraded",
        "restored", "pristine", "move-in", "turnkey", "original", "vintage"
    ],
    "location": [
        "neighborhood", "school", "district", "downtown", "suburb",
        "waterfront", "mountain", "view", "corner", "cul-de-sac",
        "gated", "community", "city", "county"
    ],
    "features": [
        "hardwood", "granite", "stainless", "vaulted", "ceiling",
        "open floor", "floor plan", "natural light", "crown molding",
        "walk-in closet", "chef kitchen", "island", "quartz"
    ],
    "financing": [
        "price", "reduced", "motivated", "seller", "financing",
        "negotiable", "equity", "investment", "opportunity", "value"
    ],
    "outdoor": [
        "yard", "lawn", "landscape", "fence", "sprinkler", "outdoor",
        "covered patio", "front yard", "back yard", "pool area"
    ]
}

#build the taxonomy list
terms = []
term_id = 1

# Add Seed Words from Predefined Categories
for category, words in categories.items():
    for word in words:
        terms.append({
            "id": f"T{term_id:04d}",
            "term": word,
            "category": category,
            "source": "seed"
        })
        term_id += 1

#Extract high-frequency bigrams from the data and add them to the taxonomy.
print("extracting high-frequency bigrams...")
existing_terms = set(t['term'] for t in terms)

for bigram, count in bigrams.most_common(300):
    phrase = ' '.join(bigram)
    if count >= 5 and phrase not in existing_terms:
        terms.append({
            "id": f"T{term_id:04d}",
            "term": phrase,
            "category": "extracted",
            "frequency": count,
            "source": "ngram"
        })
        term_id += 1
        existing_terms.add(phrase)

#Supplementing High-Frequency Words from Data
for word, count in unigrams.most_common(200):
    if count >= 10 and word not in existing_terms and len(word) > 3:
        terms.append({
            "id": f"T{term_id:04d}",
            "term": word,
            "category": "extracted",
            "frequency": count,
            "source": "unigram"
        })
        term_id += 1
        existing_terms.add(word)

    if len(terms) >= 300:
        break

# build final taxonomy
taxonomy = {
    "version": "1.0",
    "total_terms": len(terms),
    "categories": list(categories.keys()) + ["extracted"],
    "terms": terms
}

# save
os.makedirs('data/processed', exist_ok=True)
with open('data/processed/taxonomy.json', 'w') as f:
    json.dump(taxonomy, f, indent=2)

print(f"\ncompleted! generated {len(terms)} terms.")
print(f"has been saved to data/processed/taxonomy.json")
print(f"\nthe number of terms in each category:")
from collections import defaultdict
cat_count = defaultdict(int)
for t in terms:
    cat_count[t['category']] += 1
for cat, count in cat_count.items():
    print(f"  {cat}: {count}")