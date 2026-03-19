import json
import pandas as pd


TAXONOMY_PATH = 'data/processed/taxonomy.json'
SAMPLE_DATA_PATH = 'data/processed/listing_sample.csv'
EXPECTED_CATEGORIES = [
    "property_type", "bedrooms_bathrooms", "amenities",
    "condition", "location", "features", "financing", "outdoor"
]


# ── Taxonomy Tests ────────────────────────────────────────────────────────────

def test_taxonomy_loaded():
    with open(TAXONOMY_PATH) as f:
        tax = json.load(f)
    assert len(tax['terms']) >= 200, f"Expected 200+ terms, got {len(tax['terms'])}"
    assert all('id' in t and 'term' in t for t in tax['terms']), \
        "Every term must have 'id' and 'term' fields"


def test_taxonomy_has_required_categories():
    with open(TAXONOMY_PATH) as f:
        tax = json.load(f)
    for cat in EXPECTED_CATEGORIES:
        assert cat in tax['categories'], f"Missing category: {cat}"


def test_taxonomy_term_ids_unique():
    with open(TAXONOMY_PATH) as f:
        tax = json.load(f)
    ids = [t['id'] for t in tax['terms']]
    assert len(ids) == len(set(ids)), "Term IDs must be unique"


def test_taxonomy_terms_unique():
    with open(TAXONOMY_PATH) as f:
        tax = json.load(f)
    terms = [t['term'] for t in tax['terms']]
    assert len(terms) == len(set(terms)), "Term strings must be unique"


def test_taxonomy_coverage():
    """At least 30% of remarks contain at least one taxonomy term."""
    with open(TAXONOMY_PATH) as f:
        tax = json.load(f)
    df = pd.read_csv(SAMPLE_DATA_PATH)
    term_set = set(t['term'].lower() for t in tax['terms'])

    def has_term(remark):
        if pd.isna(remark):
            return False
        remark_lower = remark.lower()
        return any(term in remark_lower for term in term_set)

    coverage = df['remarks'].apply(has_term).mean()
    assert coverage >= 0.30, f"Taxonomy coverage {coverage:.1%} is below 30%"


# ── Sample Data Tests ─────────────────────────────────────────────────────────

def test_sample_data_quality():
    df = pd.read_csv(SAMPLE_DATA_PATH)
    assert len(df) >= 500, f"Expected 500+ rows, got {len(df)}"
    assert df['remarks'].str.len().min() > 50, "All remarks must be longer than 50 chars"


def test_sample_data_required_columns():
    df = pd.read_csv(SAMPLE_DATA_PATH)
    required = {'L_ListingID', 'L_Address', 'L_City', 'beds', 'baths', 'price', 'remarks'}
    assert required.issubset(set(df.columns)), \
        f"Missing columns: {required - set(df.columns)}"


def test_sample_data_no_null_remarks():
    df = pd.read_csv(SAMPLE_DATA_PATH)
    null_count = df['remarks'].isnull().sum()
    assert null_count == 0, f"Found {null_count} null remarks"


def test_sample_data_listing_ids_unique():
    df = pd.read_csv(SAMPLE_DATA_PATH)
    assert df['L_ListingID'].nunique() == len(df), "Listing IDs must be unique"
