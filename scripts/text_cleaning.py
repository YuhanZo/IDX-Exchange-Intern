import re
import pandas as pd
from collections import Counter
import nltk


class TextCleaner:
    def __init__(self):
        # 30+ real estate abbreviation mappings (longer phrases first to avoid partial matches)
        self.abbrev_map = {
            # Multi-word first
            'grt rm':   'great room',
            'fam rm':   'family room',
            'din rm':   'dining room',
            'sq ft':    'square feet',
            'w/o':      'without',
            'a/c':      'air conditioning',
            'w/':       'with',
            # Rooms
            'bdrm':     'bedroom',
            'bdr':      'bedroom',
            'mbr':      'master bedroom',
            'mba':      'master bathroom',
            'mstr':     'master',
            'br':       'bedroom',
            'bd':       'bedroom',
            'ba':       'bathroom',
            'bth':      'bathroom',
            'lr':       'living room',
            'dr':       'dining room',
            'fr':       'family room',
            'kit':      'kitchen',
            # Measurements
            'sqft':     'square feet',
            'sf':       'square feet',
            # Property features
            'gar':      'garage',
            'pkg':      'parking',
            'prkg':     'parking',
            'bkyd':     'backyard',
            'clst':     'closet',
            'flr':      'floor',
            'frnt':     'front',
            # Status / condition
            'rmd':      'remodeled',
            'reno':     'renovated',
            'upd':      'updated',
            'blt':      'built',
            # Utilities / fees
            'hoa':      'homeowners association',
            'ac':       'air conditioning',
            'util':     'utilities',
            'laund':    'laundry',
            'incl':     'included',
            'incls':    'includes',
            # Time / distance
            'approx':   'approximately',
            'mo':       'month',
            'yr':       'year',
            'min':      'minutes',
            'mi':       'miles',
            'blk':      'block',
        }

    # ── Core cleaning methods ─────────────────────────────────────────────────

    def normalize_unicode(self, text: str) -> str:
        """Replace common mojibake and non-ASCII characters with ASCII equivalents."""
        if not isinstance(text, str):
            return text
        replacements = {
            '\u2019': "'",   # right single quotation mark
            '\u2018': "'",   # left single quotation mark
            '\u201c': '"',   # left double quotation mark
            '\u201d': '"',   # right double quotation mark
            '\u2013': '-',   # en dash
            '\u2014': '-',   # em dash
            '\u2026': '...',  # ellipsis
            '\u00e2\u0080\u0099': "'",  # UTF-8 mojibake for right quote
            '\ufffd': "'",   # replacement character (common in MLS data)
        }
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        # Remove any remaining non-ASCII characters
        text = text.encode('ascii', errors='ignore').decode('ascii')
        return text

    def remove_html(self, text: str) -> str:
        """Strip HTML tags and decode common HTML entities."""
        if not isinstance(text, str):
            return text
        text = re.sub(r'<[^>]+>', ' ', text)
        html_entities = {
            '&amp;':  '&',
            '&lt;':   '<',
            '&gt;':   '>',
            '&nbsp;': ' ',
            '&quot;': '"',
            '&#39;':  "'",
        }
        for entity, char in html_entities.items():
            text = text.replace(entity, char)
        return text

    def normalize_prices(self, text: str) -> str:
        """Convert shorthand prices: 450k → 450000, 1.2m → 1200000."""
        if not isinstance(text, str):
            return text
        # Must not be followed by a letter (e.g. avoid matching 'km' in 'kilometers')
        text = re.sub(
            r'\b(\d+\.?\d*)[kK]\b',
            lambda m: str(int(float(m.group(1)) * 1_000)),
            text
        )
        text = re.sub(
            r'\b(\d+\.?\d*)[mM]\b(?!\w)',
            lambda m: str(int(float(m.group(1)) * 1_000_000)),
            text
        )
        return text

    def normalize_measurements(self, text: str) -> str:
        """Standardize square footage expressions."""
        if not isinstance(text, str):
            return text
        # Remove commas in numbers (e.g. 2,000 → 2000)
        text = re.sub(r'(\d),(\d{3})', r'\1\2', text)
        # Normalize sq.ft. / sq ft / sqft / sf patterns
        text = re.sub(r'(?i)\bsq\.ft\.?\b', 'square feet', text)
        text = re.sub(r'(?i)\bsq\s+ft\b', 'square feet', text)
        return text

    def expand_abbreviations(self, text: str) -> str:
        """Expand real estate abbreviations using word-boundary matching."""
        if not isinstance(text, str):
            return text
        # Sort by length descending so longer phrases match first
        sorted_abbrevs = sorted(self.abbrev_map.items(), key=lambda x: len(x[0]), reverse=True)
        for abbrev, expansion in sorted_abbrevs:
            if '/' in abbrev:
                # Slash-based abbreviations (w/, w/o, a/c) — no \b on slash side
                pattern = r'(?i)(?<!\w)' + re.escape(abbrev)
            else:
                pattern = r'(?i)\b' + re.escape(abbrev) + r'\b'
            text = re.sub(pattern, expansion, text)
        return text

    def normalize_whitespace(self, text: str) -> str:
        """Collapse multiple spaces/tabs and strip leading/trailing whitespace."""
        if not isinstance(text, str):
            return text
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()

    def clean_text(self, text: str) -> str:
        """Full cleaning pipeline."""
        text = self.normalize_unicode(text)
        text = self.remove_html(text)
        text = self.normalize_prices(text)
        text = self.normalize_measurements(text)
        text = self.expand_abbreviations(text)
        text = self.normalize_whitespace(text)
        return text

    # ── Data profiling ────────────────────────────────────────────────────────

    def profile_column(self, df: pd.DataFrame, column_name: str) -> dict:
        """Analyze a text column to guide cleaning strategy."""
        col = df[column_name]
        return {
            'null_rate':             col.isnull().mean(),
            'avg_length':            col.str.len().mean(),
            'common_terms':          self._extract_top_ngrams(col),
            'price_mentions':        int(col.str.contains(r'\$\d', na=False).sum()),
            'has_html':              int(col.str.contains(r'<[a-zA-Z]', na=False).sum()),
            'common_abbreviations':  self._detect_abbreviations(col),
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_top_ngrams(self, series: pd.Series, n: int = 2, top_k: int = 10) -> list:
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
        all_text = ' '.join(series.dropna().str.lower())
        tokens = nltk.word_tokenize(all_text)
        tokens = [t for t in tokens if t.isalpha()]
        grams = Counter(nltk.ngrams(tokens, n))
        return [' '.join(g) for g, _ in grams.most_common(top_k)]

    def _detect_abbreviations(self, series: pd.Series) -> dict:
        """Count how often each known abbreviation appears across the column."""
        results = {}
        combined = ' ' + ' '.join(series.dropna().str.lower()) + ' '
        for abbrev in self.abbrev_map:
            if '/' in abbrev:
                pattern = r'(?<!\w)' + re.escape(abbrev)
            else:
                pattern = r'\b' + re.escape(abbrev) + r'\b'
            results[abbrev] = len(re.findall(pattern, combined))
        return {k: v for k, v in sorted(results.items(), key=lambda x: -x[1]) if v > 0}


# ── Script entry point ────────────────────────────────────────────────────────

if __name__ == '__main__':
    df = pd.read_csv('data/processed/listing_sample.csv')
    cleaner = TextCleaner()

    # Profile before cleaning
    print("=== Data Profile (before cleaning) ===")
    profile = cleaner.profile_column(df, 'remarks')
    print(f"Null rate:         {profile['null_rate']:.1%}")
    print(f"Avg length:        {profile['avg_length']:.0f} chars")
    print(f"Price mentions:    {profile['price_mentions']}")
    print(f"HTML tags found:   {profile['has_html']}")
    print(f"Common abbrevs:    {profile['common_abbreviations']}")

    # Clean and save
    print("\nCleaning remarks...")
    df['remarks_clean'] = df['remarks'].apply(cleaner.clean_text)

    # Before / after examples
    print("\n=== Before / After Examples ===")
    mask = df['remarks'] != df['remarks_clean']
    for _, row in df[mask].head(5).iterrows():
        print(f"  BEFORE: {row['remarks'][:120]}")
        print(f"  AFTER:  {row['remarks_clean'][:120]}")
        print()

    df.to_csv('data/processed/listing_sample_clean.csv', index=False)
    print("Saved to data/processed/listing_sample_clean.csv")
