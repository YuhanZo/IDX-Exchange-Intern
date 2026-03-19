import sys
import os
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from text_cleaning import TextCleaner

cleaner = TextCleaner()


# ── Abbreviation dictionary ───────────────────────────────────────────────────

def test_abbrev_map_has_30_plus_entries():
    assert len(cleaner.abbrev_map) >= 30, \
        f"Expected 30+ abbreviations, got {len(cleaner.abbrev_map)}"


# ── normalize_prices ──────────────────────────────────────────────────────────

def test_price_k_lowercase():
    assert '450000' in cleaner.normalize_prices('priced at 450k')

def test_price_k_uppercase():
    assert '800000' in cleaner.normalize_prices('listed at 800K')

def test_price_m_lowercase():
    assert '1200000' in cleaner.normalize_prices('$1.2m home')

def test_price_m_uppercase():
    assert '2500000' in cleaner.normalize_prices('sold for 2.5M')

def test_price_with_dollar_sign():
    assert '500000' in cleaner.normalize_prices('asking $500k')

def test_price_integer_m():
    assert '3000000' in cleaner.normalize_prices('valued at 3m')

def test_price_plain_number_unchanged():
    result = cleaner.normalize_prices('3 bedroom home')
    assert result == '3 bedroom home'

def test_price_no_false_match_km():
    # 'km' (kilometers) should not be converted
    result = cleaner.normalize_prices('5km from the beach')
    assert '5000' not in result


# ── normalize_measurements ────────────────────────────────────────────────────

def test_measurement_comma_number():
    result = cleaner.normalize_measurements('2,000 sqft home')
    assert '2000' in result

def test_measurement_sq_ft_dot():
    result = cleaner.normalize_measurements('1500 sq.ft. lot')
    assert 'square feet' in result

def test_measurement_sq_ft_no_dot():
    result = cleaner.normalize_measurements('1800 sq ft living area')
    assert 'square feet' in result

def test_measurement_comma_preserved_in_non_numbers():
    result = cleaner.normalize_measurements('kitchen, dining, and living room')
    assert 'kitchen, dining, and living room' == result


# ── expand_abbreviations ──────────────────────────────────────────────────────

def test_abbrev_br_bedroom():
    assert 'bedroom' in cleaner.expand_abbreviations('3 br home')

def test_abbrev_bd_bedroom():
    assert 'bedroom' in cleaner.expand_abbreviations('2 bd available')

def test_abbrev_ba_bathroom():
    assert 'bathroom' in cleaner.expand_abbreviations('2 ba')

def test_abbrev_sqft():
    assert 'square feet' in cleaner.expand_abbreviations('1500 sqft')

def test_abbrev_sf():
    assert 'square feet' in cleaner.expand_abbreviations('1200 sf')

def test_abbrev_w_slash():
    assert 'with' in cleaner.expand_abbreviations('home w/ pool')

def test_abbrev_w_slash_o():
    assert 'without' in cleaner.expand_abbreviations('unit w/o parking')

def test_abbrev_ac():
    assert 'air conditioning' in cleaner.expand_abbreviations('central ac included')

def test_abbrev_hoa():
    assert 'homeowners association' in cleaner.expand_abbreviations('low hoa fees')

def test_abbrev_mbr():
    assert 'master bedroom' in cleaner.expand_abbreviations('spacious mbr suite')

def test_abbrev_case_insensitive():
    assert 'square feet' in cleaner.expand_abbreviations('2000 SQFT')

def test_abbrev_no_false_match_inside_word():
    # 'ba' should not match inside 'basketball'
    result = cleaner.expand_abbreviations('basketball court nearby')
    assert 'basketball' in result


# ── normalize_unicode ─────────────────────────────────────────────────────────

def test_unicode_replacement_char():
    result = cleaner.normalize_unicode('San Diego\ufffd s best')
    assert '\ufffd' not in result

def test_unicode_right_single_quote():
    result = cleaner.normalize_unicode('seller\u2019s choice')
    assert '\u2019' not in result
    assert "'" in result

def test_unicode_curly_double_quotes():
    result = cleaner.normalize_unicode('\u201chome sweet home\u201d')
    assert '\u201c' not in result
    assert '\u201d' not in result

def test_unicode_em_dash():
    result = cleaner.normalize_unicode('move-in ready\u2014no work needed')
    assert '\u2014' not in result
    assert '-' in result

def test_unicode_en_dash():
    result = cleaner.normalize_unicode('3\u20134 bedrooms')
    assert '\u2013' not in result

def test_unicode_non_string_passthrough():
    assert cleaner.normalize_unicode(None) is None


# ── remove_html ───────────────────────────────────────────────────────────────

def test_html_basic_tag():
    result = cleaner.remove_html('<p>Great home</p>')
    assert '<p>' not in result
    assert 'Great home' in result

def test_html_tag_with_attributes():
    result = cleaner.remove_html('<a href="url">click here</a>')
    assert '<a' not in result
    assert 'click here' in result

def test_html_entity_amp():
    assert '&' in cleaner.remove_html('beds &amp; baths')

def test_html_entity_nbsp():
    result = cleaner.remove_html('price&nbsp;reduced')
    assert '&nbsp;' not in result

def test_html_no_html_unchanged():
    text = 'Beautiful 3 bedroom home'
    assert cleaner.remove_html(text) == text


# ── normalize_whitespace ──────────────────────────────────────────────────────

def test_whitespace_double_space():
    result = cleaner.normalize_whitespace('beautiful  home')
    assert '  ' not in result

def test_whitespace_leading():
    assert cleaner.normalize_whitespace('   hello') == 'hello'

def test_whitespace_trailing():
    assert cleaner.normalize_whitespace('hello   ') == 'hello'

def test_whitespace_tab():
    result = cleaner.normalize_whitespace('open\tfloor plan')
    assert '\t' not in result


# ── clean_text (integration) ──────────────────────────────────────────────────

def test_clean_text_price_and_abbrev():
    result = cleaner.clean_text('3 br home priced at 500k w/ pool')
    assert 'bedroom' in result
    assert '500000' in result
    assert 'with' in result

def test_clean_text_unicode_and_measurements():
    result = cleaner.clean_text('San Diego\ufffd s finest 2,000 sqft home')
    assert '\ufffd' not in result
    assert '2000' in result
    assert 'square feet' in result

def test_clean_text_empty_string():
    assert cleaner.clean_text('') == ''

def test_clean_text_strips_whitespace():
    result = cleaner.clean_text('  lovely home  ')
    assert result == result.strip()


# ── profile_column ────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'remarks': [
            'Beautiful 3 br home w/ pool priced at 450k',
            'Spacious 2 ba condo, 1200 sqft, low hoa fees',
            None,
            '<p>Updated kitchen</p> with granite counters',
            'San Diego\ufffd s finest property near the beach',
        ]
    })

def test_profile_has_required_keys(sample_df):
    profile = cleaner.profile_column(sample_df, 'remarks')
    for key in ('null_rate', 'avg_length', 'common_terms', 'price_mentions', 'has_html', 'common_abbreviations'):
        assert key in profile, f"Missing key: {key}"

def test_profile_null_rate(sample_df):
    profile = cleaner.profile_column(sample_df, 'remarks')
    assert abs(profile['null_rate'] - 0.2) < 0.01

def test_profile_html_detection(sample_df):
    profile = cleaner.profile_column(sample_df, 'remarks')
    assert profile['has_html'] >= 1

def test_profile_price_mentions(sample_df):
    df = pd.DataFrame({'remarks': ['priced at $500k', 'no price here', '$1.2m luxury home']})
    profile = cleaner.profile_column(df, 'remarks')
    assert profile['price_mentions'] >= 1

def test_profile_abbreviations_detected(sample_df):
    profile = cleaner.profile_column(sample_df, 'remarks')
    abbrevs = profile['common_abbreviations']
    assert isinstance(abbrevs, dict)
    assert len(abbrevs) > 0
