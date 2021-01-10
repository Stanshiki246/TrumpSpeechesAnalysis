'''
Analysis on Trump Speeches - Group 1
Stanley Tantysco - 2201814670
Girindra Ado - 2201843506
Making Spacy Magic library
'''

class SpacyMagic(object):
    _spacys = {}

    @classmethod
    def get(cls, lang):
        if lang not in cls._spacys:
            import spacy
            cls._spacys[lang] = spacy.load(lang, disable=['tagger', 'ner','pos'])
        return cls._spacys[lang]

def run_spacy(text):
    nlp = SpacyMagic.get('en_core_web_sm')
    doc = nlp(text)
    return doc

def clean_text(inp):
    spacy_text = run_spacy(inp)
    out_str= ' '.join ([token.lemma_ for token in spacy_text if token.is_stop != True and token.is_punct != True\
                        and token.is_alpha ==True])
    return out_str
