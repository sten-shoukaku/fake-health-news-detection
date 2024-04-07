import nlpaug.augmenter.word as naw
import pandas as pd
import random


class EDA:

    def __init__(self, prob: float = 0.1):
        self.prob = prob
        # https://github.com/jasonwei20/eda_nlp/blob/master/code/eda.py#L9-L29
        self.stop_words = [
            'i',
            'me',
            'my',
            'myself',
            'we',
            'our',
            'ours',
            'ourselves',
            'you',
            'your',
            'yours',
            'yourself',
            'yourselves',
            'he',
            'him',
            'his',
            'himself',
            'she',
            'her',
            'hers',
            'herself',
            'it',
            'its',
            'itself',
            'they',
            'them',
            'their',
            'theirs',
            'themselves',
            'what',
            'which',
            'who',
            'whom',
            'this',
            'that',
            'these',
            'those',
            'am',
            'is',
            'are',
            'was',
            'were',
            'be',
            'been',
            'being',
            'have',
            'has',
            'had',
            'having',
            'do',
            'does',
            'did',
            'doing',
            'a',
            'an',
            'the',
            'and',
            'but',
            'if',
            'or',
            'because',
            'as',
            'until',
            'while',
            'of',
            'at',
            'by',
            'for',
            'with',
            'about',
            'against',
            'between',
            'into',
            'through',
            'during',
            'before',
            'after',
            'above',
            'below',
            'to',
            'from',
            'up',
            'down',
            'in',
            'out',
            'on',
            'off',
            'over',
            'under',
            'again',
            'further',
            'then',
            'once',
            'here',
            'there',
            'when',
            'where',
            'why',
            'how',
            'all',
            'any',
            'both',
            'each',
            'few',
            'more',
            'most',
            'other',
            'some',
            'such',
            'no',
            'nor',
            'not',
            'only',
            'own',
            'same',
            'so',
            'than',
            'too',
            'very',
            's',
            't',
            'can',
            'will',
            'just',
            'don',
            'should',
            'now',
            '',
        ]

    def __call__(self, pd: pd.DataFrame) -> pd.DataFrame:
        return self.augment(pd)

    def __synonym(self, text: str, n_min: int = 1, n_max: int = None) -> str:
        f = naw.SynonymAug(
            aug_src='wordnet',
            aug_p=1,
            aug_min=n_min,
            aug_max=n_max,
            stopwords=self.stop_words,
        )
        return ' '.join(f.augment(text.split()))

    def __swap(self, text: str, n_min: int = 1, n_max: int = None) -> str:
        f = naw.RandomWordAug(action='swap', aug_p=1, aug_min=n_min, aug_max=n_max)
        return ' '.join(f.augment(text.split()))

    def __delete(self, text: str, n_min: int = 1, n_max: int = None) -> str:
        f = naw.RandomWordAug(action='delete', aug_p=1, aug_min=n_min, aug_max=n_max)
        return ' '.join(f.augment(text.split()))

    def __insert(self, text: str, n_min: int = 1, n_max: int = 4) -> str:
        _text = text.split()
        f = naw.SynonymAug(
            aug_src='wordnet', aug_p=1, aug_max=None, stopwords=self.stop_words
        )
        syn_s = f.augment(_text)
        syn_s = (' '.join(syn_s)).split()
        syn_s = [w for w in syn_s if w not in _text]
        n_aug = random.randint(n_min, n_max)
        for i in range(n_aug):
            if len(syn_s) == 0:
                break
            _text.insert(random.randrange(len(_text) + 1), random.choice(syn_s))
        return ' '.join(_text)

    def __augment(self, text: str) -> str:
        if random.random() > self.prob:
            return text
        augmentation_techniques = ['synonym', 'swap', 'delete', 'insert']
        augmentation_technique = random.choice(augmentation_techniques)
        if augmentation_technique == 'synonym':
            return self.__synonym(text)
        if augmentation_technique == 'swap':
            return self.__swap(text)
        if augmentation_technique == 'delete':
            return self.__delete(text)
        if augmentation_technique == 'insert':
            return self.__insert(text)

    def augment(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) == 0:
            return df
        df['input'] = df['input'].apply(lambda x: self.__augment(x))
        return df
