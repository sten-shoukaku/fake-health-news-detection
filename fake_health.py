import os
import json
import pandas as pd
import typing


class FakeHealth:
    """Fake Health dataset helper"""

    def __init__(self, root_path: str):
        file_list = self.__get_filenames(
            f'{root_path}/content/HealthRelease', f'{root_path}/content/HealthStory'
        )
        review_list = [
            f'{root_path}/reviews/HealthRelease.json',
            f'{root_path}/reviews/HealthStory.json',
        ]

        reviews = []
        for r in review_list:
            reviews += self.__open_json(r, add_id=False)

        self.df_content = pd.DataFrame(
            [self.__open_json(f, id) for (f, id) in file_list]
        )
        self.df_review = pd.DataFrame(reviews)

        self.__preprocess_content()
        self.__preprocess_review()

        self.data = pd.merge(self.df_content, self.df_review, on=['id'])

    def get_data(self, input_content: str = None) -> pd.DataFrame:
        """Get dataset as Pandas DataFrame

        input_content (str): set whether to use title or text as input
        """
        assert input_content in [
            None,
            'text',
            'title',
        ], 'input_content must be either None, "title", or "text"'
        if input_content is not None:
            df = self.data.rename(columns={input_content: 'input'})
            df = df[['input', 'target']]
        else:
            df = self.data
        return df

    def __get_filenames(self, *dir_path: str) -> typing.List[str]:
        filenames = []
        for path in dir_path:
            filenames += [
                [f'{path}/{f}', f.replace('.json', '')] for f in os.listdir(path)
            ]
        return filenames

    def __open_json(
        self, file_path: str, id: str = '', add_id: bool = True
    ) -> typing.Dict[str, str]:
        with open(file_path, 'r') as f:
            res = json.load(f)
            if add_id:
                res['id'] = id
        return res

    def __preprocess_content(self) -> None:
        self.df_content = self.df_content.drop(
            columns=[
                'url',
                'images',
                'keywords',
                'authors',
                'top_img',
                'canonical_link',
                'movies',
                'publish_date',
                'summary',
                'meta_data',
                'source',
            ]
        )

        self.df_content = self.df_content.drop(
            self.df_content[self.df_content['text'] == ''].index
        )
        self.df_content['text'] = self.df_content['text'].str.lower()
        self.df_content['title'] = self.df_content['title'].str.lower()

    def __preprocess_review(self) -> None:
        self.df_review = self.df_review.rename(columns={'news_id': 'id'})

        self.df_review['target'] = self.df_review.apply(
            lambda x: 1 if x['rating'] >= 3 else 0, axis=1
        )

        self.df_review = self.df_review.drop(
            columns=[
                'link',
                'title',
                'description',
                'original_title',
                'reviewers',
                'category',
                'tags',
                'source_link',
                'rating',
                'summary',
                'criteria',
                'news_source',
            ]
        )
