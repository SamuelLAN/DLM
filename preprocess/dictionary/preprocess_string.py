import re
from lib.preprocess import utils


def process(string, _pipeline):
    for func in _pipeline:
        string = func(string)
    return string


def remove_quote(string):
    return string.replace("'", '').replace('"', '')


__reg_bracket_content = re.compile(r'[\[({【（《<][^\])}】）》>]*[\])}】）》>]')
__reg_bracket = re.compile(r'[\[({【（《<\])}】）》>]')
__reg_space = re.compile(r'\s+')


def convert_punctuations(string):
    return string.replace('，', ',').replace('。', '.').replace('！', '!').replace('？', '?'). \
        replace('：', ':').replace('；', ';').replace('“', '"').replace('”', '"')


def remove_bracket_content(string):
    return __reg_bracket_content.sub('', string)


def remove_multi_space(string):
    return __reg_space.sub(' ', string)


def remove_bracket(string):
    return __reg_bracket.sub('', string)


pipeline = [
    lambda x: str(x).lower().strip(),
    lambda x: '' if x == 'nan' else x,
    convert_punctuations,
    remove_quote,
    remove_bracket_content,
    lambda x: x.strip(),
]

__reg_not_zh = re.compile(r'[^\u4e00-\u9fa5\d ;,.，\'"+\-_]+')

__reg_not_en = re.compile(r'[^a-zA-Z\d ;,.\'"+\-_]+')
__reg_etc = re.compile(r'(,\s*)?(and)?\setc\.$', re.IGNORECASE)


def remove_not_zh(string):
    return __reg_not_zh.sub('', string)


def remove_not_en(string):
    string = __reg_not_en.sub('', string)
    string = __reg_etc.sub('', string)
    return string.strip()


def filter_duplicate(_dictionary):
    for word, val_dict in _dictionary.items():
        for key, val in val_dict.items():
            val_dict[key] = list(set(val))
    return _dictionary
