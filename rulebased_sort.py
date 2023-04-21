import re
from functools import cmp_to_key, partial
from natsort import natsort_keygen


nkeygen = natsort_keygen()

cover_pattern = re.compile(r"^(?!.*(app|map)).*(front_matter|cover).*$")
toc_pattern = re.compile(r"^(?!.*(app|map)).*(table_of_contents|toc).*$")
summary_pattern = re.compile(r"^(?!.*(app|map)).*summary.*$")
eis_pattern = re.compile(r"^(?!.*(app|map|volume|vol[^a-z]|part|chapter|chpt|section)).*eis.*$")
volume_pattern = re.compile(r"^(?!.*(app|map)).*(volume|vol[^a-z]).*$")
part_pattern = re.compile(r"^(?!.*(app|map)).*(part).*$")
chapter_pattern = re.compile(r"^(?!.*(app|map)).*(chapter|chpt).*$");
section_pattern = re.compile(r"^(?!.*(app|map)).*(section).*$");
appendix_pattern = re.compile(r"((\b|_)app(\b|_)|appendix|appendices)")
map_pattern = re.compile(r"((\b|_)map(\b|_)|(\b|_)maps(\b|_))")
noi_pattern = re.compile(r"((\b|_)noi(\b|_)|notice_of_intent)")
noa_pattern = re.compile(r"((\b|_)noa(\b|_)|notice_of_availability)")
rod_pattern = re.compile(r"((\b|_)rod(\b|_)|record_of_decision)")


def pattern_numeric_value(x):
    if cover_pattern.search(x):
        return 0
    elif toc_pattern.search(x):
        return 1
    elif summary_pattern.search(x):
        return 2
    elif eis_pattern.search(x):
        return 3
    elif volume_pattern.search(x):
        return 4
    elif part_pattern.search(x):
        return 5
    elif chapter_pattern.search(x):
        return 6
    elif section_pattern.search(x):
        return 7
    elif appendix_pattern.search(x):
        return 9
    elif map_pattern.search(x):
        return 10
    elif noi_pattern.search(x):
        return 11
    elif noa_pattern.search(x):
        return 12
    elif rod_pattern.search(x):
        return 13
    else:
        return 8


def regex_comparison(x, y):
    x = pattern_numeric_value(x)
    y = pattern_numeric_value(y)
    return (x > y) - (x < y)


def natural_comparison(x, y):
    parsed_x = nkeygen(x)
    parsed_y = nkeygen(y)
    return (parsed_x > parsed_y) - (parsed_x < parsed_y)


def pre_process(x):
    return (x.lower().replace("'", "").replace(",", "").replace(".pdf", "").replace(" ", "_")
            .replace('appendices', 'appendix').replace('volumes', 'volume').replace('chapters', 'chapter'))


def volume_comparison(x, y, key=None):
    if key is not None:
        x = key(x)
        y = key(y)
    x = pre_process(x)
    y = pre_process(y)
    comparison_result = regex_comparison(x, y)
    if comparison_result == 0:
        comparison_result = natural_comparison(x, y)
    return comparison_result


def sort(x, key=None):
    return sorted(x, key=cmp_to_key(partial(volume_comparison, key=key)))
