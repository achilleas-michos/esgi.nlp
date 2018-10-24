import re


def get_shape_category(token):
    if re.match('^[\n]+$', token):  # IS LINE BREAK
        return 'NL'
    if any(char.isdigit() for char in token) and re.match('^[0-9.,]+$', token):  # IS NUMBER (E.G., 2, 2.000)
        return 'NUMBER'
    if re.fullmatch('[^A-Za-z0-9\t\n ]+', token):  # IS SPECIAL CHARS (E.G., $, #, ., *)
        return 'SPECIAL'
    if re.fullmatch('^[A-Z\-.]+$', token):  # IS UPPERCASE (E.G., AGREEMENT, INC.)
        return 'ALL-CAPS'
    if re.fullmatch('^[A-Z][a-z\-.]+$', token):  # FIRST LETTER UPPERCASE (E.G. This, Agreement)
        return '1ST-CAP'
    if re.fullmatch('^[a-z\-.]+$', token):  # IS LOWERCASE (E.G., may, third-party)
        return 'LOWER'
    if not token.isupper() and not token.islower():  # WEIRD CASE (E.G., 3RD, E2, iPhone)
        return 'MISC'
    return 'MISC'


def get_shape_category_simple(token):
    if token.islower():
        return 'ALL-LOWER'
    elif token.isupper():
        return 'ALL-UPPER'
    elif re.fullmatch('[A-Z][a-z]+', token):
        return 'FIRST-UPPER'
    else:
        return 'MISC'
