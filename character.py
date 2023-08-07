MAX_CHARS = 1
OUTPUT_MAX_LEN = MAX_CHARS # + 2  # <GO>+groundtruth+<END>

hiraganas = [
    "あ", "い", "う", "え", "お",
    "か", "き", "く", "け", "こ",
    "さ", "し", "す", "せ", "そ",
    "た", "ち", "つ", "て", "と",
    "な", "に", "ぬ", "ね", "の",
    "は", "ひ", "ふ", "へ", "ほ",
    "ま", "み", "む", "め", "も",
    "や", "ぃ", "ゆ", "ぇ", "よ",
    "ら", "り", "る", "れ", "ろ",
    "わ", "ゐ", "ぅ", "ゑ", "を",
    "ん",
]

katakanas = [
    "ア", "イ", "ウ", "エ", "オ",
    "カ", "キ", "ク", "ケ", "コ",
    "サ", "シ", "ス", "セ", "ソ",
    "タ", "チ", "ツ", "テ", "ト",
    "ナ", "ニ", "ヌ", "ネ", "ノ",
    "ハ", "ヒ", "フ", "ヘ", "ホ",
    "マ", "ミ", "ム", "メ", "モ",
    "ヤ", "ィ", "ユ", "ェ", "ヨ",
    "ラ", "リ", "ル", "レ", "ロ",
    "ワ", "ヰ", "ゥ", "ヱ", "ヲ",
    "ン",
]

kanjis = []

char_classes = hiraganas + katakanas + kanjis
n_char_classes = len(char_classes)

char2idx = {c: n for n, c in enumerate(char_classes)}
# idx2char = {c: n for n, c in enumerate(char_classes)}

char2code = lambda c: format(ord(c), '#06x')
code2char = lambda c: chr(int(c, base=16))

tok = False
if not tok:
    tokens = {"PAD_TOKEN": n_char_classes}
else:
    tokens = {"GO_TOKEN": n_char_classes, "END_TOKEN": n_char_classes + 1, "PAD_TOKEN": n_char_classes + 2}
del tok
n_tokens = len(tokens.keys())

vocab_size = n_char_classes + n_tokens
