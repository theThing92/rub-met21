# import xml Elementtree which seems the most suitable way to parse html files and works exactly the same way as for xml files
import xml.etree.ElementTree as ET
import re
import syntok.segmenter as segmenter

def prep_bund(path: str):
    # takes the path of the corresponding xml file and returns the speech as a string.
    # it needs one speech per file. Each speech is marked with the "<text>" tag.
    tree = ET.parse(path)
    root = tree.getroot()
    speech = ""
    for body in root.iter("body"):
        for s in body.iter('s'):
            sent = s.itertext()
            for e in sent: speech = speech + e.rstrip() + " "
    return speech


def prep_wikibooks2(path: str):
    # takes the path of the corresponding html file and returns the book as a string.
    # it needs one book or chapter per file.
    tree = ET.parse(path)
    root = tree.getroot()
    book = ""
    for body in root.iter("body"):
        for s in body.iter('p'):
            if s.text != None:
                sent = s.itertext()
                for e in sent: book = book + e.rstrip() + " "
    return book


def prep_europarl(path: str):
    # function to convert europarl xml to string
    # one transcripted session per file (usually with multiple speakers)
    regex_paragraph = re.compile(r">\n*(.*)\n*<")
    text = ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        paragraphs = re.findall(regex_paragraph, text)
        text = " ".join(paragraphs)

    except Exception as e:
        raise e

    return text


def tokenize_text(text: str, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for paragraph in segmenter.process(text):
            for sentence in paragraph:
                for token in sentence:
                    # roughly reproduce the input,
                    # except for hyphenated word-breaks
                    # and replacing "n't" contractions with "not",
                    # separating tokens by single spaces
                    print(token.value, end=' ', file=f)
                print(file=f)  # print one sentence per line


if __name__ == "__main__":
    import os

    subdirs = os.listdir("../data")

    for dir in subdirs:
        dir_path = os.path.abspath(os.path.join("../data", dir))
        filepaths = [os.path.abspath(os.path.join(dir_path, f)) for f in os.listdir(dir_path)]

        for fp in filepaths:
            print(fp)
            filepath_out = fp + ".tok"
            if dir == "Bundestagsreden" and fp.endswith(".xml"):
                text = prep_bund(fp)
                tokenize_text(text, filepath_out)

            elif dir == "EUROPARL" and fp.endswith(".txt"):
                text = prep_europarl(fp)
                tokenize_text(text, filepath_out)

            elif dir == "WikiBooks" and fp.endswith(".html"):
                text = prep_wikibooks2(fp)
                tokenize_text(text, filepath_out)

