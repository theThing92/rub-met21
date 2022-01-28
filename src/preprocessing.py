# import xml Elementtree which seems the most suitable way to parse html files and works exactly the same way as for xml files
import xml.etree.ElementTree as ET
import re
import syntok.segmenter as segmenter
import csv
import os
from copy import deepcopy

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
    try:
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

    except (IOError, Exception) as e:
        raise e


def convert_to_one_token_per_line(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            tokens_per_line = [sent.split() for sent in f.readlines()]

        with open(path, "w", encoding="utf-8") as f:
            for sent in tokens_per_line:
                for token in sent:
                    print(token, file=f)
                print(file=f)

    except (IOError, Exception) as e:
        raise e


def merge_webanno_tsv_with_tagger_anno(data_directory = "data"):
    file_extension_annotated = ".tok.pos.morph.lemma"
    file_out = "merge.tsv"
    default_webanno_tsv_filename = "CURATION_USER.tsv"
    categories = ["Bundestagsreden", "EUROPARL", "WikiBooks"]
    dir_name_annotated = [data_directory+"/"+cat+"/annotated" for cat in categories]
    dir_name_annotated_files = [[path + "/"+subpath+"/"+default_webanno_tsv_filename for subpath in os.listdir(path)] for path in dir_name_annotated]
    annotated_files = []
    for sublist in dir_name_annotated_files:
        for v in sublist:
            annotated_files.append(v)
    dir_name_tokenized_pos_morph_lemma = [data_directory+"/"+cat+"/tokenized_token_per_line" for cat in categories]
    dir_name_tokenized_pos_morph_lemma_files = [[path+"/"+subpath for subpath in os.listdir(path) if file_extension_annotated in subpath] for path in dir_name_tokenized_pos_morph_lemma]
    tokenized_pos_morph_lemma_files = []
    for sublist in dir_name_tokenized_pos_morph_lemma_files:
        for v in sublist:
            tokenized_pos_morph_lemma_files.append(v)


    header =["ID_SENT-ID_TOKEN",
             "CHAR_OFFSET",
             "TOKEN",
             "MET_A|MFlag",
             "MET_A|Rolle",
             "MET_B|basic",
             "MET_B|c_mark",
             "MET_B2|b_Lex",
             "MET_D|a_Par",
             "METD|BT_webanno.custom.MET_A",
             "POS",
             "MORPH",
             "LEMMA"]

    # TODO: add list with function words (# noun, verb adverb) -> col with true/false marks interesting case (Satelliten und Kerne)
    # add list with verbs for analysis -> true marks interesting case
    # single texts per line + whole corpus as file

    # convert sem to 2sem
    # nehmen liegen die nicht als 2sem markiert sind

    # deskriptive statistik erstellen -> Wie viele Nomen, Verben etc. metaphorisch gebraucht

    rows_anno_out = []
    # iterate over annotated files
    for path_anno in annotated_files:
        print(path_anno)

        dir_name_anno = path_anno.split("/")[-2]

        with open(path_anno, newline='', encoding="utf-8") as f:
            csv_reader_anno = csv.reader(f, delimiter='\t', quotechar='"')
            for row_anno in csv_reader_anno:
                #print(row_anno)
                try:

                    row_anno_out = deepcopy(row_anno)

                    sent_id_anno = int(row_anno[0].split("-")[0])
                    token_id_anno = int(row_anno[0].split("-")[1])

                    # iterate through pos morph lemma files and add tagging info
                    file_name = None

                    for path_tagged in tokenized_pos_morph_lemma_files:
                        if dir_name_anno in path_tagged:
                            file_name = path_tagged

                    token_id_tagged = 1
                    sent_id_tagged = 1
                    rows_tagged_sent_token_id = []

                    with open(file_name, newline='', encoding="utf-8") as f:
                        csv_reader_tagged = csv.reader(f, delimiter='\t', quotechar='"')

                        for row_tagged in csv_reader_tagged:
                            row_tagged_out = deepcopy(row_tagged)

                            if row_tagged != []:

                                row_tagged_out.insert(0,token_id_tagged)
                                row_tagged_out.insert(0,sent_id_tagged)
                                token_id_tagged += 1

                                rows_tagged_sent_token_id.append(row_tagged_out)


                                # TODO: add morph info seperately
                                #morph = row_tagged_out[-2].split(".",1)[-1] if len(row_tagged_out[-2].split(".",1))==2 else "_"

                            elif row_tagged == []:
                                sent_id_tagged += 1
                                token_id_tagged = 1
                                rows_tagged_sent_token_id.append([])

                    #print(rows_tagged_sent_token_id)

                    for r in rows_tagged_sent_token_id:
                        if r != []:
                            token_id = r[1]
                            sent_id = r[0]

                            print("#######")
                            print(sent_id)
                            print(token_id)

                            print(sent_id_anno)
                            print(token_id_anno)
                            print("#######")

                            if token_id == token_id_anno and sent_id == sent_id_anno:
                                row_anno_out.append(r[-2])
                                row_anno_out.append(r[-1])

                                rows_anno_out.append(rows_anno_out)
                        elif r == []:
                            rows_anno_out.append([])
                except (IndexError, ValueError, Exception):
                    pass
        return rows_anno_out


if __name__ == "__main__":
    l = merge_webanno_tsv_with_tagger_anno()
    # import os
    #
    # subdirs = os.listdir("../data")[2:]
    #
    # for dir in subdirs:
    #     dir_path = os.path.abspath(os.path.join("../data", dir, "tokenized_token_per_line"))
    #     filepaths = [os.path.abspath(os.path.join(dir_path, f)) for f in os.listdir(dir_path)]
    #
    #     for fp in filepaths:
    #         print(fp)
    #         convert_to_one_token_per_line(fp)
    #
    #
    #     for fp in filepaths:
    #         print(fp)
    #         fp_out = fp+".pos.morph.lemma"
    #
    #         os.system("..\\rnn-tagger-german.bat " + fp + "> " + fp_out)


