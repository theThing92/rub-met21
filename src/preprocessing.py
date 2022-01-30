# import xml Elementtree which seems the most suitable way to parse html files and works exactly the same way as for xml files
import xml.etree.ElementTree as ET
import re
import syntok.segmenter as segmenter
import csv
import os
from copy import deepcopy
import re
from src.additional_merge import additional_info

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


def _split_pos_morph_info(row_tagged):
    # ['wird', 'VAFIN.3.Sg.Pres.Ind', 'werden'] -> ['wird', 'VAFIN', '.3.Sg.Pres.Ind', 'werden']
    # ['.', '$.', '.']                          -> ['.', '$.', '_', '.']
    token = row_tagged[0]
    lemma = row_tagged[-1]
    if not row_tagged[1].startswith("$") and len(row_tagged[1].split(".",1)) == 2:
        pos, morph = row_tagged[1].split(".",1)
        row_tagged_out = [token, pos, morph, lemma]
        return row_tagged_out
    else:
        pos = row_tagged[1]
        row_tagged_out = [token, pos, "_", lemma]

        return row_tagged_out

def _merge_row_anno_tagged(row_anno, row_tagged):
    # anno: ['126-11', '18637-18641', 'sein', '_', '_', '_', '_', '_', '_', '_', '_', '_']
    # tagged: ['VAFIN', '.3.Sg.Pres.Ind', 'werden']
    # merged: ['126-11', '18637-18641', 'sein', '_', '_', '_', '_', '_', '_', '_', '_', '_']
    return row_anno + row_tagged[1:]

def merge_webanno_tsv_with_tagger_anno(data_directory = "data"):
    file_extension_annotated = ".tok.pos.morph.lemma"
    file_out = "merge_webanno_pos_morph_lemma.tsv"
    file_in = "CURATION_USER.tsv"
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
             "SATELLITES",
             "POS",
             "MORPH",
             "LEMMA",
             "METAPHORICAL_FUNC_WORD",
             "METAPHORICAL_FUNC_VERB_PHRASE"]

    # TODO: add list with function words (# noun, verb adverb) -> col with true/false marks interesting case (Satelliten und Kerne)
    # add list with verbs for analysis -> true marks interesting case Apfel nehmen -> false, Abschied nehmen -> true _
    # single texts per line + whole corpus as file

    # convert sem to 2sem
    # nehmen liegen die nicht als 2sem markiert sind

    # deskriptive statistik erstellen -> Wie viele Nomen, Verben etc. metaphorisch gebraucht

    # iterate over annotated files
    for path_anno in annotated_files:
        print(path_anno)

        path_anno_out = path_anno.replace(file_in, file_out)

        rows_anno_raw = []
        rows_anno = []
        rows_tagged = []

        dir_name_anno = path_anno.split("/")[-2]

        with open(path_anno, newline='', encoding="utf-8") as f:
            csv_reader_anno = csv.reader(f, delimiter='\t', quotechar=None)
            for row_anno in csv_reader_anno:
                try:
                    rows_anno_raw.append(row_anno)

                    if row_anno[0].startswith("#") and len(row_anno) == 1:
                        continue

                    elif row_anno == []:
                        continue

                    elif "." in row_anno[0]:
                        continue

                    else:
                        rows_anno.append(row_anno[:-1])
                except IndexError:
                    continue

        # iterate through pos morph lemma files and add tagging info
        file_name = None
        for path_tagged in tokenized_pos_morph_lemma_files:
            if dir_name_anno in path_tagged:
                file_name = path_tagged
        with open(file_name, newline='', encoding="utf-8") as f:
            csv_reader_tagged = csv.reader(f, delimiter='\t', quotechar=None)
            for row_tagged in csv_reader_tagged:
                if row_tagged:

                    # tokenize slash expressions like 1998/99 90/Grüne according to syntok tokenizer before adding to list
                    # cf. 1998/99 -> 1998 / 99 (1 Token -> 3 Tokens)
                    regex_slash_alphanumeric = re.compile(r"(\w+)\/(\w+)")
                    # tokenize abbrevation points (...) to 3 seperate dots
                    regex_abbr_point = re.compile(r"\.\.\.")
                    # tokenize date and year timespans (e.g. 19.-22 [Januar] / 2020-22
                    # note: dot for second date seperated as own token with syntok
                    regex_date_timespan = re.compile(r"(\d+)(\.)(-)(\d+)")
                    regex_year_timespan = re.compile(r"(\d+)(-)(\d+)")
                    # tokenize abbr. with ampersand (e.g. S&D -> S & D)
                    regex_ampersand = re.compile(r"(\w+)(&)(\w+)")
                    # fix tokenization errors for
                    # oh"-Tarot         -> oh " - Tarot
                    # Ringe"-Tarot'     -> Ringe " - Tarot
                    # and add pos tag, lemma info for Tarot manually
                    regex_yu_gi_oh_tarot = re.compile(r"(Ringe)(\")(-)(Tarot)")
                    regex_lord_of_the_rings_tarot = re.compile(r"(oh)(\")(-)(Tarot)")

                    token = row_tagged[0]

                    match_slash_alphanumeric = regex_slash_alphanumeric.match(token)
                    match_abbr_point = regex_abbr_point.match(token)
                    match_date_timespan = regex_date_timespan.match(token)
                    match_year_timespan = regex_year_timespan.match(token)
                    match_ampersand = regex_ampersand.match(token)
                    match_yu_gi_oh_tarot = regex_yu_gi_oh_tarot.match(token)
                    match_lord_of_the_rings_tarot = regex_lord_of_the_rings_tarot.match(token)


                    if match_slash_alphanumeric:
                        #print("alpha", row_tagged)
                        alphanumeric1, alphanumeric2 = token.split("/")

                        row_tagged_alphanumeric1 = deepcopy(row_tagged)
                        row_tagged_alphanumeric1[-1] = alphanumeric1
                        row_tagged_alphanumeric1[0] = alphanumeric1
                        row_tagged_slash = deepcopy(row_tagged)
                        row_tagged_slash[-1] = "/"
                        row_tagged_slash[0] = "/"
                        row_tagged_alphanumeric2 = deepcopy(row_tagged)
                        row_tagged_alphanumeric2[-1] = alphanumeric2
                        row_tagged_alphanumeric2[0] = alphanumeric2

                        rows_tagged.append(_split_pos_morph_info(row_tagged_alphanumeric1))
                        rows_tagged.append(_split_pos_morph_info(row_tagged_slash))
                        rows_tagged.append(_split_pos_morph_info(row_tagged_alphanumeric2))

                    elif match_abbr_point:
                        #print("abbr", row_tagged)
                        row_tagged_abbr = deepcopy(row_tagged)
                        row_tagged_abbr[0] = "."
                        row_tagged_abbr[-1] = "."

                        for i in range(3):
                            rows_tagged.append(_split_pos_morph_info(row_tagged_abbr))

                    elif match_date_timespan:
                        #print("date_timespan ", row_tagged)
                        tokens = match_date_timespan.groups()

                        for token in tokens:
                            row_date_timespan = deepcopy(row_tagged)
                            row_date_timespan[0] = token
                            row_date_timespan[-1] = token

                            rows_tagged.append(_split_pos_morph_info(row_date_timespan))

                    elif match_year_timespan:
                        #print("year_timespan ", row_tagged)
                        tokens = match_year_timespan.groups()

                        for token in tokens:
                            row_year_timespan = deepcopy(row_tagged)
                            row_year_timespan[0] = token
                            row_year_timespan[-1] = token

                            rows_tagged.append(_split_pos_morph_info(row_year_timespan))

                    elif match_ampersand:
                        #print("ampersand", row_tagged)
                        tokens = match_ampersand.groups()

                        for token in tokens:
                            row_ampersand = deepcopy(row_tagged)
                            row_ampersand[0] = token
                            row_ampersand[-1] = token

                            rows_tagged.append(_split_pos_morph_info(row_ampersand))

                    elif match_yu_gi_oh_tarot or match_lord_of_the_rings_tarot:
                        #print("tarot", row_tagged)

                        tokens = [match for match in [match_yu_gi_oh_tarot, match_lord_of_the_rings_tarot ]if match is not None][0].groups()

                        for token in tokens[:-1]:
                            row_tarot_base = deepcopy(row_tagged)
                            row_tarot_base[0] = token
                            row_tarot_base[-1] = token

                            rows_tagged.append(_split_pos_morph_info(row_tarot_base))


                        # add pos info manually for Token Tarot
                        row_tarot = deepcopy(row_tagged)
                        row_tarot[0] = "Tarot"
                        row_tarot[1] = "NN.Nom.Sg.Fem"
                        row_tarot[-1] = "Tarot"
                        rows_tagged.append(_split_pos_morph_info(row_tarot))

                    else:
                        #print(row_tagged)
                        rows_tagged.append(_split_pos_morph_info(row_tagged))

        try:
            assert len(rows_tagged) == len(rows_anno), f"Mismatch between tokens from files {path_anno} and {file_name} - " \
                                                       f"annotated: {len(rows_anno)} tagged: {len(rows_tagged)}."
        except AssertionError:
            texts = {"anno": rows_anno, "tagged": rows_tagged}
            for k,v in texts.items():
                with open(k+".tsv", "w", encoding="utf-8") as f:
                    for row in v:
                        print(row,file=f)
            return rows_anno, rows_tagged

        merged = list(map(_merge_row_anno_tagged, rows_anno, rows_tagged))

        for i, row in enumerate(merged):
            sent_token_id_merged = row[0]

            for j, row_raw in enumerate(rows_anno_raw):
                try:
                    sent_token_id_raw = row_raw[0]

                    if sent_token_id_merged == sent_token_id_raw:
                        rows_anno_raw[j] = row

                    elif sent_token_id_merged != sent_token_id_raw:
                        if sent_token_id_raw.split(".")[0] == sent_token_id_merged:
                            for i in range(3):
                                rows_anno_raw[j] += ["_" for i in range(3)]


                except IndexError:
                    continue

        with open(path_anno_out, "w", encoding="utf-8") as f:
            for row in rows_anno_raw:
                print("\t".join(row), file=f)

        path_anno_out_additional_info = path_anno_out.replace("merge_webanno_pos_morph_lemma", "merge_webanno_pos_morph_lemma_additional_info")
        additional_info(path_anno_out, path_anno_out_additional_info)

        content_lines = []
        with open(path_anno_out_additional_info, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for i,line in enumerate(lines):
                if line.startswith("#"):
                    pass
                elif (line == "" or line == "\n") and (lines[i-1] != "" or lines[i-1] == "\n"):
                    content_lines.append(line)
                else:
                    content_lines.append(line)

        path_anno_out_additional_info_sent_bound = path_anno_out_additional_info.replace("merge_webanno_pos_morph_lemma_additional_info", "merge_webanno_pos_morph_lemma_additional_info_sent")
        # with sentence boundaries
        with open(path_anno_out_additional_info_sent_bound, "w", encoding="utf-8") as f:
            for h in header:
                print("#"+h, file=f, end="\t")

            for c in content_lines[1:]:
                print(c.strip("\n"), file=f)

        path_anno_out_additional_info_wo_sent_bound = path_anno_out_additional_info.replace("merge_webanno_pos_morph_lemma_additional_info", "merge_webanno_pos_morph_lemma_additional_info_token")

        # without sentence boundaries
        with open(path_anno_out_additional_info_wo_sent_bound, "w", encoding="utf-8") as f:
            header_line = "\t".join(["#"+h for h in header])
            print(header_line, file=f)

            for c in content_lines[1:]:
                if c != "\n":
                    print(c.strip("\n"), file=f)

    # merge files
    out_total_sent = "data/total/merge_webanno_pos_morph_lemma_additional_info_sent.tsv"
    out_total_token = "data/total/merge_webanno_pos_morph_lemma_additional_info_token.tsv"

    f_out_total_sent = open(out_total_sent, "a", encoding="utf-8")
    f_out_total_token = open(out_total_token, "a", encoding="utf-8")

    header_line = "\t".join(["#" + h for h in header])
    print(header_line, file=f_out_total_sent)
    print(header_line, file=f_out_total_token)

    for p in annotated_files:
        p_sent = p.replace("CURATION_USER.tsv","merge_webanno_pos_morph_lemma_additional_info_sent.tsv")
        p_token = p.replace("CURATION_USER.tsv","merge_webanno_pos_morph_lemma_additional_info_token.tsv")

        with open(p_sent, "r", encoding="utf-8") as f:
            for line in f.readlines():
                if not line.startswith("#"):
                    print(line,file=f_out_total_sent,end="")

        with open(p_token, "r", encoding="utf-8") as g:
            for line in g.readlines():
                if not line.startswith("#"):
                    print(line,file=f_out_total_token,end="")

    f_out_total_sent.close()
    f_out_total_token.close()




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


