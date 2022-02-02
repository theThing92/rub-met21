import pandas as pd

def load_data(path_to_tsv="data/total/merge_webanno_pos_morph_lemma_additional_info_token.tsv"):
    # TODO: fix bug erroneous lines for annotation within compound words
    try:
        mappings = {"Kern": "0Kern",
                    "synt.": "1synt",
                    "1synt.": "1synt",
                    "3sonst": "4sonst"}


        df = pd.read_csv(path_to_tsv, sep="\t", on_bad_lines="warn")

        for k,v in mappings.items():
            df["#MET_A|Rolle"] = df["#MET_A|Rolle"].replace(k,v)

        return df

    except (IOError, Exception) as e:
        raise e


def stats(df):
    # num tokens
    num_tokens = len(df)

    num_core = len(df[df["#MET_A|Rolle"] == "0Kern"])
    num_sem = len(df[df["#MET_A|Rolle"] == "2sem"])
    num_clash = len(df[df["#MET_A|Rolle"] == "5clash"])
    num_synt = len(df[df["#MET_A|Rolle"] == "1synt"])
    num_other = len(df[df["#MET_A|Rolle"] == "4sonst"])
    num_idiom = len(df[df["#MET_A|Rolle"] == "3idiom"])

    num_markables = num_core + num_sem + num_clash + num_synt + num_other + num_idiom

    # met_core / token
    core_per_100_token = num_core / num_tokens * 100
    print(f"core_per_100_token:\t{core_per_100_token}")

    # sem / token
    sem_per_100_token = num_sem / num_tokens * 100
    print(f"sem_per_100_token:\t{sem_per_100_token}")

    # clash / token
    clash_per_100_token = num_clash / num_tokens * 100
    print(f"clash_per_100_token:\t{clash_per_100_token}")

    # synt / token
    synt_per_100_token = num_synt / num_tokens * 100
    print(f"synt_per_100_token:\t{synt_per_100_token}")

    # sonst / token
    other_per_100_tokens = num_other / num_tokens * 100
    print(f"other_per_100_tokens:\t{other_per_100_tokens}")

    # idiom / token
    idiom_per_100_tokens = num_idiom / num_tokens * 100
    print(f"idiom_per_100_tokens:\t{idiom_per_100_tokens}")

    # markables (0Kern, 1sem...) / tokens
    markables_per_100_tokens = num_markables / num_tokens * 100
    print(f"markables_per_100_tokens:\t{markables_per_100_tokens}")

    # pie diagram (MET_A|Rolle)
    role_percent_markables = {"0Kern": num_core / num_markables,
                    "1synt": num_synt / num_markables,
                    "2sem": num_sem / num_markables,
                    "3idiom": num_idiom / num_markables,
                    "4sonst": num_other / num_markables}
    print(f"role_percent_markables:\t{role_percent_markables}")

    # avg. num. satellites per met
    avg_num_satellites_per_met = df[df["#MET_A|Rolle"] == "0Kern"][df["#SATELLITES"] != "_"]["#SATELLITES"].apply(lambda x: len(x.split("|"))).sum() / num_core
    print(f"avg_num_satellites_per_met:\t{avg_num_satellites_per_met}")

    # pos tags + met (percent / pie diagram)
    pos_percent_core = df[df["#MET_A|Rolle"] == "0Kern"]["#POS"].value_counts().to_dict()

    for k in pos_percent_core:
        pos_percent_core[k] = pos_percent_core[k] / num_core
    print(f"pos_percent_core:\t{pos_percent_core}")





if __name__ == "__main__":
    df = load_data()
    stats(df)