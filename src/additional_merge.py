def functional_verb_phrase(fnc_verbs: list, token: list):
    # gleicht Lemma mit einer Liste an Verben ab und gibt entsprechendes Ergebnis zurück
    # true wenn Funktionsverbgefüge, false wenn Verb aus der Liste aber nicht metaphorisch, _ andernfalls
    # fnc_verbs = Liste mit Verben die in Funktionsverbgefügen auftreten können
    # token = Eine Token Zeile aus unseren Daten als Liste
    
    if token[12] in fnc_verbs and token[4] != "_" and token[4] != "5clash":
        return "true"
    elif token[12] in fnc_verbs:
        return "false"
    else:
        return "_"


def function_in_metaphor(fnc_words: list, token: list):
    # gleicht POS-Tag mit einer Liste ab und gibt entsprechendes Ergebenis zurück
    # true wenn Funktionswort und Teil einer Metapher, false wenn Funktionswert und nicht Teil einer Metapher, _ andernfalls
    # fnc_words = Liste mit POS-Tags die Funktionswörter beschreiben
    # token = Eine Token Zeile aus unseren Daten als Liste

    if token[10] in fnc_words and token[4] != "_" and token[4] != "5clash":
        return "true"
    elif token[10] in fnc_words:
        return "false"
    else:
        return "_"


def define_lists():
    # definiere die Listen die abgeglichen werden sollen
    function_verbs = ["bringen", "finden", "führen", "geben", "gehen", "haben", "halten", "kommen", "liegen", "machen", "nehmen", "setzen", "stehen", "stellen", "treffen", "üben", "vertreten", "ziehen"]
    function_words = ["KOUI", "KOUS", "KON", "KOKOM", "APPR", "APPRART", "APZR", "ART", "PTKZU", "PTKVZ"]
    return function_verbs, function_words


def startswith(string: str, char: str):
    # Hilfsfunktion zu überprüfen ob ein String mit einem bestimmten Zeichen startet
    # string = beliebig langer String
    # char = String mit der Länge 1 (ein Symbol)

    if string[0] == char:
        return True
    return False

def old_version(path:str):
    # Checks if the text still contains the old columns
    with open(path, "r", encoding="utf-8") as f:
        for token in f.readlines():
            if "MET_C" in token:
                return True
            elif "MET_D" in token:
                return False
            else:
                continue

def remove_columns(token: list):

    # remove old columns
    token.pop(8)
    token.pop(9)

    # merge new string
    tok_new = "\t".join(token)

    return token, tok_new

def additional_info(inpath: str, outpath: str):
    # Fügt die zusätzlichen Informationen (Funktionswörter und Funktionsverbgefüge) hinzu
    # inpath = Dateipfad der zu lesenden Datei
    # outpath = Dateipfad der Datei in die geschrieben werden soll

    #listen erstellen/initialisieren
    fnc_verbs, fnc_words = define_lists()
    out = []
    old = old_version(inpath)

    # Daten öffnen und Infos berechnen
    with open(inpath, "r", encoding="utf-8") as f:

        # for old guidelines
        if old:
            for token in f.readlines():
                if token != "" and token!= "\n" and not startswith(token, "#"):
                    tok = token.rstrip().split("\t")
                    tok_list, tok_new = remove_columns(tok)
                    out.append((tok_new, function_in_metaphor(fnc_words, tok_list), functional_verb_phrase(fnc_verbs, tok_list)))
                elif "MET_C" in token:
                    continue
                else:
                    out.append(token)
            f.close()

        # for new guidelines
        else:
            for token in f.readlines():
                if token != "" and token!= "\n" and not startswith(token, "#"):
                    tok = token.rstrip().split("\t")
                    out.append((token, function_in_metaphor(fnc_words, tok), functional_verb_phrase(fnc_verbs, tok)))
                else:
                    out.append(token)
            f.close()

    # Daten mit neuen Infos in neue Datei schreiben
    with open(outpath, "w", encoding="utf-8") as f:
        for i in range(0, len(out)-1):
            if len(out[i]) == 3:
                f.write(out[i][0].rstrip() + "\t" + out[i][1] + "\t" + out[i][2] + "\n")
            else:
                f.write(out[i])
        f.close()