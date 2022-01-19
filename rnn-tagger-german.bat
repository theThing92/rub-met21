:: Set these paths appropriately
:: converted shell script to batch file (for usage with windows)
:: author: Maurice Vogel
@ECHO OFF
set PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'

set BASEPATH=\Users\mvogel\PycharmProjects\rub-met21\RNNTagger
set SCRIPTS=%BASEPATH%\scripts
set LIB=%BASEPATH%\lib
set PyRNN=%BASEPATH%\PyRNN
set PyNMT=%BASEPATH%\PyNMT
set TMP=\
set LANGUAGE=german

set TOKENIZER=perl %SCRIPTS%\tokenize.pl
set ABBR_LIST=%LIB%\Tokenizer\%LANGUAGE%-abbreviations
set TAGGER=py %PyRNN%\rnn-annotate.py
set RNNPAR=%LIB%\PyRNN/%LANGUAGE%
set REFORMAT=perl %SCRIPTS%\reformat.pl
set LEMMATIZER=py %PyNMT%\nmt-translate.py
set NMTPAR=%LIB%\PyNMT\%LANGUAGE%

%TOKENIZER% -g -a %ABBR_LIST% %1 > %TMP%.tok

%TAGGER% %RNNPAR% %TMP%.tok > %TMP%.tagged

%REFORMAT% %TMP%.tagged > %TMP%.reformatted

%LEMMATIZER% --print_source %NMTPAR% %TMP%.reformatted > %TMP%.lemmas

%SCRIPTS%\lemma-lookup.pl %TMP%.lemmas %TMP%.tagged 

del /f %TMP%.tok  %TMP%.tagged  %TMP%.reformatted %TMP%.lemmas
