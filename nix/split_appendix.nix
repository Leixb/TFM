{ lib
, runCommand
, pdftk
, document
, name_main ? "document.pdf"
, name_appendix ? "appendix.pdf"
, keep_original ? false
, split_at ? "Appendix"
}:

runCommand "split_document"
{
  SOURCE_DATE_EPOCH = 1685620800; # 2023-06-01
  nativeBuildInputs = [
    pdftk
  ];
}
  ''
    PAGE="$(pdftk "${document}" dump_data_utf8 | grep -A 2 'BookmarkTitle: ${split_at}' | tail -n1 | cut -d' ' -f2)"

    pdftk "${document}" cat 1-$((PAGE-1)) output "${name_main}"
    pdftk "${document}" cat $PAGE-end output "${name_appendix}"

    mkdir -p "$out"

    cp "${name_main}" "${name_appendix}" "$out"
    ${lib.optionalString keep_original "cp \"${document}\" \"$out\""}
  ''
