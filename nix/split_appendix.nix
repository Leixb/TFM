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
    pdftk "${document}" dump_data | grep Bookmark >bookmarks

    PAGE="$(grep -A 2 'BookmarkTitle: ${split_at}' bookmarks | tail -n1 | cut -d' ' -f2)"

    grep -B1000 'BookmarkTitle: ${split_at}' bookmarks | head -n-2 >bookmarks_main
    grep -B1 -A1000 'BookmarkTitle: ${split_at}' bookmarks | awk '{ if ($1 == "BookmarkPageNumber:") { print $1, $2-'$PAGE'+1 } else {print $0} }' >bookmarks_appendix

    pdftk "${document}" cat 1-$((PAGE-1)) output tmp_main.pdf
    pdftk "${document}" cat $PAGE-end output tmp_appendix.pdf

    pdftk tmp_main.pdf update_info bookmarks_main output "${name_main}"
    pdftk tmp_appendix.pdf update_info bookmarks_appendix output "${name_appendix}"

    mkdir -p "$out"

    cp "${name_main}" "${name_appendix}" "$out"
    ${lib.optionalString keep_original "cp \"${document}\" \"$out\""}
  ''
