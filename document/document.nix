{ runCommand
, texlive
, python3
, which
, outils
}:

runCommand "document.pdf"
{
  SOURCE_DATE_EPOCH = 1685620800; # 2023-06-01
  src = ./.;
  nativeBuildInputs = [
    texlive.combined.scheme-full
    python3.pkgs.pygments
    which
    outils
  ];
}
  ''
    mkdir -p build

    export HOME=$(mktemp -d)
    lndir -silent "$src" build
    cd build

    latexmk

    cp *.pdf $out
  ''
