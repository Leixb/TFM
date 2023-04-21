{ runCommand
, texlive
, python3
, which
, outils
, SOURCE_DATE_EPOCH
}:

runCommand "document.pdf"
{
  inherit SOURCE_DATE_EPOCH;
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
