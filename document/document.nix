{ runCommand
, texlive
, python3
, which
, outils
, gnuplot
, filename ? "document.pdf"
}:

runCommand filename
{
  SOURCE_DATE_EPOCH = 1685620800; # 2023-06-01
  src = ./.;
  nativeBuildInputs = [
    texlive.combined.scheme-full
    python3.pkgs.pygments
    which
    outils
    gnuplot
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
