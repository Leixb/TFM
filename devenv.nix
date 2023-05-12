{ pkgs, lib, inputs, ... }:

{
  # https://devenv.sh/basics/
  env.PROJECT = "tfm";

  # context for taskwarrior
  env.TW_CONTEXT = "tfm";

  env.JULIA_NUM_THREADS = 8;

  env.DATASETS = pkgs.callPackage ./datasets.nix { };

  # https://devenv.sh/scripts/
  scripts.build-doc.exec = ''
    latexmk -cd "$DEVENV_ROOT/document/000-main.tex" -lualatex -shell-escape -interaction=nonstopmode -file-line-error -view=none "$@"
  '';

  # Download bibliography from local zotero instance (using better-bibtex plugin)
  scripts.fetch-biblio.exec = ''
    curl -f http://127.0.0.1:23119/better-bibtex/export/collection?/1/TFM.biblatex -o "$DEVENV_ROOT/document/biblio.bib" || echo "Is Zotero running?"
  '';

  scripts.pluto.exec = "julia --project=$DEVENV_ROOT -e 'using Pkg; Pkg.instantiate(); using Pluto; Pluto.run(auto_reload_from_file=true)'";

  enterShell = ''
    task project:$PROJECT summary || echo "No summary available"
    task project:$PROJECT limit:10 next || echo "No tasks found"

    # Populate directory with raw data from nix store
    if [ -d "$DATASETS" ]; then
      if [ ! -d data/exp_raw ] || [ "$(readlink -f data/exp_raw)" != "$DATASETS" ]; then
        echo "Linking data/exp_raw to $DATASETS"
        mkdir -p data
        ln -sfn "$DATASETS" data/exp_raw
      fi
    fi
  '';

  packages = with pkgs; [
    texlab
    texlive.combined.scheme-full
    pdf2svg
    poppler_utils
    imagemagick
    inputs.poetry2nix.packages.${system}.poetry
    inputs.libsvm.packages.${system}.libsvm
    (inputs.poetry2nix.legacyPackages.${system}.mkPoetryEnv {
      projectDir = "";
      preferWheels = true;
      pyproject = ./pyproject.toml;
      poetrylock = ./poetry.lock;
      python = python3;
    })
    python3.pkgs.pygments
    nixpkgs-fmt
    gdb
    gcc
  ];

  # https://devenv.sh/languages/
  languages.nix.enable = true;
  languages.r = {
    enable = false;
    package = pkgs.rWrapper.override {
      packages = lib.attrVals
        (lib.filter (p: p != "")
          (lib.splitString "\n" (builtins.readFile ./dependencies.txt))
        )
        pkgs.rPackages;
    };
  };
  languages.julia.enable = true;
  languages.julia.package = pkgs.julia_19;

  # https://devenv.sh/pre-commit-hooks/
  pre-commit.hooks = {
    chktex.enable = true;
    nixpkgs-fmt.enable = true;
    ruff.enable = true;
    shellcheck.enable = true;
  };

  # https://devenv.sh/processes/
  # processes.jupyter.exec = "jupyter lab";
  processes.latexmk.exec = "build-doc -pvc";
  processes.pluto.exec = "pluto";
}
