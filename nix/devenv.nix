{ pkgs, lib, inputs, ... }:

let
  pyenv = inputs.poetry2nix.legacyPackages.${pkgs.system}.mkPoetryEnv {
    projectDir = "";
    preferWheels = true;
    pyproject = ../pyproject.toml;
    poetrylock = ../poetry.lock;
    python = pkgs.python3;
  };
in
{
  # https://devenv.sh/basics/
  env = {
    PROJECT = "tfm";

    # context for taskwarrior
    TW_CONTEXT = "tfm";

    PYTHON = "${pyenv}/bin/python";
    PYTHONHOME = pyenv;

    JULIA_NUM_THREADS = 8;

    JULIA_PYCALL_DEPS = pkgs.runCommand "julia-pycall-deps.jl"
      {
        pythonhome = pyenv;
        pythonversion = pyenv.python.version;
      } ''
      substituteAll ${./deps.jl} $out
    '';

    DATASETS = pkgs.callPackage ./datasets.nix { };

    # Needed for OpenGL
    LD_LIBRARY_PATH = "/run/opengl-driver/lib:/run/opengl-driver-32/lib:${pkgs.stdenv.cc.cc.lib}/lib";

    FREETYPE_ABSTRACTION_FONT_PATH = "${pkgs.lmodern}/share/fonts/opentype/public/lm";
  };

  # https://devenv.sh/scripts/
  scripts = {
    build-doc.exec = ''
      latexmk -cd "$DEVENV_ROOT/document/000-main.tex" -lualatex -shell-escape -interaction=nonstopmode -file-line-error -view=none "$@"
    '';

    # Download bibliography from local zotero instance (using better-bibtex plugin)
    fetch-biblio.exec = ''
      curl -f http://127.0.0.1:23119/better-bibtex/export/collection?/1/TFM.biblatex -o "$DEVENV_ROOT/document/biblio.bib" || echo "Is Zotero running?"
    '';

    pluto.exec = "julia --project=$DEVENV_ROOT -e 'using Pkg; Pkg.instantiate(); using Pluto; Pluto.run(auto_reload_from_file=true)'";

    sync-pycall-deps.exec = "julia ${./julia_link_pycall.jl}";
  };

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

    # Make sure julia links to the proper python version
    sync-pycall-deps || echo "Failed to sync pycall deps" >&2 &
  '';

  packages = with pkgs; [
    gnuplot
    texlab
    texlive.combined.scheme-full
    pdf2svg
    poppler_utils
    hyperfine
    imagemagick
    inputs.poetry2nix.packages.${system}.poetry
    inputs.libsvm.packages.${system}.libsvm
    pyenv
    python3.pkgs.pygments
    nixpkgs-fmt
    gdb
    gcc
  ];

  # https://devenv.sh/languages/
  languages = {
    nix.enable = true;

    r = {
      enable = false;
      package = pkgs.rWrapper.override {
        packages = lib.attrVals
          (lib.filter (p: p != "")
            (lib.splitString "\n" (builtins.readFile ../R-dependencies.txt))
          )
          pkgs.rPackages;
      };
    };

    julia = {
      enable = true;
      package = pkgs.julia_19;
    };
  };

  # https://devenv.sh/pre-commit-hooks/
  pre-commit.hooks = {
    chktex.enable = true;
    nixpkgs-fmt.enable = true;
    ruff.enable = true;
    shellcheck.enable = true;
  };

  # https://devenv.sh/processes/
  processes = {
    # jupyter.exec = "jupyter lab";
    latexmk.exec = "build-doc -pvc";
    pluto.exec = "pluto";
  };
}
