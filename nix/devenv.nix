{ pkgs, lib, inputs, ... }:

let
  pyenv = inputs.poetry2nix.legacyPackages.${pkgs.system}.mkPoetryEnv {
    projectDir = "";
    preferWheels = true;
    pyproject = ../pyproject.toml;
    poetrylock = ../poetry.lock;
    python = pkgs.python3;
  };
  Renv = pkgs.rWrapper.override {
    packages = lib.attrVals
      (lib.filter (p: p != "")
        (lib.splitString "\n" (builtins.readFile ../R-dependencies.txt))
      )
      pkgs.rPackages;
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

    R_HOME = Renv + "/lib/R";

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
      enable = true;
      package = pkgs.symlinkJoin {
        name = "R";
        paths = [ Renv ];
        buildInputs = [ pkgs.makeWrapper ];
        postBuild = ''
          wrapProgram "$out/bin/R" --set R_LIBS_SITE ""
        '';
      };
    };

    julia = {
      enable = true;
      package =
        let
          # WARN: This uses the experimental PR for Julia packages in nix:
          # https://github.com/NixOS/nixpkgs/pull/225513
          # It may not work for all packages, and may change in the future.
          # The version used is fixed with flake.lock in my own fork of the PR
          # so it will be stable as is, but do not copy this code without checking
          # if the PR has been merged and a more sensible API is available.
          #
          # One of the major caveats is that the package list is fixed through
          # the versions in an external registry, so it is not possible to use
          # the Manifest.toml file. This should change in the future when the
          # API stabilizes and julia2nix is updated to use it.

          packageOverrides = builtins.mapAttrs (name: value: pkgs.fetchFromGitHub value) (lib.importTOML ./juliaPackageOverrides.toml);
          python3 = pyenv;
          juliaWithPackages = pkgs.julia.withPackages.override { inherit python3 packageOverrides; };

          projectPackages = builtins.attrNames (lib.importTOML ../Project.toml).deps;
          brokenPackages = [ "GLMakie" ];
          isNotBroken = p: ! (builtins.elem p brokenPackages);

          packageList = builtins.filter isNotBroken projectPackages;
        in
        juliaWithPackages packageList;
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
