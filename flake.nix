{
  description = "Master thesis on infinite neural network kernels";

  nixConfig = {
    tarball-ttl = 63072000; # 2 years
    extra-substituters = [
      "https://tfm.cachix.org/"
    ];
    extra-trusted-public-keys = [
      "tfm.cachix.org-1:XGcdmGOXdUqRzemq7YoN8poxuAMUdnKlJGqzxfoFEWc="
    ];
  };

  inputs = {
    nixpkgs.url = "github:leixb/nixpkgs?rev=182ce5b195431cc51967c0d5ec00f478a762a4a2";
    flake-utils.url = "github:numtide/flake-utils";
    devenv = {
      url = "github:cachix/devenv/latest";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    libsvm = {
      url = "github:LeixB/libsvm";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };

    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };

    uciml = {
      url = "github:leixb/UCI-ML-Repository-dataset-metadata";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };
  };

  outputs = { self, nixpkgs, flake-utils, devenv, ... } @ inputs:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        devenvShell = devenv.lib.mkShell {
          inherit inputs pkgs;
          modules = [ (import ./nix/devenv.nix) ];
        };

      in
      {
        devShells.default = devenvShell;
        packages =
          let
            datasets = pkgs.callPackage ./nix/datasets.nix { };

            document = pkgs.callPackage ./nix/document.nix { };
            document-split = pkgs.callPackage ./nix/split_appendix.nix { inherit document; };

            datasets-tarball = pkgs.runCommand "datasets.tar.gz" { } ''
              mkdir -p $out
              tar -hczf $out/datasets.tar.gz -C ${datasets} .
            '';
          in
          inputs.libsvm.packages.${system} // {
            inherit (devenvShell) ci;
            inherit datasets datasets-tarball document document-split;

            default = pkgs.symlinkJoin {
              name = "document_and_datasets";
              paths = [
                datasets-tarball
                document-split
              ];
            };
          };
        apps.prefetch_datasets =
          let
            prefetcher = pkgs.writeShellApplication {
              name = "prefetch_datasets";
              runtimeInputs = with pkgs; [ git jq yq nix-prefetch ];
              text = ''
                OPTS=("fetchzip" "--stripRoot" "--expr" "false" "--url")
                GIT_ROOT="$(git rev-parse --show-toplevel)"

                FILE="''${1:-"$GIT_ROOT/datasets.toml"}"

                tomlq <"$FILE" | \
                    jq -r '.[] | select(.sha256 == "") | .url' | \
                    xargs -I{} nix-prefetch "''${OPTS[@]}" {}
              '';
            };
          in
          {
            type = "app";
            program = pkgs.lib.getExe prefetcher;
          };
      }
    );
}
