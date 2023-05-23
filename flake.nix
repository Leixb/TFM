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
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
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
  };

  outputs = { self, nixpkgs, flake-utils, devenv, ... } @ inputs:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        devenvShell = devenv.lib.mkShell {
          inherit inputs pkgs;
          modules = [ (import ./devenv.nix) ];
        };

      in
      {
        devShells.default = devenvShell;
        packages =
          let
            datasets = pkgs.callPackage ./datasets.nix { };

            document = pkgs.callPackage ./document/document.nix { };

            datasets-tarball = pkgs.runCommand "datasets.tar.gz" { } ''
              tar -hczf $out -C ${datasets} .
            '';
          in
          inputs.libsvm.packages.${system} // {
            inherit (devenvShell) ci;
            inherit datasets datasets-tarball document;

            default = pkgs.linkFarmFromDrvs "document_and_datasets" [
              datasets-tarball
              document
            ];
          };
      }
    );
}
