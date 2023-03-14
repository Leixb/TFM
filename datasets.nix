{ lib
, fetchurl
, linkFarm
}:
let
  datasets = builtins.fromTOML (builtins.readFile ./datasets.toml);
  drvList = lib.mapAttrsToList
    (name: data:
      {
        inherit name;
        path = (if lib.hasSuffix "gz" data.url then fetchTarball else fetchurl) data;
      }
    )
    datasets
  ;
in
linkFarm "datasets" drvList
