{ lib
, fetchurl
, fetchzip
, linkFarm
, writeText
}:
let
  datasets = builtins.fromTOML (builtins.readFile ../datasets.toml);
  drvList = lib.mapAttrsToList
    (name: data:
      let download_data = lib.filterAttrs (k: _: k != "meta") data; in
      {
        inherit name;
        path = (if lib.hasSuffix "gz" data.url then fetchTarball else
        (if lib.hasSuffix "zip" data.url then fetchzip else fetchurl)) download_data;
      }
    )
    datasets
  ;
  datasets_meta = builtins.toJSON (lib.mapAttrs (name: data: data.meta) datasets);
  datasets_meta_file = rec {
    name = "metadata.json";
    path = writeText name datasets_meta;
  };
in
linkFarm "datasets" (drvList ++ [ datasets_meta_file ])
