with import <nixpkgs> {};
stdenv.mkDerivation rec {
  name = "env";
  env = buildEnv { name = name; paths = buildInputs; };
  buildInputs = [
    nodejs-7_x
    yarn
    ncurses
    python3
    python3Packages.wheel
    python3Packages.virtualenv
    python3Packages.pex
  ];
}
