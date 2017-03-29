with import <nixpkgs> {};
stdenv.mkDerivation rec {
  fancyYarn = replaceDependency {
    drv = yarn;
    oldDependency = nodejs;
    newDependency = nodejs-7_x;
  };

  name = "env";
  env = buildEnv { name = name; paths = buildInputs; };
  buildInputs = [
    nodejs-7_x
    fancyYarn
    ncurses
    ruby
    bundler
    python3
    python3Packages.wheel
    python3Packages.virtualenv
  ];
}
