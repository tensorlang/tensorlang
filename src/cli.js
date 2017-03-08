/* @flow */
"use strict";

const fs = require('fs');
const process = require('process');
const path = require('path');
const meow = require('meow');
const compile = require('./compile.js');
const run = require('./run.js');
const test = require('./test.js');

const sourceMapSupport = require('source-map-support');
sourceMapSupport.install();

const opts = meow(`
`,
  {
    string: [
      'root',                  // --root ./scratch
      'source',                // --source "graph agraph { one = 1 }"
      'compile',               // --compile foo.pbtxt
      'parse',                 // --parse foo.json
      'compile-graph',         // --compile-graph foo.pbtxt
      'use-graph',             // --use-graph file.pbtxt
      // Path to GraphDef protobuf with constants to feed
      'feed-constants',        // --feed-constants inputs.pbtxt
      // Prefix to filter for (and strip from) constants
      'feed-constants-strip',  // --feed-constants-strip 'agraph/'
      // Prefix to add to constant names in feed
      'feed-constants-prefix',
      // Prefix of nodes to read result from.
      'result-prefix',         // --result-prefix 'main/'
      // Pattern to discover test graph results.
      'test-result-pattern',   // --test-result-pattern '^test/([^_].*)$'
    ],
    boolean: [
      // Run the tests graphs with given (or default) --test-* options
      'test',
      // Run the graph with given (or default) --result* and --feed-* options
      'run',
      // Whether or not feed constant protobuf is binary
      'feed-constants-binary',
      // Whether or not input is binary.
      'use-graph-binary',
      // Whether or not to result in binary.
      'result-binary',
      'compile-binary',
    ],
  }
);


const suffix = ".nao";

// Examples:
// mypackage
// mypackage.mygraph
const inputs = opts.input;
const flags = opts.flags;

if (inputs.length > 1) {
  opts.showHelp(1);
}

const shouldRunGraph = flags.run || flags.feedConstants || flags.resultPrefix || !(flags.test || flags.compile || flags.compileGraph);
const shouldTestGraph = flags.test;

console.warn(inputs);
console.warn(flags);

var input = inputs[0];
var fromFile: ?string;
var fromFileBinary: boolean = false;
var fromString: ?string;

function abortOnCatch(promise: Promise<any>) {
  promise.catch((err) => {
    console.warn(err);
    process.exit(1);
  });
}

function maybeTest() {
  if (shouldTestGraph) {
    if (fromFile) {
      abortOnCatch(test.fromFile(fromFile, fromFileBinary));
    } else if (fromString) {
      abortOnCatch(test.fromString(fromString, false));
    }
  }
}

function maybeRun() {
  if (shouldRunGraph) {
    if (fromFile) {
      abortOnCatch(run.fromFile(fromFile, fromFileBinary));
    } else if (fromString) {
      abortOnCatch(run.fromString(fromString, false));
    }
  }
}

if (input || flags.source) {
  var source;
  var pkgName: string;
  var pkgRootDir = flags.root || process.cwd();
  var parseTo = flags.parse;
  var compileTo = flags.compile;
  var compileGraphTo = flags.compileGraph;
  var compileToBinary = flags.compileBinary;

  if (flags.source) {
    source = flags.source;
    if (input) {
      console.warn("Can't provide a package name and --source option.")
      process.exit(1);
    }
    pkgName = "main";
  } else {
    pkgName = input;

    var basename = input;
    var filename = path.join(pkgRootDir, `${basename}${suffix}`);

    // TODO(adamb) Don't do this synchronously
    source = fs.readFileSync(filename).toString();
  }

  function resolvePackage(importPath: string): Promise<any> {
    var [name, scope] = importPath.split(":", 2);
    type PackageAttempt = {suffix: string, language: string}
    var trySuffixes: PackageAttempt[] = [
      {
        suffix: ".nao",
        language: "nao"
      },
      {
        suffix: ".py",
        language: "python"
      },
      {
        suffix: ".meta",
        language: "tensorflow:metagraph:pbtxt"
      }
    ];

    return trySuffixes.reduce(
      (p: Promise<any>, attempt: PackageAttempt): Promise<any> => {
        return p.catch(e => {
          return new Promise((rs, rj) => {
            fs.readFile(
                path.join(pkgRootDir, `${name}${attempt.suffix}`),
                "utf-8",
                (err, data) => {
                  if (err) {
                    rj(err);
                    return;
                  }
                  rs({language: attempt.language, name: name, scope: scope, content: data});
                });
          });
        })
      },
      Promise.reject()
    )
    .catch(() => { throw new Error("no such package: " + name); });
  }

  if (parseTo) {
    var parse = compile.parseString(pkgName, source, resolvePackage)
      .then((expr) => {
        // HACK(adamb) This is just temporary until python owns the CLI.
        if (parseTo === "-") {
          process.stdout.write(JSON.stringify(expr));
        } else {
          fs.writeFileSync(parseTo, JSON.stringify(expr));
        }
      });
    abortOnCatch(parse);
  } else {
    var compilation: Promise<any>;
    if (compileTo) {
      fromFile = compileTo;
      fromFileBinary = compileToBinary;
      compilation = compile.compile("main", source, resolvePackage,
          compileTo, compileGraphTo, compileToBinary);
    } else {
      compilation = compile.compileString("main", source, resolvePackage,
          compileGraphTo, compileToBinary);
      compilation.then((str) => { fromString = str; });
    }

    compilation.then(() => {
      maybeTest();
      maybeRun();
    });
    abortOnCatch(compilation);
  }
} else {
  fromFile = flags.useGraph;
  fromFileBinary = flags.useGraphBinary;
  maybeTest();
  maybeRun();
}
