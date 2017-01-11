/* @flow */

const fs = require('fs');
const parseOpts = require('minimist');
const compileMod = require('./compile.js');
const compile = compileMod.compile;

var opts = parseOpts(
  process.argv.slice(2),
  {
    string: ['output'],
    boolean: ['output-binary'],
  }
);

var input = opts._[0];
var output = opts.output;
var outputBinary = opts['output-binary'];

compile(input, output, outputBinary, function(err) {
  if (err) {
    console.log(err);
    process.exit(1);
  }

  process.exit(0);
});

// const fs = require('fs');
// const parseOpts = require('minimist');
// const compile = require('../src/compile').compile;
//
// var opts = parseOpts(
//   process.argv.slice(2),
//   {
//     string: [
//       'source',                // --source "graph agraph { const one scalar float = 1 }"
//       'compile',               // --compile mypackage; --compile mypackage.mygraph;
//       'use-graph',             // --use-graph file.pbtxt
//       'feed-constants',        // --feed-constants inputs.pbtxt
//       'feed-constants-strip',  // --feed-constants-strip 'agraph/'
//       'result-prefix',         // --result-prefix 'main/'
//       'test-result-pattern',   // --test-result-pattern '^test/([^_].*)$'
//     ],
//     boolean: [
//       'test',
//       'run',
//       'feed-constants-binary',
//       'use-graph-binary',
//       'result-binary',
//       'compile-binary',
//     ],
//   }
// );
//
// parser.add_argument("--binary-graphdef", nargs='?', type=bool, default=False,
//                     help="""Whether or not input is binary.""")
// parser.add_argument("--feed-constants", nargs='?', type=str,
//                     help="""Path to GraphDef protobuf with constants to feed""")
// parser.add_argument("--feed-constants-strip", nargs='?', type=str, default="",
//                     help="""Prefix to filter for (and strip from) constants""")
// parser.add_argument("--feed-constants-prefix", nargs='?', type=str,
//                     help="""Prefix to add to constant names in feed""")
// parser.add_argument("--feed-constants-binary", nargs='?', type=bool, default=False,
//                     help="""Whether or not feed constant protobuf is binary""")
//
// parser.add_argument("--run", nargs='?', type=bool, default=False,
//                     help="""Run the graph with given (or default) --result* and --feed-* options""")
// parser.add_argument("--result-prefix", nargs='?', type=str, default="main/",
//                     help="""Prefix of nodes to read result from.""")
// parser.add_argument("--result-binary", nargs='?', type=bool, default=False,
//                     help="""Whether or not to result in binary.""")
// parser.add_argument("--result", nargs='?', type=str, default="/dev/stdout")
//
// parser.add_argument("--test", nargs='?', type=bool, default=False,
//                     help="""Run the tests graphs with given (or default) --test-* options""")
// parser.add_argument("--test-result-pattern", nargs='?', type=str, default="^test[^/]*/([^_].*)$",
//                     help="""Pattern to discover test graph results.""")
//
// var input = opts._[0];
// var splitInput = input.split(".", 2);
// var filename = splitInput[0];
// var graphName = splitInput[1]; // may be null.
//
// var output = opts.output;
// var outputBinary = opts['output-binary'];
//
// compile(input, output, outputBinary, function(err) {
//   if (err) {
//     console.log(err);
//     process.exit(1);
//   }
//
//   process.exit(0);
// });
