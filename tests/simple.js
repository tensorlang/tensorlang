'use strict';

// Test parsing simple expression
// Test generating simple IR
// Use python program to convert simple IR to .pbtxt
// Use different python program to use .pbtxt to run graph with given data

var test = require('tape-catch');

var ohm = require('ohm-js');

var fs = require('fs');
var grammarText = fs.readFileSync('src/nao.ohm');
var g = ohm.grammar(grammarText);
var s = g.createSemantics();
s.addAttribute(
  'asJson',
  {
    identifier: function(_1, _2) { return this.sourceString; },
    stringExpression: function(_1, _2) { return ""; },
    nonemptyListOfLookaheadEntry: function(_1, elem1, _2, _3, _4, moreElems, _6, _7) {
      return [elem1.asJson].concat(moreElems.asJson);
    },
    invocationNamespace: function(ns, _) { return ns.sourceString; },
    nonemptyListOfLookahead: function(elems) {
      return elems.asJson.reduce(function(acc, cur) {
        return acc ? cur : acc.concat(cur);
      });
    },
    TensorLiteral_arr: function(_1, elems, _2) {
      return ["_tensor"].concat(elems.asJson);
    },
    number_whole: function(sign, _, digits, maybeImaginary) {
      // JavaScript and JSON don't support numbers with high enough
      // precision to use native types.
      var signStr = (sign.sourceString === "-") ? "-" : "";
      // return maybeImaginary.asJson ?
        // ["_complex", ["_whole", "0"], ["_whole", signStr + digits.sourceString]] :
      return ["_sf_whole", signStr + digits.sourceString];
    },
    number_fract: function(sign, _1, characteristic, _2, mantissa, maybeImaginary) {
      // JavaScript and JSON don't support numbers with high enough
      // precision to use native types.
      var signStr = (sign.sourceString === "-") ? "-" : "";
      // return maybeImaginary.asJson ?
      //   ["_complex",
      //     ["_whole", "0"],
      //     ["_fraction", signStr + characteristic.sourceString + "." + mantissa.sourceString]
      //   ] :
      return ["_sf_fraction", signStr + characteristic.sourceString + "." + mantissa.sourceString];
    },
    FuncDefinition: function(_1, _2, name, _3, _4, body, _5, _6) {
      var retvals = [];
      var args = [];
      var expressions = body.asJson.map(function(expr, ix, exprs) {
        if (expr[0] === "_named_placeholder") {
          var sub = expr.slice(1);
          args.push(sub);
          var argName = expr[1];
          return ["_sf_local", argName];
        }

        if (expr[0] === "_retval") {
          var sub = expr.slice(1);
          retvals.push(sub);
          return sub;
        }

        return sub;
      });

      return ["_sf_def_function", name.asJson, args, retvals].concat(expressions);
    },
    FuncElement: function(decl, _) {
      return decl.asJson;
    },
    GraphDefinition: function(_1, _2, name, _3, _4, body, _5, _6) {
      var emitted = 0;
      body.asJson.forEach(function(expr, ix, exprs) {
        if (expr[0] === "_retval" && !expr[1]) {
          expr[1] = "" + emitted++;
        }
      });

      return ["_sf_graph", name.asJson].concat(body.asJson);
    },
    GraphElement: function(decl, _) {
      return decl.asJson;
    },
    Expression: function(child) {
      return child.asJson;
    },
    Expression_reference: function(name) {
      return ["_sf_local", name.sourceString];
    },
    Expression_apply: function(ns, fn_name, _1, argList, _2, attrs) {
      // TODO(adamb) Support attrs.

      return [
        "_sf_apply", ns.asJson[0], fn_name.sourceString, ["_sf_attrs"],
      ].concat(argList.asJson);
    },
    PlaceholderDeclaration: function(_, name, kind) {
      return ["_named_placeholder", name.asJson, kind.asJson[1], kind.asJson[2]];
    },
    ConstantDeclaration: function(_1, name, kind, _2, value) {
      return ["_named_constant", name.asJson, kind.asJson[1], kind.asJson[2], value.asJson];
    },
    OutputDeclaration: function(_1, name, kind, _2, expr) {
      return ["_retval", null, expr.asJson];
    },
    TensorKind: function(shape, type) { return ["kind", shape.asJson, type.asJson]; },
    TensorShape_unknown: function(_) { return ["_sf_shape", null]; },
    TensorShape_scalar: function(_) { return ["_sf_shape", []]; },
    TensorShape_literal: function(_1, dims, _2) { return ["_sf_shape", dims.asJson]; },
    TensorType: function(name) { return ["_sf_type", name.sourceString]; },
  }
);

const spawn = require('child_process').spawn;
const stream = require('stream');

const env = process.env;
const CUDA_HOME = "/usr/local/cuda";

test('simple test', function (t) {
  var src = fs.readFileSync('examples/identity.nao');
  var m = g.match(src.toString());
  if (m.failed()) {
    t.end(m.message);
    return;
  }

  var n = s(m);

  var process = spawn("../bin/python",
      [
        "src/json2pb.py",
        "--input_json", "/dev/stdin",
        "--output_graph", "/dev/stdout",
      ],
      {
        env: {
          "CUDA_HOME": CUDA_HOME,
          "DYLD_LIBRARY_PATH": `${env.DYLD_LIBRARY_PATH}:${CUDA_HOME}/lib`,
          "PATH": `${CUDA_HOME}/bin:${env.PATH}`,
        },
        stdio: [
          'pipe',
          'pipe',
          2
        ]
      }
    );

  var converter = new stream.Writable();

  process.stdin.write(JSON.stringify(n.asJson));
  process.stdin.end();
  converter.data = []; // We'll store all the data inside this array
  converter._write = function (chunk) {
    this.data.push(chunk);
  };
  converter.on('end', function() { // Will be emitted when the input stream has ended, ie. no more data will be provided
    var b = Buffer.concat(this.data); // Create a buffer from all the received chunks
    console.log(b.toString());

    t.end();
  });

  process.stdout.pipe(converter);
  process.on('close', function(code, signal) {
    console.log("code", code);
    console.log("signal", signal);
  })
});
