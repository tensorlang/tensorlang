const ohm = require('ohm-js');
const fs = require('fs');

const grammarText = fs.readFileSync(`${__dirname}/nao.ohm`);

function loadGrammar() {
  return ohm.grammar(grammarText);
}

function createSemantics(grammar) {
  var s = grammar.createSemantics();
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
      TensorLiteral_false: function(_) {
        return ["_tensor", false];
      },
      TensorLiteral_true: function(_) {
        return ["_tensor", true];
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
          if (expr[0] === "_retval" && expr[1]) {
            expr[1] = "" + emitted++;
          }
        });

        return ["_sf_graph", name.asJson].concat(body.asJson);
      },
      GraphElement: function(decl, _) {
        return decl.asJson;
      },
      NameableExpression: function(child, _1, _2, _3, name) {
        var childExpr = child.asJson;
        if (name.sourceString === "") {
          return childExpr;
        }

        var childExprType = childExpr[0];
        switch (childExprType) {
        case "_sf_local":
          break;
        case "_sf_apply":
          childExpr[1] = name.sourceString;
          break;
        default:
          throw new Error("Unhandled child expression type: " + childExprType);
        }

        return childExpr;

      },
      Expression_reference: function(name) {
        return ["_sf_local", name.sourceString];
      },
      Expression_apply: function(ns, fn_name, _1, argList, _2, attrs) {
        // TODO(adamb) Support attrs.

        return [
          "_sf_apply", null, ns.asJson[0], fn_name.sourceString, ["_sf_attrs"],
        ].concat(argList.asJson);
      },
      InputDeclaration: function(_, name, kind) {
        return ["_named_placeholder", name.asJson, kind.asJson[1], kind.asJson[2]];
      },
      ConstantDeclaration: function(_1, name, kind, _2, value) {
        return ["_named_constant", name.asJson, kind.asJson[1], kind.asJson[2], value.asJson];
      },
      OutputDeclaration: function(_1, name, kind, _2, expr) {
        return ["_retval", name[0] ? name[0].asJson : null, expr.asJson];
      },
      TensorKind: function(shape, type) { return ["kind", shape.asJson, type.asJson]; },
      TensorShape_unknown: function(_) { return ["_sf_shape", null]; },
      TensorShape_scalar: function(_) { return ["_sf_shape", []]; },
      TensorShape_literal: function(_1, dims, _2) { return ["_sf_shape", dims.asJson]; },
      TensorType: function(name) { return ["_sf_type", name.sourceString]; },
    }
  );

  return s;
};

var parseExpressions = function(source) {
  var grammar = loadGrammar();
  var semantics = createSemantics(grammar);

  var m = grammar.match(source);
  if (m.failed()) {
    callback(new Error(m.message), null);
    return;
  }

  return semantics(m).asJson;
}

module.exports = parseExpressions;
