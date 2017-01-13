/* @flow */

const ohm = require('ohm-js');
const fs = require('fs');

const grammarText = fs.readFileSync(`${__dirname}/nao.ohm`);

function loadGrammar() {
  return ohm.grammar(grammarText);
}

function reduceOperandList(expr: any[][], opToTfMethod: { [key: string]: string }): any[] {
  var [ops, ...exprs] = expr;

  if (ops.length === 0) {
    return exprs[0];
  }

  return exprs.reduceRight(function(acc, e, ix) {
    var op = ops[ix];
    if (!op) {
      return e;
    }

    var method = opToTfMethod[op];
    if (!method) {
      throw new Error("Unknown operator: " + op);
    }

    return ["_sf_apply", null, "tf", method, ["_sf_attrs"], e, acc];
  });
}

function createSemantics(grammar) {
  var s = grammar.createSemantics();
  s.addAttribute(
    'asJson',
    {
      _terminal: function() { return this.sourceString; },
      identifier: function(_1, _2) { return this.sourceString; },
      stringExpression: function(_1, chars, _2) { return chars.sourceString; },
      nonemptyListOfLookaheadEntry: function(_1, elem1, _2, _3, _4, moreElems, _6, _7) {
        return [elem1.asJson].concat(moreElems.asJson);
      },
      invocationNamespace: function(ns, _) { return ns.sourceString; },
      nonemptyListOfLookahead: function(elems) {
        return elems.asJson.reduce(function(acc, cur) {
          return acc ? cur : acc.concat(cur);
        });
      },
      TensorKind: function(shape, type) { return ["kind", shape.asJson, type.asJson]; },
      TensorShape_unknown: function(_) { return ["_sf_shape", null]; },
      TensorShape_scalar: function(_) { return ["_sf_shape", []]; },
      TensorShape_literal: function(_1, dims, _2) { return ["_sf_shape", dims.asJson]; },
      TensorType: function(name) { return ["_sf_type", name.sourceString]; },

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

      ListLiteral: function(_1, elems, _2) {
        return ["_sf_list"].concat(elems.asJson);
      },

      TensorLiteralElement_false: function(_) {
        return false;
      },
      TensorLiteralElement_true: function(_) {
        return true;
      },
      TensorLiteralElement_number: function(value) {
        return value.asJson;
      },
      TensorLiteralElement_string: function(str) {
        return str.asJson;
      },
      TensorLiteralElement_arr: function(_1, elems, _2) {
        return ["_sf_list"].concat(elems.asJson);
      },
      TensorLiteral: function(child) {
        return ["_named_tensor", null, null, null, child.asJson];
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

          if (expr[0] === "__retval") {
            var retName = expr[1];
            var retVal = expr[2];
            var subName = retVal[1];

            retvals.push([retName, subName]);
            return retVal;
          }

          return expr;
        });

        return ["_sf_def_function", name.asJson, args, retvals].concat(expressions);
      },
      FuncElement: function(decl, _) {
        return decl.asJson;
      },
      GraphDefinition: function(_1, _2, name, _3, _4, body, _5, _6) {
        var emitted = 0;
        // console.log('GraphDefinition', JSON.stringify(body.asJson));
        body.asJson.forEach(function(expr, ix, exprs) {
          if (expr[0] === "__retval" && !expr[1]) {
            expr[1] = "" + emitted++;
          }
        });

        return ["_sf_graph", name.asJson].concat(body.asJson);
      },
      GraphElement: function(decl, _) {
        return decl.asJson;
      },
      AfterDeclaration: function(_1, _2, _3, _4, _5, body, _6, _7) {
        return ["__sf_after_leaves"].concat(body.asJson);
      },
      Expression: function(child, _1, _2, _3, nameExpr) {
        var childExpr = child.asJson;
        var name = nameExpr.asJson[0];
        if (name === "") {
          return childExpr;
        }

        var childExprType = childExpr[0];
        switch (childExprType) {
        case "_sf_local":
          break;
        case "_sf_list":
          break;
        case "_named_tensor":
          break;
        case "_sf_index":
          break;
        case "_sf_apply":
          childExpr[1] = name;
          break;
        default:
          throw new Error("Unhandled child expression type: " + childExprType);
        }

        return childExpr;
      },
      NonemptyListOf: function(elem, sep, rest) {
        var ops = sep.asJson;

        // Expect the last element to have no operator that goes with it.
        return [ops, elem.asJson].concat(rest.asJson);
      },
      Expression1: function(subexpr) {
        return reduceOperandList(subexpr.asJson, {
          "<=": "less_equal",
          "<": "less",
          "==": "equal",
          "!=": "not_equal",
          ">=": "greater_equal",
          ">": "greater",
        });
      },
      Expression2: function(subexpr) {
        return reduceOperandList(subexpr.asJson, {
          "+": "add",
          "-": "subtract",
        });
      },
      Expression3: function(subexpr) {
        return reduceOperandList(subexpr.asJson, {
          "*": "multiply",
          "/": "divide",
        });
      },
      indexSuffix: function(_, identifier) {
        return identifier.asJson;
      },
      Expression4: function(subexpr, indexSuffix) {
        var suffix = indexSuffix.asJson[0];
        if (!suffix) {
          return subexpr.asJson;
        }

        return ["_sf_index", subexpr.asJson, suffix];
      },
      Expression5_reference: function(name) {
        return ["_sf_local", name.sourceString];
      },
      Expression5_apply: function(ns, fn_name, attrs, _1, argList, _2) {
        // TODO(adamb) Support attrs.

        return [
          "_sf_apply", null, ns.asJson[0], fn_name.sourceString, ["_sf_attrs"],
        ].concat(argList.asJson);
      },
      InputDeclaration: function(_, name, kind) {
        return ["_named_placeholder", name.asJson, kind.asJson[1], kind.asJson[2]];
      },
      ConstantDeclaration: function(_1, name, kind, _2, tensorLiteral) {
        var childExpr = [].concat(tensorLiteral.asJson);
        childExpr[1] = name.sourceString;
        childExpr[2] = kind.asJson[1]; // shape
        childExpr[3] = kind.asJson[2]; // dtype

        return childExpr;
      },
      OutputDeclaration: function(_1, name, kind, _2, expr) {
        var rhsValue = expr.asJson[0];

        if (rhsValue) {
          if (rhsValue[0] !== ["_sf_local"]) {
            rhsValue = ["_sf_apply", name.asJson, "tf", "identity", ["_sf_attrs"], rhsValue];
          }

          if (!rhsValue[1]) {
            rhsValue[1] = name.asJson;
          }

          return ["__retval", name.asJson, rhsValue];
        }

        return ["__retval", name.asJson, ["_sf_local", name.sourceString]];
      },
    }
  );

  return s;
};

var parseExpressions = function(source: string) {
  var grammar = loadGrammar();
  var semantics = createSemantics(grammar);

  var m = grammar.match(source);
  if (m.failed()) {
    throw m.message;
  }

  return semantics(m).asJson;
}

module.exports = parseExpressions;
