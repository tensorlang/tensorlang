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

    return ["_sf_apply", null, "tf", method, null, e, acc];
  });
}

function processFunctionBody(signature, body) {
  var retvals = [];
  var [attributes, inputs] = signature;

  var expressions = body.map(function(expr, ix, exprs) {
    if (expr[0] === "__retval") {
      var retName = expr[1];
      var retVal = expr[2];
      var subName = retVal[1];
      if (!subName) {
        subName = "retval_" + retvals.length;
        retVal[1] = subName;
      }

      retvals.push([retName, subName]);
      return retVal;
    }

    return expr;
  });

  return [attributes, inputs, retvals].concat(expressions);
}

function replaceHereExpression(expr: any[], callback: (any[]) => void) {
  if (expr instanceof Array) {
    switch (expr[0]) {
    case '__sf_here':
      callback(expr);
      break;
    case "_sf_local":
    case "_sf_attr":
    case "_sf_index":
    case "_sf_cond":
    case "_sf_function":
    case "_sf_def_function":
      // Do not recurse within these.
      break;
    default:
      expr.forEach((e) => { replaceHereExpression(e, callback); });
    }
  }
}

function rewriteExpressionWithName(name: string, expr: any[]): any[] {
  var exprType = expr[0];
  switch (exprType) {
  case "_sf_local":
  case "_sf_attr":
  case "_sf_index":
  case "_sf_cond":
  case "__sf_here":
  case "_sf_list":
  case "_sf_while_loop":
    return ["_sf_define_local", name, expr];

  case "_sf_function_lookup":
  case "_named_tensor":
  case "_sf_apply":
  case "_named_apply":
    expr[1] = name;
    break;
  default:
    throw new Error("Unhandled child expression type: " + exprType);
  }

  return expr;
}

function createSemantics(grammar) {
  var s = grammar.createSemantics();
  var anonIncrement = 0;
  s.addAttribute(
    'asJson',
    {
      Program: function(importDecls, topLevelDecls) {
        return [...importDecls.asJson, ...topLevelDecls.asJson];
      },
      ImportDeclaration: function(_, body) {
        return ["_sf_import", body.asJson];
      },
      ImportDeclarationBody_single: function(spec) {
        return [spec.asJson];
      },
      ImportDeclarationBody_multi: function(_1, _2, specs, _3, _4, _5) {
        return specs.asJson;
      },
      ImportSpec: function(packageName, importPath, _) {
        var pathFragments = importPath.asJson.split("/");
        var name = packageName.asJson[0];
        if (!name) {
          name = pathFragments[pathFragments.length - 1];
        }
        return [name, pathFragments];
      },
      _terminal: function() { return this.sourceString; },
      identifier: function(_1, _2) { return this.sourceString; },
      stringLiteral: function(_1, chars, _2) { return chars.sourceString; },
      EmptyListOf: function() { },
      nonemptyListOfLookaheadEntry: function(_1, elem1, _2, _3, _4, moreElems, _6, _7) {
        return [elem1.asJson].concat(moreElems.asJson);
      },
      invocationNamespace: function(ns, _) { return ns.sourceString; },
      nonemptyListOfLookahead: function(elems) {
        return elems.asJson.reduce(function(acc, cur) {
          return acc ? acc.concat(cur) : cur;
        });
      },
      TensorKind: function(type, shape) { return ["kind", shape.asJson, type.asJson]; },
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

      FunctionLiteral: function(_, signature, block) {
        return ["_sf_function", null].concat(
          processFunctionBody(signature.asJson, block.asJson));
      },
      FunctionSignature: function(attributes, inputs) {
        return [attributes.asJson[0], inputs.asJson];
      },
      FunctionParameter: function(name, type) {
        return [name.asJson, type.asJson];
      },
      FunctionAttributeType: function(type, _1, minValue, _2, initialValue) {
        return null;
      },
      FunctionAttributes: function(_1, parameters, _2) {
        return parameters.asJson.map(function(parameter) {
          // TODO(adamb) Actually use type in the future.
          // [attrName, attrType, attrInitialValue]

          var [name, type] = parameter;
          return [name, null, null];
        })
      },
      FunctionInputs: function(_1, parameters, _2) {
        var params = parameters.asJson || [];
        return params.map(function(parameter) {
          var [name, kind] = parameter;
          var shape = kind && kind[0];
          var type = kind && kind[1];
          return [name, shape, type];
        });
      },
      FunctionDeclaration: function(_1, _2, name, signature, block) {
        return ["_sf_def_function", name.asJson].concat(
          processFunctionBody(signature.asJson, block.asJson));
      },
      FunctionBlock: function(_1, _2, body, _3, _4) {
        return body.asJson;
      },
      FunctionElement: function(decl, _) {
        return decl.asJson;
      },
      GraphDefinition: function(_1, _2, name, _3, _4, body, _5, _6) {
        var emitted = 0;
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
      AfterStatement: function(_1, _2, _3, _4, _5, body, _6, _7) {
        return ["__sf_after_leaves"].concat(body.asJson);
      },
      Assignment: function(name, _1, _2, _3, _4, rhs) {
        return rewriteExpressionWithName(name.asJson, rhs.asJson);
      },
      Expression: function(child, _1, _2, _3, nameExpr) {
        var childExpr = child.asJson;
        var name = nameExpr.asJson[0];
        if (!name) {
          return childExpr;
        }

        return rewriteExpressionWithName(name, childExpr);
      },
      NonemptyListOf: function(elem, sep, rest) {
        var ops = sep.asJson;

        // Expect the last element to have no operator that goes with it.
        return [ops, elem.asJson].concat(rest.asJson);
      },
      IfExpression: function(_1, _2, cond, _3, _4, thenClause, _5, _6, _7, _8, _9, elseClause, _10, _11) {
        return ["_sf_cond", cond.asJson, thenClause.asJson, elseClause.asJson];
      },
      ForExpression: function(_1, _2, initializers, condition, body) {
        var retvals = [];
        var bodyExprs = body.asJson.map(function(expr) {
          if (expr[0] === "__retval") {
            var retName = expr[1];
            var retVal = expr[2];
            var subName = retVal[1];
            if (!subName) {
              subName = "retval_" + retvals.length;
              retVal[1] = subName;
            }

            retvals.push([retName, subName]);
            return retVal;
          }

          return expr;
        });

        return [
          "_sf_while_loop", condition.asJson,
          body.asJson, retvals,
          initializers.asJson,
        ];
      },
      ForInitializers: function(exprs, _2) {
        return exprs.asJson;
      },
      ForBody: function(_1, _2, exprs, _4, _5) {
        return exprs.asJson;
      },
      Expression1: function(subexpr) {
        var [ops, ...exprs] = subexpr.asJson;

        if (ops.length === 0) {
          return exprs[0];
        }

        return exprs.reduce(function(acc, e, ix) {
          if (!acc) {
            return e;
          }

          // If the accumulated expression is foo(), it's a _sf_apply.
          // Recurse down its expressions and replace any ["."] with ["_sf_local", "anonResultN"].
          // Stop recursing further when we encounter a
          if (e[0] !== "_sf_apply") {
            return ["_named_apply", null, e, acc];
          } else {
            var previousName = "anon" + anonIncrement++;
            var firstReference = ["_sf_apply", previousName, "tf", "identity", null, acc];
            var otherReferences = ["_sf_local", previousName];

            replaceHereExpression(
              e,
              (expr) => {
                var reference;
                if (firstReference) {
                  reference = firstReference;
                  firstReference = null;
                } else {
                  reference = otherReferences;
                }
                expr.length = 0;
                expr.push(...reference);
              }
            );

            return e;
          }
        });
      },
      Expression2: function(subexpr) {
        return reduceOperandList(subexpr.asJson, {
          "<=": "less_equal",
          "<": "less",
          "==": "equal",
          "!=": "not_equal",
          ">=": "greater_equal",
          ">": "greater",
        });
      },
      Expression3: function(subexpr) {
        return reduceOperandList(subexpr.asJson, {
          "+": "add",
          "-": "subtract",
        });
      },
      Expression4: function(subexpr) {
        return reduceOperandList(subexpr.asJson, {
          "*": "multiply",
          "/": "divide",
        });
      },
      indexSuffix: function(_, identifier) {
        return identifier.asJson;
      },
      indexIdentifier: function(identifier) {
        return identifier.asJson;
      },
      indexNumber: function(digits) {
        return ["_sf_whole", digits.sourceString];
      },
      Expression5: function(subexpr, indexSuffix) {
        var suffix = indexSuffix.asJson[0];
        if (!suffix) {
          return subexpr.asJson;
        }

        return ["_sf_index", subexpr.asJson, suffix];
      },
      Expression6_reference: function(ns, identifier, attrs) {
        if (ns.asJson[0]) {
          return ["_sf_namespace_lookup", ns.asJson[0], identifier.asJson, attrs.asJson[0]]
        }

        if (attrs.asJson[0]) {
          return ["_sf_function_lookup", null, identifier.asJson, attrs.asJson[0]]
        }

        return ["_sf_local", identifier.sourceString];
      },
      Expression6_apply: function(ns, fn_name, attrs, _1, argList, _2) {
        return [
          "_sf_apply", null, ns.asJson[0], fn_name.sourceString, attrs.asJson[0],
        ].concat(argList.asJson);
      },
      Expression6_aboveRef: function(_) {
        return ["_sf_local", "^"];
      },
      Expression6_hereRef: function(_) {
        return ["__sf_here"];
      },
      AttributeBlock: function(_1, _2, list, _3, _4) {
        return ["_sf_attrs", ...list.asJson];
      },
      AttributeBlockWithEllipsis: function(_1, _2, list, _3, _4) {
        var entries = [];
        var hasEllipsis = false;
        list.asJson.forEach(function(elem) {
          if (elem === "...") {
            if (hasEllipsis) {
              throw new Error("An attribute block may contain up to one ellipsis");
            }

            hasEllipsis = true;
            return;
          }

          entries.push(elem);
        });

        if (hasEllipsis) {
          return ["_sf_attrs_with_ellipsis", ...entries];
        } else {
          return ["_sf_attrs", ...entries];
        }
      },
      AttributeList: function(list) {
        return list.asJson;
      },
      AttributeEntry: function(name, _, value) {
        return [name.asJson, value.asJson];
      },
      InputDeclaration: function(_, name, kind) {
        return ["_named_placeholder", name.asJson, kind.asJson[1], kind.asJson[2]];
      },
      OutputDeclaration: function(_1, name, kind, _2, expr) {
        var rhsValue = expr.asJson[0];

        if (!rhsValue) {
          return ["__retval", name.asJson, ["_sf_local", name.asJson]];
        }

        if (rhsValue[0] !== ["_sf_local"]) {
          rhsValue = ["_sf_apply", null, "tf", "identity", null, rhsValue];
        }

        return ["__retval", name.asJson, rhsValue];
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
    throw new Error(m.message);
  }

  return semantics(m).asJson;
}

module.exports = parseExpressions;
