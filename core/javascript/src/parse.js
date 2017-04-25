/* @flow */

const ohm = require('ohm-js');
const grammarText = require('./nao.ohm');

// import ohm from 'ohm-js';
// import grammarText from './nao.ohm';

function loadGrammar() {
  return ohm.grammar(grammarText);
}

function applyExpr(fn, ...args: any[]): any[] {
  return [
    "_named_apply",
    null,
    fn,
    null,
    ...args,
  ];
}

function fullyQualifiedApply(pkgName, fnName, ...args) {
  return applyExpr(
    applyExpr(["_sf_package_lookup", pkgName], fnName),
    ...args
  );
}

function identityExpr(value) {
  return fullyQualifiedApply("tf", "identity", value);
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

    return fullyQualifiedApply("tf", method, e, acc);
  });
}

function processFunctionBodyExpr(upvalNames: string[], retvals: any[], isMacro: bool, expr) {
  if (expr[0] === "__retval") {
    // console.warn('processFunctionBodyExpr', JSON.stringify(expr));
    var [_, retName, retVal: any[]] = expr;
    var subName = expressionName(retVal);
    if (!subName) {
      subName = "retval" + retvals.length;
      retVal = rewriteExpressionWithName(subName, retVal);
    }

    retvals.push([retName, subName]);
    return retVal;
  }

  if (expr[0] === "_sf_after_leaves") {
    var [_, ...rest] = expr;
    return [
      expr[0],
      ...rest.map(processFunctionBodyExpr.bind(null, upvalNames, retvals, isMacro))
    ];
  }

  if (isMacro) {
    if (expr[0] === "_named_define_local") {
      expr[0] = "_named_define_attr";
    }
  }

  if (expr[0] === "_named_define_local") {
    upvalNames.push(expr[1]);
  }

  return expr;
}

function processFunctionBody(name, signature, body) {
  var retvals = [];
  var upvals = [];
  var [attributes, inputs] = signature;

  // If inputs is null, this is a macro.
  // Otherwise it's a boring old function.
  var isMacro = !inputs;

  var expressions = body.map(processFunctionBodyExpr.bind(null, upvals, retvals, isMacro));

  if (isMacro) {
    return ["_sf_macro", name, attributes, retvals, ...expressions];
  }

  return ["_sf_function", name, attributes, inputs, retvals,
      ...expressions];
}

function replaceHereExpression(expr: any[], callback: (any[]) => void) {
  if (expr instanceof Array) {
    switch (expr[0]) {
    case '__sf_here':
      callback(expr);
      break;
    case "_sf_local":
    case "_sf_package_lookup":
    case "_sf_attr":
    case "_sf_index":
    case "_sf_cond":
    case "_sf_function":
    case "_sf_macro":
      // Do not recurse within these.
      break;
    default:
      expr.forEach((e) => { replaceHereExpression(e, callback); });
    }
  }
}

function expressionName(expr: any[]): ?string {
  var exprType = expr[0];
  switch (exprType) {
  case "_sf_index":
  case "_sf_cond":
  case "__sf_here":
  case "list":
  case "_sf_while_loop":
  case "_sf_map":
  case "apply_attrs":
    return null;

  case "assert_type":
  case "assert_shape":
    return expressionName(expr[2]);

  case "_sf_after_leaves":
    if (expr.length == 1) {
      throw new Error("Encountered empty _sf_after_leaves: " + JSON.stringify(expr));
    }
    return expressionName(expr[expr.length - 1]);

  case "_sf_function":
  case "_sf_macro":
  case "_sf_local":
  case "_sf_attr":
  case "_named_tensor":
  case "_sf_macro":
  case "_named_apply":
    return expr[1];
  }

  throw new Error("Unhandled child expression type for expressionName: " + exprType);
}

function rewriteExpressionWithName(name: string, expr: any[]): any[] {
  var exprType = expr[0];
  switch (exprType) {
  case "_sf_local":
  case "_sf_attr":
  case "_sf_index":
  case "_sf_cond":
  case "__sf_here":
  case "list":
  case "_sf_while_loop":
  case "_sf_map":
  case "apply_attrs":
  case "_sf_function":
  case "_sf_macro":
    return ["_named_define_local", name, expr];

  case "assert_type":
  case "assert_shape":
    return rewriteExpressionWithName(name, expr[2]);

  case "_sf_after_leaves":
    return rewriteExpressionWithName(name, identityExpr(expr));

  case "_named_tensor":
  case "_named_apply":
  case "_named_apply_keywords":
    expr[1] = name;
    break;
  default:
    throw new Error("Unhandled child expression type for name rewrite: " + exprType);
  }

  return expr;
}

function rewriteExpressionWithShape(shape: any[], expr: any[]): any[] {
  var exprType = expr[0];
  switch (exprType) {
  case "_sf_local":
  case "_sf_attr":
  case "_sf_index":
  case "_sf_cond":
  case "__sf_here":
  case "list":
  case "_sf_while_loop":
  case "_sf_map":
  case "apply_attrs":
  case "_named_apply":
  case "assert_type":
  case "assert_shape":
    if (shape) {
      expr = ["assert_shape", shape, expr];
    }
    break;

  case "_sf_function":
    if (shape) {
      throw new Error("Can't rewrite a function to have shape: " + JSON.stringify(shape));
    }
    break;

  case "_sf_after_leaves":
    if (shape) {
      if (expr.length == 1) {
        throw new Error("Encountered empty _sf_after_leaves: " + JSON.stringify(expr));
      }
      expr[expr.length - 1] = rewriteExpressionWithShape(shape, expr[expr.length - 1]);
    }
    break;

  case "_named_define_local":
    if (shape) {
      expr = ["_named_define_local", expr[1], ["assert_shape", shape, expr[2]]];
    }
    break;

  case "_named_tensor":
    if (shape) {
      // console.warn("rewriting ", JSON.stringify(expr));
      expr[2] = shape;
      // console.warn("to ", JSON.stringify(expr));
    }
    break;
  default:
    throw new Error("Unhandled child expression type for shape rewrite: " + exprType);
  }

  return expr;
}

function rewriteExpressionWithType(type: any[], expr: any[]): any[] {
  var exprType = expr[0];
  switch (exprType) {
  case "_sf_local":
  case "_sf_attr":
  case "_sf_index":
  case "_sf_cond":
  case "__sf_here":
  case "list":
  case "_sf_while_loop":
  case "_sf_map":
  case "apply_attrs":
  case "_named_apply":
  case "assert_type":
  case "assert_shape":
  case "_named_define_local":
    if (type) {
      expr = ["assert_type", type, expr];
    }
    break;

  case "_sf_function":
    if (type) {
      throw new Error("Can't rewrite a function to have type: " + JSON.stringify(type));
    }
    break;

  case "_sf_after_leaves":
    if (type) {
      if (expr.length == 1) {
        throw new Error("Encountered empty _sf_after_leaves: " + JSON.stringify(expr));
      }
      expr[expr.length - 1] = rewriteExpressionWithType(type, expr[expr.length - 1]);
      break;
    }

  case "_named_tensor":
    if (type) {
      // console.warn("rewriting ", JSON.stringify(expr));
      expr[3] = type;
      // console.warn("to ", JSON.stringify(expr));
    }
    break;
  default:
    throw new Error("Unhandled child expression type for type rewrite: " + exprType);
  }

  return expr;
}

function rewriteExpressionWithKind(kind: any[], expr: any[]): any[] {
  return rewriteExpressionWithType(
    kind[2],
    rewriteExpressionWithShape(kind[1], expr));
}

function doIndex(target, index) {
  if (!index) {
    return target;
  }

  return ["_sf_index", target, index];
}

function doLookup(ns, identifier) {
  var result;

  if (ns) {
    result = applyExpr(["_sf_package_lookup", ns], identifier);
  } else {
    result = ["_sf_local", identifier];
  }

  return result;
}

function doAttrLookup(ns, identifier) {
  var result;

  if (ns) {
    result = applyExpr(["_sf_package_lookup", ns], identifier);
  } else {
    result = ["_sf_attr", identifier];
  }

  return result;
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
      TopLevelDecl: function(_1, child, _2) {
        return child.asJson;
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
      ImportSpec: function(packageName, importPath, importTag, _) {
        var name = packageName.asJson[0];
        if (!name) {
          var [path, scope] = importPath.asJson.split(":", 2);
          var pathFragments = (scope || path).split("/");
          name = pathFragments[pathFragments.length - 1];
        }
        // console.warn("name", name)
        // console.warn("path", path)
        // console.warn("scope", scope)
        return [name, importPath.asJson, importTag.asJson[0]];
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
      TensorKind: function(type, shape) { return ["kind", shape.asJson[0], type.asJson]; },
      TensorShape_literal: function(_1, dims, _2) {
        return ["shape", ...(dims.asJson || [])];
      },
      unknownDimension: function(_) {
        return null;
      },
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
        return ["list"].concat(elems.asJson);
      },

      PrimitiveLiteral_false: function(_) {
        return false;
      },
      PrimitiveLiteral_true: function(_) {
        return true;
      },
      PrimitiveLiteral_number: function(value) {
        return value.asJson;
      },
      PrimitiveLiteral_string: function(str) {
        return str.asJson;
      },
      TensorLiteralElement_arr: function(_1, elems, _2) {
        return ["list"].concat(elems.asJson);
      },
      TensorLiteral: function(child) {
        return ["_named_tensor", null, null, null, child.asJson];
      },
      FunctionLiteral: function(_, signature, block) {
        return processFunctionBody(null, signature.asJson, block.asJson);
      },
      FunctionDeclaration: function(_1, _2, name, signature, block) {
        return [
          "_named_define_attr", name.asJson,
          processFunctionBody(name.asJson, signature.asJson, block.asJson),
        ];
      },
      FunctionSignature: function(attributes, inputs) {
        return [attributes.asJson[0], inputs.asJson[0]];
      },
      FunctionParameter: function(name, type) {
        return [name.asJson, type.asJson[0]];
      },
      FunctionAttributeType: function(type, _1, minValue, _2, initialValue) {
        return null;
      },
      FunctionAttributes: function(_1, parameters, _2) {
        return (parameters.asJson || []).map(function(parameter) {
          // TODO(adamb) Actually use type in the future.
          // [attrName, attrType, attrInitialValue]

          var [name, type] = parameter;
          return [name, null, null];
        })
      },
      FunctionInputs: function(_1, parameters, _2) {
        var params = parameters.asJson || [];
        return params.map(function([name, kind]) {
          var shape = kind && kind[1];
          var type = kind && kind[2];
          return [name, shape, type];
        });
      },
      FunctionBlock: function(_1, body, _2, _3) {
        return body.asJson;
      },
      FunctionElement: function(_1, decl, _2) {
        return decl.asJson;
      },
      GraphElement: function(_1, decl, _2) {
        return decl.asJson;
      },
      AfterExpression: function(_1, _2, _3, _4, _5, body, _6) {
        return ["_sf_after_leaves", ...body.asJson];
      },
      VariableUpdate: function(name, _1, _2, rhs) {
        return ["_named_var_update", name.asJson, rhs.asJson];
      },
      VariableDeclaration: function(_1, _2, name, type, shape, _3, rhs) {
        return [
          "_named_var", name.asJson,
          shape.asJson, type.asJson[0],
          rewriteExpressionWithShape(shape.asJson,
              rewriteExpressionWithType(type.asJson[0],
                  rhs.asJson))
        ];
      },
      LetAssignment: function(_1, _2, name, _3, type, shape, _4, rhs) {
        return rewriteExpressionWithType(type.asJson[0],
          rewriteExpressionWithShape(shape.asJson[0],
              rewriteExpressionWithName(name.asJson, rhs.asJson)));
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
        // console.warn('ForExpression', JSON.stringify(initializers.asJson), JSON.stringify(body.asJson));
        var retvals = [];
        var upvalNames = [];
        var bodyExprs = body.asJson.map(processFunctionBodyExpr.bind(null, upvalNames, retvals, false));
        var upvals = [];
        // var upvals = upvalNames.map((name) => { return ["_sf_local", name]; });

        return [
          "_sf_while_loop", condition.asJson,
          bodyExprs, retvals,
          [...upvals, ...initializers.asJson],
        ];
      },
      ForInitializers: function(exprs, _2) {
        return exprs.asJson;
      },
      ForBody: function(_1, exprs, _2, _3) {
        return exprs.asJson;
      },
      ForBodyExpression: function(_1, expr, _2) {
        return expr.asJson
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

          // If the accumulated expression is foo(), it's a _named_apply.
          // Recurse down its expressions and replace any ["__sf_here"]
          // appropriately.
          if (e[0] !== "_named_apply") {
            return ["_named_apply", null, e, null, acc];
          } else {
            var previousName = "anon" + anonIncrement++;
            var firstReference = identityExpr(acc)
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
          "%": "mod",
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
        return doIndex(subexpr.asJson, indexSuffix.asJson[0]);
      },
      Expression6_reference: function(ns, identifier, indexSuffix, attrs) {
        var result = doIndex(
            doLookup(ns.asJson[0], identifier.asJson),
            indexSuffix.asJson[0]);

        if (attrs.asJson[0]) {
          return ["apply_attrs", result, attrs.asJson[0]];
        }
        return result;
      },
      Expression6_applyPos: function(ns, fn_name, indexSuffix, attrs, _1, argList, _2) {
        return [
          "_named_apply", null,
          doIndex(doLookup(ns.asJson[0], fn_name.asJson), indexSuffix.asJson[0]),
          attrs.asJson[0],
          ...(argList.asJson || [])];
      },
      Expression6_applyKwd: function(ns, fn_name, indexSuffix, attrs, _1, keywordArgs, _2) {
        return [
          "_named_apply_keywords", null,
          doLookup(ns.asJson[0], fn_name.asJson),
          attrs.asJson[0],
          keywordArgs.asJson];
      },
      KeywordArguments: function(args) {
        return ["_sf_map", ...args.asJson];
      },
      KeywordArgument: function(name, _1, _2, value) {
        return [name.asJson, value.asJson];
      },
      Expression6_aboveRef: function(_) {
        return ["_sf_local", "^"];
      },
      Expression6_hereRef: function(_) {
        return ["__sf_here"];
      },
      AttributeBlock: function(_1, _2, list, _3, _4) {
        return ["_sf_map", ...list.asJson];
      },
      AttributeBlockWithEllipsis: function(_1, _2, list, _3, _4) {
        var entries = [];
        var hasEllipsis = false;
        (list.asJson || []).forEach(function(elem) {
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
          return ["_sf_map", ["_ellipsis", true], ...entries];
        } else {
          return ["_sf_map", ...entries];
        }
      },
      AttributeList: function(list) {
        return list.asJson;
      },
      AttributeValueList: function(_1, list, _2) {
        return ["list", ...(list.asJson || [])];
      },
      AttributeReference: function(ns, identifier, attrs) {
        var result = doAttrLookup(ns.asJson[0], identifier.asJson);

        if (attrs.asJson[0]) {
          return ["apply_attrs", result, attrs.asJson[0]];
        }

        return result;
      },
      AttributeEntry: function(name, _, value) {
        return [name.asJson, value.asJson];
      },
      OutputDeclaration: function(_1, name, type, shape, _2, expr) {
        var rhsValue = expr.asJson[0];

        if (!rhsValue) {
          rhsValue = ["_sf_local", name.asJson];
        }

        rhsValue = rewriteExpressionWithType(type.asJson[0] && type.asJson[0][0],
            rewriteExpressionWithShape(shape.asJson[0] && shape.asJson[0][0],
                rhsValue));
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

// export default {
//   parseExpressions: parseExpressions
// };
module.exports = {
  parseExpressions: parseExpressions
};
