/* @flow */
"use strict";

const parseExpressions = require('./compile/parseExpressions.js');
const spawnProcess = require('./util/spawnProcess.js');


type expr = any[];
type PackageName = string;
type PackageSource = {language: string, content: string};
type PackageExpression = any[];
type PalletExpression = PackageExpression[];
type PackageSourceResolver = (PackageName) => Promise<PackageSource>;

class PackageParser {
  _resolutions: {[pkgName: PackageName]: Promise<PackageExpression>};
  _sourceResolver: PackageSourceResolver;

  constructor(sourceResolver: ?PackageSourceResolver) {
    if (sourceResolver) {
      this._sourceResolver = sourceResolver;
    } else {
      this._sourceResolver = (s) => {
        return Promise.reject(
            new Error(`Could not find package to import: ${s}`));
      };
    }
    this._resolutions = {};
  }

  parsePackageAndImports(name: string, source: string): Promise<PalletExpression> {
    return new Promise((resolve, reject) => {
      var pkg = this._doParsePackage(name, source);

      pkg
      .then((pkgExpr: PackageExpression) => {
        this._doResolvePackageImports(pkgExpr)
        .then(() => {
          Promise.all(Object.keys(this._resolutions).map((k) => { return this._resolutions[k]; }))
          .then((importedPkgExprs: PackageExpression[]) => {
            resolve(
              [
                ...importedPkgExprs,
                pkgExpr,
              ]
            );
          }, reject)
        }, reject);
      }, reject);
    });
  }

  // TODO(adamb) Detect and blow up if cycles in imports are present.
  _resolvePackage(pkgName: PackageName): Promise<PackageExpression> {
    var pkg = this._resolutions[pkgName];
    if (!pkg) {
      pkg = this._doResolvePackage(pkgName);
      this._resolutions[pkgName] = pkg;
    }

    return pkg;
  }

  _doResolvePackage(pkgName: PackageName): Promise<PackageExpression> {
    return new Promise((resolve, reject) => {
      this._sourceResolver(pkgName)
      .then((pkgSource: PackageSource) => {
        switch (pkgSource.language) {
        case "python":
          resolve(["_sf_python_package", pkgName, pkgSource.content]);
          return;
        case "nao":
          this._doParsePackage(pkgName, pkgSource.language)
          .then((pkgExpr: PackageExpression) => {
            this._doResolvePackageImports(pkgExpr)
            .then(() => resolve(pkgExpr), reject);
          }, reject);
          return;
        default:
          reject(new Error(`Unknown language for package ${pkgName}: ${pkgSource.language}`));
          return;
        }
      }, reject);
    });
  }

  _doResolvePackageImports(pkgExpr: PackageExpression): Promise<PackageExpression[]> {
    return Promise.all(
      this._enumerateImports(pkgExpr).map((importedPkgName) => {
        return this._resolvePackage(importedPkgName);
      })
    );
  }

  _doParsePackage(name: string, source: string): Promise<PackageExpression> {
    try {
      return Promise.resolve(["_sf_package", name, ...parseExpressions(source)]);
    } catch (e) {
      return Promise.reject(e);
    }
  }

  _enumerateImports(pkgExpr: PackageExpression): PackageName[] {
    if (pkgExpr[0] !== "_sf_package") {
      throw new Error("Not a package expression: " + JSON.stringify(pkgExpr));
    }

    var [_, name, ...topLevelExpressions] = pkgExpr;
    var importedPackageNames = [];
    topLevelExpressions.forEach(
      ([exprType, ...exprRest]) => {
        if (exprType !== "_sf_import") {
          return;
        }

        var imported = exprRest[0];
        console.log('exprRest', JSON.stringify(imported));

        imported.forEach(
          ([importName, packageName]) => {
            importedPackageNames.push(packageName.join("/"));
          }
        );
      }
    );

    return importedPackageNames;
  }
}

module.exports = {
  compileString: function(
      packageName: string,
      source: string,
      resolveImport: ?PackageSourceResolver): Promise<string> {
    return new Promise((resolve, reject) => {
      const pp = new PackageParser(resolveImport);
      pp.parsePackageAndImports(packageName, source)
      .then((expressions) => {
        resolve(
          spawnProcess.withStdinCapturingStdout(
            "../bin/python",
            [
              `${__dirname}/cli.py`,
              "--input-json", "/dev/stdin",
              "--output-metagraphdef", "/dev/stdout",
            ],
            JSON.stringify(expressions)
          )
        );
      }, reject);
    });
  },
  compile: function(
      packageName: string,
      source: string,
      resolveImport: ?PackageSourceResolver,
      output: string,
      binary: boolean): Promise<any> {
    return new Promise((resolve, reject) => {
      const pp = new PackageParser(resolveImport);
      pp.parsePackageAndImports(packageName, source)
      .then((expressions) => {
        spawnProcess.withStdin(
          "../bin/python",
          [
            `${__dirname}/cli.py`,
            "--input-json", "/dev/stdin",
            ...(output ? ["--output-metagraphdef", output] : []),
            ...(binary ? ["--output-binary"] : [])
          ],
          JSON.stringify(expressions),
          (err) => { err ? reject(err) : resolve(); }
        );
      }, reject);
    });
  }
};
