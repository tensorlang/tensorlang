# Tensorlang, a differentiable programming language based on TensorFlow

TensorFlow is an impressive runtime for doing machine learning research at massive scale. It has an interpreter, a JIT (called XLA), and an ahead of time compiler. It generates native code for processor architectures like ARM and amd64. It has fast kernels that run on the GPU and other specialized hardware. It runs on major operating including Linux, macOS, Windows, iOS and Android.

But the end-to-end usability (and in some cases, performance) of TensorFlow lags behind mainstream programming environments like Python. While there are a number of reasons for this, we believe a big one is that TensorFlow lacks a syntax of its own.

Tensorlang is an effort to fill this gap. Tensorlang defines a set of syntax and semantics that provide access to the full performance and expressiveness of the TensorFlow runtime and toolchain.

Once complete, Tensorlang will be a programming environment specially tuned for machine learning. This environment will at once be compatible with the existing machine learning ecosystem, and also be faster, more powerful, and more enjoyable to use than anything else out there.

**NOTE** During early development, Tensorlang was code-named Nao (a pun on the Chinese word for brain). There are still many places that use this name that haven't yet been migrated over.

## Why a new programming language?

Based on experience with the existing tools, Tensorlang is designed to address a number of requirements:

- Ability to saturate a single machine's local CPU and GPU with linear scaling
- Seamless scaling to clusters of machines
- Ability to compile programs to native code that runs **fast** on major operating systems and mobile devices
- Native support for symbolic differentiation
- Easy debugging and actual stacktraces for graph errors
- Execution model that matches other programming environments (e.g. no delayed execution)
- A productive REPL environment
- Compatibility with existing libraries and models

In order to do this, we'll need to improve the state of the art on a number of dimensions:
- Debugging
- Maintenance
- Composition (building larger systems out of smaller ones)
- Clarity

Under the hood, Tensorlang compiles programs directly to TensorFlow MetaGraphDefs

## Why not just use the existing TensorFlow Python API?
TensorFlow specializes in building computation graphs. These graphs can be quite large, and their execution can be spread across a large number of machines. Part of the trick to making this work is allowing expressions to be evaluated asynchronously with respect to each other. While the existing TensorFlow software packages provide an API for defining these expressions, they do not provide a syntax, high-level toolchain, or a productive development environment.

Tensorlang has a syntax appropriate for the sorts of data flow computations present in machine learning models today. It has support for templating, type inference, and symbolic differentiation.

## Why not compile an existing language like Python directly to TensorFlow?
Compiling a language like Python directly to TensorFlow requires one of two unfortunate compromises. Either:
1. Make Python parallel by default, but that would mean most existing Python programs wouldn't work. This reduces the benefit of using Python in the first place.
2. Give up the advantages of TensorFlow's parallel model. This would dramatically reduce the language's flexibility and scaling properties.

## So we need language semantics that are a bit different those present in mainstream languages. Why define a new syntax?
A syntax is a way to summon and manipulate specific concepts in a programming language. A good syntax strikes a balance between familarity to newcomers and appropriateness to the underlying semantics. Most of our syntax is very close to existing languages (particularly Go, JavaScript, and Python). We introduce a few new forms that are a particularly good fit for concepts present in many machine learning models.

For example, many papers in machine learning include diagrams depicting transformations applied to data. These diagrams look something like `f -> g -> h`. Writing these in a mainstream language's syntax inverts the order to `h(g(f))` which obscures the more natural way people prefer to talk about it. Embracing a new syntax means we can write expressions that read like the ideas they represent. In Tensorlang, we can write:

```
f -> g -> h
```

And have it compile down to `h(g(f))`. For more advanced transformations, we might want to include additional parameters:

```
f -> g(1.0, .) -> h
```

The above expression is compiles to `h(g(1.0, f))``

There's a multi-line form of this syntax, which uses the `^` character.

```
f
g(1.0, ^)  -- intermediate
h(^)
```

## Symbolic differentiation
Because these expressions compile directly to TensorFlow graphs, and TensorFlow supports symbolic differentiation, we get symbolic differentiation for free. The syntax for this is a little kludgy still, but this is a way to define a function as well as its symbolic gradient.

```
squareAndMore = func(x) { emit x * x + x }
squareAndMoreDx = grad[squareAndMore]

// squareAndMore(1.0) == 2.0
// squareAndMoreDx(1.0) == 3.0
```

## Training and function optimization
Since neural network is just a function composed of a many other functions, each with some internal state, we can use these these concepts to train networks! Rather than expecting a human to determine the internal weights of a network, we can discover acceptable values experimentally. This process of discovery is referred to as training. To train a function, we need some example input values and a way to determine how close the function's output is to an acceptable threshold. A function trainer uses symbolic differentiation along with rules about how to update hidden state of a function.

[Check out the example of a simple MNIST classifier.](root/src/demo/digits_nb.ipynb)

## Native loops
Loops are hard to write using TensorFlow's Python API. But it doesn't have to be that way.

Compare the Python API way:

```
i = tf.constant(0)
c = lambda i: tf.less(i, 10)
b = lambda i: tf.add(i, 1)
r = tf.while_loop(c, b, [i])
```

With our way:

Note that the // lines are comments indicating what the state of `r` would look like after loop execution.

```
r = for i = 5; foo = 1; i < 10 {
  emit foo = foo * i
  emit i = i + 1
}

// r:i == 10
// r:foo == 15120
```

## Native conditionals
Compare an `if/else` statement in the TensorFlow Python API:

```
x = tf.constant(2)
y = tf.constant(5)
def f1(): return x * 17
def f2(): return y + 23
r = tf.cond(tf.less(x, y), f1, f2)
```

With

```
x = 2
y = 5
if x < y {
  x * 17
} else {
  y + 23
}
```

## Functions
A function can take any number of tensors as input and generate any number of tensors as output.

Expressions within function bodies are evaluated lazily and asynchronously. The good news is not only is are computations automatically parallelized, but no compute is wasted calculating values you don't need. To make the most of these benefits, you'll need to adjust your thinking a bit about what's executed when.

```
func add3(x, y, z) {
  emit sum = x + y + z
  emit part = x + y
}

// r = add3(1, 2, 3)
// r:sum == 6
// r:part == 3
```

In the example above you'll notice a familiar looking function definition syntax. Instead of `return` we have `emit`, as the function can `emit` tensors with different names, but the function does not cease execution when these values are emitted.

## Attributes
Sometimes you'd like to introduce flexibility into a function's implementation based on information known at **compilation time**. In these cases, use attributes.

```
func increment[amount](x) {
  return amount + x
}

// increment[amount: 1](1) == 2
// incrementByTwo = increment[amount: 2]
// incrementByTwo(1) == 3
```

As you can see above, it's possible to define a new function by providing *just the attributes* of an existing function. While function inputs and outputs can **only** be tensors, attributes can be anything. Attributes are easy to spot because they're surrounded by `[]` in both function definition and function application. Function attributes must always be given in keyword form.

## Macros
Sometimes you'd like to work with higher-order functions. This is possible using macros.
```
func incrementerFactory[amount] {
  emit fn = func(x) {
    emit sum = amount + x
  }
}
```

As you can see above, the only difference between a function definition and a macro definition is the use of `()` to specify zero or more arguments. If the `()` are present in a definition, it's a function definition. If they're absent, it's a macro definition.

## Implementation Details
The current implementation of the Tensorlang compiler is written in Python 3. A future version of it may not depend on Python at all. It can generate graphs that are usable without Python. This is important for deploying in places like TensorFlow serving.

## Python Integration
If you'd like to use graphs generated by Tensorlang in a Python project, a project like [TFI](https://github.com/ajbouh/tfi) can provide one-line access to a model from a normal Python program. TFI handles mapping mimetypes like image/png to tensors of the proper shape and dtype.
