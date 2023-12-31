# ur-smol-nabla

A minimal implementation of reverse-mode automatic differentiation ("backpropagation") in Hoon in the spirit of
[micrograd](https://github.com/karpathy/micrograd). Inspired by [backprop](https://backprop.jle.im/) and [torch.func](https://pytorch.org/docs/stable/func.html). A small deep learning framework on top of it is work in progress ...

### Overview
To simplify things, `ur-smol-nabla` works with scalar values only. A `$scalar` consists of a `@rd` (double-precision floating-point) value and an index for a location in a topologically sorted computational graph. Gradients are tracked in a `$grad-graph`. Each element of a `$grad-graph` is the local gradient (i.e. a list of partial derivatives w.r.t "input" `$scalar`s) of a corresponding `$scalar`. The (intermediate) results of computations are only stored in `$scalar`s which allows for computations to be performed in a somewhat "dynamic" fashion, as illustrated below.

### Example

To get started, copy the contents of `/lib` into the `/lib` folder of a development ships's `%base` desk. The autograd functionality is contained in `/lib/nabla.hoon` and can be made accessible in the dojo via

```
=nabla -build-file %/lib/nabla/hoon
```

The following evaluates the expression $x^2 + y^2$ at $x=3$, $y=-4$ and computes the gradients: 
```hoon
> =|  gg=grad-graph:nabla
  =^  x  gg  (new:nabla .~3.0 gg)
  =^  y  gg  (new:nabla .~-4.0 gg)
  =^  xsq  gg  (mul:nabla x x gg)
  =^  ysq  gg  (mul:nabla y y gg)
  =^  out  gg  (add:nabla xsq ysq gg)
  [out (backprop:nabla gg)]
[[val=.~25 ind=4] ~[.~6 .~-8 .~1 .~1 .~1]]
```
Note that each operation takes a `$grad-graph` as argument and in turn produces an updated `$grad-graph` (the gradient of the result gets appended) along with its result. The general pattern here is to use the `=^` rune to pin a face to the result and "update" the `$grad-graph`. The entries of the gradient `~[.~6 .~-8 .~1 .~1 .~1]` correspond to `x`, `y`, `xsq`, `ysq` and `out`, respectively.


Some more examples can be found in `/test/autograd.hoon`. When copied to the `%base` desk of a ship, the former can also be run via
```
-test %/tests/autograd ~
```

### The gradient of a gate is another gate
Similar to [torch.func](https://pytorch.org/docs/stable/func.html) and [JAX](https://github.com/google/jax?tab=readme-ov-file#transformations), there is an interface for computing gradient functions. Functions from $\mathbb{R}^n \to \mathbb{R}$ are represented by `$scalar-fn` which is a gate that takes a sample of `[(list scalar) grad-graph]` and produces a `[scalar grad-graph]`. A `$scalar-fn` can be passed to `++grad` which produces a gate that computes the gradient w.r.t the inputs of `$scalar-fn`. The gate produced by `++grad-val` additionaly produces the value of the original `$scalar-fn`.
```hoon
> =f |=  [xs=(list scalar:nabla) gg=grad-graph:nabla]
  ^-  [(list scalar:nabla) grad-graph:nabla]
  =/  x  (snag 0 xs)
  =/  y  (snag 1 xs)
  =^  xsq  gg  (mul:nabla x x gg)
  =^  ysq  gg  (mul:nabla y y gg)
  (add:nabla xsq ysq gg)

> =fprime (grad:nabla f)

> (fprime ~[.~3.0 .~-4.0])
~[.~6 .~-8]

> (fprime ~[.~1.0 .~2.0])
~[.~2 .~4]

> =fprime-f (grad-val:nabla f)

> (fprime-f ~[.~3.0 .~-4.0])
[~[.~6 .~-8] .~25]

> (fprime-f ~[.~1.0 .~2.0])
[~[.~2 .~4] .~5]
```


### Issues 
*  **Boilerplate:** A `$grad-graph` needs to be explicitly passed around for each operation. Something of the sort is probably the irreducible cost of doing business in a purely functional language but there might be a way of doing things in a less cumbersome fashion. Perhaps someone has something smart to say about this. In any case, at least a little code per operation can be removed by wrapping a `$grad-graph` and the elementary operations in a door - somewhat similar to how gall agents work. See `++grad-tracker` in `nabla.hoon` for an implementation of this idea. The example
    ```hoon
    =|  gg=grad-graph:nabla
    =^  x  gg  (new:nabla .~3.0 gg)
    =^  y  gg  (new:nabla .~-4.0 gg)
    =^  xsq  gg  (mul:nabla x x gg)
    =^  ysq  gg  (mul:nabla y y gg)
    =^  out  gg  (add:nabla xsq ysq gg)
    [out (backprop:nabla gg)]
    ```
    is equivalent to
    ```
    =/  gt  ~(. grad-tracker:nabla *grad-graph:nabla)
    =^  x  gt  (new:gt .~3.0)
    =^  y  gt  (new:gt .~-4.0)
    =^  xsq  gt  (mul:gt x x)
    =^  ysq  gt  (mul:gt y y)
    =^  out  gt  (add:gt xsq ysq)
    [out (backprop:nabla grad-graph.gt)]
    ```

*  **No higher order derivatives:** Once a `$scalar-fn` has been passed to `++grad` or `++grad-val` the music stops, since the resulting gates only take `(list @rd)` as inputs: They cannot be passed to `++grad` again. In principle this should be possible to implement (`torch.func` an `JAX` can do it after all) but I haven't really thought about how to do it.


