# ur-smol-nabla

A minimal autograd + small deep learning library in Hoon in the spirit of [micrograd](https://github.com/karpathy/micrograd). 

### Train a neural net 
`/gen/moons-demo.hoon` trains a 65 parameters neural net for 2d binary classification on a small dataset. Copy it along with the dependencies in `/lib/` into the respective directories of your ship's `%base` desk. In the dojo:

```
> |commit %base
> +moons-demo
```

The model is trained for 100 epochs which takes around half an hour on an M3 MacBook Air. Very slow 🙂.

### `/lib/nabla.hoon` under the hood

To simplify things, `ur-smol-nabla` works with scalar values only. A `$scalar` consists of a `@rd` (double-precision floating-point) value and an index of the corresponding node in a topologically sorted computational graph. Gradients are tracked in a `$grad-graph` while (intermediate) results of computations are only stored in `$scalar`s which allows for the graph to be built in a somewhat "dynamic" fashion, as illustrated below.

To play around with the autograd engine in the dojo, put `/lib/nabla.hoon` into the `/lib` folder of the `%base` desk, `|commit %base` and build it via:
```
=nabla -build-file %/lib/nabla/hoon
```

The following code evaluates the expression $x^2 + y^2$ at $x=3$, $y=-4$ and computes the gradients: 
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
The entries of the gradient `~[.~6 .~-8 .~1 .~1 .~1]` correspond to `x`, `y`, `xsq`, `ysq` and `out`, respectively.
Each operation takes a `$grad-graph` as part of its sample and produces a cell of the resulting `$scalar` and the updated `$grad-graph`. The general pattern here is to use the `=^` rune to pin a face to the result and update the `$grad-graph`. 

Some more examples can be found in `/tests/autograd.hoon` which can be run from the `%base` desk of a ship via
```
-test %/tests/autograd ~
```

Similar to [torch.func](https://pytorch.org/docs/stable/func.html) and [JAX](https://github.com/google/jax?tab=readme-ov-file#transformations), there is an interface for computing gradient functions. A `$scalar-fn` is a gate that takes a `[(list scalar) grad-graph]` sample and produces a `[scalar grad-graph]`. A `$scalar-fn` can be passed to `++grad` which produces a gate that computes the gradient w.r.t. the inputs of `$scalar-fn`. The gate produced by `++grad-val` additionally produces the value of the original `$scalar-fn`.

```hoon
> =f |=  [xs=(list scalar:nabla) gg=grad-graph:nabla]
  ^-  [scalar:nabla grad-graph:nabla]
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


### Limitations and Issues
*  **Boilerplate:** A `$grad-graph` needs to be explicitly passed around for each operation. Something of the sort might be unavoidable in a purely functional language but maybe there is a way of doing things in a less cumbersome fashion. Perhaps someone has something smart to say about this. In any case, at least a little code per operation can be removed by wrapping a `$grad-graph` and the elementary operations in a door - somewhat similar to how gall agents work. See `++grad-tracker` in `nabla.hoon` for an implementation of this idea. The example
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
    ```hoon
    =/  gt  ~(. grad-tracker:nabla *grad-graph:nabla)
    =^  x  gt  (new:gt .~3.0)
    =^  y  gt  (new:gt .~-4.0)
    =^  xsq  gt  (mul:gt x x)
    =^  ysq  gt  (mul:gt y y)
    =^  out  gt  (add:gt xsq ysq)
    [out (backprop:nabla grad-graph.gt)]
    ```

*  **No higher order derivatives:** Once a `$scalar-fn` has been passed to `++grad` or `++grad-val`, the music stops since the resulting gate has a `(list @rd)` sample and produces a `(list @rd)`. It therefore cannot be passed to `++grad` again. In principle, higher-order derivatives can probably be implemented somewhat straightforwardly (`torch.func` an `JAX` can do it after all) but I haven't really thought about how to do it.


