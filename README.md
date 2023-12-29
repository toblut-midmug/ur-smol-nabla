# ur-smol-nabla

Minimal reverse-mode automatic differentiation ("backpropagation") in Hoon in the spirit of
[micrograd](https://github.com/karpathy/micrograd). Inspired by [backprop](https://backprop.jle.im/) and [torch.func](https://pytorch.org/docs/stable/func.html).

### Description
To simplify things, `ur-smol-nabla` works with scalar values only. A `$scalar` consists of an `@rd` (double-precision floating-point) and an index in the which points to a location the computation graph. Gradients are tracked in a `$grad-graph`.

```
=nabla -build-file %/lib/nabla/hoon
```

```hoon
=|  gg=grad-graph:nabla
=^  x1  gg  (new:nabla .~2.0 gg)
=^  x2  gg  (new:nabla .~3.0 gg)
=^  out  gg  (mul:nabla x1 x2 gg)
[out gg]
```


