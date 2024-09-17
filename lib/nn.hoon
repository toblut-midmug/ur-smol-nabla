/+  *nabla
::
|%
::
+$  model  $-([(list scalar) (list scalar) grad-graph] [(list scalar) grad-graph])
::
+$  model-meta  $-(=model nparams=@ud)
::
+$  vec-fn  $-([(list scalar) grad-graph] [(list scalar) grad-graph])
::
++  bind-parameters
  |=  [m=model params=(list scalar)]
  ^-  vec-fn
  |=  [x=(list scalar) gg=grad-graph]
  ^-  [(list scalar) grad-graph]
  (m x params gg)
::
++  bind-inputs
  |=  [m=model x=(list scalar)]
  ^-  vec-fn
  |=  [params=(list scalar) gg=grad-graph]
  ^-  [(list scalar) grad-graph]
  (m x params gg)
::
++  linear 
  |=  [inputs=(list scalar) params=(list scalar) gg=grad-graph]
  ^-  [scalar grad-graph]
  ?>  (gth (lent inputs) 0)
  ?>  .=(+((lent inputs)) (lent params))
  =/  weights  (snip params)
  =/  bias  (rear params)
  =^  out  gg  (new .~0.0 gg)
  |-  
  ?:  .=((lent inputs) 0)  
    (add out bias gg)
  =^  xw  gg  (mul (rear inputs) (rear weights) gg)
  =^  out  gg  (add out xw gg)
  %=  $
    inputs  (snip inputs)
    weights  (snip weights)
    out  out
  ==
::
++  neuron
  |=  [inputs=(list scalar) params=(list scalar) gg=grad-graph]
  ^-  [scalar grad-graph]
  =^  out  gg  (linear inputs params gg)
  (relu out gg)
::
::  fully connected layer
::
++  layer
  |=  [nin=@ud nout=@ud nonlin=?]
  ^-  [model @ud]
  =/  nparams-neuron  +(nin)
  =/  nparams  (^mul nparams-neuron nout)
  :_  nparams
  |=  [inputs=(list scalar) params=(list scalar) gg=grad-graph]
  ^-  [(list scalar) grad-graph]
  ?>  .=((lent inputs) nin)
  ?>  .=((lent params) nparams)
  =|  outs=(list scalar)
  |-  
  ?~  params  
    [outs gg]
  =/  params-neuron  (scag nparams-neuron `(list scalar)`params) 
  =^  out  gg  (?:(nonlin neuron linear) inputs params-neuron gg)
  %=  $
    outs  (snoc outs out)
    params  (slag nparams-neuron `(list scalar)`params)
  ==
::
::  multilayer perceptron
::
++  mlp
  |=  [dims=(lest @ud)]
  ^-  [model @ud]
  ?>  (gte (lent dims) 2)
  =|  layers-meta=(list [model @ud])
  =.  layers-meta  
    |-
    ?:  ?=(~ t.dims)
      layers-meta
    ::  the final layer is linear
    ::
    =/  nonlin  ?!(.=((lent dims) 2))
    =/  layer-meta  (layer i.dims i.t.dims nonlin)
    %=  $
      dims  t.dims
      layers-meta  (snoc layers-meta layer-meta)
    ==
  =/  layers  (turn layers-meta head)
  ::  number of parameters of each layer
  ::
  =/  nparams-layers  (turn layers-meta tail)
  ::  total number of parameters
  ::
  =/  nparams  (reel nparams-layers ^add) 
  :_  nparams
  |=  [inputs=(list scalar) params=(list scalar) gg=grad-graph]
  ^-  [(list scalar) grad-graph]
  ?>  .=((lent inputs) i.dims)
  ?>  .=((lent params) nparams)
  =/  features  inputs
  |-
  ?:  ?=(~ layers-meta)
    [features gg]
  =/  nparams-layer  +:i.layers-meta
  =/  p  (scag nparams-layer params)
  =/  bound-layer  (bind-parameters -:i.layers-meta p)
  =^  features  gg  (bound-layer features gg)
  %=  $
    features  features
    layers-meta  t.layers-meta
    params  (slag nparams-layer params)
  ==
::    
--
