/+  *nabla
::
|%
::
+$  model  $-([(list scalar) (list scalar) grad-graph] [(list scalar) grad-graph])
::
+$  bound-model  $-([(list scalar) grad-graph] [(list scalar) grad-graph])
::
++  bind-parameters
  |=  [m=model params=(list scalar)]
  ^-  bound-model
  |=  [x=(list scalar) gg=grad-graph]
  ^-  [(list scalar) grad-graph]
  (m x params gg)
::
++  bind-inputs
  |=  [m=model x=(list scalar)]
  ^-  bound-model
  |=  [params=(list scalar) gg=grad-graph]
  ^-  [(list scalar) grad-graph]
  (m x params gg)
::
++  linear 
  |=  [x=(list scalar) params=(list scalar) gg=grad-graph]
  ^-  [scalar grad-graph]
  ?>  (gth (lent x) 0)
  ?>  .=(+((lent x)) (lent params))
  =/  weights  (snip params)
  =/  bias  (rear params)
  =/  gt  ~(. grad-tracker gg)
  =^  out  gt  (new:gt .~0.0)
  |-  
  ?:  .=((lent x) 0)  
    (add out bias grad-graph.gt)
  =^  xw  gt  (mul:gt (rear x) (rear weights))
  =^  out-new  gt  (add:gt out xw)
  %=  $
    x  (snip x)
    weights  (snip weights)
    out  out-new
    gt  gt
  ==
::
++  neuron
  |=  [x=(list scalar) params=(list scalar) gg=grad-graph]
  ^-  [scalar grad-graph]
  =^  out  gg  (linear x params gg)
  (relu out gg)
::
::  build a fully connected layer
::
++  layer
  |=  [nin=@ud nout=@ud]
  ^-  [model @ud]
  =/  nparams-neuron  +(nin)
  =/  nparams  (^mul nparams-neuron nout)
  :_  nparams
  |=  [x=(list scalar) params=(list scalar) gg=grad-graph]
  ^-  [(list scalar) grad-graph]
  ?>  .=((lent x) nin)
  ?>  .=((lent params) nparams)
  =/  outs=(list scalar)  ~
  |-  
  ?~  params  
    [outs gg]
  =^  out  gg  (neuron x (scag nparams-neuron `(list scalar)`params) gg)
  %=  $
    outs  (snoc outs out)
    params  (slag nparams-neuron `(list scalar)`params)
    gg  gg
  ==
::  build a multilayer perceptron
::
++  mlp
  |=  [dims=(list @ud)]
  ^-  [model @ud]
  ?>  (gte (lent dims) 2)
  =|  layers=(list model) 
  ?:  ?=(~ dims)  
    !!
  =|  layers-meta=(list [model @ud])
  =.  layers-meta  
    |-
    ?:  ?=(~ t.dims)
      layers-meta
    %=  $
      dims  t.dims
      layers-meta  (snoc layers-meta (layer [i.dims i.t.dims]))
    ==
  =/  layers  (turn layers-meta head)
  ::  number of parameters of each layer
  ::
  =/  nparams-layers  (turn layers-meta tail)
  ::  total number of parameters
  ::
  =/  nparams  (reel nparams-layers ^add) 
  :_  nparams
  ::  the "forward pass", roughly speaking
  ::
  |=  [x=(list scalar) params=(list scalar) gg=grad-graph]
  ^-  [(list scalar) grad-graph]
  ?>  .=((lent x) i.dims)
  ?>  .=((lent params) nparams)
  =|  bound-layers=(list bound-model)
  =.  bound-layers  
    |-
    ?:  ?=(~ layers)
      bound-layers
    ?~  nparams-layers  !!
    =/  p  (scag i.nparams-layers params)
    =/  bound-layer  (bind-parameters i.layers p)
    %=  $
      layers  t.layers
      nparams-layers  t.nparams-layers
      params  (slag i.nparams-layers params)
      bound-layers  (snoc bound-layers bound-layer)
    ==
  %+  roll  
    bound-layers
  |:  [l=*bound-model acc=[x gg]]
  ^-  _acc
  (l -.acc +.acc)
::    
--
