/+  *nabla
|%
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
  ?:  .=((lent x) 0)  (add out bias grad-graph.gt)
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
::  fully connected layer
::
++  layer
  |=  [nin=@ud nout=@ud]
  =/  nparams-neuron  +(nin)
  =/  nparams  (^mul nparams-neuron nout)
  :_  nparams
  |=  [x=(list scalar) params=(list scalar) gg=grad-graph]
  ^-  [(list scalar) grad-graph]
  ?>  .=((lent x) nin)
  ?>  .=((lent params) nparams)
  =/  outs=(list scalar)  ~
  |-  
  ?~  params  [outs gg]
  =^  out  gg  (neuron x (scag nparams-neuron `(list scalar)`params) gg)
  %=  $
    outs  (snoc outs out)
    params  (oust [0 nparams-neuron] `(list scalar)`params)
    gg  gg
  ==
--
