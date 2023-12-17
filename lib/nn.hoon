/+  *nabla
|%
::
++  linear 
  |=  [x=(list scalar) params=(list scalar) r=_grad-tracker]
  ^-  [scalar _grad-tracker]
  ?>  (gth (lent x) 0)
  ?>  .=(+((lent x)) (lent params))
  =/  weights  (snip params)
  =/  bias  (rear params)
  =^  out  r  (new:r .~0.0)
  |-  
  ?:  .=((lent x) 0)  (add:r out bias)
  =^  xw  r  (mul:r (rear x) (rear weights))
  =^  out-new  r  (add:r out xw)
  %=  $
    x  (snip x)
    weights  (snip weights)
    out  out-new
    r  r
  ==
::
++  neuron
  |=  [x=(list scalar) params=(list scalar) r=_grad-tracker]
  ^-  [scalar _grad-tracker]
  =^  out  r  (linear x params r)
  (relu:r out)
::
::  fully connected layer
::
++  layer
  |=  [nin=@ud nout=@ud]
  =/  nparams-neuron  +(nin)
  =/  nparams  (mul nparams-neuron nout)
  :_  nparams
  |=  [x=(list scalar) params=(list scalar) r=_grad-tracker]
  ^-  [(list scalar) _grad-tracker]
  ?>  .=((lent x) nin)
  ?>  .=((lent params) nparams)
  =/  outs=(list scalar)  ~
  |-  
  ?~  params  [outs r]
  =^  out  r  (neuron x (scag nparams-neuron `(list scalar)`params) r)
  %=  $
    outs  (snoc outs out)
    params  (oust [0 nparams-neuron] `(list scalar)`params)
    r  r
  ==
--
