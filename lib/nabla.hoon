|%
::  $index: ...of a node in the topologically sorted computational graph
::
+$  index  @
::  $scalar: wraps an @rd in an $index
::
+$  scalar  [val=@rd ind=index]
::  $dscalar: partial derivative of a $scalar
::
::  Same type as $scalar but interpreted differently: Local partial 
::  derivative of some $scalar w.r.t its input $scapar pointed to by 
::  ind.dscalar. 
::
+$  dscalar  scalar 
::  $local-grad: local gradient of a $scalar
::
+$  local-grad  (list dscalar)  
::  $grad-graph: local gradients of all `$scalar`s in the graph in topological order
::
+$  grad-graph  (list local-grad)
::
:: TODO: factor out ops and leave as a wrapper door only?
::
++  grad-tracker  
  |_  =grad-graph
  ++  this  .
  :: 
  :: wraps a single @rd in a $scalar and appends it to the graph
  ::
  ++  new  
    |=  v=@rd
    ^-  [scalar _this]
    :-  [val=v ind=(lent grad-graph)]  
    this(grad-graph (snoc grad-graph ~))
  :: 
  :: wraps a list of @rd in `$scalar`s and appends them to the graph
  ::
  ++  news
    |=  vs=(list @rd)
    ^-  [(list scalar) _this]
    =/  scalars=(list scalar)  
      -:(spin vs (lent grad-graph) |=([v=@rd ind=index] [`scalar`[v ind] +(ind)]))
    :-  scalars
    %=  this
      grad-graph  (weld grad-graph `^grad-graph`(reap (lent vs) ~))
    ==
  ::
  ++  add 
    |=  [a=scalar b=scalar]
    ^-  [scalar _this]
    :-  [val=(add:rd val.a val.b) ind=(lent grad-graph)]
    %=  this  grad-graph
      %+  snoc  grad-graph
      ~[[val=.~1.0 ind=ind.a] [val=.~1.0 ind=ind.b]]
    ==
  ::
  ++  sub 
    |=  [a=scalar b=scalar]
    ^-  [scalar _this]
    :-  [val=(sub:rd val.a val.b) ind=(lent grad-graph)]
    %=  this  grad-graph
      %+  snoc  grad-graph
      ~[[val=.~1.0 ind=ind.a] [val=.~-1.0 ind=ind.b]]
    ==
  ::
  ++  mul 
    |=  [a=scalar b=scalar]
    ^-  [scalar _this]
    :-  [val=(mul:rd val.a val.b) ind=(lent grad-graph)]
    %=  this  grad-graph
      %+  snoc  grad-graph
      ~[[val=val.b ind=ind.a] [val=val.a ind=ind.b]]
    ==
  ::
  ++  div
    |=  [a=scalar b=scalar]
    ^-  [scalar _this]
    :-  [val=(div:rd val.a val.b) ind=(lent grad-graph)]
    %=  this  grad-graph
      %+  snoc  grad-graph
      :~
        [val=(div:rd .~1.0 val.b) ind=ind.a] 
        [val=(div:rd (mul:rd .~-1.0 val.a) (mul:rd val.b val.b)) ind=ind.b]
      ==
    ==
  ::  
  :: Rectified linear unit. 
  :: The gradient at zero is set to zero
  :: 
  ++  relu
    |=  [a=scalar]
    ^-  [scalar _this]
    ?:  (gth:rd val.a .~0.0)
      :-  [val=val.a ind=(lent grad-graph)]
      this(grad-graph (snoc grad-graph ~[[val=.~1.0 ind=ind.a]]))
    :-  [val=.~0.0 ind=(lent grad-graph)]
    this(grad-graph (snoc grad-graph ~[[val=.~0.0 ind=ind.a]]))
  ::
  :: Accumulates the gradient of the last item in the
  :: computation graph via backpropagation
  :: 
  ++  backprop
    |:  seed=.~1.0
    :: TODO: Maybe lest instead of list?
    ::
    ^-  (list @rd)
    =/  grads-acc  `(list @rd)`(snoc (reap (dec (lent grad-graph)) .~0.0) seed)
    =/  grads  `(list @rd)`~
    |-
    ?:  .=(1 (lent grad-graph))
      :: return the gradient in the same order as the entries in grad-graph
      (flop (snoc grads (rear grads-acc)))
    %=  $
      grads-acc  (backprop-step (rear grad-graph) (rear grads-acc) (snip grads-acc))
      grads  (snoc grads (rear grads-acc))
      grad-graph  (snip grad-graph)
    == 
  :: helper gate for ++backprop
  ::
  ++  backprop-step
    |=  [=local-grad seed=@rd gacc=(list @rd)]
    ^-  (list @rd)
    %+  reel  local-grad
    |:  [p=*dscalar acc=gacc]
    %^  snap  acc  ind.p 
    %+  add:rd 
      (mul:rd val.p seed) 
    (snag ind.p acc)
  --
::  $scalar-fn: a scalar-valued function. can be passed to ++grad
::
+$  scalar-fn  $-([(list scalar) _grad-tracker] [scalar _grad-tracker])
:: 
:: (gate that computes the value and gradient of f)
:: 
++  grad-val
  |=  f=scalar-fn
  ^-  $-((list @rd) [@rd (list @rd)])
  |=  x=(list @rd)
  =/  r  ~(. grad-tracker *grad-graph)
  :: wrap the inputs in scalars and do the forward pass
  ::
  =^  ss  r  (news:r x)
  =^  y  r  (f ss r)
  :: backpropagate to obtain the full gradient
  ::
  =/  dall  (backprop:r)
  :: collect the gradient corresponding to the input values only
  ::
  =/  dx  (turn `(list scalar)`ss |=(=scalar (snag ind.scalar dall)))
  [val.y dx]
::
:: (gate that computes the gradient of f)
::
++  grad
  |=  f=scalar-fn
  ^-  $-((list @rd) (list @rd))
  |=  x=(list @rd)
  ^-  (list @rd)
  +:((grad-val f) x)
::
++  nn
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
--

