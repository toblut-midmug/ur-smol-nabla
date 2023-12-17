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
::  $scalar-fn: a scalar-valued function. can be passed to ++grad
::
+$  scalar-fn  $-([(list scalar) _grad-tracker] [scalar _grad-tracker])
:: wraps a single @rd in a $scalar and appends it to the graph
::
++  new  
  |=  [v=@rd gg=grad-graph]
  ^-  [scalar grad-graph]
  :-  [val=v ind=(lent gg)]  
  (snoc gg ~)
:: 
:: wraps a list of @rd in `$scalar`s and appends them to the graph
::
++  news
  |=  [vs=(list @rd) gg=grad-graph]
  ^-  [(list scalar) grad-graph]
  (spin vs gg new)
::
++  add 
  |=  [a=scalar b=scalar gg=grad-graph]
  ^-  [scalar grad-graph]
  :-  [val=(add:rd val.a val.b) ind=(lent gg)]
  %+  snoc  gg
  ~[[val=.~1.0 ind=ind.a] [val=.~1.0 ind=ind.b]]
::
++  sub 
  |=  [a=scalar b=scalar gg=grad-graph]
  ^-  [scalar grad-graph]
  :-  [val=(sub:rd val.a val.b) ind=(lent gg)]
  %+  snoc  gg
  ~[[val=.~1.0 ind=ind.a] [val=.~-1.0 ind=ind.b]]
::
++  mul 
  |=  [a=scalar b=scalar gg=grad-graph]
  ^-  [scalar grad-graph]
  :-  [val=(mul:rd val.a val.b) ind=(lent gg)]
  %+  snoc  gg
  ~[[val=val.b ind=ind.a] [val=val.a ind=ind.b]]
::
++  div
  |=  [a=scalar b=scalar gg=grad-graph]
  ^-  [scalar grad-graph]
  :-  [val=(div:rd val.a val.b) ind=(lent gg)]
  %+  snoc  gg
  :~
    [val=(div:rd .~1.0 val.b) ind=ind.a] 
    [val=(div:rd (mul:rd .~-1.0 val.a) (mul:rd val.b val.b)) ind=ind.b]
  ==
::  
:: Rectified linear unit. 
:: The gradient at zero is set to zero
:: 
++  relu
  |=  [a=scalar gg=grad-graph]
  ^-  [scalar grad-graph]
  ?:  (gth:rd val.a .~0.0)
    :-  [val=val.a ind=(lent gg)]
    (snoc gg ~[[val=.~1.0 ind=ind.a]])
  :-  [val=.~0.0 ind=(lent gg)]
  (snoc gg ~[[val=.~0.0 ind=ind.a]])
:: 
:: wrapper door for a $grad-graph that reduces boilerplate when working
:: with elementary operations.
::
:: Accumulates the gradient of the last item in the
:: computation graph via backpropagation
:: 
++  backprop
  |=  =grad-graph
  :: TODO: Maybe lest instead of list?
  ::
  ^-  (list @rd)
  =/  seed=@rd  .~1.0
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
  =/  dall  (backprop grad-graph.r)
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
++  grad-tracker  
  |_  =grad-graph
  ++  this  .
  :: 
  ++  wrap-unary
    |=  op=$-([scalar ^grad-graph] [scalar ^grad-graph])
    ^-  $-(scalar [scalar _this])
    |=  [a=scalar]
    ^-  [scalar _this]
    =^  out  grad-graph  (op a grad-graph)
    [out this(grad-graph grad-graph)]
  ::
  ++  wrap-binary
    |=  op=$-([scalar scalar ^grad-graph] [scalar ^grad-graph])
    ^-  $-([scalar scalar] [scalar _this])
    |=  [a=scalar b=scalar]
    ^-  [scalar _this]
    =^  out  grad-graph  (op a b grad-graph)
    [out this(grad-graph grad-graph)]
  ::
  ++  new  
    |=  v=@rd
    ^-  [scalar _this]
    =^  s  grad-graph  (^new v grad-graph)
    [s this(grad-graph grad-graph)]
  ::
  ++  news
    |=  vs=(list @rd)
    ^-  [(list scalar) _this]
    =^  s  grad-graph  (^news vs grad-graph)
    [s this(grad-graph grad-graph)]
  ::
  ++  add  (wrap-binary ^add)
  ::
  ++  sub  (wrap-binary ^sub)
  ::
  ++  mul  (wrap-binary ^mul)
  ::
  ++  div  (wrap-binary ^div)
  ::  
  ++  relu  (wrap-unary ^relu)
  ::
  --
::
--

