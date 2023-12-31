|%
::  $index: ...of a node in the topologically sorted computational graph
::
+$  index  @
::  $scalar: @rd value and its $index
::
+$  scalar  [val=@rd ind=index]
::  $dscalar: partial derivative of a $scalar
::
::  Same type as $scalar but interpreted differently: Local partial 
::  derivative of some $scalar w.r.t its input $scalar pointed to by 
::  ind.dscalar. 
::
+$  dscalar  scalar 
::  $local-grad: local gradient of a $scalar
::
+$  local-grad  (list dscalar)  
::  $grad-graph: local gradients of all `$scalar`s in the graph in topological order
::
+$  grad-graph  (list local-grad)
::  $scalar-fn: a scalar-valued function. can be passed to ++grad and
::  ++grad-val
::
+$  scalar-fn  $-([(list scalar) grad-graph] [scalar grad-graph])
::
::  wraps a single @rd in a $scalar and appends it to a $grad-graph
::
++  new  
  |=  [v=@rd gg=grad-graph]
  ^-  [scalar grad-graph]
  :-  [val=v ind=(lent gg)]  
  (snoc gg ~)
:: 
:: wraps a list of @rd in `$scalar`s and appends them to a $grad-graph
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
++  sqt
  |=  [a=scalar gg=grad-graph]
  ^-  [scalar grad-graph]
  :-  [val=(sqt:rd val.a) ind=(lent gg)]
  %+  snoc  gg 
  ~[[val=(div:rd .~0.5 (sqt:rd val.a)) ind=ind.a]]
::  
:: Rectified linear unit. The gradient at zero is set to zero.
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
:: Accumulates the gradient of the last node in the $grad-graph
:: w.r.t all strongly connected preceding nodes via backpropagation
::
++  backprop
  |=  =grad-graph
  ^-  (list @rd)
  ?:  =(~ grad-graph)
    ~
  ::  initialize all gradients of the nodes before the last one to zero;
  ::  the gradient of the very last node is one.
  ::
  =/  seed=@rd  .~1.0
  =/  grads-acc  (snoc (reap (dec (lent grad-graph)) `(unit @rd)`~) (some seed))
  =/  grads  `(list @rd)`~
  |-
  ?:  .=(0 (lent grad-graph))
     :: return the gradient in the same order as the entries in grad-graph
     ::
     (flop grads)
  ::  if the accumulated gradient of the current node is ~, it is weakly
  ::  connected and is assigned a gradient of zero.
  ::
  ?~  (rear grads-acc)
    %=  $
      grads-acc  (snip grads-acc)
      grad-graph  (snip grad-graph)
      grads  (snoc grads .~0.0)
    == 
  ::  append the gradient of a strongly connected node to the gradient
  ::  list and backpropagate
  ::
  %=  $
    grads-acc  %^     backprop-step 
                   (rear grad-graph) 
                 (need (rear grads-acc))
               (snip grads-acc)
    grads  (snoc grads (need (rear grads-acc)))
    grad-graph  (snip grad-graph)
  == 
:: helper gate for ++backprop
::
++  backprop-step
  |=  [=local-grad seed=@rd gacc=(list (unit @rd))]
  ^-  (list (unit @rd))
  %+  reel  
    local-grad
  |:  [p=*dscalar acc=gacc]
  ^-  _acc
  %^    snap  
      acc  
    ind.p 
  ^-  (unit @rd)
  =/  target-node  (snag ind.p acc)
  ?~  target-node
    (some (mul:rd val.p seed))
  %-  some
  %+  add:rd 
    (mul:rd val.p seed) 
  (need target-node)
:: 
:: (gate that computes the value and gradient of f)
:: 
++  grad-val
  |=  f=scalar-fn
  ^-  $-((list @rd) [(list @rd) @rd])
  |=  x=(list @rd)
  =|  gg=grad-graph
  :: wrap the inputs in scalars and do the forward pass
  ::
  =^  ss  gg  (news x gg)
  =^  y  gg  (f ss gg)
  :: backpropagate to obtain the full gradient
  ::
  =/  dall  (backprop gg)
  :: collect the gradient corresponding to the input values only
  ::
  =/  dx  (turn `(list scalar)`ss |=(=scalar (snag ind.scalar dall)))
  [dx val.y]
::
:: (gate that computes the gradient of f)
::
++  grad
  |=  f=scalar-fn
  ^-  $-((list @rd) (list @rd))
  |=  x=(list @rd)
  ^-  (list @rd)
  -:((grad-val f) x)
::
:: wrapper door for a $grad-graph that reduces boilerplate when working
:: with elementary operations.
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
  ++  sqt  (wrap-unary ^sqt)
  ::  
  ++  relu  (wrap-unary ^relu)
  ::
  --
::
--

