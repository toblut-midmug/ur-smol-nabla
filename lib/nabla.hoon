|%
::  index of a node in the (topologically sorted) computational graph
::
+$  index  @
::  value and index of a node
::
+$  scalar  [val=@rd ind=index]
::  Same type as `scalar` but interpreted differently: Local partial 
::  derivative of some node w.r.t the input node pointed to by 
::  ind.dscalar. 
::
+$  dscalar  scalar 
::  local gradient of a node
::
+$  local-grad  (list dscalar)  
::  local gradients of all nodes in topological order
::
+$  band  (list local-grad)
::
:: TODO: factor out ops and leave as a wrapper door only?
::
++  recorder  
  |_  =band
  ++  this  .
  ::
  ++  new  
    |=  v=@rd
    ^-  [scalar _this]
    :-  [val=v ind=(lent band)]  
    this(band (snoc band ~))
  ::
  ++  news
    |=  vs=(list @rd)
    ^-  [(list scalar) _this]
    =/  scalars=(list scalar)  
      -:(spin vs (lent band) |=([v=@rd ind=index] [`scalar`[v ind] +(ind)]))
    :-  scalars
    %=  this
      band  (weld band `^band`(reap (lent vs) ~))
    ==
  ::
  ++  add 
    |=  [a=scalar b=scalar]
    ^-  [scalar _this]
    :-  [val=(add:rd val.a val.b) ind=(lent band)]
    %=  this  band
      %+  snoc  band
      ~[[val=.~1.0 ind=ind.a] [val=.~1.0 ind=ind.b]]
    ==
  ::
  ++  sub 
    |=  [a=scalar b=scalar]
    ^-  [scalar _this]
    :-  [val=(sub:rd val.a val.b) ind=(lent band)]
    %=  this  band
      %+  snoc  band
      ~[[val=.~1.0 ind=ind.a] [val=.~-1.0 ind=ind.b]]
    ==
  ::
  ++  mul 
    |=  [a=scalar b=scalar]
    ^-  [scalar _this]
    :-  [val=(mul:rd val.a val.b) ind=(lent band)]
    %=  this  band
      %+  snoc  band
      ~[[val=val.b ind=ind.a] [val=val.a ind=ind.b]]
    ==
  ::
  ++  div
    |=  [a=scalar b=scalar]
    ^-  [scalar _this]
    :-  [val=(div:rd val.a val.b) ind=(lent band)]
    %=  this  band
      %+  snoc  band
      :~
        [val=(div:rd .~1.0 val.b) ind=ind.a] 
        [val=(div:rd (mul:rd .~-1.0 val.a) (mul:rd val.b val.b)) ind=ind.b]
      ==
    ==
  :: Accumulates the entries of the gradient via backpropagation
  :: 
  ++  backprop
    |:  seed=.~1.0
    :: TODO: Maybe lest instead of list?
    ::
    ^-  (list @rd)
    =/  grads-acc  `(list @rd)`(snoc (reap (dec (lent band)) .~0.0) seed)
    =/  grads  `(list @rd)`~
    |-
    ?:  .=(1 (lent band))
      :: return the gradient in the same order as the entries in band
      (flop (snoc grads (rear grads-acc)))
    %=  $
      grads-acc  (backprop-step (rear band) (rear grads-acc) (snip grads-acc))
      grads  (snoc grads (rear grads-acc))
      band  (snip band)
    == 
  ::
  ++  backprop-step
    |=  [=local-grad seed=@rd gacc=(list @rd)]
    ^-  (list @rd)
    %+  reel  
      local-grad
    |:  [p=*dscalar acc=gacc]
    %^  snap  acc  ind.p 
    %+  add:rd 
      (mul:rd val.p seed) 
    (snag ind.p acc)
  --
::
+$  diffable  $-([(list scalar) _recorder] [scalar _recorder])
:: 
:: Gradient and value of a function
:: 
++  grad-val
  |=  f=diffable
  ^-  $-((list @rd) [@rd (list @rd)])
  |=  x=(list @rd)
  =/  r  ~(. recorder *band)
  =^  ss  r  (news:r x)
  =^  y  r  (f ss r)
  =/  dall  (backprop:r)
  =/  dx  (turn `(list scalar)`ss |=(=scalar (snag ind.scalar dall)))
  [val.y dx]
::
:: Gradient "operator".
:: Returns a function that computes the gradient of the input function f
:: w.r.t the latter's inputs.
::
++  grad
  |=  f=diffable
  ^-  $-((list @rd) (list @rd))
  |=  x=(list @rd)
  ^-  (list @rd)
  +:((grad-val f) x)
::
++  nn
  |%
  ++  linear 
    |=  [x=(list scalar) params=(list scalar) r=_recorder]
    ^-  [scalar _recorder]
    ?>  (gth (lent x) 0)
    ?>  .=(+((lent x)) (lent params))
    =/  weights  (snip params)
    =/  bias  (rear params)
    =^  out  r  (new:r .~0.0)
    |-  
    ?:  .=((lent x) 0)
      (add:r out bias)
    =^  xw  r  (mul:r (rear x) (rear weights))
    =^  out-new  r  (add:r out xw)
    %=  $
      x  (snip x)
      weights  (snip weights)
      out  out-new
      r  r
    ==
  --
--
    
