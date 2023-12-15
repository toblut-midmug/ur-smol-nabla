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
+$  diffable  $-([(list scalar) _recorder] [scalar _recorder])
::
::  ++  news
::    |=  [vs=@rd =band]
::    ^-  [(list scalar) ^band]
::    =/  nu  |=  [v=@rd b=^band]
::            ^-  [scalar ^band]
::            =/  r  ~(. recorder b)
::            =^  s  r  (new:r v)
::            [s band.r]
::    =^  ss  band  (spin vs band nu)
::    [ss band]
::
:: TODO: factor out ops and leave as a wrapper door only?
::
++  recorder  
  |_  =band
  ++  this  .
  ::
  ++  new  
    |=  [v=@rd]
    ^-  [scalar _this]
    :-  [val=v ind=(lent band)]  
    this(band (snoc band ~))
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
    |:  [seed=.~1.0]
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
:: Gradient and value of a function
:: 
++  grad-val
  |=  [f=diffable x=(list @rd)]
  ^-  [@rd (list @rd)]
  =/  nu  |=  [v=@rd b=band]
          ^-  [scalar band]
          =/  r  ~(. recorder b)
          =^  s  r  (new:r v)
          [s band.r]
  =/  bb  *band
  =^  ss  bb  (spin x bb nu)
  =/  r  ~(. recorder bb)
  =^  y  r  (f ss r)
  =/  dall  (backprop:r)
  =/  dx  (turn `(list scalar)`ss |=(=scalar (snag ind.scalar dall)))
  [val.y dx]
::
:: Gradient of a function
:: 
++  grad
  |=  [f=diffable x=(list @rd)]
  ^-  (list @rd)
  +:(grad-val f x)
--
    



