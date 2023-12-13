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
::  local gradients in topological order
::
+$  band  (list local-grad)
::
++  new  
  |=  [v=@rd g=band]
  ^-  [scalar band]
  :-  [val=v ind=(lent g)]  
  (snoc g ~)
::
++  add 
  |=  [a=scalar b=scalar g=band]
  ^-  [scalar band]
  :-  [val=(add:rd val.a val.b) ind=(lent g)]
  %+  snoc  g
  ~[[val=.~1.0 ind=ind.a] [val=.~1.0 ind=ind.b]]
::
++  sub 
  |=  [a=scalar b=scalar g=band]
  ^-  [scalar band]
  :-  [val=(sub:rd val.a val.b) ind=(lent g)]
  %+  snoc  g
  ~[[val=.~1.0 ind=ind.a] [val=.~-1.0 ind=ind.b]]
::
++  mul 
  |=  [a=scalar b=scalar g=band]
  ^-  [scalar band]
  :-  [val=(mul:rd val.a val.b) ind=(lent g)]
  %+  snoc  g
  ~[[val=val.b ind=ind.a] [val=val.a ind=ind.b]]
::
++  div
  |=  [a=scalar b=scalar g=band]
  ^-  [scalar band]
  :-  [val=(div:rd val.a val.b) ind=(lent g)]
  %+  snoc  g
  :~
    [val=(div:rd .~1.0 val.b) ind=ind.a] 
    [val=(div:rd (mul:rd .~-1.0 val.a) (mul:rd val.b val.b)) ind=ind.b]
  ==
:: Accumulates the entries of the gradient via backpropagation and
:: returns them in reverse order i.e. the seed is the first entry in the list.
:: 
++  backprop
  |=  [g=band seed=@rd]
  :: TODO: Maybe lest instead of list?
  ::
  ^-  (list @rd)
  =/  grads-acc  `(list @rd)`(snoc (reap (dec (lent g)) .~0.0) seed)
  =/  grads  `(list @rd)`~
  |-
  ?:  .=(1 (lent g))
    (snoc grads (rear grads-acc))
  %=  $
    grads-acc  (backprop-step (rear g) (rear grads-acc) (snip grads-acc))
    grads  (snoc grads (rear grads-acc))
    g  (snip g)
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
    



