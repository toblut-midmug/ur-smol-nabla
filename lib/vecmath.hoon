/+  nbl=nabla
::
|%
::  dot product for lists of $scalar
::
++  dot
  |=  [a=(list scalar:nbl) b=(list scalar:nbl) gg=grad-graph:nbl]
  ^-  [scalar:nbl grad-graph:nbl]
  ?>  .=((lent a) (lent b))
  =^  out  gg  (new:nbl .~0.0 gg)
  |-
  ?:  |(?=(~ a) ?=(~ b))
    [out gg]
  =^  ab  gg  (mul:nbl i.a i.b gg)
  =^  out  gg  (add:nbl ab out gg)
  %=  $
    a  t.a
    b  t.b
    out  out
  ==
::
::  dot product for lists of @rd
::
++  dot-rd
  |=  [a=(list @rd) b=(list @rd)]
  ^-  @rd
  ?>  .=((lent a) (lent b))
  ?>  (gth (lent a) 0)
  =/  out  .~0.0
  |-
  ?:  |(?=(~ a) ?=(~ b))
    out
  %=  $
    a  t.a
    b  t.b
    out  (add:rd out (mul:rd i.a i.b))
  ==
::
++  add-vec
  |=  [a=(list scalar:nbl) b=(list scalar:nbl) gg=grad-graph:nbl]
  ^-  [(list scalar:nbl) grad-graph:nbl]
  ?>  .=((lent a) (lent b))
  =|  out=(list scalar:nbl)
  |-
  ?:  |(?=(~ a) ?=(~ b))
    [out gg]
  =^  component-sum  gg  (add:nbl i.a i.b gg)
  %=  $
    a  t.a
    b  t.b
    out  (snoc out component-sum)
  ==
::
++  add-vec-rd
  |=  [a=(list @rd) b=(list @rd)]
  ^-  [(list @rd)]
  ?>  .=((lent a) (lent b))
  =|  out=(list @rd)
  |-
  ?:  |(?=(~ a) ?=(~ b))
    out 
  %=  $
    a  t.a
    b  t.b
    out  (snoc out (add:rd i.a i.b))
  ==
::
++  scale-vec
  |=  [lambda=scalar:nbl v=(list scalar:nbl) gg=grad-graph:nbl]
  ^-  [(list scalar:nbl) grad-graph:nbl]
  %^  spin  v  
    gg  
  |=([a=scalar:nbl g=grad-graph:nbl] (mul:nbl a lambda g))
::
++  scale-vec-rd
  |=  [lambda=@rd v=(list @rd)]
  ^-  (list @rd)
  (turn v (cury mul:rd lambda))
::
--
