/+  nbl=nabla
::
|%
::  dot product for lists of @rd
::
++  dot-rd
  |=  [a=(list @rd) b=(list @rd)]
  ^-  @rd
  ?>  .=((lent a) (lent b))
  ?>  (gth (lent a) 0)
  =/  out  .~0.0
  |-
  ?:  |(.=(0 (lent a)) .=(0 (lent b)))
    out
  %=  $
    a  (snip a)
    b  (snip b)
    out  (add:rd out (mul:rd (rear a) (rear b)))
  ==
::  dot product for lists of $scalar
::
++  dot
  |=  [a=(list scalar:nbl) b=(list scalar:nbl) gg=grad-graph:nbl]
  ^-  [scalar:nbl grad-graph:nbl]
  ?>  .=((lent a) (lent b))
  =^  out-0  gg  (new:nbl .~0.0 gg)
  |-
  ?:  |(?=(~ a) ?=(~ b))
    [out-0 gg]
  =^  aibi  gg  (mul:nbl i.a i.b gg)
  =^  out-sum  gg  (add:nbl aibi out-0 gg)
  %=  $
    a  t.a
    b  t.b
    out-0  out-sum
    gg  gg
  ==
::
++  add-vec
  |=  [a=(list scalar:nbl) b=(list scalar:nbl) gg=grad-graph:nbl]
  ^-  [(list scalar:nbl) grad-graph:nbl]
  ?>  .=((lent a) (lent b))
  =|  out=(list scalar:nbl)
  |-
  ?:  |(?=(~ a) ?=(~ b))
    [(flop out) gg]
  =^  component-sum  gg  (add:nbl i.a i.b gg)
  %=  $
    a  t.a
    b  t.b
    out  [i=component-sum t=out]
    gg  gg
  ==
::
++  scale-vec
  |=  [lambda=scalar:nbl v=(list scalar:nbl) gg=grad-graph:nbl]
  ^-  [(list scalar:nbl) grad-graph:nbl]
  =|  out=(list scalar:nbl)
  |-
  ?:  ?=(~ v)
    [(flop out) gg]
  =^  component  gg  (mul:nbl lambda i.v gg)
  %=  $
    v  t.v
    out  [i=component t=out]
    gg  gg
  ==
::
++  add-vec-rd
  |=  [a=(list @rd) b=(list @rd)]
  ^-  [(list @rd)]
  ?>  .=((lent a) (lent b))
  =|  out=(list @rd)
  |-
  ?:  |(?=(~ a) ?=(~ b))
    (flop out)
  %=  $
    a  t.a
    b  t.b
    out  [i=(add:rd i.a i.b) t=out]
  ==
::
++  scale-vec-rd
  |=  [lambda=@rd v=(list @rd)]
  ^-  (list @rd)
  =|  out=(list @rd)
  |-
  ?:  ?=(~ v)
    (flop out)
  %=  $
    v  t.v
    out  [i=(mul:rd lambda i.v) t=out]
  ==
::
--
