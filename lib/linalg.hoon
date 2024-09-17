
/+  nbl=nabla
::
|%
::  $scalar-fn: a scalar-valued function. can be passed to ++grad and
::  ++grad-val
::
+$  scalar-fn  $-([(list scalar) grad-graph] [scalar grad-graph])
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
  |=  [a=(list scalar:usn) b=(list scalar:usn) gg=grad-graph:usn]
  ^-  [scalar:usn grad-graph:usn]
  ?>  .=((lent a) (lent b))
  =^  out-0  gg  (new:usn .~0.0 gg)
  |-
  ?:  |(?=(~ a) ?=(~ b))
    [out-0 gg]
  =^  aibi  gg  (mul:usn i.a i.b gg)
  =^  out-sum  gg  (add:usn aibi out-0 gg)
  %=  $
    a  t.a
    b  t.b
    out-0  out-sum
    gg  gg
  ==
::
++  l2-norm
  |=  [a=(list scalar:usn) gg=grad-graph:usn]
  ^-  [scalar:usn grad-graph:usn]
  =^  out  gg  (dot-scalars a a gg)
  (sqt:usn out gg)
::
++  add-vec
  |=  [a=(list scalar:usn) b=(list scalar:usn) gg=grad-graph:usn]
  ^-  [(list scalar:usn) grad-graph:usn]
  ?>  .=((lent a) (lent b))
  =|  out=(list scalar:usn)
  |-
  ?:  |(?=(~ a) ?=(~ b))
    [(flop out) gg]
  =^  component-sum  gg  (add:usn i.a i.b gg)
  %=  $
    a  t.a
    b  t.b
    out  [i=component-sum t=out]
    gg  gg
  ==
::
++  scale-vec
  |=  [lambda=scalar:usn v=(list scalar:usn) gg=grad-graph:usn]
  ^-  [(list scalar:usn) grad-graph:usn]
  =|  out=(list scalar:usn)
  |-
  ?:  ?=(~ v)
    [(flop out) gg]
  =^  component  gg  (mul:usn lambda i.v gg)
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
