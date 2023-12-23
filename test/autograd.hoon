/+  *test
/+  usn=nabla
|%
::
++  test-weakly-connected-nan
  %-  expect
  !>
  .=  ~[.~20.0]
  %.  ~[.~10.0]
  %-  grad:usn
  |=  [x=(list scalar:usn) r=grad-graph:usn]
  ^-  [scalar:usn grad-graph:usn]
  ?~  x  !!
  =^  nanval  r  (new:usn .~nan r)
  =^  dummy  r  (mul:usn i.x nanval r)
  =^  out  r  (mul:usn i.x i.x r)
  [out r]
::
++  test-sqt-of-square
  %-  expect
  !>
  %+  close-enuf
    ~[.~-1.0]
  %.  ~[.~-12.3] 
  %-  grad:usn
  |=  [x=(list scalar:usn) r=grad-graph:usn]  
  ^-  [scalar:usn grad-graph:usn]  
  ?~  x  !!  
  =^  out  r  (mul:usn i.x i.x r) 
  =^  out  r  (sqt:usn out r) 
  [out r]
::
:: absolute value
::
++  abs
  |=  a=@rd
  ^-  @rd
  ?:  (gte:rd a .~0.0)
    a
  (sub:rd .~0.0 a)
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
++  dot-scalars
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
::  close enough ...
::
++  close-enuf
  |=  [a=(list @rd) b=(list @rd)]
  ^-  ?
  ?>  .=((lent a) (lent b))
  =/  atol=@rd  .~1e-12
  |-
  ?:  |(?=(~ a) ?=(~ b))
    %.y
  ?:  (gth:rd (abs (sub:rd i.a i.b)) atol)
    %.n
  %=  $
    a  t.a
    b  t.b
  ==
::
++  test-dot-product
  %-  expect 
  !>
  =/  vs
    :~  .~-4.68699287
        .~-2.20431279
        .~2.56718151
        .~1.52540774
        .~-4.07480633
        .~2.67589856
    ==
  =/  us
    :~  .~-6.8699287
        .~2.0431279
        .~5.6718151
        .~5.2540774
        .~0.7480633
        .~6.7589856
    ==
  =/  gg  *grad-graph:usn
  =^  ss  gg  (news:usn vs gg)
  =^  rs  gg  (news:usn us gg)
  .=  (dot-rd us vs) 
  val.-:(dot-scalars ss rs gg)
::
++  f-polynomial
  |=  [x=(list scalar:usn) gg=grad-graph:usn]
  ^-  [scalar:usn grad-graph:usn]
  ?~  x  !!
  ?>  .=((lent x) 1)
  =^  x0  gg  (new:usn .~1.0 gg)
  =/  x1  i.x
  =^  x2  gg  (mul:usn x1 x1 gg)
  =^  x3  gg  (mul:usn x1 x2 gg)
  =^  x4  gg  (mul:usn x1 x3 gg)
  =^  x5  gg  (mul:usn x1 x4 gg)
  =/  powers  (limo ~[x0 x1 x2 x3 x4 x5])
  =^  coeffs  gg 
    %+  news:usn
      :~  .~-4.68699287
          .~-2.20431279
          .~2.56718151
          .~1.52540774
          .~-4.07480633
          .~2.67589856
      ==
    gg
  %^    dot-scalars 
      coeffs
    powers 
  gg
::
++  f-prime-polynomial
  |=  [x=(list @rd)]
  ^-  @rd
  ?~  x  !!
  ?>  .=((lent x) 1)
  =/  x0  .~1.0
  =/  x1  i.x
  =/  x2  (mul:rd x1 x1)
  =/  x3  (mul:rd x1 x2)
  =/  x4  (mul:rd x1 x3)
  =/  powers  (limo ~[x0 x1 x2 x3 x4])
  =/  coeffs
    :~  .~-2.20431279
        (mul:rd .~2.0 .~2.56718151)
        (mul:rd .~3.0 .~1.52540774)
        (mul:rd .~4.0 .~-4.07480633)
        (mul:rd .~5.0 .~2.67589856)
    ==
  %+  dot-rd
    coeffs
  powers 
::
++  test-polynomial
  %-  expect 
  !>
  %+  close-enuf 
    ((grad:usn f-polynomial) ~[.~2.0]) 
  (limo ~[(f-prime-polynomial ~[.~2.0])])
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
++  sub-vec
  |=  [a=(list scalar:usn) b=(list scalar:usn) gg=grad-graph:usn]
  ^-  [(list scalar:usn) grad-graph:usn]
  ?>  .=((lent a) (lent b))
  =|  out=(list scalar:usn)
  |-
  ?:  |(?=(~ a) ?=(~ b))
    [(flop out) gg]
  =^  component-diff  gg  (sub:usn i.a i.b gg)
  %=  $
    a  t.a
    b  t.b
    out  [i=component-diff t=out]
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
::  potential of an electric dipole with dipole moment of magnitude 1 in
::  gaussian units.
::
++  phi-dipole
  |=  [r=(list scalar:usn) gg=grad-graph:usn]
  ^-  [scalar:usn grad-graph:usn]
  ?~  r  !!
  ?>  .=((lent r) 3)
  =^  p  gg  (news:usn ~[.~-1.4938 .~-0.5583 .~1.2070] gg)
  =^  absr  gg  (l2-norm r gg)
  =^  r2  gg  (mul:usn absr absr gg)
  =^  r3  gg  (mul:usn absr r2 gg)
  =^  pdotr  gg  (dot-scalars p r gg)
  (div:usn pdotr r3 gg) 
:: negative electric dipole field
::
++  grad-phi-dipole
  |=  [r=(list @rd)]
  ^-  (list @rd)
  =|  out=(list @rd)
  ?>  .=((lent r) 3)
  =/  p  ~[.~-1.4938 .~-0.5583 .~1.2070]
  =/  absr  (sqt:rd (dot-rd r r))
  =/  out  (scale-vec-rd (mul:rd absr absr) p)
  =.  out  (add-vec-rd out (scale-vec-rd (mul:rd .~-3.0 (dot-rd p r)) r))
  =/  r-5  (div:rd .~1.0 (mul:rd absr (mul:rd absr (mul:rd absr (mul:rd absr absr)))))
  (scale-vec-rd r-5 out)
::
++  test-dipole
  %-  expect 
  !>
  %+  close-enuf 
    ((grad:usn phi-dipole) ~[.~2.0 .~1.0 .~0.5]) 
  (grad-phi-dipole ~[.~2.0 .~1.0 .~0.5])
::
--    
