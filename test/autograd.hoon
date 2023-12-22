/+  *test
/+  nabla
|%
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
  |=  [a=(list scalar:nabla) b=(list scalar:nabla) gg=grad-graph:nabla]
  ^-  [scalar:nabla grad-graph:nabla]
  ?>  .=((lent a) (lent b))
  =^  out-0  gg  (new:nabla .~0.0 gg)
  |-
  ?:  |(?=(~ a) ?=(~ b))
    [out-0 gg]
  =^  aibi  gg  (mul:nabla i.a i.b gg)
  =^  out-sum  gg  (add:nabla aibi out-0 gg)
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
  =/  gg  *grad-graph:nabla
  =^  ss  gg  (news:nabla vs gg)
  =^  rs  gg  (news:nabla us gg)
  .=  (dot-rd us vs) 
  val.-:(dot-scalars ss rs gg)
::
++  f-polynomial
  |=  [x=(list scalar:nabla) gg=grad-graph:nabla]
  ^-  [scalar:nabla grad-graph:nabla]
  ?~  x  !!
  ?>  .=((lent x) 1)
  =^  x0  gg  (new:nabla .~1.0 gg)
  =/  x1  i.x
  =^  x2  gg  (mul:nabla x1 x1 gg)
  =^  x3  gg  (mul:nabla x1 x2 gg)
  =^  x4  gg  (mul:nabla x1 x3 gg)
  =^  x5  gg  (mul:nabla x1 x4 gg)
  =/  powers  (limo ~[x0 x1 x2 x3 x4 x5])
  =^  coeffs  gg 
    %+  news:nabla
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
    ((grad:nabla f-polynomial) ~[.~2.0]) 
  (limo ~[(f-prime-polynomial ~[.~2.0])])
::
++  test-weakly-connected-nan
  %-  expect
  !>
  .=  ~[.~20.0]
  %.  ~[.~10.0]
  %-  grad:nabla
  |=  [x=(list scalar:nabla) r=grad-graph:nabla]
  ^-  [scalar:nabla grad-graph:nabla]
  ?~  x  !!
  =^  nanval  r  (new:nabla .~nan r)
  =^  dummy  r  (mul:nabla i.x nanval r)
  =^  out  r  (mul:nabla i.x i.x r)
  [out r]
::
++  test-sqt-of-square
  %-  expect
  !>
  %+  close-enuf
    ~[.~-1.0]
  %.  ~[.~-12.3] 
  %-  grad:nabla
  |=  [x=(list scalar:nabla) r=grad-graph:nabla]  
  ^-  [scalar:nabla grad-graph:nabla]  
  ?~  x  !!  
  =^  out  r  (mul:nabla i.x i.x r) 
  =^  out  r  (sqt:nabla out r) 
  [out r]
--    
