/+  nabla
/+  nn
:-  %say
|=  *
:-  %noun
=<
=/  fv  (grad-val:nabla sample-fn)
(gradient-descent fv ~[.~2 .~3] .~0.1 50)
|%
++  gradient-descent
  |=  $:  grad-f=$-((list @rd) [(list @rd) @rd]) 
          x=(list @rd)
          lr=@rd
          nsteps=@ud
      ==
  ^-  (list @rd)
  =/  step  0
  |-  
  ?:  .=(step nsteps)
    x
  =/  df-f  (grad-f x)
  =/  df  -:df-f
  =/  f  +:df-f
  =/  xprime  (add-vec-rd x (scale-vec-rd (mul:rd .~-1.0 lr) df))
  ~&  "step {(scow %ud step)}: f={(scow %rd f)}"
  $(x xprime, step +(step))
::
++  sample-fn
  |=  [xy=(list scalar:nabla) gg=grad-graph:nabla]
  ^-  [scalar:nabla grad-graph:nabla]
  ?>  .=(2 (lent xy))
  =/  x  (snag 0 xy)
  =/  y  (snag 1 xy)
  =^  xsq  gg  (mul:nabla x x gg)
  =^  ysq  gg  (mul:nabla y y gg)
  (add:nabla xsq ysq gg)
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
--
