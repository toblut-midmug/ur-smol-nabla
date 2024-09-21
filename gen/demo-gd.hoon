/+  nabla
/+  nn
/+  *vecmath
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
--
