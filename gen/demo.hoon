/+  nabla
/+  nn
:-  %say
|=  *
:-  %noun
=<
=/  fv  (grad-val:nabla loss-fn)
=/  p-init  (init-parameters 65 0)
(train fv p-init 100)
|%
++  train
  |=  $:  grad-f=$-((list @rd) [(list @rd) @rd]) 
          x=(list @rd)
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
  =/  lr  (sub:rd .~1 (div:rd (sun:rd step) (sun:rd nsteps)))
  =/  xprime  (add-vec-rd x (scale-vec-rd (mul:rd .~-1.0 lr) df))
  ~&  "epoch {(scow %ud step)}: loss={(scow %rd f)}"
  $(x xprime, step +(step))
::
++  loss-fn
  |=  [p=(list scalar:nabla) gg=grad-graph:nabla]
  ^-  [scalar:nabla grad-graph:nabla]
  =^  ys  gg  (news:nabla y-train gg)
  =^  xs  gg  `[(list (list scalar:nabla)) grad-graph:nabla]`(spin x-train gg news:nabla)
  =/  m  (bind-parameters:nn -:(mlp:nn ~[2 8 4 1]) p)
  =^  scores  gg  `[(list (list scalar:nabla)) grad-graph:nabla]`(spin xs gg m)
  =/  scores  `(list scalar:nabla)`(zing scores)
  ::[(snag 0 scores) gg]
  =^  data-loss  gg  (hinge-loss scores ys gg)
  =^  sqsm  gg  (dot-scalars p p gg)
  =^  alpha  gg  (new:nabla .~1e-4 gg)
  =^  reg-loss  gg  (mul:nabla alpha sqsm gg)
  (add:nabla data-loss reg-loss gg)
::
++  hinge-loss
  |=  [scores=(list scalar:nabla) labels=(list scalar:nabla) gg=grad-graph:nabla]
  ^-  [scalar:nabla grad-graph:nabla]
  ?>  .=((lent scores) (lent labels))
  =^  nsamples  gg  (new:nabla (sun:rd (lent scores)) gg)
  =^  out-0  gg  (new:nabla .~0.0 gg)
  |-
  ?:  |(?=(~ scores) ?=(~ labels))
    (div:nabla out-0 nsamples gg)
  =^  lss  gg  (mul:nabla i.scores i.labels gg)
  =^  o  gg  (new:nabla .~1 gg)
  =^  lss  gg  (sub:nabla o lss gg)
  =^  lss  gg  (relu:nabla lss gg)
  =^  out-sum  gg  (add:nabla out-0 lss gg)
  %=  $
    scores  t.scores
    labels  t.labels
    out-0  out-sum
    gg  gg
  ==
::
++  init-parameters
  |=  [n=@ud seed=@]
  ^-  (list @rd)
  =|  parameters=(list @rd)
  =/  rng  ~(. og seed)
  |-
  ?:  .=(n 0)
    parameters
  =^  rval  rng  (rads:rng 2.000.000)
  =/  p  (sub:rd (div:rd (sun:rd rval) .~1e6) .~1)
  $(rng rng, parameters (snoc parameters p), n (dec n))
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
::
++  x-train 
  ^-  (list (list @rd))
  :~  ~[.~0.2045645509097321 .~0.33361554024744483]
      ~[.~0.6106948307994285 .~-0.5039833298329776]
      ~[.~1.308320731336601 .~-0.4675917099653295]
      ~[.~1.0878811829707866 .~-0.3533041310638866]
      ~[.~0.9757628774974012 .~0.22187329676921597]
      ~[.~0.7965575374287077 .~0.4841821126583557]
      ~[.~1.8433388903775587 .~0.3407638184356191]
      ~[.~0.12829216806383592 .~-0.07881051069122398]
      ~[.~1.9306505112710275 .~0.013425975034904322]
      ~[.~0.26652547411508026 .~-0.15328172432736986]
      ~[.~0.5440645956056949 .~-0.0976856550559434]
      ~[.~1.306639080325393 .~-0.45399130531434767]
      ~[.~-1.0096260530636227 .~0.30648865196012126]
      ~[.~0.9334042976802118 .~-0.48669225605391686]
      ~[.~-0.48186997480161264 .~0.5555152697071535]
      ~[.~1.6433197965984385 .~-0.3359047445903782]
      ~[.~1.776122862397418 .~0.675539695931188]
      ~[.~-0.0178641976843584 .~0.4541032545293611]
      ~[.~-0.7572525761908537 .~0.3114668763327555]
      ~[.~-0.032145292005859666 .~0.7992248600305809]
      ~[.~1.4105174771031799 .~-0.33301912163062924]
      ~[.~-0.5039023291094005 .~0.9283432752711508]
      ~[.~1.990592914498692 .~-0.06911914720960897]
      ~[.~-0.8718926621606518 .~-0.07270643100637506]
      ~[.~-1.0713617308106116 .~0.1301716300352114]
      ~[.~0.9172848673482008 .~0.48321391357899285]
      ~[.~0.6346336282116163 .~-0.4275746015668872]
      ~[.~-0.33312271471200755 .~1.0675194395228533]
      ~[.~0.15113111832791853 .~-0.06513549289248849]
      ~[.~0.24415007359780014 .~0.9682751087257009]
      ~[.~0.0502902489153729 .~0.3012321867061522]
      ~[.~1.7739543815188918 .~0.0486517180070761]
      ~[.~0.6930066144854172 .~-0.4765919591141241]
      ~[.~-0.7824497287288859 .~0.7161014448649095]
      ~[.~-0.6978541783117385 .~0.48093254863609763]
      ~[.~-0.01590843955979566 .~0.36268689989020764]
      ~[.~0.7119100429853578 .~0.5500734834881762]
      ~[.~0.4792391247959088 .~0.8008079397670146]
      ~[.~1.8777305079890845 .~0.21789039326339515]
      ~[.~0.8959349919002718 .~0.13288120946099455]
      ~[.~-0.5970284643971744 .~0.789600310964643]
      ~[.~0.973485459964959 .~-0.017292172022354915]
      ~[.~-0.5710335046984093 .~1.0846752443934895]
      ~[.~0.9078835825610537 .~0.560402998045406]
      ~[.~0.2324665490694537 .~0.8928580704364164]
      ~[.~0.25818784430437847 .~1.1162657096569684]
      ~[.~0.45570492618643227 .~1.020577816322083]
      ~[.~-0.19184364546733765 .~0.9005081644407977]
      ~[.~1.666301565318532 .~-0.19687656146494484]
      ~[.~0.9133377821875416 .~-0.4638764384614954]
  ==
::
++  y-train  
  ^-  (list @rd)
  :~  .~1
      .~1
      .~1
      .~1
      .~-1
      .~-1
      .~1
      .~1
      .~1
      .~1
      .~1
      .~1
      .~-1
      .~1
      .~-1
      .~1
      .~1
      .~1
      .~-1
      .~-1
      .~1
      .~-1
      .~1
      .~-1
      .~-1
      .~-1
      .~1
      .~-1
      .~1
      .~-1
      .~1
      .~1
      .~1
      .~-1
      .~-1
      .~1
      .~-1
      .~-1
      .~1
      .~-1
      .~-1
      .~-1
      .~-1
      .~-1
      .~-1
      .~-1
      .~-1
      .~-1
      .~1
      .~1
  ==
--