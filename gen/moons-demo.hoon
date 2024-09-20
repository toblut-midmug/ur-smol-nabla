:: Train a multilayer perceptron for binary classification on synthetic
:: 2d data.
:: Cf.:
:: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html
:: https://github.com/karpathy/micrograd/blob/master/demo.ipynb
::
/+  nbl=nabla
/+  nn
/+  vm=vecmath
::
:-  %say
|=  *
:-  %noun
=<
=/  m-meta  (mlp:nn ~[2 8 4 1])
::  "It's a good seed, sir."
::
[model.m-meta (train m-meta 1 100 moons-x-train moons-y-train moons-x-val moons-y-val)]
::
|%
++  train
  |=  $:  =model-meta:nn 
          seed=@
          nepochs=@ud
          x-train=(list (list @rd))
          y-train=(list @rd)
          x-val=(list (list @rd))
          y-val=(list @rd)
      ==
  ^-  (list @rd)
  =/  loss-fn  (construct-loss-fn model.model-meta x-train y-train)
  =/  params  (init-parameters nparams.model-meta seed)
  =/  epoch  0
  |-  
  ?:  .=(epoch nepochs)
    params
  =/  grads-loss  ((grad-val:nbl loss-fn) params)
  =/  grads  -:grads-loss
  =/  loss  +:grads-loss
  :: linear learning rate decay
  ::
  =/  lr  (sub:rd .~1 (div:rd (sun:rd epoch) (sun:rd nepochs)))
  =/  params  (add-vec-rd:vm params (scale-vec-rd:vm (mul:rd .~-1.0 lr) grads))
  =/  preds  (predict model.model-meta x-train params)
  =/  acc  (accuracy preds y-train)
  =/  preds-val  (predict model.model-meta x-val params)
  =/  acc-val  (accuracy preds-val y-val)
  ~&  "epoch {(scow %ud epoch)}: loss={(scow %rd loss)}, train-acc={(scow %rd acc)}, val-acc={(scow %rd acc-val)}"
  $(params params, epoch +(epoch))
::
::  Construct the loss function from the model and the full data set.
::
++  construct-loss-fn
  |=  [m=model:nn x-train=(list (list @rd)) y-train=(list @rd)]
  ^-  scalar-fn:nbl
  |=  [p=(list scalar:nbl) gg=grad-graph:nbl]
  ^-  [scalar:nbl grad-graph:nbl]
  =^  ys  gg  (news:nbl y-train gg)
  =^  xs  gg  [p q]:(spin x-train gg news:nbl)
  =/  m-forward  (bind-parameters:nn m p)
  =^  scores  gg  [p q]:(spin xs gg m-forward)
  =/  scores  `(list scalar:nbl)`(zing scores)
  =^  data-loss  gg  (hinge-loss scores ys gg)
  =^  sqsm  gg  (dot:vm p p gg)
  =^  alpha  gg  (new:nbl .~1e-4 gg)
  =^  reg-loss  gg  (mul:nbl alpha sqsm gg)
  (add:nbl data-loss reg-loss gg)
::
++  hinge-loss
  |=  [scores=(list scalar:nbl) labels=(list scalar:nbl) gg=grad-graph:nbl]
  ^-  [scalar:nbl grad-graph:nbl]
  ?>  .=((lent scores) (lent labels))
  =^  nsamples  gg  (new:nbl (sun:rd (lent scores)) gg)
  =^  out-0  gg  (new:nbl .~0.0 gg)
  |-
  ?:  |(?=(~ scores) ?=(~ labels))
    (div:nbl out-0 nsamples gg)
  =^  lss  gg  (mul:nbl i.scores i.labels gg)
  =^  const-1  gg  (new:nbl .~1 gg)
  =^  lss  gg  (sub:nbl const-1 lss gg)
  =^  lss  gg  (relu:nbl lss gg)
  =^  out-sum  gg  (add:nbl out-0 lss gg)
  %=  $
    scores  t.scores
    labels  t.labels
    out-0  out-sum
  ==
::
++  predict
  |=  [=model:nn inputs-batch=(list (list @rd)) params=(list @rd)]
  ^-  (list @rd)
  =/  gg  *grad-graph:nbl
  =^  xs  gg  [p q]:(spin inputs-batch gg news:nbl)
  =^  p  gg  (news:nbl params gg)
  =/  forward  (bind-parameters:nn model p)
  =^  scores  gg  [p q]:(spin xs gg forward)
  =/  scores  `(list scalar:nbl)`(zing scores)
  (turn scores |=(s=scalar:nbl ?:((gth:rd val.s .~0) .~1 .~-1)))
::
++  accuracy
  |=  [scores=(list @rd) labels=(list @rd)]
  ^-  @rd
  ?>  .=((lent scores) (lent labels))
  =/  preds  (turn scores |=(s=@rd ?:((gth:rd s .~0) .~1 .~-1)))
  (add:rd .~0.5 (div:rd (dot-rd:vm preds labels) (sun:rd (mul 2 (lent scores)))))
::
::  Uniform random numbers in the interval [-1, 1]. 
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
:: Data for training and evaluation.
:: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html
::
++  moons-x-val
  ^-  (list (list @rd))
  :~  ~[.~0.9774801645765868 .~-0.8406262908843175]
      ~[.~2.0929306738908475 .~0.1922594655232861]
      ~[.~-0.041478488804055386 .~0.14517670648350328]
      ~[.~0.4826219578766875 .~0.9238816777230495]
      ~[.~-0.049895384738241136 .~0.8012447719477833]
      ~[.~0.8311282377903513 .~-0.19343733477608077]
      ~[.~1.8128682455196217 .~0.04282266226680259]
      ~[.~1.9297734176728623 .~0.17781002788323116]
      ~[.~1.2367964870535793 .~-0.5347834067669726]
      ~[.~-1.0206340778922465 .~0.23979326109936097]
      ~[.~-0.85952679944456 .~0.05398016473433371]
      ~[.~1.938616626634409 .~0.28504773471252154]
      ~[.~-0.827977747569135 .~0.8301507272122078]
      ~[.~0.019915185596359967 .~0.708050152828684]
      ~[.~-0.08185093973084093 .~0.1560382035258669]
      ~[.~0.6127214456916381 .~-0.40442686968279007]
      ~[.~-0.5722558354766104 .~1.0076467022335185]
      ~[.~1.0229399529534677 .~0.09980454392849986]
      ~[.~-0.006352437910997065 .~0.25582661999066725]
      ~[.~-0.4384109881128897 .~0.8634486530527119]
      ~[.~0.25107709146207563 .~-0.10397200387637077]
      ~[.~0.7595882131723216 .~0.5387466636434798]
      ~[.~0.2767047654128552 .~-0.2301341635487303]
      ~[.~1.1731145587225618 .~0.02538732979724251]
      ~[.~0.03408561885669907 .~0.048950182907453337]
      ~[.~0.769359962311405 .~-0.4272414179317413]
      ~[.~1.5771259807313402 .~-0.3017649894512119]
      ~[.~-0.9426804555501221 .~0.6319473077241292]
      ~[.~1.0168888088911825 .~-0.4232524241277457]
      ~[.~0.13191461607963553 .~0.8833207597534741]
      ~[.~0.8503045773293658 .~0.581070588450721]
      ~[.~0.9340179919406211 .~0.7912060435644241]
      ~[.~0.9663773309559074 .~0.4948924167848406]
      ~[.~-0.8317599911819381 .~0.3786115216958393]
      ~[.~0.3058700942100425 .~0.3028598031830417]
      ~[.~1.0265833357940566 .~-0.4098569228339265]
      ~[.~-0.9265963831899826 .~0.38049711904202765]
      ~[.~1.4648169511310578 .~-0.2024446485682735]
      ~[.~0.20072155807928943 .~-0.27702362163348127]
      ~[.~0.5931165601138068 .~-0.6053543623884078]
      ~[.~0.727637529663275 .~0.8104027910951772]
      ~[.~2.0665918618390613 .~0.3755246198290243]
      ~[.~0.20125955449252791 .~0.9087616765690204]
      ~[.~-0.5947741070234241 .~0.7646177771747118]
      ~[.~1.655649380926572 .~-0.18157661150934937]
      ~[.~-0.9739181657460108 .~0.2325586842450934]
      ~[.~0.44097368946283866 .~0.8778605281495014]
      ~[.~2.029264805833413 .~-0.10161982827521332]
      ~[.~-0.35320463526142853 .~0.9002213045155135]
      ~[.~1.5805640360647213 .~-0.5357043647658047]
  ==
::
++  moons-y-val
  ^-  (list @rd)
  :~  .~1
      .~1
      .~1
      .~-1
      .~-1
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
      .~1
      .~-1
      .~-1
      .~1
      .~-1
      .~1
      .~-1
      .~1
      .~-1
      .~1
      .~1
      .~1
      .~-1
      .~1
      .~-1
      .~-1
      .~-1
      .~-1
      .~-1
      .~1
      .~1
      .~-1
      .~1
      .~1
      .~1
      .~-1
      .~1
      .~-1
      .~-1
      .~1
      .~-1
      .~-1
      .~1
      .~-1
      .~1
  ==
::
++  moons-x-train
  ^-  (list (list @rd))
  :~  ~[.~1.606548108086409 .~-0.19861287493127355]
      ~[.~-0.4947482475221469 .~1.1003181584089208]
      ~[.~-0.9339511018934178 .~0.11156346742691137]
      ~[.~0.3945225345195033 .~-0.2359220641700401]
      ~[.~0.614075191205101 .~0.776292173140289]
      ~[.~0.3429395369000665 .~-0.24504810879297242]
      ~[.~1.6422242614567473 .~-0.24504450947962517]
      ~[.~-0.8878959373156354 .~0.44653492084922497]
      ~[.~0.0820800133364609 .~0.1468843233699894]
      ~[.~0.8423898936969823 .~0.47363248636145666]
      ~[.~0.989204901718078 .~-0.5473461766668748]
      ~[.~0.49990085404233475 .~0.8240469038732791]
      ~[.~0.7107741942361889 .~-0.49410251849066494]
      ~[.~0.38258158178293433 .~-0.3467582336931977]
      ~[.~-1.0460235123746064 .~0.3812301591852637]
      ~[.~2.0942148064571837 .~0.3785387765418395]
      ~[.~-0.42748309967409975 .~0.7857877308489627]
      ~[.~0.25391444582550404 .~0.9056486289656493]
      ~[.~0.21688942538186964 .~0.8976729509824117]
      ~[.~0.34495478955882974 .~0.7169826132175211]
      ~[.~1.0558009290851103 .~-0.5570724293610295]
      ~[.~2.157641955403414 .~0.31182838842437177]
      ~[.~1.8254889533407774 .~0.09549761104475343]
      ~[.~1.4670621479052406 .~-0.4116808887926374]
      ~[.~0.9362371946975949 .~-0.013605707518403383]
      ~[.~0.7125378354851141 .~0.5982938101194445]
      ~[.~-0.08810272336285659 .~0.5712282426608808]
      ~[.~2.041330246921498 .~-0.06001420658992862]
      ~[.~0.7437819888313673 .~0.2102786103697616]
      ~[.~1.5033170753478655 .~-0.31782880631838684]
      ~[.~-0.18007574438922597 .~0.9809458431188908]
      ~[.~0.29967564709905087 .~-0.23169454452136284]
      ~[.~-0.16258755118880003 .~0.31482750932393716]
      ~[.~0.3374201847949682 .~0.8746585037069142]
      ~[.~-0.02484924149958473 .~0.18310698918016416]
      ~[.~1.8918271777883382 .~0.5824415788681049]
      ~[.~-0.7361114775019407 .~0.9064420111490108]
      ~[.~-0.7915337606287999 .~0.955944451573406]
      ~[.~0.8307776971327462 .~-0.5752697354388759]
      ~[.~-0.9716305933793322 .~0.10960359808018055]
      ~[.~0.03459343822218372 .~0.10963468748465718]
      ~[.~0.9486559529472033 .~0.18617386716970655]
      ~[.~-0.9519159732226851 .~0.4433438122041971]
      ~[.~1.3157729405733907 .~-0.46997255036344654]
      ~[.~-0.41632045989121647 .~0.9678877447097638]
      ~[.~-0.04535722606236721 .~1.180052517064173]
      ~[.~0.6200417097199296 .~-0.5130851255810395]
      ~[.~0.9493546690646611 .~0.46050689101194436]
      ~[.~-0.8665979467604934 .~0.7422421994613781]
      ~[.~1.3872231696365998 .~-0.2569770907382608]
  ==
::
++  moons-y-train
  ^-  (list @rd)
  :~  .~1
      .~-1
      .~-1
      .~1
      .~-1
      .~1
      .~1
      .~-1
      .~1
      .~-1
      .~1
      .~-1
      .~1
      .~1
      .~-1
      .~1
      .~-1
      .~-1
      .~-1
      .~-1
      .~1
      .~1
      .~1
      .~1
      .~-1
      .~-1
      .~1
      .~1
      .~-1
      .~1
      .~-1
      .~1
      .~1
      .~-1
      .~1
      .~1
      .~-1
      .~-1
      .~1
      .~-1
      .~1
      .~-1
      .~-1
      .~1
      .~-1
      .~-1
      .~1
      .~-1
      .~-1
      .~1
  ==
--
