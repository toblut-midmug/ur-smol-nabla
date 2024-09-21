/+  *test
/+  *nn
/+  nbl=nabla
|%
::
++  test-forward
  %-  expect
  !>
  =|  gg=grad-graph:nbl
  =/  n-out  4
  =/  m-meta  (mlp ~[2 16 16 n-out])
  =^  p  gg  (news:nbl (reap nparams.m-meta .~1.23456789) gg)
  =^  x  gg  (news:nbl ~[.~1.0 .~2.0] gg)
  =^  out  gg  (model.m-meta x p gg)
  .=(n-out (lent out))
::
--
