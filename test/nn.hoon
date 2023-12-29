/+  *test
/+  *nn
/+  usn=nabla
|%
::
++  test-forward
  %-  expect
  !>
  =|  gg=grad-graph:usn
  =/  m-meta  (mlp ~[2 16 16 1])
  =/  m  -:m-meta
  =/  nparams  +:m-meta
  =^  p  gg  (news:usn (reap nparams .~1.23456789) gg)
  =^  x  gg  (news:usn ~[.~1.0 .~2.0] gg)
  =^  out  gg  (m x p gg)
  %.y
::
--
