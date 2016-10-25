import problem as p

unforced = p.poisson1d(**{'dmleft':-1.,
                          'dmright':1.,
                          'bcleft':-0.5,
                          'bcright':1.1
                          }
                       )

quad = p.poisson1dquad(**{'dmleft':-1.,
                          'dmright':1.,
                          'bcleft':-0.5,
                          'bcright':1.1,
                          'p2':1.2,
                          'p1':-0.86,
                          'p0':0.023
                          }
                       )

