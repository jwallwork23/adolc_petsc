import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', help='Forward mode')
parser.add_argument('-b', help='Reverse mode')
args = parser.parse_args()

b = bool(args.b)
d = bool(args.d)
assert(not(b and d))
e = 'b' if b else 'd'

# Post-process RHS_d/RHS_b to have the right includes
f = open('RHS_'+e+'.c', 'r')
contents = f.readlines()
contents[3] += '#include "mxm_'+e+'.h"\n'
contents = ''.join(contents)
f.close()
f = open('RHS_'+e+'.c', 'w')
f.write(contents)
f.close()

# Post-process header file
f = open('mxm_'+e+'.h', 'r')
contents = f.readlines()
for i in range(2):
    contents[i] = contents[i].replace('MXM', 'MXM_B' if b else 'MXM_D')
contents[2] = '#include "naive_dgemm_'       + e +'.c"\n' + \
              '#include "extra_naive_mtmv_'  + e +'.c"\n' + \
              '#include "naive_mpm_'         + e +'.c"\n' + \
              '#include "naive_matrix_axpy_' + e +'.c"\n'
contents = ''.join(contents)
f.close()
f = open('mxm_'+e+'.h', 'w')
f.write(contents)
f.close()
