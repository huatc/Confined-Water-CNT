# build_cnt_itp.py  -> writes full cnt.itp with 728 atoms from cnt.gro
gro = input('Enter .gro filename')
n = None

with open(gro,"r") as f:
    title = f.readline()
    n = int(f.readline().strip())   # atom count from .gro

with open("cnt.itp","w") as f:
    f.write("; ======= cnt.itp (auto-generated) =======\n")
    f.write("[ moleculetype ]\n; name   nrexcl\nCNT      3\n\n")
    f.write("[ atoms ]\n;  nr   type       resnr resid  atom  cgnr  charge   mass\n")
    for i in range(1, n+1):
        f.write(f"{i:5d} opls_145   1     CNT   C{i:<4}{i:6d}{0.0:8.3f}{12.011:9.3f}\n")

print('Ran Built CNT ITP')