
# for accuracy in 3e-3 1e-3 3e-4 1e-4 7e-5
for accuracy in 7e-5
do
    python misc/tetra.py data/objs/tiny-box.obj \
        data/objs/tiny-box-remesh-${accuracy} --switches "pq1.1a${accuracy}"
    python datagen/elast_twist.py basic.max_count=500 basic.prefix=generated/twist-tiny-box-remesh-${accuracy} \
        +mesh=data/objs/tiny-box-remesh-${accuracy} visualize=false
done