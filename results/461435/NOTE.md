Superseded by 461490. In this run the v2 binary also timed native DGEMM right
before each shape emulated runs, while v1 ran with --no-native; the native run
heats the GPU and slows the emulated runs after it (~5-7%, see native_post).
That asymmetry, not the library, made v2 look 6-8% slower at 4096^3 and
8192^2x1024. 461490 (both --no-native, script in run.sh) shows v1 == v2.
scripts/abtest.sbatch now runs both builds identically.
