flist=$(ls test_*)
for f in $flist; do
    python $f --generate-gold
done