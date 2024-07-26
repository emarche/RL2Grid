# launch 50 jobs of singlejob_submit.sbatch
for j in $(seq 20 27)
do
  sbatch --job-name=$j.run --output=$j.out ./singlejob_submit.sh
done
