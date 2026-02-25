set -e

ech " Running DVC pipeline for training... "
dvc repro
echo " DVC pipeline completed. Training is done! "