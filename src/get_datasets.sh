#!/bin/bash
# requires git 2.19
git clone \
  --depth 1 \
  --filter=blob:none \
  --no-checkout \
  https://github.com/gvrkiran/BalancedExposure.git datasets \
;
cd datasets
git checkout master -- datasets
mv datasets balanced_exposure
for f in `ls */*/*.gz`; do
	gzip -d $f;
done

