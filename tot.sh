#sh /root/workspace/diff_util/collector/collect_diff.sh
#python /root/workspace/diff_util/encoder/diff_encoder.py
#python /root/workspace/diff_util/encoder/gen_feature.py

#python /root/workspace/run_Bug2Commit.py
python /root/workspace/analyze/result_gen.py

# Post-hoc test
Rscript /root/workspace/analyze/post_hoc.R -b FALSE -f "use_stopword:True,use_br:True,HSFL:False"
