#!/usr/bin/env bash


stage=0
stop_stage=1000

. utils/parse_options.sh
#. path_for_nn_vad.sh
#. path_for_fsq_sptt.sh # hltsz
. path_for_fsq_speechtext.sh # sribd
# 2024-12-12 maduo note:
# cluster method sota setting is as follows:
# vad threshold=0.6, vad model is trained at
# chunk_size=3s, shift_size=3s, skip_chunk_size=0.93s,
# cluster method : spectral clustering
# speaker embedding is extracted from cam++ (cnceleb and voxceleb) /mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt
#
# sota dev set predict rttm file : /mntnfs/lee_data1/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/data/magicdata-RAMC/dev/rttm_predict_0.6_transformer_vad_chunk_size_3_step_size_3_skip_chunk_size_0.93
# groundtruth dev set rttm file : /mntnfs/lee_data1/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/data/magicdata-RAMC/dev/rttm_openslr_gt_dev_nog0

# sota test set predict rttm file : /mntnfs/lee_data1/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/data/magicdata-RAMC/test/rttm_predict_0.6_transformer_vad_chunk_size_3_step_size_3_skip_chunk_size_0.93
# groundtruth test set rttm file : /mntnfs/lee_data1/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/data/magicdata-RAMC/test/rttm_openslr_gt_test_nog0

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ];then
  echo "spectral clustering ...."
  python scripts/magicdata-RAMC/020_spectral_cluster.py
  # SCTK-2.4.12/src/md-eval/md-eval.pl -c 0.25 -r data/magicdata-RAMC_debug/test/rttm_debug2 -s data/magicdata-RAMC/test/rttm_predict
  # result:35.04/30.22/1.70/3.12
  #  SCTK-2.4.12/src/md-eval/md-eval.pl -c 0.25 -r data/magicdata-RAMC_debug/test/rttm -s data/magicdata-RAMC/test/rttm_predict
  # result:
  # WARNING:  speaker G00000169 speaking more than once at time 124.835
#WARNING:  speaker G00000000 speaking more than once at time 242.056
#WARNING:  speaker G00000168 speaking more than once at time 359.176
#WARNING:  speaker G00000168 speaking more than once at time 388.331
#WARNING:  speaker G00000000 speaking more than once at time 646.813
#WARNING:  speaker G00000168 speaking more than once at time 906.248
#WARNING:  speaker G00000000 speaking more than once at time 993.704
#WARNING:  speaker G00000168 speaking more than once at time 1238.088
#WARNING:  speaker G00000000 speaking more than once at time 1256.272
#WARNING:  speaker G00000169 speaking more than once at time 1363.458
#WARNING:  speaker G00000168 speaking more than once at time 1471.048
#WARNING:  speaker G00000000 speaking more than once at time 1602.984
#WARNING:  speaker G00000169 speaking more than once at time 1780.168
#WARNING:  speaker G00000169 speaking more than once at time 124.835
#WARNING:  speaker G00000000 speaking more than once at time 242.056
#WARNING:  speaker G00000168 speaking more than once at time 359.176
#WARNING:  speaker G00000168 speaking more than once at time 388.331
#WARNING:  speaker G00000000 speaking more than once at time 646.813
#WARNING:  speaker G00000168 speaking more than once at time 906.248
#WARNING:  speaker G00000000 speaking more than once at time 993.704
#WARNING:  speaker G00000168 speaking more than once at time 1238.088
#WARNING:  speaker G00000000 speaking more than once at time 1256.272
#WARNING:  speaker G00000169 speaking more than once at time 1363.458
#WARNING:  speaker G00000168 speaking more than once at time 1471.048
#WARNING:  speaker G00000000 speaking more than once at time 1602.984
#WARNING:  speaker G00000169 speaking more than once at time 1780.168
#WARNING:  speaker G00000000 speaking more than once at time 447.416
#WARNING:  speaker G00000000 speaking more than once at time 447.416
#WARNING:  speaker G00000464 speaking more than once at time 452.378
#WARNING:  speaker G00000464 speaking more than once at time 452.378
#WARNING:  speaker G00000131 speaking more than once at time 528.962
#WARNING:  speaker G00000131 speaking more than once at time 532.848
#WARNING:  speaker G00000131 speaking more than once at time 528.962
#WARNING:  speaker G00000131 speaking more than once at time 532.848
#WARNING:  speaker G00000185 speaking more than once at time 1421.653
#WARNING:  speaker G00000185 speaking more than once at time 1421.653
#WARNING:  speaker G00000290 speaking more than once at time 1526.856
#WARNING:  speaker G00000290 speaking more than once at time 1526.856
#WARNING:  speaker G00000583 speaking more than once at time 280.592
#WARNING:  speaker G00000583 speaking more than once at time 406.701
#WARNING:  speaker G00000583 speaking more than once at time 971.588
#WARNING:  speaker G00000584 speaking more than once at time 1181.130
#WARNING:  speaker G00000583 speaking more than once at time 1333.227
#WARNING:  speaker G00000583 speaking more than once at time 1438.255
#WARNING:  speaker G00000583 speaking more than once at time 1455.758
#WARNING:  speaker G00000583 speaking more than once at time 1714.851
#WARNING:  speaker G00000583 speaking more than once at time 280.592
#WARNING:  speaker G00000583 speaking more than once at time 406.701
#WARNING:  speaker G00000583 speaking more than once at time 971.588
#WARNING:  speaker G00000584 speaking more than once at time 1181.130
#WARNING:  speaker G00000583 speaking more than once at time 1333.227
#WARNING:  speaker G00000583 speaking more than once at time 1438.255
#WARNING:  speaker G00000583 speaking more than once at time 1455.758
#WARNING:  speaker G00000583 speaking more than once at time 1714.851
#WARNING:  speaker G00000639 speaking more than once at time 1024.776
#WARNING:  speaker G00000000 speaking more than once at time 1499.859
#WARNING:  speaker G00000639 speaking more than once at time 1024.776
#WARNING:  speaker G00000000 speaking more than once at time 1499.859
#WARNING:  speaker G00000364 speaking more than once at time 324.970
#WARNING:  speaker G00000364 speaking more than once at time 1355.424
#WARNING:  speaker G00000364 speaking more than once at time 324.970
#WARNING:  speaker G00000364 speaking more than once at time 1355.424
#WARNING:  speaker G00000231 speaking more than once at time 371.369
#WARNING:  speaker G00000231 speaking more than once at time 933.760
#WARNING:  speaker G00000000 speaking more than once at time 1094.816
#WARNING:  speaker G00000231 speaking more than once at time 1173.609
#WARNING:  speaker G00000230 speaking more than once at time 1183.944
#WARNING:  speaker G00000231 speaking more than once at time 1236.736
#WARNING:  speaker G00000231 speaking more than once at time 1503.376
#WARNING:  speaker G00000231 speaking more than once at time 1542.736
#WARNING:  speaker G00000231 speaking more than once at time 1584.552
#WARNING:  speaker G00000231 speaking more than once at time 1600.000
#WARNING:  speaker G00000231 speaking more than once at time 1656.000
#WARNING:  speaker G00000231 speaking more than once at time 1671.752
#WARNING:  speaker G00000230 speaking more than once at time 1737.416
#WARNING:  speaker G00000230 speaking more than once at time 1752.809
#WARNING:  speaker G00000231 speaking more than once at time 1802.152
#WARNING:  speaker G00000231 speaking more than once at time 371.369
#WARNING:  speaker G00000231 speaking more than once at time 933.760
#WARNING:  speaker G00000000 speaking more than once at time 1094.816
#WARNING:  speaker G00000231 speaking more than once at time 1173.609
#WARNING:  speaker G00000230 speaking more than once at time 1183.944
#WARNING:  speaker G00000231 speaking more than once at time 1236.736
#WARNING:  speaker G00000231 speaking more than once at time 1503.376
#WARNING:  speaker G00000231 speaking more than once at time 1542.736
#WARNING:  speaker G00000231 speaking more than once at time 1584.552
#WARNING:  speaker G00000231 speaking more than once at time 1600.000
#WARNING:  speaker G00000231 speaking more than once at time 1656.000
#WARNING:  speaker G00000231 speaking more than once at time 1671.752
#WARNING:  speaker G00000230 speaking more than once at time 1737.416
#WARNING:  speaker G00000230 speaking more than once at time 1752.809
#WARNING:  speaker G00000231 speaking more than once at time 1802.152
#WARNING:  speaker G00000000 speaking more than once at time 824.512
#WARNING:  speaker G00000283 speaking more than once at time 1151.936
#WARNING:  speaker G00000282 speaking more than once at time 1329.656
#WARNING:  speaker G00000283 speaking more than once at time 1500.328
#WARNING:  speaker G00000283 speaking more than once at time 1501.544
#WARNING:  speaker G00000283 speaking more than once at time 1765.672
#WARNING:  speaker G00000283 speaking more than once at time 1795.024
#WARNING:  speaker G00000283 speaking more than once at time 1844.736
#WARNING:  speaker G00000283 speaking more than once at time 1915.304
#WARNING:  speaker G00000000 speaking more than once at time 824.512
#WARNING:  speaker G00000283 speaking more than once at time 1151.936
#WARNING:  speaker G00000282 speaking more than once at time 1329.656
#WARNING:  speaker G00000283 speaking more than once at time 1500.328
#WARNING:  speaker G00000283 speaking more than once at time 1501.544
#WARNING:  speaker G00000283 speaking more than once at time 1765.672
#WARNING:  speaker G00000283 speaking more than once at time 1795.024
#WARNING:  speaker G00000283 speaking more than once at time 1844.736
#WARNING:  speaker G00000283 speaking more than once at time 1915.304
#WARNING:  speaker G00000000 speaking more than once at time 581.612
#WARNING:  speaker G00000000 speaking more than once at time 745.872
#WARNING:  speaker G00000000 speaking more than once at time 1265.952
#WARNING:  speaker G00000000 speaking more than once at time 581.612
#WARNING:  speaker G00000000 speaking more than once at time 745.872
#WARNING:  speaker G00000000 speaking more than once at time 1265.952
#WARNING:  speaker G00000000 speaking more than once at time 490.930
#WARNING:  speaker G00000671 speaking more than once at time 922.378
#WARNING:  speaker G00000000 speaking more than once at time 1101.386
#WARNING:  speaker G00000000 speaking more than once at time 1101.954
#WARNING:  speaker G00000672 speaking more than once at time 1106.322
#WARNING:  speaker G00000672 speaking more than once at time 1110.122
#WARNING:  speaker G00000672 speaking more than once at time 1151.858
#WARNING:  speaker G00000672 speaking more than once at time 1172.058
#WARNING:  speaker G00000671 speaking more than once at time 1292.322
#WARNING:  speaker G00000000 speaking more than once at time 1512.938
#WARNING:  speaker G00000000 speaking more than once at time 1541.690
#WARNING:  speaker G00000672 speaking more than once at time 1841.738
#WARNING:  speaker G00000672 speaking more than once at time 1848.514
#WARNING:  speaker G00000000 speaking more than once at time 490.930
#WARNING:  speaker G00000671 speaking more than once at time 922.378
#WARNING:  speaker G00000000 speaking more than once at time 1101.386
#WARNING:  speaker G00000000 speaking more than once at time 1101.954
#WARNING:  speaker G00000672 speaking more than once at time 1106.322
#WARNING:  speaker G00000672 speaking more than once at time 1110.122
#WARNING:  speaker G00000672 speaking more than once at time 1151.858
#WARNING:  speaker G00000672 speaking more than once at time 1172.058
#WARNING:  speaker G00000671 speaking more than once at time 1292.322
#WARNING:  speaker G00000000 speaking more than once at time 1512.938
#WARNING:  speaker G00000000 speaking more than once at time 1541.690
#WARNING:  speaker G00000672 speaking more than once at time 1841.738
#WARNING:  speaker G00000672 speaking more than once at time 1848.514
#WARNING:  speaker G00000000 speaking more than once at time 51.944
#WARNING:  speaker G00000000 speaking more than once at time 168.248
#WARNING:  speaker G00000000 speaking more than once at time 929.808
#WARNING:  speaker G00000000 speaking more than once at time 959.712
#WARNING:  speaker G00000000 speaking more than once at time 1400.736
#WARNING:  speaker G00000000 speaking more than once at time 51.944
#WARNING:  speaker G00000000 speaking more than once at time 168.248
#WARNING:  speaker G00000000 speaking more than once at time 929.808
#WARNING:  speaker G00000000 speaking more than once at time 959.712
#WARNING:  speaker G00000000 speaking more than once at time 1400.736
#WARNING:  speaker G00000045 speaking more than once at time 103.272
#WARNING:  speaker G00000045 speaking more than once at time 506.368
#WARNING:  speaker G00000045 speaking more than once at time 735.968
#WARNING:  speaker G00000046 speaking more than once at time 1088.608
#WARNING:  speaker G00000045 speaking more than once at time 103.272
#WARNING:  speaker G00000045 speaking more than once at time 506.368
#WARNING:  speaker G00000045 speaking more than once at time 735.968
#WARNING:  speaker G00000046 speaking more than once at time 1088.608
#WARNING:  speaker G00000000 speaking more than once at time 799.053
#WARNING:  speaker G00000000 speaking more than once at time 1017.744
#WARNING:  speaker G00000000 speaking more than once at time 799.053
#WARNING:  speaker G00000000 speaking more than once at time 1017.744
#WARNING:  speaker G00000000 speaking more than once at time 53.256
#WARNING:  speaker G00000000 speaking more than once at time 268.505
#WARNING:  speaker G00000000 speaking more than once at time 407.201
#WARNING:  speaker G00000000 speaking more than once at time 433.528
#WARNING:  speaker G00000000 speaking more than once at time 757.544
#WARNING:  speaker G00000000 speaking more than once at time 871.482
#WARNING:  speaker G00000000 speaking more than once at time 1035.712
#WARNING:  speaker G00000000 speaking more than once at time 1057.248
#WARNING:  speaker G00000092 speaking more than once at time 1142.634
#WARNING:  speaker G00000000 speaking more than once at time 1162.832
#WARNING:  speaker G00000000 speaking more than once at time 1164.312
#WARNING:  speaker G00000000 speaking more than once at time 1241.203
#WARNING:  speaker G00000000 speaking more than once at time 1260.024
#WARNING:  speaker G00000000 speaking more than once at time 1308.680
#WARNING:  speaker G00000000 speaking more than once at time 1588.259
#WARNING:  speaker G00000000 speaking more than once at time 1731.002
#WARNING:  speaker G00000000 speaking more than once at time 1732.232
#WARNING:  speaker G00000000 speaking more than once at time 1741.378
#WARNING:  speaker G00000000 speaking more than once at time 1746.944
#WARNING:  speaker G00000000 speaking more than once at time 1751.770
#WARNING:  speaker G00000000 speaking more than once at time 1791.312
#WARNING:  speaker G00000000 speaking more than once at time 1791.968
#WARNING:  speaker G00000000 speaking more than once at time 1825.248
#WARNING:  speaker G00000000 speaking more than once at time 53.256
#WARNING:  speaker G00000000 speaking more than once at time 268.505
#WARNING:  speaker G00000000 speaking more than once at time 407.201
#WARNING:  speaker G00000000 speaking more than once at time 433.528
#WARNING:  speaker G00000000 speaking more than once at time 757.544
#WARNING:  speaker G00000000 speaking more than once at time 871.482
#WARNING:  speaker G00000000 speaking more than once at time 1035.712
#WARNING:  speaker G00000000 speaking more than once at time 1057.248
#WARNING:  speaker G00000092 speaking more than once at time 1142.634
#WARNING:  speaker G00000000 speaking more than once at time 1162.832
#WARNING:  speaker G00000000 speaking more than once at time 1164.312
#WARNING:  speaker G00000000 speaking more than once at time 1241.203
#WARNING:  speaker G00000000 speaking more than once at time 1260.024
#WARNING:  speaker G00000000 speaking more than once at time 1308.680
#WARNING:  speaker G00000000 speaking more than once at time 1588.259
#WARNING:  speaker G00000000 speaking more than once at time 1731.002
#WARNING:  speaker G00000000 speaking more than once at time 1732.232
#WARNING:  speaker G00000000 speaking more than once at time 1741.378
#WARNING:  speaker G00000000 speaking more than once at time 1746.944
#WARNING:  speaker G00000000 speaking more than once at time 1751.770
#WARNING:  speaker G00000000 speaking more than once at time 1791.312
#WARNING:  speaker G00000000 speaking more than once at time 1791.968
#WARNING:  speaker G00000000 speaking more than once at time 1825.248
#WARNING:  speaker G00000243 speaking more than once at time 73.258
#WARNING:  speaker G00000243 speaking more than once at time 73.258
#WARNING:  speaker G00000537 speaking more than once at time 297.482
#WARNING:  speaker G00000537 speaking more than once at time 297.482
#WARNING:  speaker G00000284 speaking more than once at time 357.992
#WARNING:  speaker G00000284 speaking more than once at time 482.832
#WARNING:  speaker G00000285 speaking more than once at time 1135.592
#WARNING:  speaker G00000284 speaking more than once at time 1421.712
#WARNING:  speaker G00000285 speaking more than once at time 1442.656
#WARNING:  speaker G00000284 speaking more than once at time 1508.323
#WARNING:  speaker G00000284 speaking more than once at time 1630.563
#WARNING:  speaker G00000284 speaking more than once at time 1798.032
#WARNING:  speaker G00000285 speaking more than once at time 1886.980
#WARNING:  speaker G00000284 speaking more than once at time 357.992
#WARNING:  speaker G00000284 speaking more than once at time 482.832
#WARNING:  speaker G00000285 speaking more than once at time 1135.592
#WARNING:  speaker G00000284 speaking more than once at time 1421.712
#WARNING:  speaker G00000285 speaking more than once at time 1442.656
#WARNING:  speaker G00000284 speaking more than once at time 1508.323
#WARNING:  speaker G00000284 speaking more than once at time 1630.563
#WARNING:  speaker G00000284 speaking more than once at time 1798.032
#WARNING:  speaker G00000285 speaking more than once at time 1886.980
#WARNING:  speaker G00000326 speaking more than once at time 387.722
#WARNING:  speaker G00000327 speaking more than once at time 632.136
#WARNING:  speaker G00000326 speaking more than once at time 839.436
#WARNING:  speaker G00000000 speaking more than once at time 1013.128
#WARNING:  speaker G00000327 speaking more than once at time 1241.096
#WARNING:  speaker G00000327 speaking more than once at time 1695.756
#WARNING:  speaker G00000326 speaking more than once at time 387.722
#WARNING:  speaker G00000327 speaking more than once at time 632.136
#WARNING:  speaker G00000326 speaking more than once at time 839.436
#WARNING:  speaker G00000000 speaking more than once at time 1013.128
#WARNING:  speaker G00000327 speaking more than once at time 1241.096
#WARNING:  speaker G00000327 speaking more than once at time 1695.756
#WARNING:  speaker G00000490 speaking more than once at time 405.167
#WARNING:  speaker G00000489 speaking more than once at time 678.490
#WARNING:  speaker G00000490 speaking more than once at time 1127.747
#WARNING:  speaker G00000490 speaking more than once at time 1860.903
#WARNING:  speaker G00000490 speaking more than once at time 405.167
#WARNING:  speaker G00000489 speaking more than once at time 678.490
#WARNING:  speaker G00000490 speaking more than once at time 1127.747
#WARNING:  speaker G00000490 speaking more than once at time 1860.903
#WARNING:  speaker G00000000 speaking more than once at time 24.746
#WARNING:  speaker G00000000 speaking more than once at time 1266.368
#WARNING:  speaker G00000000 speaking more than once at time 24.746
#WARNING:  speaker G00000000 speaking more than once at time 1266.368
#WARNING:  speaker G00000528 speaking more than once at time 1780.944
#WARNING:  speaker G00000528 speaking more than once at time 1780.944
#WARNING:  speaker G00000534 speaking more than once at time 116.488
#WARNING:  speaker G00000000 speaking more than once at time 361.608
#WARNING:  speaker G00000533 speaking more than once at time 543.008
#WARNING:  speaker G00000533 speaking more than once at time 666.832
#WARNING:  speaker G00000533 speaking more than once at time 1035.624
#WARNING:  speaker G00000533 speaking more than once at time 1793.704
#WARNING:  speaker G00000534 speaking more than once at time 116.488
#WARNING:  speaker G00000000 speaking more than once at time 361.608
#WARNING:  speaker G00000533 speaking more than once at time 543.008
#WARNING:  speaker G00000533 speaking more than once at time 666.832
#WARNING:  speaker G00000533 speaking more than once at time 1035.624
#WARNING:  speaker G00000533 speaking more than once at time 1793.704
#WARNING:  speaker G00000621 speaking more than once at time 377.309
#WARNING:  speaker G00000621 speaking more than once at time 477.125
#WARNING:  speaker G00000000 speaking more than once at time 547.109
#WARNING:  speaker G00000622 speaking more than once at time 855.493
#WARNING:  speaker G00000622 speaking more than once at time 864.773
#WARNING:  speaker G00000621 speaking more than once at time 899.869
#WARNING:  speaker G00000621 speaking more than once at time 916.613
#WARNING:  speaker G00000621 speaking more than once at time 992.837
#WARNING:  speaker G00000621 speaking more than once at time 1235.525
#WARNING:  speaker G00000621 speaking more than once at time 1722.885
#WARNING:  speaker G00000622 speaking more than once at time 1774.429
#WARNING:  speaker G00000621 speaking more than once at time 1818.725
#WARNING:  speaker G00000621 speaking more than once at time 377.309
#WARNING:  speaker G00000621 speaking more than once at time 477.125
#WARNING:  speaker G00000000 speaking more than once at time 547.109
#WARNING:  speaker G00000622 speaking more than once at time 855.493
#WARNING:  speaker G00000622 speaking more than once at time 864.773
#WARNING:  speaker G00000621 speaking more than once at time 899.869
#WARNING:  speaker G00000621 speaking more than once at time 916.613
#WARNING:  speaker G00000621 speaking more than once at time 992.837
#WARNING:  speaker G00000621 speaking more than once at time 1235.525
#WARNING:  speaker G00000621 speaking more than once at time 1722.885
#WARNING:  speaker G00000622 speaking more than once at time 1774.429
#WARNING:  speaker G00000621 speaking more than once at time 1818.725
#WARNING:  speaker G00000680 speaking more than once at time 130.024
#WARNING:  speaker G00000000 speaking more than once at time 209.704
#WARNING:  speaker G00000680 speaking more than once at time 256.099
#WARNING:  speaker G00000000 speaking more than once at time 280.673
#WARNING:  speaker G00000000 speaking more than once at time 312.328
#WARNING:  speaker G00000680 speaking more than once at time 804.904
#WARNING:  speaker G00000680 speaking more than once at time 1129.232
#WARNING:  speaker G00000680 speaking more than once at time 1223.816
#WARNING:  speaker G00000680 speaking more than once at time 1411.528
#WARNING:  speaker G00000680 speaking more than once at time 1536.809
#WARNING:  speaker G00000000 speaking more than once at time 1730.288
#WARNING:  speaker G00000680 speaking more than once at time 130.024
#WARNING:  speaker G00000000 speaking more than once at time 209.704
#WARNING:  speaker G00000680 speaking more than once at time 256.099
#WARNING:  speaker G00000000 speaking more than once at time 280.673
#WARNING:  speaker G00000000 speaking more than once at time 312.328
#WARNING:  speaker G00000680 speaking more than once at time 804.904
#WARNING:  speaker G00000680 speaking more than once at time 1129.232
#WARNING:  speaker G00000680 speaking more than once at time 1223.816
#WARNING:  speaker G00000680 speaking more than once at time 1411.528
#WARNING:  speaker G00000680 speaking more than once at time 1536.809
#WARNING:  speaker G00000000 speaking more than once at time 1730.288
#WARNING:  speaker G00000392 speaking more than once at time 795.224
#WARNING:  speaker G00000391 speaking more than once at time 809.608
#WARNING:  speaker G00000391 speaking more than once at time 1701.352
#WARNING:  speaker G00000392 speaking more than once at time 795.224
#WARNING:  speaker G00000391 speaking more than once at time 809.608
#WARNING:  speaker G00000391 speaking more than once at time 1701.352
#WARNING:  speaker G00000430 speaking more than once at time 1307.916
#WARNING:  speaker G00000430 speaking more than once at time 1667.180
#WARNING:  speaker G00000429 speaking more than once at time 1823.436
#WARNING:  speaker G00000430 speaking more than once at time 1307.916
#WARNING:  speaker G00000430 speaking more than once at time 1667.180
#WARNING:  speaker G00000429 speaking more than once at time 1823.436
#WARNING:  speaker G00000662 speaking more than once at time 31.144
#WARNING:  speaker G00000000 speaking more than once at time 265.232
#WARNING:  speaker G00000661 speaking more than once at time 358.408
#WARNING:  speaker G00000661 speaking more than once at time 752.928
#WARNING:  speaker G00000661 speaking more than once at time 900.488
#WARNING:  speaker G00000662 speaking more than once at time 1328.008
#WARNING:  speaker G00000662 speaking more than once at time 1504.744
#WARNING:  speaker G00000662 speaking more than once at time 1852.488
#WARNING:  speaker G00000662 speaking more than once at time 31.144
#WARNING:  speaker G00000000 speaking more than once at time 265.232
#WARNING:  speaker G00000661 speaking more than once at time 358.408
#WARNING:  speaker G00000661 speaking more than once at time 752.928
#WARNING:  speaker G00000661 speaking more than once at time 900.488
#WARNING:  speaker G00000662 speaking more than once at time 1328.008
#WARNING:  speaker G00000662 speaking more than once at time 1504.744
#WARNING:  speaker G00000662 speaking more than once at time 1852.488
#WARNING:  speaker G00000000 speaking more than once at time 424.078
#WARNING:  speaker G00000000 speaking more than once at time 726.736
#WARNING:  speaker G00000141 speaking more than once at time 1731.995
#WARNING:  speaker G00000000 speaking more than once at time 424.078
#WARNING:  speaker G00000000 speaking more than once at time 726.736
#WARNING:  speaker G00000141 speaking more than once at time 1731.995
#WARNING:  speaker G00000322 speaking more than once at time 1789.851
#WARNING:  speaker G00000322 speaking more than once at time 1789.851
#WARNING:  speaker G00000155 speaking more than once at time 1816.256
#WARNING:  speaker G00000155 speaking more than once at time 1816.256
#WARNING:  speaker G00000000 speaking more than once at time 950.478
#WARNING:  speaker G00000341 speaking more than once at time 1127.215
#WARNING:  speaker G00000000 speaking more than once at time 1579.866
#WARNING:  speaker G00000000 speaking more than once at time 950.478
#WARNING:  speaker G00000341 speaking more than once at time 1127.215
#WARNING:  speaker G00000000 speaking more than once at time 1579.866
#WARNING:  speaker G00000000 speaking more than once at time 334.738
#WARNING:  speaker G00000000 speaking more than once at time 783.496
#WARNING:  speaker G00000000 speaking more than once at time 783.784
#WARNING:  speaker G00000094 speaking more than once at time 1272.240
#WARNING:  speaker G00000000 speaking more than once at time 1351.680
#WARNING:  speaker G00000000 speaking more than once at time 1725.120
#WARNING:  speaker G00000000 speaking more than once at time 334.738
#WARNING:  speaker G00000000 speaking more than once at time 783.496
#WARNING:  speaker G00000000 speaking more than once at time 783.784
#WARNING:  speaker G00000094 speaking more than once at time 1272.240
#WARNING:  speaker G00000000 speaking more than once at time 1351.680
#WARNING:  speaker G00000000 speaking more than once at time 1725.120
#34.59/30.26/0.01/4.32

fi

#if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
#   echo "cpu get embedding"
#	python scripts/magicdata-RAMC/020_spectral_cluster_v2.py
#   echo "finish"
#fi

# if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
#     echo "clustering ......."
#     python scripts/magicdata-RAMC/020_spectral_cluster2.py
# fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
    echo "oracle clustering ......."
    python scripts/magicdata-RAMC/020_spectral_cluster2_ora.py
   #  SCTK-2.4.12/src/md-eval/md-eval.pl -c 0.25 -r data/magicdata-RAMC_debug/test/rttm -s data/magicdata-RAM
#C_debug/test/rttm_debug2
#WARNING:  speaker G00000169 speaking more than once at time 124.835
#WARNING:  speaker G00000169 speaking more than once at time 124.835
#WARNING:  speaker G00000000 speaking more than once at time 242.056
#WARNING:  speaker G00000168 speaking more than once at time 359.176
#WARNING:  speaker G00000168 speaking more than once at time 359.176
#WARNING:  speaker G00000168 speaking more than once at time 388.331
#WARNING:  speaker G00000168 speaking more than once at time 388.331
#WARNING:  speaker G00000000 speaking more than once at time 646.813
#WARNING:  speaker G00000168 speaking more than once at time 906.248
#WARNING:  speaker G00000168 speaking more than once at time 906.248
#WARNING:  speaker G00000000 speaking more than once at time 993.704
#WARNING:  speaker G00000168 speaking more than once at time 1238.088
#WARNING:  speaker G00000168 speaking more than once at time 1238.088
#WARNING:  speaker G00000000 speaking more than once at time 1256.272
#WARNING:  speaker G00000169 speaking more than once at time 1363.458
#WARNING:  speaker G00000169 speaking more than once at time 1363.458
#WARNING:  speaker G00000168 speaking more than once at time 1471.048
#WARNING:  speaker G00000168 speaking more than once at time 1471.048
#WARNING:  speaker G00000000 speaking more than once at time 1602.984
#WARNING:  speaker G00000169 speaking more than once at time 1780.168
#WARNING:  speaker G00000169 speaking more than once at time 1780.168
#WARNING:  speaker G00000169 speaking more than once at time 124.835
#WARNING:  speaker G00000169 speaking more than once at time 124.835
#WARNING:  speaker G00000000 speaking more than once at time 242.056
#WARNING:  speaker G00000168 speaking more than once at time 359.176
#WARNING:  speaker G00000168 speaking more than once at time 359.176
#WARNING:  speaker G00000168 speaking more than once at time 388.331
#WARNING:  speaker G00000168 speaking more than once at time 388.331
#WARNING:  speaker G00000000 speaking more than once at time 646.813
#WARNING:  speaker G00000168 speaking more than once at time 906.248
#WARNING:  speaker G00000168 speaking more than once at time 906.248
#WARNING:  speaker G00000000 speaking more than once at time 993.704
#WARNING:  speaker G00000168 speaking more than once at time 1238.088
#WARNING:  speaker G00000168 speaking more than once at time 1238.088
#WARNING:  speaker G00000000 speaking more than once at time 1256.272
#WARNING:  speaker G00000169 speaking more than once at time 1363.458
#WARNING:  speaker G00000169 speaking more than once at time 1363.458
#WARNING:  speaker G00000168 speaking more than once at time 1471.048
#WARNING:  speaker G00000168 speaking more than once at time 1471.048
#WARNING:  speaker G00000000 speaking more than once at time 1602.984
#WARNING:  speaker G00000169 speaking more than once at time 1780.168
#WARNING:  speaker G00000169 speaking more than once at time 1780.168
#WARNING:  speaker G00000000 speaking more than once at time 447.416
#WARNING:  speaker G00000000 speaking more than once at time 447.416
#WARNING:  speaker G00000464 speaking more than once at time 452.378
#WARNING:  speaker G00000464 speaking more than once at time 452.378
#WARNING:  speaker G00000464 speaking more than once at time 452.378
#WARNING:  speaker G00000464 speaking more than once at time 452.378
#WARNING:  speaker G00000131 speaking more than once at time 528.962
#WARNING:  speaker G00000131 speaking more than once at time 528.962
#WARNING:  speaker G00000131 speaking more than once at time 532.848
#WARNING:  speaker G00000131 speaking more than once at time 532.848
#WARNING:  speaker G00000131 speaking more than once at time 528.962
#WARNING:  speaker G00000131 speaking more than once at time 528.962
#WARNING:  speaker G00000131 speaking more than once at time 532.848
#WARNING:  speaker G00000131 speaking more than once at time 532.848
#WARNING:  speaker G00000185 speaking more than once at time 1421.653
#WARNING:  speaker G00000185 speaking more than once at time 1421.653
#WARNING:  speaker G00000185 speaking more than once at time 1421.653
#WARNING:  speaker G00000185 speaking more than once at time 1421.653
#WARNING:  speaker G00000290 speaking more than once at time 1526.856
#WARNING:  speaker G00000290 speaking more than once at time 1526.856
#WARNING:  speaker G00000290 speaking more than once at time 1526.856
#WARNING:  speaker G00000290 speaking more than once at time 1526.856
#WARNING:  speaker G00000583 speaking more than once at time 280.592
#WARNING:  speaker G00000583 speaking more than once at time 280.592
#WARNING:  speaker G00000583 speaking more than once at time 406.701
#WARNING:  speaker G00000583 speaking more than once at time 406.701
#WARNING:  speaker G00000583 speaking more than once at time 971.588
#WARNING:  speaker G00000583 speaking more than once at time 971.588
#WARNING:  speaker G00000584 speaking more than once at time 1181.130
#WARNING:  speaker G00000584 speaking more than once at time 1181.130
#WARNING:  speaker G00000583 speaking more than once at time 1333.227
#WARNING:  speaker G00000583 speaking more than once at time 1333.227
#WARNING:  speaker G00000583 speaking more than once at time 1438.255
#WARNING:  speaker G00000583 speaking more than once at time 1438.255
#WARNING:  speaker G00000583 speaking more than once at time 1455.758
#WARNING:  speaker G00000583 speaking more than once at time 1455.758
#WARNING:  speaker G00000583 speaking more than once at time 1714.851
#WARNING:  speaker G00000583 speaking more than once at time 1714.851
#WARNING:  speaker G00000583 speaking more than once at time 280.592
#WARNING:  speaker G00000583 speaking more than once at time 280.592
#WARNING:  speaker G00000583 speaking more than once at time 406.701
#WARNING:  speaker G00000583 speaking more than once at time 406.701
#WARNING:  speaker G00000583 speaking more than once at time 971.588
#WARNING:  speaker G00000583 speaking more than once at time 971.588
#WARNING:  speaker G00000584 speaking more than once at time 1181.130
#WARNING:  speaker G00000584 speaking more than once at time 1181.130
#WARNING:  speaker G00000583 speaking more than once at time 1333.227
#WARNING:  speaker G00000583 speaking more than once at time 1333.227
#WARNING:  speaker G00000583 speaking more than once at time 1438.255
#WARNING:  speaker G00000583 speaking more than once at time 1438.255
#WARNING:  speaker G00000583 speaking more than once at time 1455.758
#WARNING:  speaker G00000583 speaking more than once at time 1455.758
#WARNING:  speaker G00000583 speaking more than once at time 1714.851
#WARNING:  speaker G00000583 speaking more than once at time 1714.851
#WARNING:  speaker G00000639 speaking more than once at time 1024.776
#WARNING:  speaker G00000639 speaking more than once at time 1024.776
#WARNING:  speaker G00000000 speaking more than once at time 1499.859
#WARNING:  speaker G00000639 speaking more than once at time 1024.776
#WARNING:  speaker G00000639 speaking more than once at time 1024.776
#WARNING:  speaker G00000000 speaking more than once at time 1499.859
#WARNING:  speaker G00000364 speaking more than once at time 324.970
#WARNING:  speaker G00000364 speaking more than once at time 324.970
#WARNING:  speaker G00000364 speaking more than once at time 1355.424
#WARNING:  speaker G00000364 speaking more than once at time 1355.424
#WARNING:  speaker G00000364 speaking more than once at time 324.970
#WARNING:  speaker G00000364 speaking more than once at time 324.970
#WARNING:  speaker G00000364 speaking more than once at time 1355.424
#WARNING:  speaker G00000364 speaking more than once at time 1355.424
#WARNING:  speaker G00000231 speaking more than once at time 371.369
#WARNING:  speaker G00000231 speaking more than once at time 371.369
#WARNING:  speaker G00000231 speaking more than once at time 933.760
#WARNING:  speaker G00000231 speaking more than once at time 933.760
#WARNING:  speaker G00000000 speaking more than once at time 1094.816
#WARNING:  speaker G00000231 speaking more than once at time 1173.609
#WARNING:  speaker G00000231 speaking more than once at time 1173.609
#WARNING:  speaker G00000230 speaking more than once at time 1183.944
#WARNING:  speaker G00000230 speaking more than once at time 1183.944
#WARNING:  speaker G00000231 speaking more than once at time 1236.736
#WARNING:  speaker G00000231 speaking more than once at time 1236.736
#WARNING:  speaker G00000231 speaking more than once at time 1503.376
#WARNING:  speaker G00000231 speaking more than once at time 1503.376
#WARNING:  speaker G00000231 speaking more than once at time 1542.736
#WARNING:  speaker G00000231 speaking more than once at time 1542.736
#WARNING:  speaker G00000231 speaking more than once at time 1584.552
#WARNING:  speaker G00000231 speaking more than once at time 1584.552
#WARNING:  speaker G00000231 speaking more than once at time 1600.000
#WARNING:  speaker G00000231 speaking more than once at time 1600.000
#WARNING:  speaker G00000231 speaking more than once at time 1656.000
#WARNING:  speaker G00000231 speaking more than once at time 1656.000
#WARNING:  speaker G00000231 speaking more than once at time 1671.752
#WARNING:  speaker G00000231 speaking more than once at time 1671.752
#WARNING:  speaker G00000230 speaking more than once at time 1737.416
#WARNING:  speaker G00000230 speaking more than once at time 1737.416
#WARNING:  speaker G00000230 speaking more than once at time 1752.809
#WARNING:  speaker G00000230 speaking more than once at time 1752.809
#WARNING:  speaker G00000231 speaking more than once at time 1802.152
#WARNING:  speaker G00000231 speaking more than once at time 1802.152
#WARNING:  speaker G00000231 speaking more than once at time 371.369
#WARNING:  speaker G00000231 speaking more than once at time 371.369
#WARNING:  speaker G00000231 speaking more than once at time 933.760
#WARNING:  speaker G00000231 speaking more than once at time 933.760
#WARNING:  speaker G00000000 speaking more than once at time 1094.816
#WARNING:  speaker G00000231 speaking more than once at time 1173.609
#WARNING:  speaker G00000231 speaking more than once at time 1173.609
#WARNING:  speaker G00000230 speaking more than once at time 1183.944
#WARNING:  speaker G00000230 speaking more than once at time 1183.944
#WARNING:  speaker G00000231 speaking more than once at time 1236.736
#WARNING:  speaker G00000231 speaking more than once at time 1236.736
#WARNING:  speaker G00000231 speaking more than once at time 1503.376
#WARNING:  speaker G00000231 speaking more than once at time 1503.376
#WARNING:  speaker G00000231 speaking more than once at time 1542.736
#WARNING:  speaker G00000231 speaking more than once at time 1542.736
#WARNING:  speaker G00000231 speaking more than once at time 1584.552
#WARNING:  speaker G00000231 speaking more than once at time 1584.552
#WARNING:  speaker G00000231 speaking more than once at time 1600.000
#WARNING:  speaker G00000231 speaking more than once at time 1600.000
#WARNING:  speaker G00000231 speaking more than once at time 1656.000
#WARNING:  speaker G00000231 speaking more than once at time 1656.000
#WARNING:  speaker G00000231 speaking more than once at time 1671.752
#WARNING:  speaker G00000231 speaking more than once at time 1671.752
#WARNING:  speaker G00000230 speaking more than once at time 1737.416
#WARNING:  speaker G00000230 speaking more than once at time 1737.416
#WARNING:  speaker G00000230 speaking more than once at time 1752.809
#WARNING:  speaker G00000230 speaking more than once at time 1752.809
#WARNING:  speaker G00000231 speaking more than once at time 1802.152
#WARNING:  speaker G00000231 speaking more than once at time 1802.152
#WARNING:  speaker G00000000 speaking more than once at time 824.512
#WARNING:  speaker G00000283 speaking more than once at time 1151.936
#WARNING:  speaker G00000283 speaking more than once at time 1151.936
#WARNING:  speaker G00000282 speaking more than once at time 1329.656
#WARNING:  speaker G00000282 speaking more than once at time 1329.656
#WARNING:  speaker G00000283 speaking more than once at time 1500.328
#WARNING:  speaker G00000283 speaking more than once at time 1500.328
#WARNING:  speaker G00000283 speaking more than once at time 1501.544
#WARNING:  speaker G00000283 speaking more than once at time 1501.544
#WARNING:  speaker G00000283 speaking more than once at time 1765.672
#WARNING:  speaker G00000283 speaking more than once at time 1765.672
#WARNING:  speaker G00000283 speaking more than once at time 1795.024
#WARNING:  speaker G00000283 speaking more than once at time 1795.024
#WARNING:  speaker G00000283 speaking more than once at time 1844.736
#WARNING:  speaker G00000283 speaking more than once at time 1844.736
#WARNING:  speaker G00000283 speaking more than once at time 1915.304
#WARNING:  speaker G00000283 speaking more than once at time 1915.304
#WARNING:  speaker G00000000 speaking more than once at time 824.512
#WARNING:  speaker G00000283 speaking more than once at time 1151.936
#WARNING:  speaker G00000283 speaking more than once at time 1151.936
#WARNING:  speaker G00000282 speaking more than once at time 1329.656
#WARNING:  speaker G00000282 speaking more than once at time 1329.656
#WARNING:  speaker G00000283 speaking more than once at time 1500.328
#WARNING:  speaker G00000283 speaking more than once at time 1500.328
#WARNING:  speaker G00000283 speaking more than once at time 1501.544
#WARNING:  speaker G00000283 speaking more than once at time 1501.544
#WARNING:  speaker G00000283 speaking more than once at time 1765.672
#WARNING:  speaker G00000283 speaking more than once at time 1765.672
#WARNING:  speaker G00000283 speaking more than once at time 1795.024
#WARNING:  speaker G00000283 speaking more than once at time 1795.024
#WARNING:  speaker G00000283 speaking more than once at time 1844.736
#WARNING:  speaker G00000283 speaking more than once at time 1844.736
#WARNING:  speaker G00000283 speaking more than once at time 1915.304
#WARNING:  speaker G00000283 speaking more than once at time 1915.304
#WARNING:  speaker G00000000 speaking more than once at time 581.612
#WARNING:  speaker G00000000 speaking more than once at time 745.872
#WARNING:  speaker G00000000 speaking more than once at time 1265.952
#WARNING:  speaker G00000000 speaking more than once at time 581.612
#WARNING:  speaker G00000000 speaking more than once at time 745.872
#WARNING:  speaker G00000000 speaking more than once at time 1265.952
#WARNING:  speaker G00000000 speaking more than once at time 490.930
#WARNING:  speaker G00000671 speaking more than once at time 922.378
#WARNING:  speaker G00000671 speaking more than once at time 922.378
#WARNING:  speaker G00000000 speaking more than once at time 1101.386
#WARNING:  speaker G00000000 speaking more than once at time 1101.954
#WARNING:  speaker G00000672 speaking more than once at time 1106.322
#WARNING:  speaker G00000672 speaking more than once at time 1106.322
#WARNING:  speaker G00000672 speaking more than once at time 1110.122
#WARNING:  speaker G00000672 speaking more than once at time 1110.122
#WARNING:  speaker G00000672 speaking more than once at time 1151.858
#WARNING:  speaker G00000672 speaking more than once at time 1151.858
#WARNING:  speaker G00000672 speaking more than once at time 1172.058
#WARNING:  speaker G00000672 speaking more than once at time 1172.058
#WARNING:  speaker G00000671 speaking more than once at time 1292.322
#WARNING:  speaker G00000671 speaking more than once at time 1292.322
#WARNING:  speaker G00000000 speaking more than once at time 1512.938
#WARNING:  speaker G00000000 speaking more than once at time 1541.690
#WARNING:  speaker G00000672 speaking more than once at time 1841.738
#WARNING:  speaker G00000672 speaking more than once at time 1841.738
#WARNING:  speaker G00000672 speaking more than once at time 1848.514
#WARNING:  speaker G00000672 speaking more than once at time 1848.514
#WARNING:  speaker G00000000 speaking more than once at time 490.930
#WARNING:  speaker G00000671 speaking more than once at time 922.378
#WARNING:  speaker G00000671 speaking more than once at time 922.378
#WARNING:  speaker G00000000 speaking more than once at time 1101.386
#WARNING:  speaker G00000000 speaking more than once at time 1101.954
#WARNING:  speaker G00000672 speaking more than once at time 1106.322
#WARNING:  speaker G00000672 speaking more than once at time 1106.322
#WARNING:  speaker G00000672 speaking more than once at time 1110.122
#WARNING:  speaker G00000672 speaking more than once at time 1110.122
#WARNING:  speaker G00000672 speaking more than once at time 1151.858
#WARNING:  speaker G00000672 speaking more than once at time 1151.858
#WARNING:  speaker G00000672 speaking more than once at time 1172.058
#WARNING:  speaker G00000672 speaking more than once at time 1172.058
#WARNING:  speaker G00000671 speaking more than once at time 1292.322
#WARNING:  speaker G00000671 speaking more than once at time 1292.322
#WARNING:  speaker G00000000 speaking more than once at time 1512.938
#WARNING:  speaker G00000000 speaking more than once at time 1541.690
#WARNING:  speaker G00000672 speaking more than once at time 1841.738
#WARNING:  speaker G00000672 speaking more than once at time 1841.738
#WARNING:  speaker G00000672 speaking more than once at time 1848.514
#WARNING:  speaker G00000672 speaking more than once at time 1848.514
#WARNING:  speaker G00000000 speaking more than once at time 51.944
#WARNING:  speaker G00000000 speaking more than once at time 168.248
#WARNING:  speaker G00000000 speaking more than once at time 929.808
#WARNING:  speaker G00000000 speaking more than once at time 959.712
#WARNING:  speaker G00000000 speaking more than once at time 1400.736
#WARNING:  speaker G00000000 speaking more than once at time 51.944
#WARNING:  speaker G00000000 speaking more than once at time 168.248
#WARNING:  speaker G00000000 speaking more than once at time 929.808
#WARNING:  speaker G00000000 speaking more than once at time 959.712
#WARNING:  speaker G00000000 speaking more than once at time 1400.736
#WARNING:  speaker G00000045 speaking more than once at time 103.272
#WARNING:  speaker G00000045 speaking more than once at time 103.272
#WARNING:  speaker G00000045 speaking more than once at time 506.368
#WARNING:  speaker G00000045 speaking more than once at time 506.368
#WARNING:  speaker G00000045 speaking more than once at time 735.968
#WARNING:  speaker G00000045 speaking more than once at time 735.968
#WARNING:  speaker G00000046 speaking more than once at time 1088.608
#WARNING:  speaker G00000046 speaking more than once at time 1088.608
#WARNING:  speaker G00000045 speaking more than once at time 103.272
#WARNING:  speaker G00000045 speaking more than once at time 103.272
#WARNING:  speaker G00000045 speaking more than once at time 506.368
#WARNING:  speaker G00000045 speaking more than once at time 506.368
#WARNING:  speaker G00000045 speaking more than once at time 735.968
#WARNING:  speaker G00000045 speaking more than once at time 735.968
#WARNING:  speaker G00000046 speaking more than once at time 1088.608
#WARNING:  speaker G00000046 speaking more than once at time 1088.608
#WARNING:  speaker G00000000 speaking more than once at time 799.053
#WARNING:  speaker G00000000 speaking more than once at time 1017.744
#WARNING:  speaker G00000000 speaking more than once at time 799.053
#WARNING:  speaker G00000000 speaking more than once at time 1017.744
#WARNING:  speaker G00000000 speaking more than once at time 53.256
#WARNING:  speaker G00000000 speaking more than once at time 268.505
#WARNING:  speaker G00000000 speaking more than once at time 407.201
#WARNING:  speaker G00000000 speaking more than once at time 433.528
#WARNING:  speaker G00000000 speaking more than once at time 757.544
#WARNING:  speaker G00000000 speaking more than once at time 871.482
#WARNING:  speaker G00000000 speaking more than once at time 1035.712
#WARNING:  speaker G00000000 speaking more than once at time 1057.248
#WARNING:  speaker G00000092 speaking more than once at time 1142.634
#WARNING:  speaker G00000092 speaking more than once at time 1142.634
#WARNING:  speaker G00000000 speaking more than once at time 1162.832
#WARNING:  speaker G00000000 speaking more than once at time 1164.312
#WARNING:  speaker G00000000 speaking more than once at time 1241.203
#WARNING:  speaker G00000000 speaking more than once at time 1260.024
#WARNING:  speaker G00000000 speaking more than once at time 1308.680
#WARNING:  speaker G00000000 speaking more than once at time 1588.259
#WARNING:  speaker G00000000 speaking more than once at time 1731.002
#WARNING:  speaker G00000000 speaking more than once at time 1732.232
#WARNING:  speaker G00000000 speaking more than once at time 1741.378
#WARNING:  speaker G00000000 speaking more than once at time 1746.944
#WARNING:  speaker G00000000 speaking more than once at time 1751.770
#WARNING:  speaker G00000000 speaking more than once at time 1791.312
#WARNING:  speaker G00000000 speaking more than once at time 1791.968
#WARNING:  speaker G00000000 speaking more than once at time 1825.248
#WARNING:  speaker G00000000 speaking more than once at time 53.256
#WARNING:  speaker G00000000 speaking more than once at time 268.505
#WARNING:  speaker G00000000 speaking more than once at time 407.201
#WARNING:  speaker G00000000 speaking more than once at time 433.528
#WARNING:  speaker G00000000 speaking more than once at time 757.544
#WARNING:  speaker G00000000 speaking more than once at time 871.482
#WARNING:  speaker G00000000 speaking more than once at time 1035.712
#WARNING:  speaker G00000000 speaking more than once at time 1057.248
#WARNING:  speaker G00000092 speaking more than once at time 1142.634
#WARNING:  speaker G00000092 speaking more than once at time 1142.634
#WARNING:  speaker G00000000 speaking more than once at time 1162.832
#WARNING:  speaker G00000000 speaking more than once at time 1164.312
#WARNING:  speaker G00000000 speaking more than once at time 1241.203
#WARNING:  speaker G00000000 speaking more than once at time 1260.024
#WARNING:  speaker G00000000 speaking more than once at time 1308.680
#WARNING:  speaker G00000000 speaking more than once at time 1588.259
#WARNING:  speaker G00000000 speaking more than once at time 1731.002
#WARNING:  speaker G00000000 speaking more than once at time 1732.232
#WARNING:  speaker G00000000 speaking more than once at time 1741.378
#WARNING:  speaker G00000000 speaking more than once at time 1746.944
#WARNING:  speaker G00000000 speaking more than once at time 1751.770
#WARNING:  speaker G00000000 speaking more than once at time 1791.312
#WARNING:  speaker G00000000 speaking more than once at time 1791.968
#WARNING:  speaker G00000000 speaking more than once at time 1825.248
#WARNING:  speaker G00000243 speaking more than once at time 73.258
#WARNING:  speaker G00000243 speaking more than once at time 73.258
#WARNING:  speaker G00000243 speaking more than once at time 73.258
#WARNING:  speaker G00000243 speaking more than once at time 73.258
#WARNING:  speaker G00000537 speaking more than once at time 297.482
#WARNING:  speaker G00000537 speaking more than once at time 297.482
#WARNING:  speaker G00000537 speaking more than once at time 297.482
#WARNING:  speaker G00000537 speaking more than once at time 297.482
#WARNING:  speaker G00000284 speaking more than once at time 357.992
#WARNING:  speaker G00000284 speaking more than once at time 357.992
#WARNING:  speaker G00000284 speaking more than once at time 482.832
#WARNING:  speaker G00000284 speaking more than once at time 482.832
#WARNING:  speaker G00000285 speaking more than once at time 1135.592
#WARNING:  speaker G00000285 speaking more than once at time 1135.592
#WARNING:  speaker G00000284 speaking more than once at time 1421.712
#WARNING:  speaker G00000284 speaking more than once at time 1421.712
#WARNING:  speaker G00000285 speaking more than once at time 1442.656
#WARNING:  speaker G00000285 speaking more than once at time 1442.656
#WARNING:  speaker G00000284 speaking more than once at time 1508.323
#WARNING:  speaker G00000284 speaking more than once at time 1508.323
#WARNING:  speaker G00000284 speaking more than once at time 1630.563
#WARNING:  speaker G00000284 speaking more than once at time 1630.563
#WARNING:  speaker G00000284 speaking more than once at time 1798.032
#WARNING:  speaker G00000284 speaking more than once at time 1798.032
#WARNING:  speaker G00000285 speaking more than once at time 1886.980
#WARNING:  speaker G00000285 speaking more than once at time 1886.980
#WARNING:  speaker G00000284 speaking more than once at time 357.992
#WARNING:  speaker G00000284 speaking more than once at time 357.992
#WARNING:  speaker G00000284 speaking more than once at time 482.832
#WARNING:  speaker G00000284 speaking more than once at time 482.832
#WARNING:  speaker G00000285 speaking more than once at time 1135.592
#WARNING:  speaker G00000285 speaking more than once at time 1135.592
#WARNING:  speaker G00000284 speaking more than once at time 1421.712
#WARNING:  speaker G00000284 speaking more than once at time 1421.712
#WARNING:  speaker G00000285 speaking more than once at time 1442.656
#WARNING:  speaker G00000285 speaking more than once at time 1442.656
#WARNING:  speaker G00000284 speaking more than once at time 1508.323
#WARNING:  speaker G00000284 speaking more than once at time 1508.323
#WARNING:  speaker G00000284 speaking more than once at time 1630.563
#WARNING:  speaker G00000284 speaking more than once at time 1630.563
#WARNING:  speaker G00000284 speaking more than once at time 1798.032
#WARNING:  speaker G00000284 speaking more than once at time 1798.032
#WARNING:  speaker G00000285 speaking more than once at time 1886.980
#WARNING:  speaker G00000285 speaking more than once at time 1886.980
#WARNING:  speaker G00000326 speaking more than once at time 387.722
#WARNING:  speaker G00000326 speaking more than once at time 387.722
#WARNING:  speaker G00000327 speaking more than once at time 632.136
#WARNING:  speaker G00000327 speaking more than once at time 632.136
#WARNING:  speaker G00000326 speaking more than once at time 839.436
#WARNING:  speaker G00000326 speaking more than once at time 839.436
#WARNING:  speaker G00000000 speaking more than once at time 1013.128
#WARNING:  speaker G00000327 speaking more than once at time 1241.096
#WARNING:  speaker G00000327 speaking more than once at time 1241.096
#WARNING:  speaker G00000327 speaking more than once at time 1695.756
#WARNING:  speaker G00000327 speaking more than once at time 1695.756
#WARNING:  speaker G00000326 speaking more than once at time 387.722
#WARNING:  speaker G00000326 speaking more than once at time 387.722
#WARNING:  speaker G00000327 speaking more than once at time 632.136
#WARNING:  speaker G00000327 speaking more than once at time 632.136
#WARNING:  speaker G00000326 speaking more than once at time 839.436
#WARNING:  speaker G00000326 speaking more than once at time 839.436
#WARNING:  speaker G00000000 speaking more than once at time 1013.128
#WARNING:  speaker G00000327 speaking more than once at time 1241.096
#WARNING:  speaker G00000327 speaking more than once at time 1241.096
#WARNING:  speaker G00000327 speaking more than once at time 1695.756
#WARNING:  speaker G00000327 speaking more than once at time 1695.756
#WARNING:  speaker G00000490 speaking more than once at time 405.167
#WARNING:  speaker G00000490 speaking more than once at time 405.167
#WARNING:  speaker G00000489 speaking more than once at time 678.490
#WARNING:  speaker G00000489 speaking more than once at time 678.490
#WARNING:  speaker G00000490 speaking more than once at time 1127.747
#WARNING:  speaker G00000490 speaking more than once at time 1127.747
#WARNING:  speaker G00000490 speaking more than once at time 1860.903
#WARNING:  speaker G00000490 speaking more than once at time 1860.903
#WARNING:  speaker G00000490 speaking more than once at time 405.167
#WARNING:  speaker G00000490 speaking more than once at time 405.167
#WARNING:  speaker G00000489 speaking more than once at time 678.490
#WARNING:  speaker G00000489 speaking more than once at time 678.490
#WARNING:  speaker G00000490 speaking more than once at time 1127.747
#WARNING:  speaker G00000490 speaking more than once at time 1127.747
#WARNING:  speaker G00000490 speaking more than once at time 1860.903
#WARNING:  speaker G00000490 speaking more than once at time 1860.903
#WARNING:  speaker G00000000 speaking more than once at time 24.746
#WARNING:  speaker G00000000 speaking more than once at time 1266.368
#WARNING:  speaker G00000000 speaking more than once at time 24.746
#WARNING:  speaker G00000000 speaking more than once at time 1266.368
#WARNING:  speaker G00000528 speaking more than once at time 1780.944
#WARNING:  speaker G00000528 speaking more than once at time 1780.944
#WARNING:  speaker G00000528 speaking more than once at time 1780.944
#WARNING:  speaker G00000528 speaking more than once at time 1780.944
#WARNING:  speaker G00000534 speaking more than once at time 116.488
#WARNING:  speaker G00000534 speaking more than once at time 116.488
#WARNING:  speaker G00000000 speaking more than once at time 361.608
#WARNING:  speaker G00000533 speaking more than once at time 543.008
#WARNING:  speaker G00000533 speaking more than once at time 543.008
#WARNING:  speaker G00000533 speaking more than once at time 666.832
#WARNING:  speaker G00000533 speaking more than once at time 666.832
#WARNING:  speaker G00000533 speaking more than once at time 1035.624
#WARNING:  speaker G00000533 speaking more than once at time 1035.624
#WARNING:  speaker G00000533 speaking more than once at time 1793.704
#WARNING:  speaker G00000533 speaking more than once at time 1793.704
#WARNING:  speaker G00000534 speaking more than once at time 116.488
#WARNING:  speaker G00000534 speaking more than once at time 116.488
#WARNING:  speaker G00000000 speaking more than once at time 361.608
#WARNING:  speaker G00000533 speaking more than once at time 543.008
#WARNING:  speaker G00000533 speaking more than once at time 543.008
#WARNING:  speaker G00000533 speaking more than once at time 666.832
#WARNING:  speaker G00000533 speaking more than once at time 666.832
#WARNING:  speaker G00000533 speaking more than once at time 1035.624
#WARNING:  speaker G00000533 speaking more than once at time 1035.624
#WARNING:  speaker G00000533 speaking more than once at time 1793.704
#WARNING:  speaker G00000533 speaking more than once at time 1793.704
#WARNING:  speaker G00000621 speaking more than once at time 377.309
#WARNING:  speaker G00000621 speaking more than once at time 377.309
#WARNING:  speaker G00000621 speaking more than once at time 477.125
#WARNING:  speaker G00000621 speaking more than once at time 477.125
#WARNING:  speaker G00000000 speaking more than once at time 547.109
#WARNING:  speaker G00000622 speaking more than once at time 855.493
#WARNING:  speaker G00000622 speaking more than once at time 855.493
#WARNING:  speaker G00000622 speaking more than once at time 864.773
#WARNING:  speaker G00000622 speaking more than once at time 864.773
#WARNING:  speaker G00000621 speaking more than once at time 899.869
#WARNING:  speaker G00000621 speaking more than once at time 899.869
#WARNING:  speaker G00000621 speaking more than once at time 916.613
#WARNING:  speaker G00000621 speaking more than once at time 916.613
#WARNING:  speaker G00000621 speaking more than once at time 992.837
#WARNING:  speaker G00000621 speaking more than once at time 992.837
#WARNING:  speaker G00000621 speaking more than once at time 1235.525
#WARNING:  speaker G00000621 speaking more than once at time 1235.525
#WARNING:  speaker G00000621 speaking more than once at time 1722.885
#WARNING:  speaker G00000621 speaking more than once at time 1722.885
#WARNING:  speaker G00000622 speaking more than once at time 1774.429
#WARNING:  speaker G00000622 speaking more than once at time 1774.429
#WARNING:  speaker G00000621 speaking more than once at time 1818.725
#WARNING:  speaker G00000621 speaking more than once at time 1818.725
#WARNING:  speaker G00000621 speaking more than once at time 377.309
#WARNING:  speaker G00000621 speaking more than once at time 377.309
#WARNING:  speaker G00000621 speaking more than once at time 477.125
#WARNING:  speaker G00000621 speaking more than once at time 477.125
#WARNING:  speaker G00000000 speaking more than once at time 547.109
#WARNING:  speaker G00000622 speaking more than once at time 855.493
#WARNING:  speaker G00000622 speaking more than once at time 855.493
#WARNING:  speaker G00000622 speaking more than once at time 864.773
#WARNING:  speaker G00000622 speaking more than once at time 864.773
#WARNING:  speaker G00000621 speaking more than once at time 899.869
#WARNING:  speaker G00000621 speaking more than once at time 899.869
#WARNING:  speaker G00000621 speaking more than once at time 916.613
#WARNING:  speaker G00000621 speaking more than once at time 916.613
#WARNING:  speaker G00000621 speaking more than once at time 992.837
#WARNING:  speaker G00000621 speaking more than once at time 992.837
#WARNING:  speaker G00000621 speaking more than once at time 1235.525
#WARNING:  speaker G00000621 speaking more than once at time 1235.525
#WARNING:  speaker G00000621 speaking more than once at time 1722.885
#WARNING:  speaker G00000621 speaking more than once at time 1722.885
#WARNING:  speaker G00000622 speaking more than once at time 1774.429
#WARNING:  speaker G00000622 speaking more than once at time 1774.429
#WARNING:  speaker G00000621 speaking more than once at time 1818.725
#WARNING:  speaker G00000621 speaking more than once at time 1818.725
#WARNING:  speaker G00000680 speaking more than once at time 130.024
#WARNING:  speaker G00000680 speaking more than once at time 130.024
#WARNING:  speaker G00000000 speaking more than once at time 209.704
#WARNING:  speaker G00000680 speaking more than once at time 256.099
#WARNING:  speaker G00000680 speaking more than once at time 256.099
#WARNING:  speaker G00000000 speaking more than once at time 280.673
#WARNING:  speaker G00000000 speaking more than once at time 312.328
#WARNING:  speaker G00000680 speaking more than once at time 804.904
#WARNING:  speaker G00000680 speaking more than once at time 804.904
#WARNING:  speaker G00000680 speaking more than once at time 1129.232
#WARNING:  speaker G00000680 speaking more than once at time 1129.232
#WARNING:  speaker G00000680 speaking more than once at time 1223.816
#WARNING:  speaker G00000680 speaking more than once at time 1223.816
#WARNING:  speaker G00000680 speaking more than once at time 1411.528
#WARNING:  speaker G00000680 speaking more than once at time 1411.528
#WARNING:  speaker G00000680 speaking more than once at time 1536.809
#WARNING:  speaker G00000680 speaking more than once at time 1536.809
#WARNING:  speaker G00000000 speaking more than once at time 1730.288
#WARNING:  speaker G00000680 speaking more than once at time 130.024
#WARNING:  speaker G00000680 speaking more than once at time 130.024
#WARNING:  speaker G00000000 speaking more than once at time 209.704
#WARNING:  speaker G00000680 speaking more than once at time 256.099
#WARNING:  speaker G00000680 speaking more than once at time 256.099
#WARNING:  speaker G00000000 speaking more than once at time 280.673
#WARNING:  speaker G00000000 speaking more than once at time 312.328
#WARNING:  speaker G00000680 speaking more than once at time 804.904
#WARNING:  speaker G00000680 speaking more than once at time 804.904
#WARNING:  speaker G00000680 speaking more than once at time 1129.232
#WARNING:  speaker G00000680 speaking more than once at time 1129.232
#WARNING:  speaker G00000680 speaking more than once at time 1223.816
#WARNING:  speaker G00000680 speaking more than once at time 1223.816
#WARNING:  speaker G00000680 speaking more than once at time 1411.528
#WARNING:  speaker G00000680 speaking more than once at time 1411.528
#WARNING:  speaker G00000680 speaking more than once at time 1536.809
#WARNING:  speaker G00000680 speaking more than once at time 1536.809
#WARNING:  speaker G00000000 speaking more than once at time 1730.288
#WARNING:  speaker G00000392 speaking more than once at time 795.224
#WARNING:  speaker G00000392 speaking more than once at time 795.224
#WARNING:  speaker G00000391 speaking more than once at time 809.608
#WARNING:  speaker G00000391 speaking more than once at time 809.608
#WARNING:  speaker G00000391 speaking more than once at time 1701.352
#WARNING:  speaker G00000391 speaking more than once at time 1701.352
#WARNING:  speaker G00000392 speaking more than once at time 795.224
#WARNING:  speaker G00000392 speaking more than once at time 795.224
#WARNING:  speaker G00000391 speaking more than once at time 809.608
#WARNING:  speaker G00000391 speaking more than once at time 809.608
#WARNING:  speaker G00000391 speaking more than once at time 1701.352
#WARNING:  speaker G00000391 speaking more than once at time 1701.352
#WARNING:  speaker G00000430 speaking more than once at time 1307.916
#WARNING:  speaker G00000430 speaking more than once at time 1307.916
#WARNING:  speaker G00000430 speaking more than once at time 1667.180
#WARNING:  speaker G00000430 speaking more than once at time 1667.180
#WARNING:  speaker G00000429 speaking more than once at time 1823.436
#WARNING:  speaker G00000429 speaking more than once at time 1823.436
#WARNING:  speaker G00000430 speaking more than once at time 1307.916
#WARNING:  speaker G00000430 speaking more than once at time 1307.916
#WARNING:  speaker G00000430 speaking more than once at time 1667.180
#WARNING:  speaker G00000430 speaking more than once at time 1667.180
#WARNING:  speaker G00000429 speaking more than once at time 1823.436
#WARNING:  speaker G00000429 speaking more than once at time 1823.436
#WARNING:  speaker G00000662 speaking more than once at time 31.144
#WARNING:  speaker G00000662 speaking more than once at time 31.144
#WARNING:  speaker G00000000 speaking more than once at time 265.232
#WARNING:  speaker G00000661 speaking more than once at time 358.408
#WARNING:  speaker G00000661 speaking more than once at time 358.408
#WARNING:  speaker G00000661 speaking more than once at time 752.928
#WARNING:  speaker G00000661 speaking more than once at time 752.928
#WARNING:  speaker G00000661 speaking more than once at time 900.488
#WARNING:  speaker G00000661 speaking more than once at time 900.488
#WARNING:  speaker G00000662 speaking more than once at time 1328.008
#WARNING:  speaker G00000662 speaking more than once at time 1328.008
#WARNING:  speaker G00000662 speaking more than once at time 1504.744
#WARNING:  speaker G00000662 speaking more than once at time 1504.744
#WARNING:  speaker G00000662 speaking more than once at time 1852.488
#WARNING:  speaker G00000662 speaking more than once at time 1852.488
#WARNING:  speaker G00000662 speaking more than once at time 31.144
#WARNING:  speaker G00000662 speaking more than once at time 31.144
#WARNING:  speaker G00000000 speaking more than once at time 265.232
#WARNING:  speaker G00000661 speaking more than once at time 358.408
#WARNING:  speaker G00000661 speaking more than once at time 358.408
#WARNING:  speaker G00000661 speaking more than once at time 752.928
#WARNING:  speaker G00000661 speaking more than once at time 752.928
#WARNING:  speaker G00000661 speaking more than once at time 900.488
#WARNING:  speaker G00000661 speaking more than once at time 900.488
#WARNING:  speaker G00000662 speaking more than once at time 1328.008
#WARNING:  speaker G00000662 speaking more than once at time 1328.008
#WARNING:  speaker G00000662 speaking more than once at time 1504.744
#WARNING:  speaker G00000662 speaking more than once at time 1504.744
#WARNING:  speaker G00000662 speaking more than once at time 1852.488
#WARNING:  speaker G00000662 speaking more than once at time 1852.488
#WARNING:  speaker G00000000 speaking more than once at time 424.078
#WARNING:  speaker G00000000 speaking more than once at time 726.736
#WARNING:  speaker G00000141 speaking more than once at time 1731.995
#WARNING:  speaker G00000141 speaking more than once at time 1731.995
#WARNING:  speaker G00000000 speaking more than once at time 424.078
#WARNING:  speaker G00000000 speaking more than once at time 726.736
#WARNING:  speaker G00000141 speaking more than once at time 1731.995
#WARNING:  speaker G00000141 speaking more than once at time 1731.995
#WARNING:  speaker G00000322 speaking more than once at time 1789.851
#WARNING:  speaker G00000322 speaking more than once at time 1789.851
#WARNING:  speaker G00000322 speaking more than once at time 1789.851
#WARNING:  speaker G00000322 speaking more than once at time 1789.851
#WARNING:  speaker G00000155 speaking more than once at time 1816.256
#WARNING:  speaker G00000155 speaking more than once at time 1816.256
#WARNING:  speaker G00000155 speaking more than once at time 1816.256
#WARNING:  speaker G00000155 speaking more than once at time 1816.256
#WARNING:  speaker G00000000 speaking more than once at time 950.478
#WARNING:  speaker G00000341 speaking more than once at time 1127.215
#WARNING:  speaker G00000341 speaking more than once at time 1127.215
#WARNING:  speaker G00000000 speaking more than once at time 1579.866
#WARNING:  speaker G00000000 speaking more than once at time 950.478
#WARNING:  speaker G00000341 speaking more than once at time 1127.215
#WARNING:  speaker G00000341 speaking more than once at time 1127.215
#WARNING:  speaker G00000000 speaking more than once at time 1579.866
#WARNING:  speaker G00000000 speaking more than once at time 334.738
#WARNING:  speaker G00000000 speaking more than once at time 783.496
#WARNING:  speaker G00000000 speaking more than once at time 783.784
#WARNING:  speaker G00000094 speaking more than once at time 1272.240
#WARNING:  speaker G00000094 speaking more than once at time 1272.240
#WARNING:  speaker G00000000 speaking more than once at time 1351.680
#WARNING:  speaker G00000000 speaking more than once at time 1725.120
#WARNING:  speaker G00000000 speaking more than once at time 334.738
#WARNING:  speaker G00000000 speaking more than once at time 783.496
#WARNING:  speaker G00000000 speaking more than once at time 783.784
#WARNING:  speaker G00000094 speaking more than once at time 1272.240
#WARNING:  speaker G00000094 speaking more than once at time 1272.240
#WARNING:  speaker G00000000 speaking more than once at time 1351.680
#WARNING:  speaker G00000000 speaking more than once at time 1725.120
#1.85/1.85/0.00/0.00
fi
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
    echo "clustering ......."
    vad_threshold="0.6 0.7 0.8 0.9 0.91 0.92"
    #vad_threshold="0.5"
    vad_type="transformer_vad"
    chunk_size=3
    skip_chunk_size=0.93
    test_set_dir="data/magicdata-RAMC/test/"
    for name in $vad_threshold;do
     python scripts/magicdata-RAMC/020_spectral_cluster_common.py\
         --vad_threshold $name\
         --predict_vad_path_dir $test_set_dir/predict_vad_${name} \
         --test_set_dir $test_set_dir\
         --predict_rttm_path $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_skip_chunk_size_${skip_chunk_size}\
         --chunk_size $chunk_size\
         --skip_chunk_size $skip_chunk_size\
         --vad_type $vad_type


    done
fi
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
   echo "DER score"
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.5 0.6 0.7 0.8 0.9 0.91 0.92"
   test_set_dir="data/magicdata-RAMC/test/"
   vad_type="transformer_vad"
   chunk_size=3
   skip_chunk_size=0.93
   for name in $vad_threshold;do
     $sctk_dir/md-eval.pl -c 0.25 -r $test_set_dir/rttm -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_skip_chunk_size_${skip_chunk_size}
   done
   #  bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 4 > logs/020_spectral_cluster_stage4.log 2 >&1
   # result: cat logs/020_spectral_cluster_stage4.log
   # DER score
   # DER  MS FA SC
   #10.00/2.83/0.22/6.95
   #11.24/4.40/0.13/6.71
   #13.47/7.01/0.07/6.39
   #18.60/12.79/0.03/5.78
   #34.59/30.26/0.01/4.32
   #37.19/33.05/0.01/4.12
   #40.03/36.10/0.01/3.92

 fi

 if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
   echo "CDER score"
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.5 0.6 0.7 0.8 0.9 0.91 0.92"
   test_set_dir="data/magicdata-RAMC/test/"
   vad_type="transformer_vad"
   chunk_size=3
   skip_chunk_size=0.93
   . path_for_nn_vad.sh
   for name in $vad_threshold;do
     python clustering_based/cder/score.py  -r $test_set_dir/rttm -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_skip_chunk_size_${skip_chunk_size}
   done
   # bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 5 > logs/020_spectral_cluster_stage5 2>&1
   #  grep -r "Avg CDER" logs/020_spectral_cluster_stage5
   # Avg CDER : 0.218
   # Avg CDER : 0.218
   # Avg CDER : 0.224
   # Avg CDER : 0.218
   # Avg CDER : 0.234
   # Avg CDER : 0.237
   # Avg CDER : 0.249
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ];then
   echo "DER score collar=0.0"
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.5 0.6 0.7 0.8 0.9 0.91 0.92"
   test_set_dir="data/magicdata-RAMC/test/"
   vad_type="transformer_vad"
   chunk_size=3
   skip_chunk_size=0.93
   for name in $vad_threshold;do
     $sctk_dir/md-eval.pl -c 0.0 -r $test_set_dir/rttm -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_skip_chunk_size_${skip_chunk_size}
   done
   #  bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 6 > logs/020_spectral_cluster_stage6.log 2>&1
   #  grep -v 'WARNING' logs/020_spectral_cluster_stage6.log
   # DER score collar=0.0
   # 18.67/4.36/4.06/10.24
   # 19.36/6.82/2.75/9.79
   # 21.57/10.72/1.63/9.22
   # 27.07/18.17/0.82/8.08
   # 42.03/35.99/0.28/5.76
   # 44.42/38.71/0.25/5.46
   # 47.01/41.64/0.22/5.15
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ];then
    echo "clustering ......."
    vad_threshold="0.1 0.2 0.3 0.4 0.45 0.46 0.47 0.51 0.52 0.53"
    #vad_threshold="0.5"
    vad_type="transformer_vad"
    chunk_size=3
    skip_chunk_size=0.93
    test_set_dir="data/magicdata-RAMC/test/"
    for name in $vad_threshold;do
     python scripts/magicdata-RAMC/020_spectral_cluster_common.py\
         --vad_threshold $name\
         --predict_vad_path_dir $test_set_dir/predict_vad_${name} \
         --test_set_dir $test_set_dir\
         --predict_rttm_path $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_skip_chunk_size_${skip_chunk_size}\
         --chunk_size $chunk_size\
         --skip_chunk_size $skip_chunk_size\
         --vad_type $vad_type


    done
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ];then
   echo "DER score"
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.1 0.2 0.3 0.4 0.45 0.46 0.47 0.51 0.52 0.53"
   test_set_dir="data/magicdata-RAMC/test/"
   vad_type="transformer_vad"
   chunk_size=3
   skip_chunk_size=0.93
   for name in $vad_threshold;do
     $sctk_dir/md-eval.pl -c 0.25 -r $test_set_dir/rttm -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_skip_chunk_size_${skip_chunk_size}
   done
   #  bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 9 --stop-stage 9 > logs/020_spectral_cluster_stage9.log 2 >&1
   # grep -v 'WARNING' logs/020_spectral_cluster_stage9.log
#DER score
#11.09/0.18/3.03/7.89
#9.22/0.42/1.12/7.69
#8.94/0.95/0.57/7.42
#9.33/1.76/0.34/7.24
#9.63/2.24/0.27/7.12
#9.70/2.36/0.26/7.09
#9.74/2.45/0.25/7.05
#10.09/2.95/0.21/6.93
#10.18/3.08/0.20/6.91
#10.25/3.20/0.19/6.86
 fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ];then
   echo "CDER score"
   #note:collar of CDER score default set is equal to zero.
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.1 0.2 0.3 0.4 0.45 0.46 0.47 0.51 0.52 0.53"
   test_set_dir="data/magicdata-RAMC/test/"
   vad_type="transformer_vad"
   chunk_size=3
   skip_chunk_size=0.93
   . path_for_nn_vad.sh
   for name in $vad_threshold;do
     python clustering_based/cder/score.py  -r $test_set_dir/rttm -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_skip_chunk_size_${skip_chunk_size}
   done
   # bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 10  --stop-stage 10 > logs/020_spectral_cluster_stage10.log 2>&1
   #  grep -r 'Avg CDER' logs/020_spectral_cluster_stage10.log
   # Avg CDER : 0.234
#Avg CDER : 0.225
#Avg CDER : 0.221
#Avg CDER : 0.218
#Avg CDER : 0.219
#Avg CDER : 0.217
#Avg CDER : 0.217
#Avg CDER : 0.218
#Avg CDER : 0.217
#Avg CDER : 0.217
fi

if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ];then
   echo "DER score collar=0.0"
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.1 0.2 0.3 0.4 0.45 0.46 0.47 0.51 0.52 0.53"
   test_set_dir="data/magicdata-RAMC/test/"
   vad_type="transformer_vad"
   chunk_size=3
   skip_chunk_size=0.93
   for name in $vad_threshold;do
     $sctk_dir/md-eval.pl -c 0.0 -r $test_set_dir/rttm -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_skip_chunk_size_${skip_chunk_size}
   done
   #  bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 11 --stop-stage 11 > logs/020_spectral_cluster_stage11.log 2>&1
   #  grep -v 'WARNING' logs/020_spectral_cluster_stage11.log
   # DER score collar=0.0
#24.44/0.41/12.33/11.70
#21.69/0.86/9.44/11.39
#19.71/1.76/6.95/10.99
#18.85/2.96/5.22/10.67
#18.70/3.60/4.61/10.49
#18.69/3.75/4.50/10.44
#18.67/3.88/4.39/10.39
#18.69/4.53/3.95/10.21
#18.72/4.70/3.85/10.17
#18.72/4.87/3.74/10.11
fi

if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ];then
    echo "clustering ......."
    vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
    #vad_threshold="0.5"
    vad_type="transformer_vad"
    chunk_size=3
    skip_chunk_size=0.93
    test_set_dir="data/magicdata-RAMC/test/"
    for name in $vad_threshold;do
     python scripts/magicdata-RAMC/020_spectral_cluster_common.py\
         --vad_threshold $name\
         --predict_vad_path_dir $test_set_dir/predict_vad_${name} \
         --test_set_dir $test_set_dir\
         --predict_rttm_path $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_skip_chunk_size_${skip_chunk_size}\
         --chunk_size $chunk_size\
         --skip_chunk_size $skip_chunk_size\
         --vad_type $vad_type


    done
fi

if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ];then
   echo "DER score"
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
   test_set_dir="data/magicdata-RAMC/test/"
   vad_type="transformer_vad"
   chunk_size=3
   skip_chunk_size=0.93
   for name in $vad_threshold;do
     $sctk_dir/md-eval.pl -c 0.25 -r $test_set_dir/rttm -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_skip_chunk_size_${skip_chunk_size}
   done
   #  bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 13 --stop-stage 13 > logs/020_spectral_cluster_stage13.log 2 >&1
   # grep -v 'WARNING' logs/020_spectral_cluster_stage13.log
   # DER score
#9.22/0.42/1.12/7.69
#9.07/0.48/0.96/7.62
#9.00/0.56/0.84/7.60
#8.97/0.61/0.79/7.57
#8.92/0.72/0.69/7.51
#8.92/0.81/0.64/7.47
#8.93/0.87/0.60/7.46
#8.94/0.95/0.57/7.42  #for vad_threshold==0.30
#8.98/1.03/0.54/7.41
#9.01/1.10/0.52/7.40
#9.09/1.23/0.47/7.39
#9.15/1.38/0.42/7.34
#9.24/1.60/0.37/7.27
fi

if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ];then
   echo "CDER score"
   #note:collar of CDER score default set is equal to zero.
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
   test_set_dir="data/magicdata-RAMC/test/"
   vad_type="transformer_vad"
   chunk_size=3
   skip_chunk_size=0.93
   . path_for_nn_vad.sh
   for name in $vad_threshold;do
     python clustering_based/cder/score.py  -r $test_set_dir/rttm -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_skip_chunk_size_${skip_chunk_size}
   done
   # bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 14  --stop-stage 14 > logs/020_spectral_cluster_stage14.log 2>&1
   #  grep -r 'Avg CDER' logs/020_spectral_cluster_stage14.log
   # Avg CDER : 0.225
#Avg CDER : 0.223
#Avg CDER : 0.222
#Avg CDER : 0.221
#Avg CDER : 0.220
#Avg CDER : 0.220
#Avg CDER : 0.220
#Avg CDER : 0.221
#Avg CDER : 0.220
#Avg CDER : 0.220
#Avg CDER : 0.221
#Avg CDER : 0.219
#Avg CDER : 0.219
fi

if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ];then
   echo "DER score collar=0.0"
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
   test_set_dir="data/magicdata-RAMC/test/"
   vad_type="transformer_vad"
   chunk_size=3
   skip_chunk_size=0.93
   for name in $vad_threshold;do
     $sctk_dir/md-eval.pl -c 0.0 -r $test_set_dir/rttm -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_skip_chunk_size_${skip_chunk_size}
   done
   #  #  bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 15 --stop-stage 15 > logs/020_spectral_cluster_stage15.log 2>&1
   #  grep -v 'WARNING' logs/020_spectral_cluster_stage15.log
   # DER score collar=0.0
#21.69/0.86/9.44/11.39
#21.31/0.99/9.02/11.30
#21.02/1.12/8.65/11.25
#20.86/1.21/8.44/11.22
#20.35/1.40/7.84/11.12
#20.10/1.54/7.50/11.06
#19.89/1.64/7.20/11.05
#19.71/1.76/6.95/10.99
#19.58/1.89/6.72/10.97
#19.46/1.99/6.52/10.95
#19.26/2.20/6.16/10.90
#19.09/2.44/5.83/10.83
#18.95/2.73/5.50/10.72

fi

if [ ${stage} -le 17 ] && [ ${stop_stage} -ge 17 ];then
    echo "clustering ......."
    vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
    #vad_threshold="0.5"
    vad_type="transformer_vad"
    chunk_size=2
    skip_chunk_size=0.93
    test_set_dir="data/magicdata-RAMC/test/"
    for name in $vad_threshold;do
     python scripts/magicdata-RAMC/020_spectral_cluster_common.py\
         --vad_threshold $name\
         --predict_vad_path_dir $test_set_dir/predict_vad_${name} \
         --test_set_dir $test_set_dir\
         --predict_rttm_path $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_skip_chunk_size_${skip_chunk_size}\
         --chunk_size $chunk_size\
         --skip_chunk_size $skip_chunk_size\
         --vad_type $vad_type


    done
fi

if [ ${stage} -le 18 ] && [ ${stop_stage} -ge 18 ];then
   echo "DER score"
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
   test_set_dir="data/magicdata-RAMC/test/"
   vad_type="transformer_vad"
   chunk_size=2
   skip_chunk_size=0.93
   for name in $vad_threshold;do
     $sctk_dir/md-eval.pl -c 0.25 -r $test_set_dir/rttm -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_skip_chunk_size_${skip_chunk_size}
   done
   #  bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 18 --stop-stage 18 > logs/020_spectral_cluster_stage18.log 2 >&1
   # grep -v 'WARNING' logs/020_spectral_cluster_stage18.log
   # DER score
#8.06/0.70/1.05/6.31
#8.00/0.79/0.90/6.31
#7.96/0.87/0.79/6.30
#7.97/0.94/0.74/6.29
#8.01/1.11/0.65/6.24
#8.06/1.22/0.61/6.24
#8.11/1.32/0.57/6.23
#8.20/1.42/0.54/6.24
#8.26/1.51/0.51/6.23
#8.32/1.61/0.49/6.23
#8.45/1.79/0.44/6.22
#8.57/2.01/0.39/6.17
#8.71/2.26/0.35/6.10
fi

if [ ${stage} -le 19 ] && [ ${stop_stage} -ge 19 ];then
   echo "CDER score"
   #note:collar of CDER score default set is equal to zero.
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
   test_set_dir="data/magicdata-RAMC/test/"
   vad_type="transformer_vad"
   chunk_size=2
   skip_chunk_size=0.93
   . path_for_nn_vad.sh
   for name in $vad_threshold;do
     python clustering_based/cder/score.py  -r $test_set_dir/rttm -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_skip_chunk_size_${skip_chunk_size}
   done
   # bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 19  --stop-stage 19 > logs/020_spectral_cluster_stage19.log 2>&1
   #  grep -r 'Avg CDER' logs/020_spectral_cluster_stage19.log
   # Avg CDER : 0.233
#Avg CDER : 0.231
#Avg CDER : 0.229
#Avg CDER : 0.229
#Avg CDER : 0.227
#Avg CDER : 0.228
#Avg CDER : 0.228
#Avg CDER : 0.229
#Avg CDER : 0.227
#Avg CDER : 0.226
#Avg CDER : 0.225
#Avg CDER : 0.224
#Avg CDER : 0.223
fi

if [ ${stage} -le 20 ] && [ ${stop_stage} -ge 20 ];then
   echo "DER score collar=0.0"
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
   test_set_dir="data/magicdata-RAMC/test/"
   vad_type="transformer_vad"
   chunk_size=2
   skip_chunk_size=0.93
   for name in $vad_threshold;do
     $sctk_dir/md-eval.pl -c 0.0 -r $test_set_dir/rttm -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_skip_chunk_size_${skip_chunk_size}
   done
   #  #  bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 20 --stop-stage 20 > logs/020_spectral_cluster_stage20.log 2>&1
   #  grep -v 'WARNING' logs/020_spectral_cluster_stage20.log
#   DER score collar=0.0
#20.43/1.40/9.06/9.97
#20.14/1.57/8.63/9.94
#19.90/1.69/8.29/9.92
#19.79/1.81/8.09/9.89
#19.38/2.07/7.50/9.81
#19.21/2.25/7.18/9.79
#19.04/2.41/6.88/9.76
#18.94/2.55/6.63/9.76
#18.84/2.69/6.42/9.73
#18.76/2.84/6.21/9.72
#18.63/3.11/5.85/9.68
#18.55/3.43/5.51/9.61
#18.46/3.76/5.20/9.51
fi


if [ ${stage} -le 21 ] && [ ${stop_stage} -ge 21 ];then
    echo "clustering ......."
    #vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
    vad_threshold="0.31 0.32 0.34 0.36 0.38"
    vad_type="transformer_vad"
    chunk_size=1.5
    skip_chunk_size=0.93
    test_set_dir="data/magicdata-RAMC/test/"
    for name in $vad_threshold;do
     python scripts/magicdata-RAMC/020_spectral_cluster_common.py\
         --vad_threshold $name\
         --predict_vad_path_dir $test_set_dir/predict_vad_${name} \
         --test_set_dir $test_set_dir\
         --predict_rttm_path $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_skip_chunk_size_${skip_chunk_size}\
         --chunk_size $chunk_size\
         --skip_chunk_size $skip_chunk_size\
         --vad_type $vad_type


    done
fi

if [ ${stage} -le 22 ] && [ ${stop_stage} -ge 22 ];then
   echo "DER score"
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
   test_set_dir="data/magicdata-RAMC/test/"
   vad_type="transformer_vad"
   chunk_size=1.5
   skip_chunk_size=0.93
   for name in $vad_threshold;do
     $sctk_dir/md-eval.pl -c 0.25 -r $test_set_dir/rttm -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_skip_chunk_size_${skip_chunk_size}
   done
   #  #  bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 22 --stop-stage 22 > logs/020_spectral_cluster_stage22.log 2>&1
   #  grep -v 'WARNING' logs/020_spectral_cluster_stage22.log
   # DER score
#7.60/0.92/0.99/5.69
#7.58/1.05/0.85/5.68
#7.59/1.18/0.75/5.65
#7.63/1.27/0.70/5.66
#7.75/1.50/0.61/5.64
#7.81/1.63/0.57/5.61
#7.88/1.74/0.54/5.60
#7.99/1.87/0.51/5.61
#8.05/1.99/0.48/5.58
#8.11/2.09/0.46/5.57
#8.26/2.31/0.41/5.53
#8.48/2.59/0.37/5.51
#8.74/2.95/0.33/5.47
fi

if [ ${stage} -le 23 ] && [ ${stop_stage} -ge 23 ];then
   echo "CDER score"
   #note:collar of CDER score default set is equal to zero.
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
   test_set_dir="data/magicdata-RAMC/test/"
   vad_type="transformer_vad"
   chunk_size=1.5
   skip_chunk_size=0.93
   . path_for_nn_vad.sh
   for name in $vad_threshold;do
     python clustering_based/cder/score.py  -r $test_set_dir/rttm -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_skip_chunk_size_${skip_chunk_size}
   done
   # bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 23  --stop-stage 23 > logs/020_spectral_cluster_stage23.log 2>&1
   #  grep -r 'Avg CDER' logs/020_spectral_cluster_stage23.log
   # Avg CDER : 0.244
#Avg CDER : 0.242
#Avg CDER : 0.240
#Avg CDER : 0.243
#Avg CDER : 0.245
#Avg CDER : 0.242
#Avg CDER : 0.242
#Avg CDER : 0.240
#Avg CDER : 0.238
#Avg CDER : 0.236
#Avg CDER : 0.236
#Avg CDER : 0.236
#Avg CDER : 0.227
fi

if [ ${stage} -le 24 ] && [ ${stop_stage} -ge 24 ];then
   echo "DER score collar=0.0"
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
   test_set_dir="data/magicdata-RAMC/test/"
   vad_type="transformer_vad"
   chunk_size=1.5
   skip_chunk_size=0.93
   for name in $vad_threshold;do
     $sctk_dir/md-eval.pl -c 0.0 -r $test_set_dir/rttm -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_skip_chunk_size_${skip_chunk_size}
   done
   #  #  bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 24 --stop-stage 24 > logs/020_spectral_cluster_stage24.log 2>&1
   #  grep -v 'WARNING' logs/020_spectral_cluster_stage24.log
   # DER score collar=0.0
#19.80/1.86/8.70/9.24
#19.56/2.08/8.28/9.21
#19.37/2.29/7.93/9.15
#19.31/2.45/7.72/9.14
#19.02/2.80/7.12/9.09
#18.86/3.00/6.81/9.05
#18.73/3.18/6.52/9.03
#18.66/3.35/6.28/9.03
#18.58/3.53/6.06/8.99
#18.52/3.69/5.86/8.97
#18.42/4.01/5.51/8.90
#18.43/4.41/5.18/8.84
#18.48/4.85/4.87/8.77
fi


if [ ${stage} -le 25 ] && [ ${stop_stage} -ge 25 ];then
    echo "clustering ......."
    vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
    #vad_threshold="0.31 0.32 0.34 0.36 0.38"
    vad_type="transformer_vad"
    chunk_size=1.5
    step_size=0.75
    skip_chunk_size=0.5
    test_set_dir="data/magicdata-RAMC/test/"
    for name in $vad_threshold;do
     python scripts/magicdata-RAMC/020_spectral_cluster_common.py\
         --vad_threshold $name\
         --predict_vad_path_dir $test_set_dir/predict_vad_${name} \
         --test_set_dir $test_set_dir\
         --predict_rttm_path $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_skip_chunk_size_${skip_chunk_size}\
         --chunk_size $chunk_size\
         --step_size $step_size\
         --skip_chunk_size $skip_chunk_size\
         --vad_type $vad_type


    done
fi
if [ ${stage} -le 26 ] && [ ${stop_stage} -ge 26 ];then
   echo "DER score"
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
   test_set_dir="data/magicdata-RAMC/test/"
   vad_type="transformer_vad"
   chunk_size=1.5
   step_size=0.75
   skip_chunk_size=0.5
   for name in $vad_threshold;do
     $sctk_dir/md-eval.pl -c 0.25 -r $test_set_dir/rttm -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_skip_chunk_size_${skip_chunk_size}
   done
   #  #  bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 26 --stop-stage 26 > logs/020_spectral_cluster_stage26.log 2>&1
   #  grep -v 'WARNING' logs/020_spectral_cluster_stage26.log
   # DER score
#6.92/0.07/1.14/5.71
#6.56/0.09/1.00/5.47
#6.45/0.11/0.89/5.45
#6.41/0.13/0.83/5.45
#6.36/0.18/0.73/5.45
#6.34/0.21/0.69/5.44
#6.35/0.24/0.65/5.45
#6.35/0.28/0.62/5.45
#6.34/0.31/0.59/5.45
#6.33/0.34/0.56/5.43
#5.93/0.40/0.51/5.03
#5.96/0.48/0.46/5.02
#6.02/0.59/0.41/5.02
fi

if [ ${stage} -le 27 ] && [ ${stop_stage} -ge 27 ];then
   echo "CDER score"
   #note:collar of CDER score default set is equal to zero.
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
   test_set_dir="data/magicdata-RAMC/test/"
   vad_type="transformer_vad"
   chunk_size=1.5
   step_size=0.75
   skip_chunk_size=0.5
   . path_for_nn_vad.sh
   for name in $vad_threshold;do
     python clustering_based/cder/score.py  -r $test_set_dir/rttm -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_skip_chunk_size_${skip_chunk_size}
   done
   #  #  bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 27 --stop-stage 27 > logs/020_spectral_cluster_stage27.log 2>&1
   # grep -r 'Avg CDER' logs/020_spectral_cluster_stage27.log
#Avg CDER : 0.372
#Avg CDER : 0.326
#Avg CDER : 0.312
#Avg CDER : 0.324
#Avg CDER : 0.298
#Avg CDER : 0.295
#Avg CDER : 0.281
#Avg CDER : 0.296
#Avg CDER : 0.294
#Avg CDER : 0.302
#Avg CDER : 0.293
#Avg CDER : 0.300
#Avg CDER : 0.271
fi
if [ ${stage} -le 28 ] && [ ${stop_stage} -ge 28 ];then
   echo "DER score"
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
   test_set_dir="data/magicdata-RAMC/test/"
   vad_type="transformer_vad"
   chunk_size=1.5
   step_size=0.75
   skip_chunk_size=0.5
   for name in $vad_threshold;do
     $sctk_dir/md-eval.pl -c 0.0 -r $test_set_dir/rttm -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_skip_chunk_size_${skip_chunk_size}
   done
   #  #  bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 28 --stop-stage 28 > logs/020_spectral_cluster_stage28.log 2>&1
   #  grep -v 'WARNING' logs/020_spectral_cluster_stage28.log
#DER score
#18.53/0.16/9.25/9.13
#17.94/0.20/8.82/8.92
#17.56/0.25/8.44/8.88
#17.38/0.29/8.22/8.86
#16.86/0.41/7.60/8.86
#16.58/0.47/7.27/8.83
#16.35/0.55/6.96/8.84
#16.15/0.61/6.71/8.83
#15.96/0.68/6.47/8.81
#15.79/0.74/6.27/8.79
#15.14/0.86/5.89/8.39
#14.92/0.99/5.55/8.38
#14.74/1.14/5.23/8.37
 fi


if [ ${stage} -le 29 ] && [ ${stop_stage} -ge 29 ];then
    echo "clustering ......."
    vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
    #vad_threshold="0.31 0.32 0.34 0.36 0.38"
    vad_type="transformer_vad"
    chunk_size=1.5
    step_size=0.75
    skip_chunk_size=0.5
    test_set_dir="data/magicdata-RAMC/dev/"
    for name in $vad_threshold;do
     python scripts/magicdata-RAMC/020_spectral_cluster_common.py\
         --vad_threshold $name\
         --predict_vad_path_dir $test_set_dir/predict_vad_${name} \
         --test_set_dir $test_set_dir\
         --predict_rttm_path $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}\
         --chunk_size $chunk_size\
         --step_size $step_size\
         --skip_chunk_size $skip_chunk_size\
         --vad_type $vad_type
    done

fi

if [ ${stage} -le 30 ] && [ ${stop_stage} -ge 30 ];then
   echo "DER score"
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
   test_set_dir="data/magicdata-RAMC/dev/"
   vad_type="transformer_vad"
   chunk_size=1.5
   step_size=0.75
   skip_chunk_size=0.5
   for name in $vad_threshold;do
     $sctk_dir/md-eval.pl -c 0.25 -r $test_set_dir/rttm -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}
   done
    #  #  bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 30 --stop-stage 30 > logs/020_spectral_cluster_stage30.log 2>&1
   #  grep -v 'WARNING' logs/020_spectral_cluster_stage30.log
#   DER score
#10.62/0.03/2.37/8.22
#10.38/0.04/2.12/8.22
#10.20/0.06/1.92/8.22
#8.37/0.07/1.82/6.47
#8.20/0.09/1.65/6.46
#8.13/0.10/1.58/6.44
#9.81/0.12/1.51/8.18
#9.78/0.14/1.46/8.18
#9.76/0.16/1.41/8.19
#9.71/0.18/1.35/8.19
#9.66/0.22/1.23/8.20
#9.62/0.29/1.14/8.18
#7.73/0.37/1.04/6.32
fi
if [ ${stage} -le 31 ] && [ ${stop_stage} -ge 31 ];then
   echo "CDER score"
   #note:collar of CDER score default set is equal to zero.
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
   test_set_dir="data/magicdata-RAMC/dev/"
   vad_type="transformer_vad"
   chunk_size=1.5
   step_size=0.75
   skip_chunk_size=0.5
   . path_for_nn_vad.sh
   for name in $vad_threshold;do
     python clustering_based/cder/score.py  -r $test_set_dir/rttm -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}
   done
   #  #  bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 31 --stop-stage 31 > logs/020_spectral_cluster_stage31.log 2>&1
   #  grep -r 'Avg CDER' logs/020_spectral_cluster_stage31.log
   # Avg CDER : 0.324
#Avg CDER : 0.316
#Avg CDER : 0.312
#Avg CDER : 0.278
#Avg CDER : 0.274
#Avg CDER : 0.270
#Avg CDER : 0.305
#Avg CDER : 0.308
#Avg CDER : 0.311
#Avg CDER : 0.310
#Avg CDER : 0.303
#Avg CDER : 0.298
#Avg CDER : 0.264
fi

if [ ${stage} -le 32 ] && [ ${stop_stage} -ge 32 ];then
   echo "DER score"
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
   test_set_dir="data/magicdata-RAMC/dev/"
   vad_type="transformer_vad"
   chunk_size=1.5
   step_size=0.75
   skip_chunk_size=0.5
   for name in $vad_threshold;do
     $sctk_dir/md-eval.pl -c 0.0 -r $test_set_dir/rttm -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}
   done
    #  #  bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 32 --stop-stage 32 > logs/020_spectral_cluster_stage32.log 2>&1
   #  grep -v 'WARNING' logs/020_spectral_cluster_stage32.log
   # DER score
#DER score
#22.19/0.11/11.50/10.58
#21.71/0.14/11.00/10.57
#21.28/0.19/10.53/10.56
#19.24/0.21/10.28/8.75
#18.55/0.29/9.55/8.71
#18.26/0.33/9.24/8.68
#19.79/0.39/8.94/10.46
#19.57/0.43/8.68/10.46
#19.37/0.48/8.44/10.45
#19.15/0.53/8.18/10.44
#18.78/0.63/7.70/10.45
#18.38/0.75/7.22/10.41
#16.24/0.88/6.82/8.55
fi
if [ ${stage} -le 33 ] && [ ${stop_stage} -ge 33 ];then
   echo "DER score"
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
   test_set_dir="data/magicdata-RAMC/dev/"
   vad_type="transformer_vad"
   chunk_size=1.5
   step_size=0.75
   skip_chunk_size=0.5
   test_set_dir_groundtruth=/mntcephfs/lab_data/maduo/datasets/voice-activity-detection/rttm_total/rttm_gt_dev_nog0
   for name in $vad_threshold;do
     $sctk_dir/md-eval.pl -c 0.25 -r $test_set_dir_groundtruth -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}
   done
  #  bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 33 --stop-stage 33 > logs/020_spectral_cluster_stage33.log 2>&1
  #  grep -v 'WARNING' logs/020_spectral_cluster_stage33.log
  # DER score
#11.46/0.03/4.07/7.36
#11.20/0.04/3.80/7.36
#10.99/0.06/3.58/7.36
#9.14/0.07/3.47/5.60
#8.95/0.08/3.28/5.59
#8.87/0.10/3.20/5.57
#10.56/0.11/3.12/7.32
#10.53/0.13/3.06/7.33
#10.50/0.15/3.01/7.34
#10.45/0.17/2.94/7.33
#10.37/0.21/2.81/7.35
#10.32/0.28/2.70/7.34
#8.40/0.36/2.59/5.46

fi

if [ ${stage} -le 34 ] && [ ${stop_stage} -ge 34 ];then
   echo "CDER score"
   #note:collar of CDER score default set is equal to zero.
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
   test_set_dir="data/magicdata-RAMC/dev/"
   vad_type="transformer_vad"
   chunk_size=1.5
   step_size=0.75
   skip_chunk_size=0.5
   . path_for_nn_vad.sh
   test_set_dir_groundtruth=/mntcephfs/lab_data/maduo/datasets/voice-activity-detection/rttm_total/rttm_gt_dev_nog0
   for name in $vad_threshold;do
     python clustering_based/cder/score.py  -r $test_set_dir_groundtruth -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}
   done
   #  #  bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 34 --stop-stage 34 > logs/020_spectral_cluster_stage34.log 2>&1
   #  grep -r 'Avg CDER' logs/020_spectral_cluster_stage34.log
   # Avg CDER : 0.248
#Avg CDER : 0.236
#Avg CDER : 0.232
#Avg CDER : 0.196
#Avg CDER : 0.194
#Avg CDER : 0.192
#Avg CDER : 0.226
#Avg CDER : 0.230
#Avg CDER : 0.234
#Avg CDER : 0.232
#Avg CDER : 0.226
#Avg CDER : 0.215
#Avg CDER : 0.178
fi



if [ ${stage} -le 35 ] && [ ${stop_stage} -ge 35 ];then
   echo "DER score"
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.1 0.2 0.3 0.4 0.45 0.46 0.47 0.51 0.52 0.53"
   test_set_dir="data/magicdata-RAMC/test/"
   vad_type="transformer_vad"
   chunk_size=3
   skip_chunk_size=0.93
   echo "remove G00000000 segement"
   for name in $vad_threshold;do
     $sctk_dir/md-eval.pl -c 0.25 -r $test_set_dir/rttm_debug2 -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_skip_chunk_size_${skip_chunk_size}
   done
   #  #  bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 35 --stop-stage 35 > logs/020_spectral_cluster_stage35.log 2>&1
   #  grep -v 'WARNING' logs/020_spectral_cluster_stage35.log
#   DER score
#remove G00000000 segement
#12.66/0.17/6.33/6.16
#10.66/0.40/4.29/5.97
#10.23/0.92/3.59/5.71
#10.52/1.73/3.24/5.55
#10.78/2.21/3.12/5.44
#10.84/2.33/3.10/5.41
#10.87/2.42/3.08/5.37
#11.18/2.92/2.99/5.27
#11.26/3.04/2.97/5.25
#11.31/3.16/2.94/5.20
fi

if [ ${stage} -le 40 ] && [ ${stop_stage} -ge 40 ];then
    echo "clustering ......."
    vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
    #vad_threshold="0.31 0.32 0.34 0.36 0.38"
    vad_type="transformer_vad"
    chunk_size=2
    step_size=2
    skip_chunk_size=0.93
    test_set_dir="data/magicdata-RAMC/dev/"
    for name in $vad_threshold;do
     python scripts/magicdata-RAMC/020_spectral_cluster_common.py\
         --vad_threshold $name\
         --predict_vad_path_dir $test_set_dir/predict_vad_${name} \
         --test_set_dir $test_set_dir\
         --predict_rttm_path $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}\
         --chunk_size $chunk_size\
         --step_size $step_size\
         --skip_chunk_size $skip_chunk_size\
         --vad_type $vad_type


    done
fi

if [ ${stage} -le 41 ] && [ ${stop_stage} -ge 41 ];then
   echo "DER score"
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
   test_set_dir="data/magicdata-RAMC/dev/"
   vad_type="transformer_vad"
   chunk_size=2
   step_size=2
   skip_chunk_size=0.93
   for name in $vad_threshold;do
     $sctk_dir/md-eval.pl -c 0.25 -r $test_set_dir/rttm -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}
   done
   #  #  bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 41 --stop-stage 41 > logs/020_spectral_cluster_stage41.log 2>&1
   #  grep -v 'WARNING' logs/020_spectral_cluster_stage41.log
   # DER score
#11.87/0.60/2.29/8.99
#10.04/0.68/2.02/7.35
#8.14/0.75/1.77/5.61
#8.11/0.80/1.68/5.62
#8.03/0.94/1.51/5.58
#8.00/0.98/1.44/5.58
#7.97/1.06/1.36/5.55
#7.96/1.15/1.31/5.51
#7.98/1.22/1.26/5.50
#9.60/1.28/1.20/7.13
#7.96/1.41/1.08/5.48
#8.03/1.60/0.99/5.44
#8.11/1.81/0.91/5.40

fi

# ## Comparing stage42 and stage43 shows that there is no problem with the vad model I trained.
 ## and if remove g0 segments from groundtruth rttm
 #  and CDER will descreased and better than his paper best dev_CDER(i.e.13.2%),
 # because I use more shorter window(chunk_size=2,step_size=2) than his paper setting(i.e. chunk_size=3, step_size=3)
if [ ${stage} -le 42 ] && [ ${stop_stage} -ge 42 ];then
   echo "CDER score"
   #note:collar of CDER score default set is equal to zero.
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
   test_set_dir="data/magicdata-RAMC/dev/"
   vad_type="transformer_vad"
   chunk_size=2
   step_size=2
   skip_chunk_size=0.93
   . path_for_nn_vad.sh
   for name in $vad_threshold;do
     python clustering_based/cder/score.py  -r $test_set_dir/rttm -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}
   done
   #  #  bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 42 --stop-stage 42 > logs/020_spectral_cluster_stage42.log 2>&1
   # grep -r 'Avg CDER' logs/020_spectral_cluster_stage42.log
#Avg CDER : 0.274
#Avg CDER : 0.247
#Avg CDER : 0.222
#Avg CDER : 0.222
#Avg CDER : 0.225
#Avg CDER : 0.223
#Avg CDER : 0.221
#Avg CDER : 0.218
#Avg CDER : 0.216
#Avg CDER : 0.240
#Avg CDER : 0.216
#Avg CDER : 0.217
#Avg CDER : 0.216
fi

if [ ${stage} -le 43 ] && [ ${stop_stage} -ge 43 ];then
   echo "CDER score"
   #note:collar of CDER score default set is equal to zero.
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
   test_set_dir="data/magicdata-RAMC/dev/"
   vad_type="transformer_vad"
   chunk_size=2
   step_size=2
   skip_chunk_size=0.93
   . path_for_nn_vad.sh
   test_set_dir_groundtruth=/mntcephfs/lab_data/maduo/datasets/voice-activity-detection/rttm_total/rttm_gt_dev_nog0
   for name in $vad_threshold;do
     python clustering_based/cder/score.py  -r $test_set_dir_groundtruth -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}
   done
 #  bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 43 --stop-stage 43 >logs/020_spectral_cluster_stage43.log 2>&1
 #  grep -r 'Avg CDER' logs/020_spectral_cluster_stage43.log
 # Avg CDER : 0.180
#Avg CDER : 0.151
#Avg CDER : 0.126
#Avg CDER : 0.126
#Avg CDER : 0.130
#Avg CDER : 0.127
#Avg CDER : 0.125
#Avg CDER : 0.121
#Avg CDER : 0.119
#Avg CDER : 0.143
#Avg CDER : 0.117
#Avg CDER : 0.119
#Avg CDER : 0.117
 fi

 if [ ${stage} -le 44 ] && [ ${stop_stage} -ge 44 ];then
   echo "DER score"
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
   test_set_dir="data/magicdata-RAMC/dev/"
   vad_type="transformer_vad"
   chunk_size=2
   step_size=2
   skip_chunk_size=0.93
   test_set_dir_groundtruth=/mntcephfs/lab_data/maduo/datasets/voice-activity-detection/rttm_total/rttm_gt_dev_nog0
   for name in $vad_threshold;do
     $sctk_dir/md-eval.pl -c 0.25 -r $test_set_dir_groundtruth -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}
   done
   #  bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 44 --stop-stage 44 >logs/020_spectral_cluster_stage44.log 2>&1
   # grep -v 'WARNING' logs/020_spectral_cluster_stage44.log
   # DER score
#12.67/0.58/3.93/8.15
#10.81/0.66/3.65/6.50
#8.87/0.73/3.38/4.75
#8.82/0.79/3.27/4.76
#8.72/0.92/3.07/4.73
#8.69/0.96/3.00/4.72
#8.64/1.04/2.91/4.69
#8.63/1.14/2.85/4.64
#8.63/1.20/2.78/4.64
#10.26/1.26/2.72/6.29
#8.58/1.38/2.57/4.62
#8.63/1.58/2.46/4.59
#8.69/1.79/2.36/4.55
 fi

if [ ${stage} -le 45 ] && [ ${stop_stage} -ge 45 ];then
    echo "clustering ......."
    vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
    #vad_threshold="0.31 0.32 0.34 0.36 0.38"
    vad_type="transformer_vad"
    chunk_size=1.5
    step_size=0.75
    skip_chunk_size=0.025 # # 25ms, because window of fbank feature is 25ms
    test_set_dir="data/magicdata-RAMC/test/"
    for name in $vad_threshold;do
     python scripts/magicdata-RAMC/020_spectral_cluster_common.py\
         --vad_threshold $name\
         --predict_vad_path_dir $test_set_dir/predict_vad_${name} \
         --test_set_dir $test_set_dir\
         --predict_rttm_path $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}\
         --chunk_size $chunk_size\
         --step_size $step_size\
         --skip_chunk_size $skip_chunk_size\
         --vad_type $vad_type
    done

fi

if [ ${stage} -le 51 ] && [ ${stop_stage} -ge 51 ];then
    echo "clustering ......."
    vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
    #vad_threshold="0.31 0.32 0.34 0.36 0.38"
    vad_type="transformer_vad"
    chunk_size=1.0
    step_size=0.5
    skip_chunk_size=0.025 # 25ms, because window of fbank feature is 25ms
    test_set_dir="data/magicdata-RAMC/test/"
    for name in $vad_threshold;do
     python scripts/magicdata-RAMC/020_spectral_cluster_common.py\
         --vad_threshold $name\
         --predict_vad_path_dir $test_set_dir/predict_vad_${name} \
         --test_set_dir $test_set_dir\
         --predict_rttm_path $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}\
         --chunk_size $chunk_size\
         --step_size $step_size\
         --skip_chunk_size $skip_chunk_size\
         --vad_type $vad_type
    done

fi

if [ ${stage} -le 60 ] && [ ${stop_stage} -ge 60 ];then
    echo "clustering ......."
    vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
    #vad_threshold="0.31 0.32 0.34 0.36 0.38"
    vad_type="transformer_vad"
    chunk_size=1.5
    step_size=0.75
    skip_chunk_size=0.025 # # 25ms, because window of fbank feature is 25ms
    test_set_dir="data/magicdata-RAMC/dev/"
    for name in $vad_threshold;do
     python scripts/magicdata-RAMC/020_spectral_cluster_common.py\
         --vad_threshold $name\
         --predict_vad_path_dir $test_set_dir/predict_vad_other_people${name} \
         --test_set_dir $test_set_dir\
         --predict_rttm_path $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}\
         --chunk_size $chunk_size\
         --step_size $step_size\
         --skip_chunk_size $skip_chunk_size\
         --vad_type $vad_type
    done

fi



## note: I will use cam++200k speaker model
if [ ${stage} -le 70 ] && [ ${stop_stage} -ge 70 ];then
    echo "clustering ......."
    vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
    #vad_threshold="0.31 0.32 0.34 0.36 0.38"
    vad_type="transformer_vad"
    chunk_size=1.5
    step_size=0.75
    skip_chunk_size=0.5
    pretrain_speaker_model_ckpt=/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin
    for sub in dev test;do
     test_set_dir="data/magicdata-RAMC/$sub/"
     for name in $vad_threshold;do
      python scripts/magicdata-RAMC/020_spectral_cluster_common.py\
         --pretrain_speaker_model_ckpt $pretrain_speaker_model_ckpt \
         --vad_threshold $name\
         --predict_vad_path_dir $test_set_dir/predict_vad_${name} \
         --test_set_dir $test_set_dir\
         --predict_rttm_path $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}_cam++_zh_200k\
         --chunk_size $chunk_size\
         --step_size $step_size\
         --skip_chunk_size $skip_chunk_size\
         --vad_type $vad_type


    done
   done
fi

if [ ${stage} -le 71 ] && [ ${stop_stage} -ge 71 ];then
   echo "DER score"
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
   test_set_dir="data/magicdata-RAMC/dev/"
   vad_type="transformer_vad"
   chunk_size=1.5
   step_size=0.75
   skip_chunk_size=0.5
   #collar="0.25 0.0"
   collar="0.25"
   testset="dev"
   #testset="test"
   for sub in $testset;do
    test_set_dir="data/magicdata-RAMC/$sub/"
    test_set_dir_groundtruth=/mntcephfs/lab_data/maduo/datasets/voice-activity-detection/rttm_total/rttm_gt_${sub}_nog0
    for c in $collar ;do
     for name in $vad_threshold;do
       #echo "score for $sub in $c $name"
       $sctk_dir/md-eval.pl -c $c -r $test_set_dir_groundtruth -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}_cam++_zh_200k
     done
    done
   done

 #  bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 71 --stop-stage 71 >logs/020_spectral_cluster_stage71_dev.log 2>&1
 # grep -v 'WARNING' logs/020_spectral_cluster_stage71_dev.log


    #  bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 71 --stop-stage 71 >logs/020_spectral_cluster_stage71_test.log 2>&1
   # grep -v 'WARNING' logs/020_spectral_cluster_stage71_test.log
   # DER score
#7.72/0.06/4.19/3.47
#7.56/0.08/4.03/3.45
#7.44/0.09/3.90/3.45
#7.40/0.11/3.84/3.46
#7.33/0.16/3.71/3.45
#7.29/0.19/3.66/3.45
#7.26/0.22/3.60/3.43
#7.25/0.26/3.56/3.43
#6.81/0.29/3.52/3.01
#6.82/0.32/3.48/3.02
#6.82/0.38/3.41/3.03
#6.84/0.46/3.34/3.03
#6.86/0.57/3.28/3.02
fi

if [ ${stage} -le 72 ] && [ ${stop_stage} -ge 72 ];then
   echo "CDER score"
   #note:collar of CDER score default set is equal to zero.
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
   test_set_dir="data/magicdata-RAMC/dev/"
   vad_type="transformer_vad"
   chunk_size=1.5
   step_size=0.75
   skip_chunk_size=0.5
   . path_for_nn_vad.sh
   #testset="dev test"
   #testset="dev"
   testset="test"
   for sub in $testset;do
    test_set_dir="data/magicdata-RAMC/$sub/"
    test_set_dir_groundtruth=/mntcephfs/lab_data/maduo/datasets/voice-activity-detection/rttm_total/rttm_gt_${sub}_nog0
     for name in $vad_threshold;do
       python clustering_based/cder/score.py  -r $test_set_dir_groundtruth -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}_cam++_zh_200k
     done
    done
    #  bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 72 --stop-stage 72 >logs/020_spectral_cluster_stage72_dev.log 2>&1
    # Avg CDER : 0.231
#Avg CDER : 0.257
#Avg CDER : 0.179
#Avg CDER : 0.194
#Avg CDER : 0.182
#Avg CDER : 0.174
#Avg CDER : 0.198
#Avg CDER : 0.203
#Avg CDER : 0.203
#Avg CDER : 0.203
#Avg CDER : 0.194
#Avg CDER : 0.160
#Avg CDER : 0.188
#  bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 72 --stop-stage 72 >logs/020_spectral_cluster_stage72_test.log 2>&1
#Avg CDER : 0.249
#Avg CDER : 0.201
#Avg CDER : 0.249
#Avg CDER : 0.263
#Avg CDER : 0.218
#Avg CDER : 0.241
#Avg CDER : 0.202
#Avg CDER : 0.176
#Avg CDER : 0.167
#Avg CDER : 0.167
#Avg CDER : 0.158
#Avg CDER : 0.179
#Avg CDER : 0.175
fi


 ## using tao liu's vad model and its paper setting
 ## Comparing stage76 and stage77 shows that there is no problem with the rttm I prepared.
 ## Comparing stage77 and stage78 shows that if remove g0 segments from groundtruth rttm
 #  and CDER will descreased and match his paper best dev_CDER(i.e.13.2%) .
 if [ ${stage} -le 75 ] && [ ${stop_stage} -ge 75 ];then
    echo "clustering ......."
    #vad_threshold="0.3 0.47 0.5 0.6 0.7 0.8 0.9"
    vad_threshold="0.9"
    #vad_threshold="0.31 0.32 0.34 0.36 0.38"
    vad_type="transformer_vad"
    chunk_size=3
    step_size=3
    skip_chunk_size=0.93
    pretrain_speaker_model_ckpt=/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt
    #for sub in dev test;do
    #testset="dev test"
    testset="dev"
    for sub in $testset;do
     test_set_dir="data/magicdata-RAMC/$sub/"
     for name in $vad_threshold;do
      python scripts/magicdata-RAMC/020_spectral_cluster_common.py\
         --pretrain_speaker_model_ckpt $pretrain_speaker_model_ckpt \
         --vad_threshold $name\
         --predict_vad_path_dir $test_set_dir/predict_vad_other_people$name \
         --test_set_dir $test_set_dir\
         --predict_rttm_path $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}_other_people_vad\
         --chunk_size $chunk_size\
         --step_size $step_size\
         --skip_chunk_size $skip_chunk_size\
         --vad_type $vad_type


    done
   done
fi
if [ ${stage} -le 76 ] && [ ${stop_stage} -ge 76 ];then
   echo "CDER score"
   #note:collar of CDER score default set is equal to zero.
   #vad_threshold="0.3 0.47 0.5 0.6 0.7 0.8 0.9"
   vad_threshold="0.9"
   vad_type="transformer_vad"
   chunk_size=3
   step_size=3
   skip_chunk_size=0.93
   testset="dev"
   . path_for_nn_vad.sh
   for sub in $testset;do
    test_set_dir="data/magicdata-RAMC/$sub/"
     for name in $vad_threshold;do
       python clustering_based/cder/score.py  -r $test_set_dir/rttm -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}_other_people_vad
     done
    done
# bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 76
# CDER score
#CTS-CN-F2F-2019-11-15-1421 CDER = 0.210
#CTS-CN-F2F-2019-11-15-1422 CDER = 0.448
#CTS-CN-F2F-2019-11-15-1423 CDER = 0.170
#CTS-CN-F2F-2019-11-15-1426 CDER = 0.242
#CTS-CN-F2F-2019-11-15-1428 CDER = 0.282
#CTS-CN-F2F-2019-11-15-1434 CDER = 0.147
#CTS-CN-F2F-2019-11-15-1447 CDER = 0.203
#CTS-CN-F2F-2019-11-15-1448 CDER = 0.194
#CTS-CN-F2F-2019-11-15-1449 CDER = 0.190
#CTS-CN-F2F-2019-11-15-1452 CDER = 0.187
#CTS-CN-F2F-2019-11-15-1458 CDER = 0.266
#CTS-CN-F2F-2019-11-15-1461 CDER = 0.143
#CTS-CN-F2F-2019-11-15-1463 CDER = 0.204
#CTS-CN-F2F-2019-11-15-1468 CDER = 0.171
#CTS-CN-F2F-2019-11-15-1469 CDER = 0.237
#CTS-CN-F2F-2019-11-15-1470 CDER = 0.278
#CTS-CN-F2F-2019-11-15-1473 CDER = 0.163
#CTS-CN-F2F-2019-11-15-1475 CDER = 0.367
#CTS-CN-F2F-2019-11-15-1477 CDER = 0.162
#Avg CDER : 0.224
fi

if [ ${stage} -le 77 ] && [ ${stop_stage} -ge 77 ];then
   echo "CDER score"
   #note:collar of CDER score default set is equal to zero.
   #vad_threshold="0.3 0.47 0.5 0.6 0.7 0.8 0.9"
   vad_threshold="0.9"
   vad_type="transformer_vad"
   chunk_size=3
   step_size=3
   skip_chunk_size=0.93
   testset="dev"
   . path_for_nn_vad.sh
   for sub in $testset;do
    test_set_dir="data/magicdata-RAMC/$sub/"
    test_set_dir_groundtruth=/mntcephfs/lab_data/maduo/datasets/voice-activity-detection/rttm_total/rttm_gt_dev
     for name in $vad_threshold;do
       python clustering_based/cder/score.py  -r $test_set_dir_groundtruth -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}_other_people_vad
     done
    done
# CDER score
#CTS-CN-F2F-2019-11-15-1421 CDER = 0.210
#CTS-CN-F2F-2019-11-15-1422 CDER = 0.448
#CTS-CN-F2F-2019-11-15-1423 CDER = 0.170
#CTS-CN-F2F-2019-11-15-1426 CDER = 0.242
#CTS-CN-F2F-2019-11-15-1428 CDER = 0.282
#CTS-CN-F2F-2019-11-15-1434 CDER = 0.147
#CTS-CN-F2F-2019-11-15-1447 CDER = 0.203
#CTS-CN-F2F-2019-11-15-1448 CDER = 0.194
#CTS-CN-F2F-2019-11-15-1449 CDER = 0.190
#CTS-CN-F2F-2019-11-15-1452 CDER = 0.187
#CTS-CN-F2F-2019-11-15-1458 CDER = 0.266
#CTS-CN-F2F-2019-11-15-1461 CDER = 0.143
#CTS-CN-F2F-2019-11-15-1463 CDER = 0.204
#CTS-CN-F2F-2019-11-15-1468 CDER = 0.171
#CTS-CN-F2F-2019-11-15-1469 CDER = 0.237
#CTS-CN-F2F-2019-11-15-1470 CDER = 0.278
#CTS-CN-F2F-2019-11-15-1473 CDER = 0.163
#CTS-CN-F2F-2019-11-15-1475 CDER = 0.367
#CTS-CN-F2F-2019-11-15-1477 CDER = 0.162
#Avg CDER : 0.224
fi

if [ ${stage} -le 78 ] && [ ${stop_stage} -ge 78 ];then
   echo "CDER score"
   #note:collar of CDER score default set is equal to zero.
   #vad_threshold="0.3 0.47 0.5 0.6 0.7 0.8 0.9"
   vad_threshold="0.9"
   vad_type="transformer_vad"
   chunk_size=3
   step_size=3
   skip_chunk_size=0.93
   testset="dev"
   . path_for_nn_vad.sh
   for sub in $testset;do
    test_set_dir="data/magicdata-RAMC/$sub/"
    test_set_dir_groundtruth=/mntcephfs/lab_data/maduo/datasets/voice-activity-detection/rttm_total/rttm_gt_dev_nog0
     for name in $vad_threshold;do
       python clustering_based/cder/score.py  -r $test_set_dir_groundtruth -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}_other_people_vad
     done
    done
# CDER score
#CTS-CN-F2F-2019-11-15-1421 CDER = 0.128
#CTS-CN-F2F-2019-11-15-1422 CDER = 0.362
#CTS-CN-F2F-2019-11-15-1423 CDER = 0.140
#CTS-CN-F2F-2019-11-15-1426 CDER = 0.120
#CTS-CN-F2F-2019-11-15-1428 CDER = 0.174
#CTS-CN-F2F-2019-11-15-1434 CDER = 0.092
#CTS-CN-F2F-2019-11-15-1447 CDER = 0.126
#CTS-CN-F2F-2019-11-15-1448 CDER = 0.152
#CTS-CN-F2F-2019-11-15-1449 CDER = 0.101
#CTS-CN-F2F-2019-11-15-1452 CDER = 0.103
#CTS-CN-F2F-2019-11-15-1458 CDER = 0.136
#CTS-CN-F2F-2019-11-15-1461 CDER = 0.000
#CTS-CN-F2F-2019-11-15-1463 CDER = 0.054
#CTS-CN-F2F-2019-11-15-1468 CDER = 0.082
#CTS-CN-F2F-2019-11-15-1469 CDER = 0.129
#CTS-CN-F2F-2019-11-15-1470 CDER = 0.161
#CTS-CN-F2F-2019-11-15-1473 CDER = 0.062
#CTS-CN-F2F-2019-11-15-1475 CDER = 0.208
#CTS-CN-F2F-2019-11-15-1477 CDER = 0.139
#Avg CDER : 0.130
fi


if [ ${stage} -le 80 ] && [ ${stop_stage} -ge 80 ];then
    echo "clustering ......."
    #vad_threshold="0.3 0.47 0.5 0.6 0.7 0.8 0.9"
    vad_threshold="0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91 0.92"
    #vad_threshold="0.31 0.32 0.34 0.36 0.38"
    vad_type="transformer_vad"
    chunk_size=2
    step_size=2
    skip_chunk_size=0.93
    pretrain_speaker_model_ckpt=/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt
    #for sub in dev test;do
    #testset="dev test"
    testset="dev"
    for sub in $testset;do
     test_set_dir="data/magicdata-RAMC/$sub/"
     for name in $vad_threshold;do
      python scripts/magicdata-RAMC/020_spectral_cluster_common.py\
         --pretrain_speaker_model_ckpt $pretrain_speaker_model_ckpt \
         --vad_threshold $name\
         --predict_vad_path_dir $test_set_dir/predict_vad_$name \
         --test_set_dir $test_set_dir\
         --predict_rttm_path $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}\
         --chunk_size $chunk_size\
         --step_size $step_size\
         --skip_chunk_size $skip_chunk_size\
         --vad_type $vad_type


    done
   done
fi

if [ ${stage} -le 81 ] && [ ${stop_stage} -ge 81 ];then
   echo "CDER score"
   #note:collar of CDER score default set is equal to zero.
   #vad_threshold="0.3 0.47 0.5 0.6 0.7 0.8 0.9"
   vad_threshold="0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91 0.92"
   vad_type="transformer_vad"
   chunk_size=2
   step_size=2
   skip_chunk_size=0.93
   testset="dev"
   . path_for_nn_vad.sh
   for sub in $testset;do
    test_set_dir="data/magicdata-RAMC/$sub/"
    test_set_dir_groundtruth=/mntcephfs/lab_data/maduo/datasets/voice-activity-detection/rttm_total/rttm_gt_dev_nog0
     for name in $vad_threshold;do
       python clustering_based/cder/score.py  -r $test_set_dir_groundtruth -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}
     done
    done
  # grep -r 'Avg CDER :' logs/020_spectral_cluster_V100_stage80-81_debug_1.log
#Avg CDER : 0.120
#Avg CDER : 0.141
#Avg CDER : 0.142
#Avg CDER : 0.119
#Avg CDER : 0.118
#Avg CDER : 0.119
#Avg CDER : 0.119
#Avg CDER : 0.117
#Avg CDER : 0.117
#Avg CDER : 0.121
#Avg CDER : 0.121
#Avg CDER : 0.131
#Avg CDER : 0.140
#Avg CDER : 0.142
fi

if [ ${stage} -le 82 ] && [ ${stop_stage} -ge 82 ];then
    echo "clustering ......."
    #vad_threshold="0.3 0.47 0.5 0.6 0.7 0.8 0.9"
    vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91 0.92"
    #vad_threshold="0.31 0.32 0.34 0.36 0.38"
    vad_type="transformer_vad"
    chunk_size=3
    step_size=3
    skip_chunk_size=0.93
    pretrain_speaker_model_ckpt=/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt
    #for sub in dev test;do
    #testset="dev test"
    testset="dev"
    for sub in $testset;do
     test_set_dir="data/magicdata-RAMC/$sub/"
     for name in $vad_threshold;do
      python scripts/magicdata-RAMC/020_spectral_cluster_common.py\
         --pretrain_speaker_model_ckpt $pretrain_speaker_model_ckpt \
         --vad_threshold $name\
         --predict_vad_path_dir $test_set_dir/predict_vad_$name \
         --test_set_dir $test_set_dir\
         --predict_rttm_path $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}\
         --chunk_size $chunk_size\
         --step_size $step_size\
         --skip_chunk_size $skip_chunk_size\
         --vad_type $vad_type
    done
   done
fi

if [ ${stage} -le 83 ] && [ ${stop_stage} -ge 83 ];then
   echo "CDER score"
   #note:collar of CDER score default set is equal to zero.
   #vad_threshold="0.3 0.47 0.5 0.6 0.7 0.8 0.9"
   vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91 0.92"
   vad_type="transformer_vad"
   chunk_size=3
   step_size=3
   skip_chunk_size=0.93
   testset="dev"
   . path_for_nn_vad.sh
   for sub in $testset;do
    test_set_dir="data/magicdata-RAMC/$sub/"
    test_set_dir_groundtruth=/mntcephfs/lab_data/maduo/datasets/voice-activity-detection/rttm_total/rttm_gt_dev_nog0 # The groundtruth version provided by the challenge
     for name in $vad_threshold;do
       python clustering_based/cder/score.py  -r $test_set_dir_groundtruth -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}
     done
    done
 # grep -r 'Avg CDER :' logs/020_spectral_cluster_V100_stage82-83_debug_1.log
#Avg CDER : 0.158
#Avg CDER : 0.148
#Avg CDER : 0.124
#Avg CDER : 0.124
#Avg CDER : 0.122
#Avg CDER : 0.123
#Avg CDER : 0.121
#Avg CDER : 0.121
#Avg CDER : 0.119
#Avg CDER : 0.120
#Avg CDER : 0.119
#Avg CDER : 0.117
#Avg CDER : 0.119
#Avg CDER : 0.119
#Avg CDER : 0.119
#Avg CDER : 0.118
#Avg CDER : 0.119
#Avg CDER : 0.117
#Avg CDER : 0.119
#Avg CDER : 0.118
#Avg CDER : 0.117
#Avg CDER : 0.115
#Avg CDER : 0.111
#Avg CDER : 0.112
#Avg CDER : 0.117
#Avg CDER : 0.130
#Avg CDER : 0.136
#Avg CDER : 0.142

## The statistics of dev set are as follows:
# durations of g0 segement=507.52s, total duration=29986.2s, rate=1.7%
# num of g0 segement = 597, num of total =11037 ; rate=5.41%
fi


if [ ${stage} -le 84 ] && [ ${stop_stage} -ge 84 ];then
    echo "clustering ......."
    #vad_threshold="0.3 0.47 0.5 0.6 0.7 0.8 0.9"
    vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91 0.92"
    #vad_threshold="0.31 0.32 0.34 0.36 0.38"
    vad_type="transformer_vad"
    chunk_size=3
    step_size=3
    skip_chunk_size=0.93
    pretrain_speaker_model_ckpt=/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt
    #for sub in dev test;do
    #testset="dev test"
    testset="test"
    for sub in $testset;do
     test_set_dir="data/magicdata-RAMC/$sub/"
     for name in $vad_threshold;do
      python scripts/magicdata-RAMC/020_spectral_cluster_common.py\
         --pretrain_speaker_model_ckpt $pretrain_speaker_model_ckpt \
         --vad_threshold $name\
         --predict_vad_path_dir $test_set_dir/predict_vad_$name \
         --test_set_dir $test_set_dir\
         --predict_rttm_path $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}\
         --chunk_size $chunk_size\
         --step_size $step_size\
         --skip_chunk_size $skip_chunk_size\
         --vad_type $vad_type
    done
   done
fi
if [ ${stage} -le 85 ] && [ ${stop_stage} -ge 85 ];then
   echo "CDER score"
   #note:collar of CDER score default set is equal to zero.
   #vad_threshold="0.3 0.47 0.5 0.6 0.7 0.8 0.9"
   vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91 0.92"
   vad_type="transformer_vad"
   chunk_size=3
   step_size=3
   skip_chunk_size=0.93
   testset="test"
   . path_for_nn_vad.sh
   for sub in $testset;do
    test_set_dir="data/magicdata-RAMC/$sub/"
    test_set_dir_groundtruth=/mntcephfs/lab_data/maduo/datasets/voice-activity-detection/rttm_total/rttm_gt_${sub}_nog0
     for name in $vad_threshold;do
       python clustering_based/cder/score.py  -r $test_set_dir_groundtruth -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}
     done
    done
# grep -r 'Avg CDER :' logs/020_spectral_cluster_V100_stage84-85_debug_1.log
#Avg CDER : 0.119
#Avg CDER : 0.108
#Avg CDER : 0.105
#Avg CDER : 0.103
#Avg CDER : 0.103
#Avg CDER : 0.102
#Avg CDER : 0.101
#Avg CDER : 0.102
#Avg CDER : 0.102
#Avg CDER : 0.102
#Avg CDER : 0.102
#Avg CDER : 0.104
#Avg CDER : 0.102
#Avg CDER : 0.102
#Avg CDER : 0.102
#Avg CDER : 0.102
#Avg CDER : 0.100
#Avg CDER : 0.100
#Avg CDER : 0.099
#Avg CDER : 0.100
#Avg CDER : 0.100
#Avg CDER : 0.100
#Avg CDER : 0.100
#Avg CDER : 0.109
#Avg CDER : 0.099
#Avg CDER : 0.121
#Avg CDER : 0.125
#Avg CDER : 0.128
## The statistics of test set are as follows:
# grep G00000000 data/magicdata-RAMC/test/rttm_gt_test> rttm_gt_test_g0
#  awk -F ' ' '{sum+=$5;}END{print sum}' rttm_gt_test_g0
# awk -F ' ' '{sum+=$5;}END{print sum}' data/magicdata-RAMC/test/rttm_gt_test
# durations of g0 segement=1978.5s, total duration=63446.5s, rate=3.12%
# num of g0 segement = 2306, num of total =25369 ; rate=9.1%

fi
if [ ${stage} -le 86 ] && [ ${stop_stage} -ge 86 ];then
   echo "CDER score"
   #note:collar of CDER score default set is equal to zero.
   #vad_threshold="0.3 0.47 0.5 0.6 0.7 0.8 0.9"
   vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91 0.92"
   vad_type="transformer_vad"
   chunk_size=3
   step_size=3
   skip_chunk_size=0.93
   testset="test"
   . path_for_nn_vad.sh
   for sub in $testset;do
    test_set_dir="data/magicdata-RAMC/$sub/"
    #test_set_dir_groundtruth=/mntcephfs/lab_data/maduo/datasets/voice-activity-detection/rttm_total/rttm_gt_${sub}_nog0 #The groundtruth version provided by the challenge
    test_set_dir_groundtruth=data/magicdata-RAMC/test/rttm_openslr_gt_test_nog0 # the groundtruth version provided by the openslr.
     for name in $vad_threshold;do
       python clustering_based/cder/score.py  -r $test_set_dir_groundtruth -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}
     done
    done
  #grep -v G00000000 data/magicdata-RAMC/test/rttm > data/magicdata-RAMC/test/rttm_openslr_gt_test_nog0
    # bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 86 --stop-stage 86 > logs/020_spectral_cluster_stage86.log 2>&1
  #  grep -r 'Avg' logs/020_spectral_cluster_stage86.log
#Avg CDER : 0.120
#Avg CDER : 0.110
#Avg CDER : 0.107
#Avg CDER : 0.105
#Avg CDER : 0.105
#Avg CDER : 0.105
#Avg CDER : 0.103
#Avg CDER : 0.104
#Avg CDER : 0.104
#Avg CDER : 0.104
#Avg CDER : 0.104
#Avg CDER : 0.106
#Avg CDER : 0.104
#Avg CDER : 0.104
#Avg CDER : 0.104
#Avg CDER : 0.104
#Avg CDER : 0.102
#Avg CDER : 0.102
#Avg CDER : 0.101
#Avg CDER : 0.102
#Avg CDER : 0.101
#Avg CDER : 0.101
#Avg CDER : 0.102 as report vad_threshod=0.6, path : /mntnfs/lee_data1/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/data/magicdata-RAMC/test/rttm_predict_0.6_transformer_vad_chunk_size_3_step_size_3_skip_chunk_size_0.93
#Avg CDER : 0.110
#Avg CDER : 0.101
#Avg CDER : 0.123
#Avg CDER : 0.127
#Avg CDER : 0.130

# 2025-2-26, more specify short term DER:
# for example : less than 1s segment
# python3 ../multi_datasets/ts_vad2/short_term_statistics.py 1 /mntnfs/lee_data1/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/data/magicdata-RAMC/test/rttm_predict_0.6_transformer_vad_chunk_size_3_step_size_3_skip_chunk_size_0.93 /mntnfs/lee_data1/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/data/magicdata-RAMC/test/rttm_openslr_gt_test_nog0
# der=56.63,miss=29.89, false=0.0,confusion=26.74

#for example : less than 2s segment
# python3 ../multi_datasets/ts_vad2/short_term_statistics.py 2 /mntnfs/lee_data1/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/data/magicdata-RAMC/test/rttm_predict_0.6_transformer_vad_chunk_size_3_step_size_3_skip_chunk_size_0.93 /mntnfs/lee_data1/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/data/magicdata-RAMC/test/rttm_openslr_gt_test_nog0
# der=27.97,miss=9.71, false=0.0,confusion=18.25
fi

if [ ${stage} -le 87 ] && [ ${stop_stage} -ge 87 ];then
   echo "CDER score"
   #note:collar of CDER score default set is equal to zero.
   #vad_threshold="0.3 0.47 0.5 0.6 0.7 0.8 0.9"
   vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91 0.92"
   vad_type="transformer_vad"
   chunk_size=3
   step_size=3
   skip_chunk_size=0.93
   testset="dev"
   . path_for_nn_vad.sh
   for sub in $testset;do
    test_set_dir="data/magicdata-RAMC/$sub/"
    #test_set_dir_groundtruth=/mntcephfs/lab_data/maduo/datasets/voice-activity-detection/rttm_total/rttm_gt_dev_nog0 # The groundtruth version provided by the challenge
    test_set_dir_groundtruth=data/magicdata-RAMC/dev/rttm_openslr_gt_dev_nog0 # the groundtruth version provided by the openslr.
     for name in $vad_threshold;do
       python clustering_based/cder/score.py  -r $test_set_dir_groundtruth -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}
     done
    done
    #grep -v G00000000 data/magicdata-RAMC/dev/rttm > data/magicdata-RAMC/dev/rttm_openslr_gt_dev_nog0
    # bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 87 --stop-stage 87 > logs/020_spectral_cluster_stage87.log 2>&1
    # grep -r 'Avg' logs/020_spectral_cluster_stage87.log
#Avg CDER : 0.158
#Avg CDER : 0.148
#Avg CDER : 0.124
#Avg CDER : 0.124
#Avg CDER : 0.122
#Avg CDER : 0.123
#Avg CDER : 0.121
#Avg CDER : 0.121
#Avg CDER : 0.119
#Avg CDER : 0.120
#Avg CDER : 0.119
#Avg CDER : 0.117
#Avg CDER : 0.119
#Avg CDER : 0.119
#Avg CDER : 0.119
#Avg CDER : 0.118
#Avg CDER : 0.119
#Avg CDER : 0.117
#Avg CDER : 0.119
#Avg CDER : 0.118
#Avg CDER : 0.117
#Avg CDER : 0.115
#Avg CDER : 0.111 as report vad_threshod=0.6 path : /mntnfs/lee_data1/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/data/magicdata-RAMC/dev/rttm_predict_0.6_transformer_vad_chunk_size_3_step_size_3_skip_chunk_size_0.93
#Avg CDER : 0.112
#Avg CDER : 0.117
#Avg CDER : 0.130
#Avg CDER : 0.136
#Avg CDER : 0.142
# 2025-2-26, more specify short term DER:
# for example : less than 1s segment
# python3 ../multi_datasets/ts_vad2/short_term_statistics.py 1 /mntnfs/lee_data1/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/data/magicdata-RAMC/dev/rttm_predict_0.6_transformer_vad_chunk_size_3_step_size_3_skip_chunk_size_0.93 /mntnfs/lee_data1/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/data/magicdata-RAMC/dev/rttm_openslr_gt_dev_nog0
# der=56.68,miss=30.89, false=0.0,confusion=25.79

# for example : less than 2s segment
# python3 ../multi_datasets/ts_vad2/short_term_statistics.py 2 /mntnfs/lee_data1/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/data/magicdata-RAMC/dev/rttm_predict_0.6_transformer_vad_chunk_size_3_step_size_3_skip_chunk_size_0.93 /mntnfs/lee_data1/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/data/magicdata-RAMC/dev/rttm_openslr_gt_dev_nog0
# der=25.64,miss=10.62, false=0.0,confusion=15.03


fi

if [ ${stage} -le 88 ] && [ ${stop_stage} -ge 88 ];then
   echo "DER score"
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91 0.92"
   vad_type="transformer_vad"
   chunk_size=3
   step_size=3
   skip_chunk_size=0.93
   testset="dev"
   . path_for_nn_vad.sh
   #collar=0.0
   collar=0.25
   for c in $collar;do
      echo "compute collar=$c mode"
   for sub in $testset;do
    test_set_dir="data/magicdata-RAMC/$sub/"
    #test_set_dir_groundtruth=/mntcephfs/lab_data/maduo/datasets/voice-activity-detection/rttm_total/rttm_gt_dev_nog0 # The groundtruth version provided by the challenge
    test_set_dir_groundtruth=data/magicdata-RAMC/dev/rttm_openslr_gt_dev_nog0 # the groundtruth version provided by the openslr.

    for name in $vad_threshold;do
     $sctk_dir/md-eval.pl -c $c -r $test_set_dir_groundtruth -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}
   done
   done
 done
 # grep -v WARNING: logs/020_spectral_cluster_V100_stage88_debug_1.log
 # DER score
#compute collar=0.0 mode
#27.14/0.29/17.16/9.69
#23.59/0.82/13.54/9.23
#21.54/0.93/13.01/7.61
#21.15/1.05/12.53/7.57
#20.93/1.13/12.27/7.53
#20.30/1.31/11.52/7.46
#20.06/1.42/11.20/7.43
#19.81/1.52/10.89/7.40
#19.59/1.63/10.61/7.35
#19.42/1.72/10.37/7.33
#19.17/1.80/10.11/7.26
#18.79/1.94/9.67/7.17
#18.47/2.18/9.20/7.09
#18.28/2.44/8.79/7.04
#18.11/2.68/8.43/7.00
#17.79/3.21/7.73/6.85
#17.77/3.38/7.57/6.81
#17.71/3.52/7.41/6.79
#17.67/4.05/6.97/6.65
#17.64/4.19/6.85/6.61
#17.61/4.31/6.71/6.58
#17.63/4.49/6.57/6.57
#17.69/6.18/5.27/6.24
#19.37/9.88/3.78/5.70
#24.75/17.40/2.51/4.84
#40.70/36.76/1.35/2.59
#43.45/39.89/1.23/2.34
#46.32/43.06/1.14/2.13
#grep -v WARNING: logs/020_spectral_cluster_V100_stage88_debug_2.log
#DER score
#compute collar=0.25 mode
#14.77/0.11/7.09/7.57
#11.68/0.37/4.04/7.27
#9.84/0.42/3.73/5.69
#9.63/0.49/3.47/5.67
#9.53/0.54/3.36/5.64
#9.37/0.62/3.15/5.60
#9.31/0.67/3.07/5.58
#9.26/0.72/2.98/5.56
#9.21/0.77/2.91/5.52
#9.18/0.81/2.85/5.51
#9.08/0.86/2.77/5.45
#8.97/0.93/2.64/5.40
#8.94/1.09/2.53/5.33
#8.98/1.26/2.43/5.29
#9.01/1.42/2.33/5.27
#9.12/1.80/2.16/5.17
#9.19/1.91/2.13/5.15
#9.22/2.00/2.08/5.13
#9.44/2.40/1.97/5.06
#9.48/2.51/1.94/5.03
#9.54/2.61/1.91/5.02
#9.64/2.73/1.89/5.01
#10.38/3.90/1.68/4.80
#12.40/6.50/1.44/4.46
#17.59/12.50/1.17/3.91
#34.46/31.59/0.81/2.06
#37.44/34.83/0.76/1.85
#40.52/38.13/0.71/1.68
 fi

 if [ ${stage} -le 89 ] && [ ${stop_stage} -ge 89 ];then
   echo "DER score"
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91 0.92"
   vad_type="transformer_vad"
   chunk_size=3
   step_size=3
   skip_chunk_size=0.93
   testset="test"
   . path_for_nn_vad.sh
   #collar=0.0
   collar=0.25
   for c in $collar;do
      echo "compute collar=$c mode"
   for sub in $testset;do
    test_set_dir="data/magicdata-RAMC/$sub/"
    #test_set_dir_groundtruth=/mntcephfs/lab_data/maduo/datasets/voice-activity-detection/rttm_total/rttm_gt_test_nog0 # The groundtruth version provided by the challenge
    test_set_dir_groundtruth=data/magicdata-RAMC/test/rttm_openslr_gt_test_nog0 # the groundtruth version provided by the openslr.

    for name in $vad_threshold;do
     $sctk_dir/md-eval.pl -c $c -r $test_set_dir_groundtruth -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}
   done
   done
 done
 #grep -v WARNING: logs/020_spectral_cluster_V100_stage89_debug_1.log
#DER score
#compute collar=0.0 mode
#25.22/0.41/16.04/8.77
#22.35/0.85/13.02/8.48
#21.94/0.96/12.57/8.41
#21.64/1.09/12.19/8.36
#21.47/1.18/11.96/8.33
#20.93/1.36/11.33/8.24
#20.66/1.50/10.97/8.19
#20.44/1.61/10.66/8.18
#20.25/1.72/10.40/8.13
#20.11/1.85/10.16/8.10
#19.99/1.95/9.95/8.09
#19.77/2.16/9.57/8.05
#19.59/2.39/9.21/7.98
#19.43/2.69/8.86/7.88
#19.32/2.91/8.56/7.84
#19.14/3.56/7.91/7.67
#19.12/3.70/7.79/7.63
#19.09/3.83/7.67/7.59
#19.07/4.31/7.31/7.46
#19.09/4.47/7.19/7.43
#19.11/4.63/7.07/7.40
#19.10/4.80/6.95/7.34
#19.68/6.74/5.85/7.09
#21.81/10.61/4.54/6.66
#27.23/18.04/3.43/5.75
#42.15/35.92/2.34/3.89
#44.54/38.65/2.22/3.67
#47.10/41.56/2.08/3.46
#grep -v WARNING: logs/020_spectral_cluster_V100_stage89_debug_2.log
#DER score
#compute collar=0.25 mode
#12.66/0.17/6.33/6.16
#10.66/0.40/4.29/5.97
#10.48/0.47/4.10/5.91
#10.39/0.54/3.96/5.89
#10.34/0.59/3.89/5.86
#10.25/0.70/3.75/5.80
#10.24/0.78/3.69/5.77
#10.23/0.84/3.63/5.76
#10.23/0.92/3.59/5.71
#10.26/1.01/3.55/5.71
#10.28/1.07/3.52/5.69
#10.34/1.20/3.44/5.69
#10.37/1.36/3.37/5.65
#10.44/1.57/3.30/5.58
#10.52/1.73/3.24/5.55
#10.78/2.21/3.12/5.44
#10.84/2.33/3.10/5.41
#10.87/2.42/3.08/5.37
#11.09/2.80/3.01/5.28
#11.18/2.92/2.99/5.27
#11.26/3.04/2.97/5.25
#11.31/3.16/2.94/5.20
#12.19/4.35/2.75/5.08
#14.26/6.94/2.49/4.83
#19.22/12.71/2.18/4.32
#35.04/30.22/1.70/3.12
#37.62/33.02/1.63/2.96
#40.42/36.06/1.54/2.82
 fi

 if [ ${stage} -le 90 ] && [ ${stop_stage} -ge 90 ];then
    echo "clustering ......."
    #vad_threshold="0.3 0.47 0.5 0.6 0.7 0.8 0.9"
    vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91 0.92"
    #vad_threshold="0.31 0.32 0.34 0.36 0.38"
    vad_type="transformer_vad"
    chunk_size=2
    step_size=2
    skip_chunk_size=0.93
    pretrain_speaker_model_ckpt=/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt
    #for sub in dev test;do
    #testset="dev test"
    testset="dev"
    #testset="test"
    for sub in $testset;do
     test_set_dir="data/magicdata-RAMC/$sub/"
     for name in $vad_threshold;do
      python scripts/magicdata-RAMC/020_spectral_cluster_common.py\
         --pretrain_speaker_model_ckpt $pretrain_speaker_model_ckpt \
         --vad_threshold $name\
         --predict_vad_path_dir $test_set_dir/predict_vad_$name \
         --test_set_dir $test_set_dir\
         --predict_rttm_path $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}\
         --chunk_size $chunk_size\
         --step_size $step_size\
         --skip_chunk_size $skip_chunk_size\
         --vad_type $vad_type
    done
   done
fi


 if [ ${stage} -le 91 ] && [ ${stop_stage} -ge 91 ];then
    echo "clustering ......."
    #vad_threshold="0.3 0.47 0.5 0.6 0.7 0.8 0.9"
    vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91 0.92"
    #vad_threshold="0.31 0.32 0.34 0.36 0.38"
    vad_type="transformer_vad"
    chunk_size=2
    step_size=2
    skip_chunk_size=0.93
    pretrain_speaker_model_ckpt=/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt
    #for sub in dev test;do
    #testset="dev test"
    #testset="dev"
    testset="test"
    for sub in $testset;do
     test_set_dir="data/magicdata-RAMC/$sub/"
     for name in $vad_threshold;do
      python scripts/magicdata-RAMC/020_spectral_cluster_common.py\
         --pretrain_speaker_model_ckpt $pretrain_speaker_model_ckpt \
         --vad_threshold $name\
         --predict_vad_path_dir $test_set_dir/predict_vad_$name \
         --test_set_dir $test_set_dir\
         --predict_rttm_path $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}\
         --chunk_size $chunk_size\
         --step_size $step_size\
         --skip_chunk_size $skip_chunk_size\
         --vad_type $vad_type
    done
   done
fi

if [ ${stage} -le 92 ] && [ ${stop_stage} -ge 92 ];then
    echo "clustering ......."
    #vad_threshold="0.3 0.47 0.5 0.6 0.7 0.8 0.9"
    vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91 0.92"
    #vad_threshold="0.31 0.32 0.34 0.36 0.38"
    vad_type="transformer_vad"
    chunk_size=1.5
    step_size=1.5
    skip_chunk_size=0.93
    pretrain_speaker_model_ckpt=/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt
    #for sub in dev test;do
    #testset="dev test"
    testset="dev"
    #testset="test"
    for sub in $testset;do
     test_set_dir="data/magicdata-RAMC/$sub/"
     for name in $vad_threshold;do
      python scripts/magicdata-RAMC/020_spectral_cluster_common.py\
         --pretrain_speaker_model_ckpt $pretrain_speaker_model_ckpt \
         --vad_threshold $name\
         --predict_vad_path_dir $test_set_dir/predict_vad_$name \
         --test_set_dir $test_set_dir\
         --predict_rttm_path $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}\
         --chunk_size $chunk_size\
         --step_size $step_size\
         --skip_chunk_size $skip_chunk_size\
         --vad_type $vad_type
    done
   done
fi

if [ ${stage} -le 93 ] && [ ${stop_stage} -ge 93 ];then
    echo "clustering ......."
    #vad_threshold="0.3 0.47 0.5 0.6 0.7 0.8 0.9"
    vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91 0.92"
    #vad_threshold="0.31 0.32 0.34 0.36 0.38"
    vad_type="transformer_vad"
    chunk_size=1.5
    step_size=1.5
    skip_chunk_size=0.93
    pretrain_speaker_model_ckpt=/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt
    #for sub in dev test;do
    #testset="dev test"
    #testset="dev"
    testset="test"
    for sub in $testset;do
     test_set_dir="data/magicdata-RAMC/$sub/"
     for name in $vad_threshold;do
      python scripts/magicdata-RAMC/020_spectral_cluster_common.py\
         --pretrain_speaker_model_ckpt $pretrain_speaker_model_ckpt \
         --vad_threshold $name\
         --predict_vad_path_dir $test_set_dir/predict_vad_$name \
         --test_set_dir $test_set_dir\
         --predict_rttm_path $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}\
         --chunk_size $chunk_size\
         --step_size $step_size\
         --skip_chunk_size $skip_chunk_size\
         --vad_type $vad_type
    done
   done
fi

if [ ${stage} -le 94 ] && [ ${stop_stage} -ge 94 ];then
   echo "CDER score"
   #note:collar of CDER score default set is equal to zero.
   #vad_threshold="0.3 0.47 0.5 0.6 0.7 0.8 0.9"
   vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91 0.92"
   vad_type="transformer_vad"
   chunk_size=2
   step_size=2
   skip_chunk_size=0.93
   #testset="dev"
   testset="test"
   . path_for_nn_vad.sh
   for sub in $testset;do
    echo "compute $sub CDER score"
    test_set_dir="data/magicdata-RAMC/$sub/"
    #test_set_dir_groundtruth=/mntcephfs/lab_data/maduo/datasets/voice-activity-detection/rttm_total/rttm_gt_dev_nog0 # The groundtruth version provided by the challenge
    test_set_dir_groundtruth=data/magicdata-RAMC/$sub/rttm_openslr_gt_${sub}_nog0 # the groundtruth version provided by the openslr.
     for name in $vad_threshold;do
       python clustering_based/cder/score.py  -r $test_set_dir_groundtruth -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}
     done
    done

    #grep -r 'Avg' logs/020_spectral_cluster_V100_stage94_debug_1.log
    # compute dev CDER score
#Avg CDER : 0.195
#Avg CDER : 0.180
#Avg CDER : 0.151
#Avg CDER : 0.126
#Avg CDER : 0.126
#Avg CDER : 0.130
#Avg CDER : 0.127
#Avg CDER : 0.125
#Avg CDER : 0.121
#Avg CDER : 0.119
#Avg CDER : 0.143
#Avg CDER : 0.117
#Avg CDER : 0.119
#Avg CDER : 0.117
#Avg CDER : 0.120
#Avg CDER : 0.141
#Avg CDER : 0.142
#Avg CDER : 0.119
#Avg CDER : 0.118
#Avg CDER : 0.119
#Avg CDER : 0.119
#Avg CDER : 0.117
#Avg CDER : 0.117
#Avg CDER : 0.121
#Avg CDER : 0.121
#Avg CDER : 0.131
#Avg CDER : 0.140
#Avg CDER : 0.142

#grep -r 'Avg' logs/020_spectral_cluster_V100_stage94_debug_2.log
# compute test CDER score
#Avg CDER : 0.148
#Avg CDER : 0.122
#Avg CDER : 0.118
#Avg CDER : 0.116
#Avg CDER : 0.115
#Avg CDER : 0.114
#Avg CDER : 0.115
#Avg CDER : 0.116
#Avg CDER : 0.116
#Avg CDER : 0.115
#Avg CDER : 0.113
#Avg CDER : 0.112
#Avg CDER : 0.110
#Avg CDER : 0.108
#Avg CDER : 0.108
#Avg CDER : 0.108
#Avg CDER : 0.107
#Avg CDER : 0.106
#Avg CDER : 0.108
#Avg CDER : 0.107
#Avg CDER : 0.107
#Avg CDER : 0.107
#Avg CDER : 0.106
#Avg CDER : 0.118
#Avg CDER : 0.109
#Avg CDER : 0.131
#Avg CDER : 0.136
#Avg CDER : 0.138
fi

if [ ${stage} -le 95 ] && [ ${stop_stage} -ge 95 ];then
   echo "DER score"
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91 0.92"
   vad_type="transformer_vad"
   chunk_size=2
   step_size=2
   skip_chunk_size=0.93
   testset="dev"
   . path_for_nn_vad.sh
   #collar=0.0
   collar=0.25
   for c in $collar;do
      echo "compute collar=$c mode"
   for sub in $testset;do
       #echo "compute $sub DER score"
    test_set_dir="data/magicdata-RAMC/$sub/"
    #test_set_dir_groundtruth=/mntcephfs/lab_data/maduo/datasets/voice-activity-detection/rttm_total/rttm_gt_test_nog0 # The groundtruth version provided by the challenge
    test_set_dir_groundtruth=data/magicdata-RAMC/$sub/rttm_openslr_gt_${sub}_nog0 # the groundtruth version provided by the openslr.

    for name in $vad_threshold;do
     $sctk_dir/md-eval.pl -c $c -r $test_set_dir_groundtruth -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}
   done
   done
 done
#grep -v WARNING: logs/020_spectral_cluster_V100_stage95_debug_1.log
#DER score
#compute collar=0.0 mode
#27.62/0.50/16.87/10.26
#24.37/1.26/13.14/9.98
#22.35/1.42/12.62/8.31
#20.27/1.54/12.14/6.60
#20.08/1.63/11.88/6.57
#19.54/1.89/11.13/6.52
#19.32/2.00/10.82/6.51
#19.10/2.14/10.51/6.45
#18.93/2.30/10.23/6.40
#18.79/2.42/9.98/6.38
#20.29/2.54/9.73/8.02
#18.36/2.75/9.28/6.33
#18.14/3.07/8.80/6.27
#17.98/3.39/8.38/6.21
#17.87/3.65/8.04/6.17
#19.42/4.41/7.32/7.68
#19.41/4.60/7.17/7.64
#17.73/4.77/7.02/5.95
#17.75/5.33/6.60/5.82
#17.80/5.51/6.48/5.82
#17.85/5.70/6.35/5.80
#17.87/5.86/6.23/5.78
#18.38/7.84/5.01/5.54
#20.40/11.74/3.62/5.04
#26.24/19.55/2.42/4.28
#42.35/38.74/1.30/2.31
#45.15/41.87/1.18/2.10
#47.88/44.90/1.09/1.89

#grep -v WARNING: logs/020_spectral_cluster_V100_stage95_debug_2.log
#DER score
#compute collar=0.25 mode
#15.44/0.20/6.97/8.27
#12.67/0.58/3.93/8.15
#10.81/0.66/3.65/6.50
#8.87/0.73/3.38/4.75
#8.82/0.79/3.27/4.76
#8.72/0.92/3.07/4.73
#8.69/0.96/3.00/4.72
#8.64/1.04/2.91/4.69
#8.63/1.14/2.85/4.64
#8.63/1.20/2.78/4.64
#10.26/1.26/2.72/6.29
#8.58/1.38/2.57/4.62
#8.63/1.58/2.46/4.59
#8.69/1.79/2.36/4.55
#8.78/1.96/2.27/4.54
#10.74/2.53/2.10/6.11
#10.83/2.68/2.07/6.08
#9.23/2.80/2.02/4.41
#9.46/3.22/1.92/4.32
#9.58/3.36/1.89/4.32
#9.71/3.53/1.87/4.31
#9.79/3.66/1.84/4.29
#10.89/5.07/1.65/4.17
#13.20/7.92/1.41/3.86
#18.88/14.36/1.16/3.36
#36.06/33.47/0.80/1.79
#39.11/36.74/0.75/1.62
#42.06/39.90/0.71/1.45
fi

 if [ ${stage} -le 96 ] && [ ${stop_stage} -ge 96 ];then
   echo "DER score"
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91 0.92"
   vad_type="transformer_vad"
   chunk_size=2
   step_size=2
   skip_chunk_size=0.93
   testset="test"
   . path_for_nn_vad.sh
   #collar=0.0
   collar=0.25
   for c in $collar;do
      echo "compute collar=$c mode"
   for sub in $testset;do
      #echo "compute $sub DER score"
    test_set_dir="data/magicdata-RAMC/$sub/"
    #test_set_dir_groundtruth=/mntcephfs/lab_data/maduo/datasets/voice-activity-detection/rttm_total/rttm_gt_test_nog0 # The groundtruth version provided by the challenge
    test_set_dir_groundtruth=data/magicdata-RAMC/$sub/rttm_openslr_gt_${sub}_nog0 # the groundtruth version provided by the openslr.

    for name in $vad_threshold;do
     $sctk_dir/md-eval.pl -c $c -r $test_set_dir_groundtruth -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}
   done
   done
 done
#grep -v WARNING: logs/020_spectral_cluster_V100_stage96_debug_1.log
#DER score
#compute collar=0.0 mode
#23.64/0.61/15.68/7.35
#21.01/1.37/12.59/7.05
#20.70/1.53/12.14/7.03
#20.45/1.66/11.78/7.01
#20.33/1.77/11.57/6.99
#19.90/2.03/10.95/6.91
#19.72/2.21/10.61/6.90
#19.54/2.37/10.30/6.87
#19.43/2.51/10.04/6.88
#19.32/2.65/9.81/6.85
#19.23/2.80/9.59/6.85
#19.09/3.07/9.20/6.82
#18.99/3.39/8.85/6.75
#18.89/3.71/8.51/6.67
#18.85/4.01/8.20/6.63
#18.93/4.83/7.55/6.55
#18.95/5.01/7.43/6.50
#19.00/5.21/7.32/6.47
#19.11/5.77/6.96/6.38
#19.17/5.98/6.84/6.36
#19.20/6.16/6.72/6.31
#19.26/6.37/6.61/6.29
#20.14/8.52/5.57/6.05
#22.78/12.67/4.36/5.74
#28.56/20.24/3.29/5.04
#43.74/38.01/2.24/3.49
#46.09/40.66/2.12/3.31
#48.59/43.45/2.00/3.14

#grep -v WARNING: logs/020_spectral_cluster_V100_stage96_debug_2.log
#DER score
#compute collar=0.25 mode
#11.19/0.25/6.17/4.77
#9.42/0.67/4.17/4.59
#9.33/0.76/3.99/4.58
#9.28/0.84/3.86/4.58
#9.28/0.91/3.80/4.57
#9.29/1.09/3.68/4.52
#9.33/1.19/3.61/4.52
#9.36/1.29/3.56/4.51
#9.43/1.39/3.52/4.52
#9.48/1.48/3.48/4.52
#9.53/1.58/3.43/4.52
#9.63/1.76/3.36/4.51
#9.73/1.98/3.28/4.47
#9.85/2.23/3.21/4.41
#10.00/2.46/3.16/4.39
#10.49/3.08/3.03/4.38
#10.60/3.24/3.02/4.35
#10.72/3.39/3.00/4.33
#11.05/3.84/2.92/4.29
#11.17/3.99/2.90/4.28
#11.27/4.14/2.88/4.25
#11.39/4.29/2.86/4.24
#12.48/5.70/2.67/4.11
#15.01/8.60/2.43/3.98
#20.37/14.62/2.10/3.64
#36.62/32.23/1.64/2.76
#39.18/34.97/1.56/2.65
#41.93/37.90/1.48/2.54
fi

if [ ${stage} -le 97 ] && [ ${stop_stage} -ge 97 ];then
   echo "CDER score"
   #note:collar of CDER score default set is equal to zero.
   #vad_threshold="0.3 0.47 0.5 0.6 0.7 0.8 0.9"
   vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91 0.92"
   vad_type="transformer_vad"
   chunk_size=1.5
   step_size=1.5
   skip_chunk_size=0.93
   #testset="dev"
   testset="test"
   . path_for_nn_vad.sh
   for sub in $testset;do
    echo "compute $sub CDER score"
    test_set_dir="data/magicdata-RAMC/$sub/"
    #test_set_dir_groundtruth=/mntcephfs/lab_data/maduo/datasets/voice-activity-detection/rttm_total/rttm_gt_dev_nog0 # The groundtruth version provided by the challenge
    test_set_dir_groundtruth=data/magicdata-RAMC/$sub/rttm_openslr_gt_${sub}_nog0 # the groundtruth version provided by the openslr.
     for name in $vad_threshold;do
       python clustering_based/cder/score.py  -r $test_set_dir_groundtruth -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}
     done
    done
#grep -r Avg logs/020_spectral_cluster_V100_stage97_debug_1.log
#compute dev CDER score
#Avg CDER : 0.228
#Avg CDER : 0.166
#Avg CDER : 0.191
#Avg CDER : 0.188
#Avg CDER : 0.188
#Avg CDER : 0.187
#Avg CDER : 0.164
#Avg CDER : 0.185
#Avg CDER : 0.162
#Avg CDER : 0.162
#Avg CDER : 0.160
#Avg CDER : 0.156
#Avg CDER : 0.155
#Avg CDER : 0.154
#Avg CDER : 0.131
#Avg CDER : 0.125
#Avg CDER : 0.124
#Avg CDER : 0.123
#Avg CDER : 0.123
#Avg CDER : 0.124
#Avg CDER : 0.123
#Avg CDER : 0.122
#Avg CDER : 0.174
#Avg CDER : 0.122
#Avg CDER : 0.127
#Avg CDER : 0.137
#Avg CDER : 0.143
#Avg CDER : 0.146
#grep -r Avg logs/020_spectral_cluster_V100_stage97_debug_2.log
#compute test CDER score
#Avg CDER : 0.197
#Avg CDER : 0.142
#Avg CDER : 0.139
#Avg CDER : 0.136
#Avg CDER : 0.142
#Avg CDER : 0.144
#Avg CDER : 0.138
#Avg CDER : 0.138
#Avg CDER : 0.134
#Avg CDER : 0.132
#Avg CDER : 0.130
#Avg CDER : 0.130
#Avg CDER : 0.131
#Avg CDER : 0.118
#Avg CDER : 0.118
#Avg CDER : 0.119
#Avg CDER : 0.118
#Avg CDER : 0.117
#Avg CDER : 0.117
#Avg CDER : 0.117
#Avg CDER : 0.115
#Avg CDER : 0.114
#Avg CDER : 0.114
#Avg CDER : 0.118
#Avg CDER : 0.113
#Avg CDER : 0.137
#Avg CDER : 0.139
#Avg CDER : 0.142
fi
if [ ${stage} -le 98 ] && [ ${stop_stage} -ge 98 ];then
   echo "DER score"
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91 0.92"
   vad_type="transformer_vad"
   chunk_size=1.5
   step_size=1.5
   skip_chunk_size=0.93
   testset="test"
   . path_for_nn_vad.sh
   #collar=0.0
   collar=0.25
   for c in $collar;do
      echo "compute collar=$c mode"
   for sub in $testset;do
      #echo "compute $sub DER score"
    test_set_dir="data/magicdata-RAMC/$sub/"
    #test_set_dir_groundtruth=/mntcephfs/lab_data/maduo/datasets/voice-activity-detection/rttm_total/rttm_gt_test_nog0 # The groundtruth version provided by the challenge
    test_set_dir_groundtruth=data/magicdata-RAMC/$sub/rttm_openslr_gt_${sub}_nog0 # the groundtruth version provided by the openslr.

    for name in $vad_threshold;do
     $sctk_dir/md-eval.pl -c $c -r $test_set_dir_groundtruth -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}
   done
   done
 done
#grep -v WARNING:  logs/020_spectral_cluster_V100_stage98_debug_1.log
#DER score
#compute collar=0.0 mode
#23.03/0.89/15.33/6.81
#20.37/1.85/12.22/6.29
#20.11/2.06/11.77/6.27
#19.90/2.27/11.41/6.22
#19.84/2.43/11.19/6.22
#19.52/2.77/10.55/6.19
#19.34/2.97/10.22/6.15
#19.19/3.14/9.91/6.14
#19.12/3.32/9.66/6.14
#19.03/3.50/9.42/6.11
#18.96/3.66/9.22/6.09
#18.85/3.98/8.84/6.03
#18.84/4.37/8.47/6.00
#18.88/4.82/8.14/5.92
#18.85/5.15/7.84/5.87
#19.06/6.09/7.19/5.78
#19.17/6.34/7.07/5.77
#19.24/6.53/6.96/5.76
#19.51/7.23/6.60/5.68
#19.61/7.47/6.48/5.66
#19.71/7.70/6.37/5.63
#19.78/7.92/6.26/5.60
#20.98/10.25/5.30/5.43
#23.80/14.48/4.17/5.15
#30.29/22.66/3.16/4.48
#45.96/40.62/2.13/3.21
#48.23/43.17/2.03/3.03
#50.69/45.90/1.90/2.89
#grep -v WARNING:  logs/020_spectral_cluster_V100_stage98_debug_2.log
#DER score
#compute collar=0.25 mode
#10.80/0.38/6.03/4.39
#8.96/0.90/4.11/3.95
#8.91/1.03/3.93/3.95
#8.89/1.16/3.81/3.92
#8.93/1.25/3.75/3.93
#9.00/1.47/3.61/3.92
#9.04/1.60/3.54/3.89
#9.08/1.71/3.49/3.89
#9.18/1.84/3.45/3.89
#9.23/1.96/3.41/3.87
#9.28/2.06/3.37/3.86
#9.41/2.28/3.30/3.83
#9.59/2.56/3.22/3.81
#9.85/2.91/3.16/3.77
#10.00/3.15/3.11/3.74
#10.59/3.90/2.98/3.71
#10.78/4.11/2.95/3.72
#10.90/4.26/2.93/3.71
#11.38/4.85/2.86/3.67
#11.53/5.03/2.84/3.66
#11.68/5.21/2.83/3.65
#11.81/5.37/2.81/3.63
#13.16/6.96/2.62/3.58
#15.87/10.02/2.37/3.48
#21.99/16.76/2.06/3.17
#38.85/34.74/1.58/2.53
#41.34/37.41/1.51/2.41
#44.07/40.29/1.43/2.34
 fi
 if [ ${stage} -le 99 ] && [ ${stop_stage} -ge 99 ];then
   echo "DER score"
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.1 0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38 0.4 0.45 0.46 0.47 0.5 0.51 0.52 0.53 0.6 0.7 0.8 0.9 0.91 0.92"
   vad_type="transformer_vad"
   chunk_size=1.5
   step_size=1.5
   skip_chunk_size=0.93
   testset="dev"
   . path_for_nn_vad.sh
   #collar=0.0
   collar=0.25
   for c in $collar;do
      echo "compute collar=$c mode"
   for sub in $testset;do
      #echo "compute $sub DER score"
    test_set_dir="data/magicdata-RAMC/$sub/"
    #test_set_dir_groundtruth=/mntcephfs/lab_data/maduo/datasets/voice-activity-detection/rttm_total/rttm_gt_test_nog0 # The groundtruth version provided by the challenge
    test_set_dir_groundtruth=data/magicdata-RAMC/$sub/rttm_openslr_gt_${sub}_nog0 # the groundtruth version provided by the openslr.

    for name in $vad_threshold;do
     $sctk_dir/md-eval.pl -c $c -r $test_set_dir_groundtruth -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_step_size_${step_size}_skip_chunk_size_${skip_chunk_size}
   done
   done
 done
 # grep -v WARNING:  logs/020_spectral_cluster_V100_stage99_debug_1.log
#DER score
#compute collar=0.0 mode
#26.88/0.66/16.58/9.64
#22.13/1.68/12.74/7.71
#23.43/1.84/12.23/9.36
#23.15/2.08/11.75/9.32
#23.02/2.24/11.49/9.30
#22.59/2.57/10.75/9.27
#20.67/2.75/10.44/7.48
#22.25/2.88/10.15/9.22
#20.35/3.05/9.86/7.44
#20.23/3.19/9.61/7.42
#20.09/3.31/9.35/7.43
#19.86/3.57/8.91/7.37
#19.74/3.99/8.41/7.34
#19.72/4.39/8.00/7.33
#17.98/4.76/7.64/5.58
#17.97/5.57/6.93/5.47
#18.05/5.81/6.77/5.47
#18.10/6.01/6.63/5.45
#18.26/6.73/6.20/5.33
#18.33/6.93/6.09/5.31
#18.39/7.13/5.97/5.29
#18.50/7.39/5.83/5.28
#20.88/9.54/4.71/6.63
#21.69/13.68/3.41/4.59
#28.11/21.98/2.27/3.86
#44.67/41.34/1.21/2.11
#47.24/44.22/1.10/1.92
#49.92/47.16/1.02/1.73
#grep -v WARNING:  logs/020_spectral_cluster_V100_stage99_debug_2.log
#DER score
#compute collar=0.25 mode
#14.85/0.26/6.82/7.77
#10.63/0.79/3.83/6.00
#12.07/0.87/3.55/7.66
#11.98/1.01/3.32/7.65
#11.95/1.10/3.21/7.64
#11.93/1.28/3.02/7.64
#10.13/1.38/2.94/5.82
#11.91/1.46/2.85/7.60
#10.13/1.55/2.79/5.79
#10.14/1.63/2.72/5.79
#10.13/1.69/2.64/5.80
#10.15/1.87/2.51/5.77
#10.28/2.15/2.38/5.75
#10.46/2.42/2.29/5.75
#8.91/2.68/2.19/4.03
#9.29/3.27/2.03/3.99
#9.46/3.45/2.00/4.01
#9.56/3.61/1.96/3.98
#9.96/4.20/1.85/3.91
#10.08/4.36/1.83/3.90
#10.21/4.53/1.80/3.88
#10.39/4.73/1.78/3.88
#13.30/6.35/1.59/5.36
#14.31/9.45/1.36/3.50
#20.57/16.44/1.10/3.02
#38.38/36.01/0.75/1.62
#41.20/39.04/0.70/1.46
#44.11/42.14/0.66/1.31

 fi
