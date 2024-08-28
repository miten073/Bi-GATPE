import matplotlib.pyplot as plt
import os
import re

# 解析第一种日志数据
logs1 = """
2024-04-24 14:23:37 [INFO]: Epoch 001 - training loss: 1.1450, validating loss: 0.3559
2024-04-24 14:23:41 [INFO]: Epoch 002 - training loss: 0.8716, validating loss: 0.2468
2024-04-24 14:23:44 [INFO]: Epoch 003 - training loss: 0.7839, validating loss: 0.1911
2024-04-24 14:23:48 [INFO]: Epoch 004 - training loss: 0.7353, validating loss: 0.1846
2024-04-24 14:23:52 [INFO]: Epoch 005 - training loss: 0.7060, validating loss: 0.1768
2024-04-24 14:23:57 [INFO]: Epoch 006 - training loss: 0.6816, validating loss: 0.1726
2024-04-24 14:24:05 [INFO]: Epoch 007 - training loss: 0.6669, validating loss: 0.1467
2024-04-24 14:24:13 [INFO]: Epoch 008 - training loss: 0.6525, validating loss: 0.1446
2024-04-24 14:24:21 [INFO]: Epoch 009 - training loss: 0.6426, validating loss: 0.1390
2024-04-24 14:24:29 [INFO]: Epoch 010 - training loss: 0.6328, validating loss: 0.1328
2024-04-24 14:24:37 [INFO]: Epoch 011 - training loss: 0.6251, validating loss: 0.1309
2024-04-24 14:24:45 [INFO]: Epoch 012 - training loss: 0.6175, validating loss: 0.1294
2024-04-24 14:24:52 [INFO]: Epoch 013 - training loss: 0.6123, validating loss: 0.1235
2024-04-24 14:25:00 [INFO]: Epoch 014 - training loss: 0.6064, validating loss: 0.1203
2024-04-24 14:25:08 [INFO]: Epoch 015 - training loss: 0.6006, validating loss: 0.1302
2024-04-24 14:25:16 [INFO]: Epoch 016 - training loss: 0.5964, validating loss: 0.1150
2024-04-24 14:25:24 [INFO]: Epoch 017 - training loss: 0.5920, validating loss: 0.1281
2024-04-24 14:25:32 [INFO]: Epoch 018 - training loss: 0.5904, validating loss: 0.1026
2024-04-24 14:25:39 [INFO]: Epoch 019 - training loss: 0.5863, validating loss: 0.1067
2024-04-24 14:25:47 [INFO]: Epoch 020 - training loss: 0.5816, validating loss: 0.1208
2024-04-24 14:25:55 [INFO]: Epoch 021 - training loss: 0.5797, validating loss: 0.1058
2024-04-24 14:26:03 [INFO]: Epoch 022 - training loss: 0.5773, validating loss: 0.1053
2024-04-24 14:26:11 [INFO]: Epoch 023 - training loss: 0.5745, validating loss: 0.1151
2024-04-24 14:26:19 [INFO]: Epoch 024 - training loss: 0.5721, validating loss: 0.0994
2024-04-24 14:26:27 [INFO]: Epoch 025 - training loss: 0.5688, validating loss: 0.0944
2024-04-24 14:26:35 [INFO]: Epoch 026 - training loss: 0.5678, validating loss: 0.0952
2024-04-24 14:26:42 [INFO]: Epoch 027 - training loss: 0.5646, validating loss: 0.0948
2024-04-24 14:26:50 [INFO]: Epoch 028 - training loss: 0.5631, validating loss: 0.1040
2024-04-24 14:26:58 [INFO]: Epoch 029 - training loss: 0.5622, validating loss: 0.0935
2024-04-24 14:27:06 [INFO]: Epoch 030 - training loss: 0.5591, validating loss: 0.1007
2024-04-24 14:27:14 [INFO]: Epoch 031 - training loss: 0.5584, validating loss: 0.0950
2024-04-24 14:27:22 [INFO]: Epoch 032 - training loss: 0.5559, validating loss: 0.0880
2024-04-24 14:27:30 [INFO]: Epoch 033 - training loss: 0.5556, validating loss: 0.0892
2024-04-24 14:27:38 [INFO]: Epoch 034 - training loss: 0.5520, validating loss: 0.0947
2024-04-24 14:27:46 [INFO]: Epoch 035 - training loss: 0.5517, validating loss: 0.0876
2024-04-24 14:27:54 [INFO]: Epoch 036 - training loss: 0.5506, validating loss: 0.0845
2024-04-24 14:28:02 [INFO]: Epoch 037 - training loss: 0.5497, validating loss: 0.0854
2024-04-24 14:28:09 [INFO]: Epoch 038 - training loss: 0.5477, validating loss: 0.0858
2024-04-24 14:28:17 [INFO]: Epoch 039 - training loss: 0.5477, validating loss: 0.0814
2024-04-24 14:28:25 [INFO]: Epoch 040 - training loss: 0.5458, validating loss: 0.0792
2024-04-24 14:28:33 [INFO]: Epoch 041 - training loss: 0.5439, validating loss: 0.0805
2024-04-24 14:28:41 [INFO]: Epoch 042 - training loss: 0.5429, validating loss: 0.0846
2024-04-24 14:28:49 [INFO]: Epoch 043 - training loss: 0.5423, validating loss: 0.0792
2024-04-24 14:28:57 [INFO]: Epoch 044 - training loss: 0.5412, validating loss: 0.0784
2024-04-24 14:29:04 [INFO]: Epoch 045 - training loss: 0.5398, validating loss: 0.0808
2024-04-24 14:29:12 [INFO]: Epoch 046 - training loss: 0.5397, validating loss: 0.0776
2024-04-24 14:29:20 [INFO]: Epoch 047 - training loss: 0.5388, validating loss: 0.0814
2024-04-24 14:29:28 [INFO]: Epoch 048 - training loss: 0.5373, validating loss: 0.0786
2024-04-24 14:29:59 [INFO]: Epoch 049 - training loss: 0.5355, validating loss: 0.0777
2024-04-24 14:30:36 [INFO]: Epoch 050 - training loss: 0.5361, validating loss: 0.0801
2024-04-24 14:31:13 [INFO]: Epoch 051 - training loss: 0.5346, validating loss: 0.0754
2024-04-24 14:31:49 [INFO]: Epoch 052 - training loss: 0.5334, validating loss: 0.0765
2024-04-24 14:32:26 [INFO]: Epoch 053 - training loss: 0.5331, validating loss: 0.0732
2024-04-24 14:33:02 [INFO]: Epoch 054 - training loss: 0.5321, validating loss: 0.0805
2024-04-24 14:33:39 [INFO]: Epoch 055 - training loss: 0.5309, validating loss: 0.0790
2024-04-24 14:34:10 [INFO]: Epoch 056 - training loss: 0.5305, validating loss: 0.0720
2024-04-24 14:34:30 [INFO]: Epoch 057 - training loss: 0.5293, validating loss: 0.0751
2024-04-24 14:34:51 [INFO]: Epoch 058 - training loss: 0.5290, validating loss: 0.0773
2024-04-24 14:35:12 [INFO]: Epoch 059 - training loss: 0.5284, validating loss: 0.0737
2024-04-24 14:35:32 [INFO]: Epoch 060 - training loss: 0.5275, validating loss: 0.0769
2024-04-24 14:35:52 [INFO]: Epoch 061 - training loss: 0.5261, validating loss: 0.0739
2024-04-24 14:36:13 [INFO]: Epoch 062 - training loss: 0.5266, validating loss: 0.0713
2024-04-24 14:36:33 [INFO]: Epoch 063 - training loss: 0.5252, validating loss: 0.0721
2024-04-24 14:36:53 [INFO]: Epoch 064 - training loss: 0.5254, validating loss: 0.0732
2024-04-24 14:37:14 [INFO]: Epoch 065 - training loss: 0.5249, validating loss: 0.0683
2024-04-24 14:37:35 [INFO]: Epoch 066 - training loss: 0.5246, validating loss: 0.0666
2024-04-24 14:37:56 [INFO]: Epoch 067 - training loss: 0.5241, validating loss: 0.0735
2024-04-24 14:38:16 [INFO]: Epoch 068 - training loss: 0.5223, validating loss: 0.0720
2024-04-24 14:38:36 [INFO]: Epoch 069 - training loss: 0.5220, validating loss: 0.0695
2024-04-24 14:38:56 [INFO]: Epoch 070 - training loss: 0.5224, validating loss: 0.0671
2024-04-24 14:39:16 [INFO]: Epoch 071 - training loss: 0.5224, validating loss: 0.0666
2024-04-24 14:39:37 [INFO]: Epoch 072 - training loss: 0.5195, validating loss: 0.0708
2024-04-24 14:39:57 [INFO]: Epoch 073 - training loss: 0.5198, validating loss: 0.0708
2024-04-24 14:40:17 [INFO]: Epoch 074 - training loss: 0.5199, validating loss: 0.0706
2024-04-24 14:40:38 [INFO]: Epoch 075 - training loss: 0.5184, validating loss: 0.0752
2024-04-24 14:40:58 [INFO]: Epoch 076 - training loss: 0.5195, validating loss: 0.0677
2024-04-24 14:41:18 [INFO]: Epoch 077 - training loss: 0.5176, validating loss: 0.0682
2024-04-24 14:41:31 [INFO]: Epoch 078 - training loss: 0.5176, validating loss: 0.0664
2024-04-24 14:41:49 [INFO]: Epoch 079 - training loss: 0.5166, validating loss: 0.0643
2024-04-24 14:42:00 [INFO]: Epoch 080 - training loss: 0.5168, validating loss: 0.0658
2024-04-24 14:42:13 [INFO]: Epoch 081 - training loss: 0.5162, validating loss: 0.0670
2024-04-24 14:42:25 [INFO]: Epoch 082 - training loss: 0.5160, validating loss: 0.0690
2024-04-24 14:42:38 [INFO]: Epoch 083 - training loss: 0.5155, validating loss: 0.0628
2024-04-24 14:42:45 [INFO]: Epoch 084 - training loss: 0.5154, validating loss: 0.0675
2024-04-24 14:42:57 [INFO]: Epoch 085 - training loss: 0.5144, validating loss: 0.0688
2024-04-24 14:43:11 [INFO]: Epoch 086 - training loss: 0.5148, validating loss: 0.0655
2024-04-24 14:43:25 [INFO]: Epoch 087 - training loss: 0.5131, validating loss: 0.0634
2024-04-24 14:43:39 [INFO]: Epoch 088 - training loss: 0.5136, validating loss: 0.0652
2024-04-24 14:43:53 [INFO]: Epoch 089 - training loss: 0.5132, validating loss: 0.0647
2024-04-24 14:44:07 [INFO]: Epoch 090 - training loss: 0.5134, validating loss: 0.0641
2024-04-24 14:44:21 [INFO]: Epoch 091 - training loss: 0.5125, validating loss: 0.0634
2024-04-24 14:44:34 [INFO]: Epoch 092 - training loss: 0.5125, validating loss: 0.0664
2024-04-24 14:44:48 [INFO]: Epoch 093 - training loss: 0.5120, validating loss: 0.0689
2024-04-24 14:45:02 [INFO]: Epoch 094 - training loss: 0.5103, validating loss: 0.0584
2024-04-24 14:45:16 [INFO]: Epoch 095 - training loss: 0.5101, validating loss: 0.0664
2024-04-24 14:45:29 [INFO]: Epoch 096 - training loss: 0.5106, validating loss: 0.0634
2024-04-24 14:45:43 [INFO]: Epoch 097 - training loss: 0.5103, validating loss: 0.0622
2024-04-24 14:45:57 [INFO]: Epoch 098 - training loss: 0.5092, validating loss: 0.0611
2024-04-24 14:46:10 [INFO]: Epoch 099 - training loss: 0.5093, validating loss: 0.0689
2024-04-24 14:46:24 [INFO]: Epoch 100 - training loss: 0.5094, validating loss: 0.0667
"""

epochs1 = []
train_losses1 = []
val_losses1 = []
for line in logs1.strip().split('\n'):
    match = re.match(r'.*Epoch (\d+) - training loss: ([\d\.]+), validating loss: ([\d\.]+)', line)
    if match:
        epoch = int(match.group(1))
        train_loss = float(match.group(2))
        val_loss = float(match.group(3))
        epochs1.append(epoch)
        train_losses1.append(train_loss)
        val_losses1.append(val_loss)

# 解析第二种日志数据
logs2 = """
Epoch 0 completed in 5.17 seconds
epoch (0 / 100) (Train Loss:0.68085769)
Validation Loss: 0.367747
Epoch 1 completed in 4.60 seconds
epoch (1 / 100) (Train Loss:0.42014325)
Validation Loss: 0.318100
Epoch 2 completed in 4.34 seconds
epoch (2 / 100) (Train Loss:0.37160412)
Validation Loss: 0.249818
Epoch 3 completed in 4.27 seconds
epoch (3 / 100) (Train Loss:0.33857139)
Validation Loss: 0.223939
Epoch 4 completed in 4.27 seconds
epoch (4 / 100) (Train Loss:0.31398502)
Validation Loss: 0.207406
Epoch 5 completed in 4.55 seconds
epoch (5 / 100) (Train Loss:0.29220408)
Validation Loss: 0.188195
Epoch 6 completed in 4.61 seconds
epoch (6 / 100) (Train Loss:0.27412802)
Validation Loss: 0.160174
Epoch 7 completed in 4.12 seconds
epoch (7 / 100) (Train Loss:0.26161894)
Validation Loss: 0.146582
Epoch 8 completed in 4.09 seconds
epoch (8 / 100) (Train Loss:0.24874551)
Validation Loss: 0.150248
Epoch 9 completed in 4.09 seconds
epoch (9 / 100) (Train Loss:0.24202323)
Validation Loss: 0.118762
Epoch 10 completed in 4.19 seconds
epoch (10 / 100) (Train Loss:0.23151002)
Validation Loss: 0.123550
Epoch 11 completed in 4.15 seconds
epoch (11 / 100) (Train Loss:0.22309433)
Validation Loss: 0.116228
Epoch 12 completed in 4.13 seconds
epoch (12 / 100) (Train Loss:0.21782705)
Validation Loss: 0.100374
Epoch 13 completed in 4.10 seconds
epoch (13 / 100) (Train Loss:0.21193372)
Validation Loss: 0.103463
Epoch 14 completed in 4.11 seconds
epoch (14 / 100) (Train Loss:0.20629530)
Validation Loss: 0.086531
Epoch 15 completed in 4.12 seconds
epoch (15 / 100) (Train Loss:0.20250473)
Validation Loss: 0.093192
Epoch 16 completed in 4.11 seconds
epoch (16 / 100) (Train Loss:0.19651138)
Validation Loss: 0.089865
Epoch 17 completed in 4.11 seconds
epoch (17 / 100) (Train Loss:0.19413686)
Validation Loss: 0.081954
Epoch 18 completed in 4.15 seconds
epoch (18 / 100) (Train Loss:0.19179585)
Validation Loss: 0.077497
Epoch 19 completed in 4.15 seconds
epoch (19 / 100) (Train Loss:0.18693624)
Validation Loss: 0.073610
Epoch 20 completed in 4.12 seconds
epoch (20 / 100) (Train Loss:0.18342512)
Validation Loss: 0.073367
Epoch 21 completed in 4.14 seconds
epoch (21 / 100) (Train Loss:0.18225907)
Validation Loss: 0.076761
Epoch 22 completed in 4.08 seconds
epoch (22 / 100) (Train Loss:0.18083535)
Validation Loss: 0.069039
Epoch 23 completed in 4.08 seconds
epoch (23 / 100) (Train Loss:0.17589213)
Validation Loss: 0.066034
Epoch 24 completed in 4.13 seconds
epoch (24 / 100) (Train Loss:0.17546886)
Validation Loss: 0.061372
Epoch 25 completed in 4.18 seconds
epoch (25 / 100) (Train Loss:0.17252002)
Validation Loss: 0.073875
Epoch 26 completed in 4.16 seconds
epoch (26 / 100) (Train Loss:0.17173897)
Validation Loss: 0.058614
Epoch 27 completed in 4.11 seconds
epoch (27 / 100) (Train Loss:0.16962922)
Validation Loss: 0.059129
Epoch 28 completed in 4.08 seconds
epoch (28 / 100) (Train Loss:0.16718782)
Validation Loss: 0.061594
Epoch 29 completed in 4.11 seconds
epoch (29 / 100) (Train Loss:0.16685649)
Validation Loss: 0.055769
Epoch 30 completed in 4.08 seconds
epoch (30 / 100) (Train Loss:0.16499284)
Validation Loss: 0.056943
Epoch 31 completed in 4.16 seconds
epoch (31 / 100) (Train Loss:0.16313301)
Validation Loss: 0.057209
Epoch 32 completed in 4.17 seconds
epoch (32 / 100) (Train Loss:0.16157027)
Validation Loss: 0.053883
Epoch 33 completed in 4.31 seconds
epoch (33 / 100) (Train Loss:0.16117652)
Validation Loss: 0.057822
Epoch 34 completed in 4.16 seconds
epoch (34 / 100) (Train Loss:0.15845572)
Validation Loss: 0.051769
Epoch 35 completed in 4.22 seconds
epoch (35 / 100) (Train Loss:0.15799211)
Validation Loss: 0.049906
Epoch 36 completed in 4.31 seconds
epoch (36 / 100) (Train Loss:0.15630714)
Validation Loss: 0.052089
Epoch 37 completed in 4.30 seconds
epoch (37 / 100) (Train Loss:0.15591111)
Validation Loss: 0.054922
Epoch 38 completed in 4.37 seconds
epoch (38 / 100) (Train Loss:0.15357942)
Validation Loss: 0.048985
Epoch 39 completed in 4.40 seconds
epoch (39 / 100) (Train Loss:0.15396955)
Validation Loss: 0.049099
Epoch 40 completed in 4.29 seconds
epoch (40 / 100) (Train Loss:0.15212681)
Validation Loss: 0.048296
Epoch 41 completed in 4.36 seconds
epoch (41 / 100) (Train Loss:0.15185323)
Validation Loss: 0.045893
Epoch 42 completed in 4.30 seconds
epoch (42 / 100) (Train Loss:0.15028064)
Validation Loss: 0.051378
Epoch 43 completed in 4.60 seconds
epoch (43 / 100) (Train Loss:0.14941180)
Validation Loss: 0.045905
Epoch 44 completed in 4.18 seconds
epoch (44 / 100) (Train Loss:0.14857745)
Validation Loss: 0.044556
Epoch 45 completed in 4.13 seconds
epoch (45 / 100) (Train Loss:0.14639927)
Validation Loss: 0.047494
Epoch 46 completed in 4.05 seconds
epoch (46 / 100) (Train Loss:0.14692364)
Validation Loss: 0.041884
Epoch 47 completed in 4.08 seconds
epoch (47 / 100) (Train Loss:0.14579711)
Validation Loss: 0.043487
Epoch 48 completed in 4.10 seconds
epoch (48 / 100) (Train Loss:0.14584086)
Validation Loss: 0.047046
Epoch 49 completed in 4.15 seconds
epoch (49 / 100) (Train Loss:0.14443335)
Validation Loss: 0.047898
Epoch 50 completed in 4.11 seconds
epoch (50 / 100) (Train Loss:0.14328700)
Validation Loss: 0.043026
Epoch 51 completed in 4.09 seconds
epoch (51 / 100) (Train Loss:0.14276436)
Validation Loss: 0.039716
Epoch 52 completed in 4.14 seconds
epoch (52 / 100) (Train Loss:0.14267244)
Validation Loss: 0.039075
Epoch 53 completed in 4.13 seconds
epoch (53 / 100) (Train Loss:0.14186316)
Validation Loss: 0.041406
Epoch 54 completed in 4.20 seconds
epoch (54 / 100) (Train Loss:0.14009014)
Validation Loss: 0.038598
Epoch 55 completed in 4.16 seconds
epoch (55 / 100) (Train Loss:0.13922147)
Validation Loss: 0.043875
Epoch 56 completed in 4.13 seconds
epoch (56 / 100) (Train Loss:0.13882057)
Validation Loss: 0.040353
Epoch 57 completed in 4.10 seconds
epoch (57 / 100) (Train Loss:0.13883723)
Validation Loss: 0.041027
Epoch 58 completed in 4.14 seconds
epoch (58 / 100) (Train Loss:0.13847530)
Validation Loss: 0.037204
Epoch 59 completed in 4.33 seconds
epoch (59 / 100) (Train Loss:0.13783393)
Validation Loss: 0.037814
Epoch 60 completed in 4.15 seconds
epoch (60 / 100) (Train Loss:0.13670375)
Validation Loss: 0.039218
Epoch 61 completed in 4.25 seconds
epoch (61 / 100) (Train Loss:0.13646464)
Validation Loss: 0.042308
Epoch 62 completed in 4.29 seconds
epoch (62 / 100) (Train Loss:0.13532351)
Validation Loss: 0.036900
Epoch 63 completed in 4.32 seconds
epoch (63 / 100) (Train Loss:0.13562097)
Validation Loss: 0.039777
Epoch 64 completed in 4.35 seconds
epoch (64 / 100) (Train Loss:0.13498967)
Validation Loss: 0.035878
Epoch 65 completed in 4.41 seconds
epoch (65 / 100) (Train Loss:0.13411163)
Validation Loss: 0.036032
Epoch 66 completed in 4.40 seconds
epoch (66 / 100) (Train Loss:0.13353580)
Validation Loss: 0.040623
Epoch 67 completed in 4.33 seconds
epoch (67 / 100) (Train Loss:0.13310048)
Validation Loss: 0.038542
Epoch 68 completed in 4.30 seconds
epoch (68 / 100) (Train Loss:0.13235621)
Validation Loss: 0.035994
Epoch 69 completed in 4.57 seconds
epoch (69 / 100) (Train Loss:0.13291314)
Validation Loss: 0.042614
Epoch 70 completed in 7.46 seconds
epoch (70 / 100) (Train Loss:0.13211821)
Validation Loss: 0.035998
Epoch 71 completed in 7.56 seconds
epoch (71 / 100) (Train Loss:0.13153659)
Validation Loss: 0.035192
Epoch 72 completed in 7.61 seconds
epoch (72 / 100) (Train Loss:0.13126401)
Validation Loss: 0.036585
Epoch 73 completed in 7.55 seconds
epoch (73 / 100) (Train Loss:0.12959431)
Validation Loss: 0.036043
Epoch 74 completed in 7.52 seconds
epoch (74 / 100) (Train Loss:0.12969936)
Validation Loss: 0.034288
Epoch 75 completed in 7.45 seconds
epoch (75 / 100) (Train Loss:0.12892415)
Validation Loss: 0.034301
Epoch 76 completed in 7.59 seconds
epoch (76 / 100) (Train Loss:0.12946943)
Validation Loss: 0.035987
Epoch 77 completed in 7.59 seconds
epoch (77 / 100) (Train Loss:0.12856072)
Validation Loss: 0.035489
Epoch 78 completed in 7.49 seconds
epoch (78 / 100) (Train Loss:0.12848394)
Validation Loss: 0.032784
Epoch 79 completed in 7.61 seconds
epoch (79 / 100) (Train Loss:0.12799806)
Validation Loss: 0.034990
Epoch 80 completed in 7.54 seconds
epoch (80 / 100) (Train Loss:0.12671302)
Validation Loss: 0.036705
Epoch 81 completed in 7.46 seconds
epoch (81 / 100) (Train Loss:0.12697826)
Validation Loss: 0.032515
Epoch 82 completed in 7.55 seconds
epoch (82 / 100) (Train Loss:0.12690022)
Validation Loss: 0.033962
Epoch 83 completed in 7.51 seconds
epoch (83 / 100) (Train Loss:0.12559393)
Validation Loss: 0.033257
Epoch 84 completed in 7.46 seconds
epoch (84 / 100) (Train Loss:0.12610793)
Validation Loss: 0.031856
Epoch 85 completed in 7.53 seconds
epoch (85 / 100) (Train Loss:0.12493795)
Validation Loss: 0.032594
Epoch 86 completed in 7.55 seconds
epoch (86 / 100) (Train Loss:0.12563274)
Validation Loss: 0.031214
Epoch 87 completed in 7.55 seconds
epoch (87 / 100) (Train Loss:0.12412823)
Validation Loss: 0.035968
Epoch 88 completed in 7.68 seconds
epoch (88 / 100) (Train Loss:0.12415902)
Validation Loss: 0.030367
Epoch 89 completed in 7.59 seconds
epoch (89 / 100) (Train Loss:0.12418767)
Validation Loss: 0.032184
Epoch 90 completed in 7.54 seconds
epoch (90 / 100) (Train Loss:0.12342068)
Validation Loss: 0.030576
Epoch 91 completed in 7.64 seconds
epoch (91 / 100) (Train Loss:0.12346606)
Validation Loss: 0.030775
Epoch 92 completed in 7.64 seconds
epoch (92 / 100) (Train Loss:0.12416131)
Validation Loss: 0.032907
Epoch 93 completed in 7.49 seconds
epoch (93 / 100) (Train Loss:0.12270211)
Validation Loss: 0.032537
Epoch 94 completed in 7.51 seconds
epoch (94 / 100) (Train Loss:0.12258772)
Validation Loss: 0.030177
Epoch 95 completed in 7.54 seconds
epoch (95 / 100) (Train Loss:0.12095291)
Validation Loss: 0.033749
Epoch 96 completed in 7.60 seconds
epoch (96 / 100) (Train Loss:0.12168286)
Validation Loss: 0.030677
Epoch 97 completed in 7.56 seconds
epoch (97 / 100) (Train Loss:0.12090054)
Validation Loss: 0.031083
Epoch 98 completed in 7.67 seconds
epoch (98 / 100) (Train Loss:0.12107252)
Validation Loss: 0.028585
Epoch 99 completed in 7.51 seconds
epoch (99 / 100) (Train Loss:0.12170219)
Validation Loss: 0.032802
"""


epochs2 = []
train_losses2 = []
val_losses2 = []
for line in logs2.strip().split('\n'):
    match = re.match(r'epoch \((\d+) \/ 100\) \(Train Loss:([\d\.]+)\)', line)
    if match:
        epoch = int(match.group(1))
        train_loss = float(match.group(2))
        epochs2.append(epoch)
        train_losses2.append(train_loss)
    match = re.match(r'Validation Loss: ([\d\.]+)', line)
    if match:
        val_loss = float(match.group(1))
        val_losses2.append(val_loss)

# 绘制 log1 的训练损失曲线
plt.figure(figsize=(8, 6))
plt.plot(epochs1, train_losses1)
plt.title('SAITS Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.savefig('pic/saits_train_loss.png', dpi=300, bbox_inches='tight')
plt.close()

# 绘制 log2 的训练损失曲线
plt.figure(figsize=(8, 6))
plt.plot(epochs2, train_losses2)
plt.title('GDN Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.savefig('pic/gdn_train_loss.png', dpi=300, bbox_inches='tight')
plt.close()

# 绘制 log1 和 log2 的验证损失曲线
plt.figure(figsize=(8, 6))
plt.plot(epochs1, val_losses1, label='saits Validation Loss')
plt.plot(epochs2, val_losses2, label='gdn Validation Loss')
plt.title('Validation Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.legend()
plt.savefig('pic/validation_loss.png', dpi=300, bbox_inches='tight')
plt.close()