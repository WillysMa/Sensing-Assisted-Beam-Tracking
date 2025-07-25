# Sensing-Assisted Beam Tracking
Simulator for paper ''Knowledge Distillation for Sensing-Assisted Beam Tracking in mmWave Communications''


## Explaination
This project contains 10 trained models:

-- Model 1: bblox_best：The baseline scheme of the paper [1]

[1] S. Jiang and A. Alkhateeb, “Computer vision aided beam tracking in a real-world millimeter wave deployment,” in Proc. IEEE Global Commun. Conf. Workshop, 2022.

-- Model 2: Teacher_no_MHA: Teacher model without multi-head attention (MHA) mechanism

-- Model 3: Teacher_with_MHA: Teacher model with multi-head attention (MHA) mechanism

-- Model 4: Teacher_self-KD: Teacher model (with MHA module) is trained with self-KD. The teacher model is Model 3.

-- Model 5: Student_L8_no_KD: Student model with L=8 trained from dataset 

-- Model 6: Student_L5_no_KD: Student model with L=5 trained from dataset

-- Model 7: Student_L3_no_KD: Student model with L=3 trained from dataset

-- Model 8: Student_L8_with_KD: Student model with L=8 trained with KD. The teacher model is Model 4. 

-- Model 9: Student_L5_no_KD: Student model with L=5 trained with KD.  The teacher model is Model 4. 

-- Model 10: Student_L3_no_KD: Student model with L=3 trained with KD. The teacher model is Model 4. 

## Dataset
We use Scenario 9 of the Deepsense 6G dataset https://www.deepsense6g.net/scenario-9/

## Train model
1. Download the dataset scenrio 9 to local workstation.
2. Run train_weighted.py

