## 1. Original Dataset

The original dataset is from *CICMalDroid 2020*. The link is https://www.unb.ca/cic/datasets/maldroid-2020.html.

## 2. Obfuscation Tool

The links to the obfuscation tool we used are:

*  Obfuscapk: https://github.com/ClaudiuGeorgiu/Obfuscapk

## 3. Reports from Virustotal

* **VT_Reports/Benign/01Original/**: Virustotal detection reports for the original samples  from **Benign** category.
* **VT_Reports/Benign/ConstStringEncryption/** : Virustotal detection reports for the **ConstStringEncryption** type of obfuscated samples from **Benign** category.
* **VT_Reports/Malware/01Original** : Virustotal detection reports for the original samples  from **Malware** category.
* **VT_Reports/Malware/VT_Reports/ConstStringEncryption** : Virustotal detection reports for the **ConstStringEncryption** type of obfuscated samples from **Malware** category.

## 4. Noise Dataset

* RDMnoise Dataset
  * train_{*noise ratio*}.txt, e.g., train_0.45.txt. 
    * *noise ratio = {0.0, 0.05, 0.15, 0.25, 0.35, 0.45}*
  
  * val.txt
  
  * test.txt
  
* IDNnoise Dataset
  * train_{*noise ratio*}.txt, e.g., train_0.45.txt. 
    * *noise ratio = {0.0, 0.05, 0.15, 0.25, 0.35, 0.45}*
  
  * val.txt
  
  * test.txt
  
* Mixed noise Dataset
  * train_{*RDMnoise ratio*}{*IDNnoise ratio*}.txt, e.g., train_0.20_0.20.txt
    * *RDMnoise ratio = {0.0, 0.05, 0.10, 0.15, 0.20}*
  
    * *IDNnoise ratio = {0.0, 0.05, 0.10, 0.15, 0.20}*
  
  * val.txt
  
  * test.txt

## 5. CoNoMAD Code

* train_ori_model.py： is used to obtain a model for predicting pseudo labels
* CoNoMAD_main.py： is the CoNoMAD main function to get the robust detector
* test.py： is used to get Precision (P), Recall (R), F1 metrics
