import os
import subprocess
import glossharvester

### scan PDFs
pdf_input_dir = "/Users/Stiph002/Projects/igtdetect/igtdetect/data/1_pdf_input"
path_to_txt_scans_dir = "/Users/Stiph002/Projects/igtdetect/igtdetect/data/2_txt_scans"
for pdf_file in os.listdir(pdf_input_dir):
    path_to_pdf = os.path.join(pdf_input_dir, pdf_file)
    filename = os.path.basename(path_to_pdf).split('.')[0] + '-scanned.txt'
    path_to_txt = os.path.join(path_to_txt_scans_dir, filename) 
    subprocess.run(['pdf2txt.py', '-t', 'xml', '-o', path_to_txt, path_to_pdf])

### detect features
path_to_txt_features_dir = "/Users/Stiph002/Projects/igtdetect/igtdetect/data/3_txt_features/"
for txt_scan in os.listdir(path_to_txt_scans_dir):
    path_to_txt = os.path.join(path_to_txt_scans_dir, txt_scan)
    filename = os.path.basename(txt_scan).split('-')[0] + '-features.txt'
    path_to_features = os.path.join(path_to_txt_features_dir, filename)
    subprocess.run(['freki', path_to_txt, path_to_features, '-r', 'pdfminer'])

### detect IGTs with Freki
model_path = "/Users/Stiph002/Projects/igtdetect/sample/new-model.pkl.gz"
config_path = "/Users/Stiph002/Projects/igtdetect/defaults.ini.sample"
path_to_freki_features_dir = "/Users/Stiph002/Projects/igtdetect/igtdetect/data/4_freki_features"
subprocess.run(['python', 'detect-igt', 'test', '--config', config_path, '--classifier-path', model_path, '--test-files', path_to_txt_features_dir, '--classified-dir', path_to_freki_features_dir])


### Harvest IGTs with the harvester
for freki_file in os.listdir(path_to_freki_features_dir):
    path_to_freki_feature_file = os.path.join(path_to_freki_features_dir, freki_file)
    print(path_to_freki_feature_file)
    IGT_list = glossharvester.harvest_IGTs(path_to_freki_feature_file)

    if len(IGT_list) > 0:
        print('yay')

for item in IGT_list:
    print(item)