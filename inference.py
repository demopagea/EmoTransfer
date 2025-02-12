#!/home/ubuntu/OpenVoice/.openvoiceenv/bin/python
# import matplotlib.pyplot as plt
# import IPython.display as ipd
import torch
import utils
from api import STS
import os
import pandas as pd
from glob import glob

hps = utils.get_hparams_from_file("/users/PAS2062/delijingyic/project/OpenVoice/emotion_STS/melo_sts9/logs5/emotion_STS/config.json")
print('hps',hps)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

config_path='/users/PAS2062/delijingyic/project/OpenVoice/emotion_STS/melo_sts9/logs5/emotion_STS/config.json'
ckpt_path='/users/PAS2062/delijingyic/project/OpenVoice/emotion_STS/melo_sts9/logs5/emotion_STS/G_63000.pth'
language='EN'

model = STS(language,device=device,config_path=config_path,ckpt_path=ckpt_path)
speaker_ids = model.hps.data.spk2id
cwd=os.getcwd()
output_directory=ckpt_path.split('/')[-3] + '/' + ckpt_path.split('/')[-2]
output_dir=f'{cwd}/{output_directory}/evaluation/inference_syn_difspk_difftxt'
ckpt_num=ckpt_path.split('/')[-1].split('.')[0]
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


real_audio_directory='/users/PAS2062/delijingyic/project/OpenVoice/data_syn/inference_syn_difspk_difftxt'
emo_directatry='/users/PAS2062/delijingyic/project/OpenVoice/data_syn/inference_syn_difspk_difftxt'
wav_pattern='**/*.wav'
emo_pattern='**/*.wav.emo.npy'
df=pd.DataFrame()
inf={}

for file in glob(os.path.join(real_audio_directory, wav_pattern), recursive=True):
    audiopath=file
    file_emo=file.split('/')[-1].split('_')[-1].split('.')[0].lower()
    speaker=file.split('/')[-1].split('_')[1] + '_' + file.split('/')[-1].split('_')[2].split('.')[0]
    for file2 in glob(os.path.join(emo_directatry, wav_pattern), recursive=True):
        file2_emo=file2.split('/')[-1].split('_')[-1].split('.')[0].lower()

        if file_emo !=file2_emo:
            inf['reference_audio']=file2
            inf['input_audio']=file
            output_path=f'{output_dir}/{speaker}_to_{file2_emo}_{ckpt_num}.wav'
            output_path_syn=f'{output_dir}/{speaker}_to_{file2_emo}_{ckpt_num}_syn.wav'
            inf['output_audio']=output_path
            emo_input=f'{file}.emo.npy'
            emo=f'{file2}.emo.npy'
            model.sts_to_file(audiopath,emo_input,emo,output_path,output_path_syn,file2)
            df = pd.concat([df, pd.DataFrame([inf])], ignore_index=True)
df.to_csv(f'{output_dir}/eval.csv')

