import glob
import os
import argparse
import numpy
from pathlib import Path

def generate_file_lists(args):
    dataset_path=args.dataset_path
    '''
    dev set:
    tree -L 3 -d /F00120240032/mlc-slm_task2/MLC-SLM_Workshop-Development_Set/MLC-SLM_Workshop-Development_Set/data/
/F00120240032/mlc-slm_task2/MLC-SLM_Workshop-Development_Set/MLC-SLM_Workshop-Development_Set/data/
├── English
│   ├── American
      --.*wav
      --...
│   │   └── asr
│   ├── Australian
│   │   └── asr
│   ├── British
│   │   └── asr
│   ├── Filipino
│   │   └── asr
│   └── Indian
│       └── asr
├── French
├── German
├── Italian
├── Japanese
├── Korean
├── Portuguese
├── Russian
├── Spanish
├── Thai
└── Vietnamese


    train set:
     tree -L 2 -d /F00120240032/mlc-slm_task2/data/
/F00120240032/mlc-slm_task2/data/
├── English
│   ├── American_English
      ├── 0022
        --.*wav
        --...
│   ├── Australian_English
│   ├── British_English
│   ├── Filipino_English
│   └── Indian_English
├── dev_asr
└── train_asr
    '''
    ref_file_dict={}
    audio_file_dict={}
    ref_file_lists=glob.glob(f'{dataset_path}/**/**/*.txt', recursive=False)
    audio_file_lists=glob.glob(f'{dataset_path}/**/**/*.wav', recursive=False)
    for ref_file in ref_file_lists:
        ref_data=numpy.genfromtxt(ref_file, dtype=str, delimiter='\t', encoding='utf-8')

        language_id=ref_file.split('/')[-2] if args.dataset_part=='dev' else ref_file.split('/')[-3]
        if language_id not in ref_file_dict:
            ref_file_dict[language_id]={}
        session_id=os.path.basename(ref_file).split('.')[0]
        if session_id not in ref_file_dict[language_id]:
            ref_file_dict[language_id][language_id+'_'+session_id]=[]
        
        for line in ref_data:
            if len(line)!=0:
                '''
                origin 1.680273886484085       2.3401385301640696      O1      Hi Will
                target SPEAKER 2speakers_example 0 40.932 7.801 <NA> <NA> 2 <NA> <NA>
                '''
                print(f"line: {line}, ref_file: {ref_file}")
                ref_file_dict[language_id][language_id+'_'+session_id].append(\
                    'SPEAKER {}_{} 1 {} {} <NA> <NA> {} <NA> <NA>'.format(\
                    language_id, session_id, round(float(line[0]), 4), round(float(line[1])-float(line[0]), 4), line[2]))
    for audio_file in audio_file_lists:
        print(f"audio_file: {audio_file}")
        language_id=audio_file.split('/')[-2] if args.dataset_part=='dev' else audio_file.split('/')[-3]
        language_output_path=os.path.join(args.output_path, f'{args.dataset_part}_wav/{language_id}')
        os.makedirs(language_output_path, exist_ok=True)
        audio_output_path=language_output_path+'/'+language_id+'_'+os.path.basename(audio_file)
        if os.path.exists(audio_output_path):
            if Path(audio_output_path).is_symlink():
                continue
        
        else:
            os.symlink(audio_file, audio_output_path)
        
        audio_file_dict[os.path.basename(audio_output_path).split('.')[0]]=audio_output_path

    # write audio lists
    with open(os.path.join(args.output_path, f'{args.dataset_part}_wav.list'), 'w') as f:
        for line in audio_file_dict.values():
            f.write(f'{line}\n')
    # write ref files
    for language, language_dict in ref_file_dict.items():
        os.makedirs(os.path.join(args.output_path, f'{args.dataset_part}_rttm/{language}'), exist_ok=True)
        for session, session_list in language_dict.items():
            with open(os.path.join(args.output_path, f'{args.dataset_part}_rttm/{language}/{session}.rttm'), 'w') as f:
                for line in session_list:
                    f.write(f'{line}\n')
    # write ref lists
    with open(os.path.join(args.output_path, f'{args.dataset_part}_rttm.list'), 'w') as f:
        for language, language_dict in ref_file_dict.items():
            for session, session_list in language_dict.items():
                f.write(os.path.join(args.output_path, f'{args.dataset_part}_rttm/{language}/{session}.rttm')+'\n')


def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./MLC-SLM_Workshop-Development_Set/data')
    parser.add_argument('--output_path', type=str, default='./examples')
    parser.add_argument('--dataset_part', type=str, default='dev')
    args=parser.parse_args()
    return args

if __name__=='__main__':
    args=get_args()
    generate_file_lists(args)
