
import json
from tqdm import tqdm
import argparse

dataset = "dev"


def make_argparse():
    # Set up an argument parser.
    parser = argparse.ArgumentParser(description='convert simulate libricss json to rttm.')
    parser.add_argument('--mixspec_json', metavar='<file>', required='True',
                        help='Input mixspec.json of simulated libricss data.')
    parser.add_argument('--output_rttm', metavar='<file>', required='True',
                        help='Output rttm format.')

    return parser

def json2rttm(rttm_path, json_file):
    with open(rttm_path, 'w') as fw, open(json_file,'r')as fr:
        mix_info = json.load(fr)
        for utt_info in tqdm(mix_info):
            utt_id = utt_info["output"].split("/")[-1][:-4]
            for sub_utt_info in utt_info["inputs"]:
                spk_id = sub_utt_info["speaker_id"]
                start = sub_utt_info["offset"]
                duration = sub_utt_info["length_in_seconds"]
                print(
                    f"SPEAKER {utt_id} 1 {start} {duration} <NA> <NA> {spk_id} <NA> <NA>",
                    file=fw,
                )


if __name__== "__main__":
    parser = make_argparse()
    args = parser.parse_args()
    json2rttm(args.output_rttm, args.mixspec_json)
