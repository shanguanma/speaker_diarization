import logging
from pathlib import Path
import textgrid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)
def build_spk2int(textgrid_dir: str):
    """
    统计给定TextGrid目录下的说话人，生成spk2int。
    """
    spk_ids = set()
    #spk_ids =[]
    tg_dir = Path(textgrid_dir)
    for tg_file in tg_dir.glob('*.TextGrid'):
        try:
            tg = textgrid.TextGrid.fromFile(str(tg_file))
            for tier in tg:
                #logging.info(f"str(tg_file):{str(tg_file)} , tier.name: {tier.name}")
                if tier.name.strip():
                    logging.info(f"tier.name: {tier.name[-9:]}")
                    spk_ids.add(tier.name[-9:])
                    #logging.info(f"tier.name: {tier.name}")
                    #spk_ids.append(tier.name[:-9])
        except Exception as e:
            logging.warning(f"Could not process {tg_file}: {e}")
    spk2int = {spk: i for i, spk in enumerate(sorted(list(spk_ids)))}
    logging.info(f"Found {len(spk2int)} unique speakers in the training set.")
    return spk2int

if __name__ == "__main__":
    textgrid_dir="/data/maduo/datasets/alimeeting/Train_Ali_far/textgrid_dir"
    #textgrid_dir="/data/maduo/datasets/alimeeting/Eval_Ali/Eval_Ali_far/textgrid_dir"
    spk2int = build_spk2int(textgrid_dir)
