import glob
from pathlib import Path
import os



image_data = "/se/team/p3dr/data/machine_vision_training/hpatches/hpatches-sequences-release/"
app = "/se/work/oscsj/affineImageMatching/cpp/bin/viewChange"
outdir = "/se/work/oscsj/hpatches_test_root_sift/"


fls = glob.glob(f"{image_data}/*", recursive=False)
# print(fls)

for f in fls:
    seq = Path(f).stem
    outseq = outdir + seq
    if(not Path(outseq).exists()):
        os.mkdir(outseq)

    imgs = glob.glob(f"{f}/[!1]*.ppm", recursive=False)
    imgs.sort(key=lambda x: int(Path(x).stem))
    im = [f"{f}/1.ppm"] * len(imgs)

    for i, pair in enumerate(zip(im, imgs)):
        ref = Path(pair[0]).stem
        quer = Path(pair[1]).stem
        imseq = outseq + '/' + f"{ref}_{quer}"

        if(not Path(imseq).exists()):
            os.mkdir(imseq)

        print(f"{app} "
              f"-im0 {pair[0]} "
              f"-im1 {pair[1]} "
              "-k root-sift "
              "--flann 1 "
              f"-mo eufr_H "
              f"-mo2 gt_H "
              f"-mi {f}/H_1_{i+2} "
              "--vis 0 "
              f"-of {imseq}/"
             )

        # exit()

    # for
    # print(imgs)
