import glob
from pathlib import Path
import os



image_data = "/home/oscar/sw/vision/AffineMapsTest/imageData/hpatches-sequences-release/"
app = "/home/oscar/sw/vision/AffineMapsTest/cpp/bin/viewChange"
outdir = "/home/oscar/sw/vision/AffineMapsTest/cpp/hpatches_test/"


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
              f"-mo {imseq}/eufr_H "
              f"-mo2 {imseq}/gt_H "
              f"-mi {f}/H_1_{i+2} "
              "--vis 0"
             )

        # exit()

    # for
    # print(imgs)
