import glob
from pathlib import Path



image_data = "/home/oscar/sw/vision/AffineMapsTest/imageData/hpatches-sequences-release/"
app = "/home/oscar/sw/vision/AffineMapsTest/cpp/bin/viewChange"


fls = glob.glob(f"{image_data}/*", recursive=False)
# print(fls)

for f in fls:
    seq = Path(f).stem

    imgs = glob.glob(f"{f}/[!1]*.ppm", recursive=False)
    imgs.sort(key=lambda x: int(Path(x).stem))
    im = [f"{f}/1.ppm"] * len(imgs)

    for pair in zip(im, imgs):
        print(f"{app} "

             )

    # for
    # print(imgs)
