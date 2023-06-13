import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

import glob
import argparse
from pathlib import Path
from PIL import Image

from typing import List



def read_Hmat(input_path: Path) -> np.ndarray:
    out = np.zeros((3,3))

    with open(input_path, "r") as file:
        lines = file.readlines()

    for i in range(3):
        line = lines[i].strip().split(' ')
        for j in range(3):
            out[i, j] = float(line[j])

    return out

def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size


def create_mma_list(inpt_path: Path) -> List[List[float]]:
    fldrs = [Path(x) for x in glob.glob(f"{inpt_path}/*", recursive=False)]
    head_dict = {"seq_name": [],
                 "image_ref": [],
                 "image_quer": [],
                 "i": [],
                 "j": [],
                 "u": [],
                 "v": [],
                 "gu": [],
                 "gv": [],
                }

    for i, seq in enumerate(fldrs):
        seq_name = seq.stem
        for trans in glob.glob(f"{seq}/*_*", recursive=False):

            requer = Path(trans).stem.split("_")
            ref = requer[0]
            quer = requer[1]

            gtH = read_Hmat(Path(f"{trans}/gt_H"))
            gtH = gtH / gtH[2,2]

            predH = read_Hmat(Path(f"{trans}/eufr_H"))
            predH = predH / predH[2,2]
            im0 = f"{trans}/0.ppm" if Path(f"{trans}/0.ppm").exists() else f"{trans}/0.png"
            im1 = f"{trans}/1.ppm" if Path(f"{trans}/1.ppm").exists() else f"{trans}/1.png"

            im0size = get_image_size(im0)
            im1size = get_image_size(im1)

            upts = np.linspace(0.0, im1size[0], 5) + 0.5
            vpts = np.linspace(0.0, im1size[1], 5) + 0.5


            for i, u in enumerate(upts):
                for j, v in enumerate(vpts):
                    pred_pos = np.dot(predH, np.array([u, v, 1.0]))
                    gt_pos = np.dot(gtH, np.array([u, v, 1.0]))

                    pred_pos = pred_pos /pred_pos[2]
                    gt_pos = gt_pos /gt_pos[2]

                    head_dict["seq_name"].append(seq_name)
                    head_dict["image_ref"].append(ref)
                    head_dict["image_quer"].append(quer)

                    head_dict["i"].append(i)
                    head_dict["j"].append(j)

                    head_dict["u"].append(pred_pos[0])
                    head_dict["v"].append(pred_pos[1])

                    head_dict["gu"].append(gt_pos[0])
                    head_dict["gv"].append(gt_pos[1])


    df = pd.DataFrame(head_dict)

    df["uerr"] = df["u"] - df["gu"]
    df["verr"] = df["v"] - df["gv"]
    df["rmse"] = np.sqrt((df["v"] - df["gv"])**2 + 
                         (df["v"] - df["gv"])**2)

    out = []
    for i, k in enumerate(["v", "i", ""]):
        if(k != ""):
            dftemp = df[df["seq_name"].str[0] == k]
        else:
            dftemp = df

        dftemp.sort_values(by=["rmse"], ascending=True)

        list_of_mma = []
        for threshold in np.linspace(0.01, 40.0, 200):
            list_of_mma.append((dftemp["rmse"] <= threshold).sum() / dftemp.shape[0])

        out.append(list_of_mma)

    return out

if __name__ == "__main__":
    prser = argparse.ArgumentParser("Hom interface")
    prser.add_argument("--inputfolders", "-ifs", type=Path, required=True,
                       nargs='+')
    prser.add_argument("--labels", "-l", type=Path, required=True,
                       nargs='+')

    args = prser.parse_args()

    fig, ax = plt.subplots(1, 3)

    for path, label in zip(args.inputfolders, args.labels):
        for i, mma in enumerate(create_mma_list(path)):
            if i == 0:
                imtypes = "viewpoint"
            elif i == 1:
                imtypes = "illumination"
            else:
                imtypes = "all"

            ax[i].plot(mma, label=label)
            ax[i].set_xlabel("pixel error")
            ax[i].set_ylabel("MMA")
            ax[i].title.set_text(f"{imtypes}")

    plt.legend()

    plt.show()

