import os
import glob
import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader, Subset

from MT_MP import MTMP
from MT_Net import MTNet
from MT_MIX import MTMIX

from MT_MP_wo import MTMP_wo
from MT_Net_wo import MTNet_wo
from MT_MIX_wo import MTMIX_wo

from MT_MP_shcbam import MTMP_sh
from MT_Net_shcbam import MTNet_sh
from MT_MIX_shcbam import MTMIX_sh
import os
import yaml
from dataloader import MultiTaskUltrasoundDataset
device = "cuda" if torch.cuda.is_available() else "cpu"

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

data_root = config["dataset"]["root_dir"]
model_root = config["paths"]["result_root"]
prediction_csv = os.path.join(model_root,"MT_MIX_results","all_predictions.csv")
save_csv = os.path.join(model_root,"segmentation_metrics_results.csv")


folds = [1,2,3,4,5]

full_dataset = MultiTaskUltrasoundDataset(
    root_dir=data_root,
    image_size=(768,1024),
    require_mask=True
)

path_to_idx = {
    s["image_path"]:i
    for i,s in enumerate(full_dataset.samples)
}




def build_model(folder_name):

    if folder_name=="MT_MP_models":
        return MTMP(in_channels=1,bilinear=True,dropout=0.3)

    elif folder_name=="MT_MIX_models":
        return MTMIX(in_channels=1,bilinear=True,dropout=0.3)

    elif folder_name=="MT_Net_models":
        return MTNet(in_channels=1,bilinear=True,dropout=0.3)

    elif folder_name=="MT_MP_wo_models":
        return MTMP_wo(in_channels=1,bilinear=True,dropout=0.3)


    elif folder_name=="MT_MIX_wo_models":
        return MTMIX_wo(in_channels=1,bilinear=True,dropout=0.3)

    elif folder_name=="MT_Net_wo_models":
        return MTNet_wo(in_channels=1,bilinear=True,dropout=0.3)

    elif folder_name=="MT_MP_shcbam_models":
        return MTMP_sh(in_channels=1,bilinear=True,dropout=0.3)

    elif folder_name=="MT_MIX_shcbam_models":
        return MTMIX_sh(in_channels=1,bilinear=True,dropout=0.3)


    elif folder_name=="MT_Net_shcbam_models":
        return MTNet_sh(in_channels=1,bilinear=True,dropout=0.3)

    else:
        raise ValueError(folder_name)


def get_model_path(folder_name, fold):

    folder=os.path.join(
        model_root,
        folder_name
    )


    if not os.path.exists(folder):
        raise FileNotFoundError(folder)


    files=glob.glob(
        os.path.join(folder,"*.pth")
    )

    base_name=folder_name.replace(
        "_models",
        ""
    )

    if "_shcbam" in base_name:

        base_name=base_name.replace(
            "_shcbam",
            ""
        )


    if "_wo" in base_name:

        base_name=base_name.replace(
            "_wo",
            ""
        )


    keyword=base_name.lower()



    candidates=[]


    for f in files:

        name=os.path.basename(f).lower()


        if (
            keyword in name
            and f"fold{fold}" in name
        ):

            candidates.append(f)

    if len(candidates)==0:
        raise FileNotFoundError(
            f"""
    Cannot find weight:

    Folder:
    {folder}

    Required:
    {keyword}_fold{fold}.pth

    Available:
    {files}
    """
        )

    if len(candidates)>1:

        print(
            "Warning multiple candidates:"
        )

        for c in candidates:
            print(c)
    return candidates[0]

def load_model(folder_name, weight_path):
    model=build_model(folder_name)
    model=model.to(device)

    checkpoint=torch.load(
        weight_path,
        map_location=device
    )

    if isinstance(checkpoint,dict):


        if "model_state_dict" in checkpoint:

            state=checkpoint["model_state_dict"]


        elif "state_dict" in checkpoint:
            state=checkpoint["state_dict"]

        else:

            state=checkpoint
    else:

        state=checkpoint

    new_state={}


    for k,v in state.items():

        if k.startswith("module."):

            k=k.replace(
                "module.",
                ""
            )

        new_state[k]=v


    print(
        "Loading:",
        folder_name,
        "\nWeight:",
        weight_path
    )



    model.load_state_dict(
        new_state,
        strict=True
    )
    model.eval()
    return model

# Dice IoU
def dice_iou(pred,gt):
    pred=pred.astype(bool)
    gt=gt.astype(bool)
    inter=np.logical_and(pred,gt).sum()
    dice=(2*inter+1e-7)/(pred.sum()+gt.sum()+1e-7)
    union=np.logical_or(pred,gt).sum()
    iou=(inter+1e-7)/(union+1e-7)
    return dice,iou


def evaluate(model, loader):
    dice_list=[]
    iou_list=[]
    with torch.no_grad():

        for batch in loader:
            image=batch["image"].to(device)

            mask=batch["mask"]

            output=model(image)

            seg_logits=output["seg_logits"]



            pred=torch.sigmoid(
                seg_logits
            )

            pred=(pred>0.5).cpu().numpy()

            mask=mask.numpy()

            for p,g in zip(pred,mask):

                d,i=dice_iou(
                    p[0],
                    g[0]
                )


                dice_list.append(d)
                iou_list.append(i)

    return (
        np.mean(dice_list),
        np.mean(iou_list)
    )

def main():


    pred_df=pd.read_csv(
        prediction_csv
    )

    model_list=[
        "MT_MP_models",
        "MT_MIX_models",
        "MT_Net_models",

        "MT_MP_wo_models",
        "MT_MIX_wo_models",
        "MT_Net_wo_models",

        "MT_MP_shcbam_models",
        "MT_MIX_shcbam_models",
        "MT_Net_shcbam_models"
    ]

    results=[]

    for model_name in model_list:

        fold_dice=[]
        fold_iou=[]

        for fold in folds:

            print(
                "\n======",
                model_name,
                "fold",
                fold,
                "======"
            )

            fold_df=pred_df[
                pred_df.fold==fold
            ]

            indices=[]
            for p in fold_df.image_path:

                if p in path_to_idx:
                    indices.append(
                        path_to_idx[p]
                    )

                else:
                    print(
                        "Missing:",
                        p
                    )



            dataset=Subset(
                full_dataset,
                indices
            )


            loader=DataLoader(dataset,batch_size=8,
                shuffle=False,num_workers=4
            )

            weight=get_model_path(model_name,fold)
            model=load_model(model_name,weight)
            dice,iou=evaluate(model,loader)
            print(
                "Dice:",
                dice,
                "IoU:",
                iou
            )

            fold_dice.append(dice)
            fold_iou.append(iou)

        results.append({
            "model":model_name,
            "Dice_mean":
                np.mean(fold_dice),
            "Dice_std":
                np.std(fold_dice),
            "IoU_mean":
                np.mean(fold_iou),
            "IoU_std":
                np.std(fold_iou)

        })

    result=pd.DataFrame(results)

    print(result)

    result.to_csv(
        save_csv,
        index=False
    )

if __name__=="__main__":

    main()