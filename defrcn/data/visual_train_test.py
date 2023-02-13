from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
import meta_voc
import random
import matplotlib.pyplot as plot 
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import argparse
import os
from itertools import chain
import cv2
import tqdm
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.data.build import filter_images_with_few_keypoints
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
import numpy as np


def setup(args):
    cfg = get_cfg()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.freeze()
    return cfg

def parse_args(in_args=None):
    parser = argparse.ArgumentParser(description="Visualize ground-truth data for test or train")
    parser.add_argument(
        "--source",
        choices=["train", "test"],
        required=True,
        help="visualize the annotations",
    )
    parser.add_argument("--config-file", metavar="FILE", help="path to config file")
    parser.add_argument("--output-dir", default="./", help="path to output directory")
    parser.add_argument("--show", action="store_true", help="show output in a window")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args(in_args)

if __name__ == "__main__":
    args = parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup(args)

    dirname = args.output_dir
    os.makedirs(dirname, exist_ok=True)
    if args.source == "train":
        #visualize test 
        meta_voc.MetadataCatalog.get("voc_2007_trainval_novel1_1shot_seed0").set(thing_classes=["Japanese Knotweed"])
        meta =  meta_voc.MetadataCatalog.get("voc_2007_trainval_novel1_1shot_seed0")
        print("meta-------->",meta)
        mydict = meta_voc.load_filtered_voc_instances("voc_2007_trainval_novel1_1shot_seed0","datasets/VOC2007","trainval",['Japanese Knotweed'])
        print("mydict---------->",mydict)
        for d in mydict:
            img = cv2.imread(d["file_name"])
            # print(img)
            # r = np.array(img)
            # flattened_image = r.reshape(-1, r.shape[-1])
            # np.savetxt('/home/jovyan/thesis_s2577712/DeFRCN_voc_format/thesis-pascal_voc_format/datasets/VOC2007array_data.txt', flattened_image, fmt='%.2f')
            visualizer = Visualizer(img[:, :, ::-1], metadata=meta, scale=0.5)
            out = visualizer.draw_dataset_dict(d)
            #cv2.imshow("window",out.get_image()[:, :, ::-1])
            #cvShowImage("window",out.get_image()[:, :, ::-1])
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            #cv2.waitKey(1)
            plot.figure(figsize=(15,15))
            plot.imshow(out.get_image()[:, :, ::-1])
            plot.show()
            #plot.savefig("/home/jovyan/thesis_s2577712/DeFRCN_voc_format/thesis-pascal_voc_format/datasets/VOC2007/image.jpg")
           
    else:    
        # vizualise predictions
        data = meta_voc.load_filtered_voc_instances("voc_2007_test_novel1","datasets/VOC2007","test",['japanese Knotweed'])
        meta =  meta_voc.MetadataCatalog.get("voc_2007_test_novel1")
        predictor = DefaultPredictor(cfg)
        for d in random.sample(data, 1):
            img = cv2.imread(d["file_name"])    
            print(img)
        outputs = predictor(img)
        # print(outputs["instances"].pred_classes)
        # print(outputs["instances"].pred_boxes)
        instances = outputs["instances"].to("cpu")
        # Add bounding box and label name for debug
        v = Visualizer(img[:, :, ::-1], metadata=meta, scale=0.5, instance_mode=ColorMode.IMAGE_BW)
        for i, box in enumerate(instances.pred_boxes):
             v.draw_box(box)
        out = v.draw_instance_predictions(instances)
        plot.subplot(1,2,1)
        plot.imshow(out.get_image()[:, :, ::-1])
        plot.subplot(1,2,2)
        plot.imshow(img[:, :, ::-1])
        plot.show()
