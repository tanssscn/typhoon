import os
import argparse
import numpy as np
import torch
import cv2
from tqdm import tqdm

import arg_config
from models.factory import create_model
import utils.tools as util
import evaluation
from typh_Generation.utils.datasetTest import TrainDataSetTest
from utils.DataSet import TestDataSet, TestLabelDataSet


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    # 实例化验证数据集
    # val_dataset = TestDataSet(images_path=os.path.join(args.data_path, 'TEST_INPUT'),
    #                           label_path=os.path.join(args.data_path, 'TEST_LABEL.npy'),
    #                           status="test")
    # val_dataset = TestLabelDataSet(images_path=os.path.join(args.data_path, 'test'),
    #                                status="test")
    val_dataset = TrainDataSetTest(images_path=args.data_path)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)
    model_name = args.model
    weights_path = args.weights
    status = "test"
    model = create_model(model_name, device, weights_path, status)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        print(model.load_state_dict(weights_dict, strict=False))

    result_root_path = "runs/testing/" + model_name
    test(model, val_loader, device, result_root_path)


@torch.no_grad()
def test(model, data_loader, device, result_root_path):
    model.eval()
    image_size = arg_config.img_size
    sample_num = 0
    data_loader = tqdm(data_loader)

    save_npy = np.ndarray(shape=(data_loader.__len__() + 1, 2), dtype=np.float64)
    val_images_label = np.ndarray(shape=(data_loader.__len__() + 1, 2), dtype=np.float64)
    save_npy[0][0], save_npy[0][1] = -1, -1

    for step, data in enumerate(data_loader):
        images, labels, image, index = data
        index = index[0]
        sample_num += images.shape[0]
        pred = model(images.to(device))
        # print(pred)
        image = image[0]

        target_point = labels.cpu().data.numpy()
        regression_point = pred.cpu().data.numpy()

        dist = np.sqrt(np.power((regression_point[0][0] * image_size - target_point[0][0] * image_size), 2) + np.power(
            (regression_point[0][1] * image_size - target_point[0][1] * image_size), 2))

        dist_img = cv2.circle(image,
                              (int(regression_point[0][0] * image_size), int(regression_point[0][1] * image_size)),
                              2, (255, 0, 0), -1)
        dist_img = cv2.circle(dist_img, (int(target_point[0][0] * image_size), int(target_point[0][1] * image_size)), 2,
                              (0, 0, 255), -1)
        cv2.putText(dist_img, "DIST:" + str(round(dist, 4)), (5, 25), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2)

        util.mkdir(result_root_path)

        save_npy[index][0], save_npy[index][1] = regression_point[0][0], regression_point[0][1]
        val_images_label[index][0], val_images_label[index][1] = target_point[0][0], target_point[0][1]
        util.save_image(dist_img, os.path.join(result_root_path, str(index) + ".png"))

        np.save(os.path.join(result_root_path, 'pre.npy'), save_npy)
        np.save(os.path.join(result_root_path, 'label.npy'), val_images_label)

    l2_dist = evaluation.evaluate_typhoon(save_npy, val_images_label)
    print(f"Testing npy result have been saved! Evaluation distance: {l2_dist:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--model', default=arg_config.model)
    parser.add_argument('--weights', type=str, default=arg_config.weights,
                        help='initial weights path you should redirect manully')
    parser.add_argument('--data-path', type=str, default=arg_config.data_path)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()

    main(opt)
