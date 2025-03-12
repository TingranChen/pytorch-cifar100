#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""

import argparse

from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from conf import settings
from utils import get_network, get_test_dataloader

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='attention56', help='net type')
    parser.add_argument('-weights', type=str, default='./checkpoint/vgg16/Tuesday_24_December_2024_14h_33m_41s/vgg16-47-best.pth', help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=1, help='batch size for dataloader')
    parser.add_argument('-quan', action='store_true', default=False, help='Quantiization Aware')
    parser.add_argument('-mre', action='store_true', default=False, help='MRE Aware')
    parser.add_argument('-throu', action='store_true', default=True, help='Throughput Exam')
    parser.add_argument('-modeltest', action='store_true', default=False, help='breath model test')
    args = parser.parse_args()

    net = get_network(args)

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        #settings.CIFAR100_PATH,
        num_workers=4,
        batch_size=args.b,
    )
    pth_list = {'vgg16': './checkpoint/vgg16/Tuesday_24_December_2024_14h_33m_41s/vgg16-47-best.pth',
                'xception': './checkpoint/xception/Tuesday_24_December_2024_20h_38m_27s/xception-16-best.pth',
                'stochasticdepth34': './checkpoint/stochasticdepth34/Tuesday_24_December_2024_12h_13m_41s/stochasticdepth34-45-best.pth',
                'squeezenet': './checkpoint/squeezenet/Tuesday_24_December_2024_20h_08m_25s/squeezenet-23-best.pth',
                'shufflenetv2': './checkpoint/shufflenetv2/Wednesday_25_December_2024_03h_19m_46s/shufflenetv2-41-best.pth',
                'resnet34': './checkpoint/resnet34/Wednesday_25_December_2024_03h_27m_03s/resnet34-92-best.pth',
                'mobilenetv2': './checkpoint/mobilenetv2/Tuesday_24_December_2024_15h_54m_54s/mobilenetv2-50-best.pth',
                'inceptionv3': './checkpoint/inceptionv3/Tuesday_24_December_2024_17h_26m_39s/inceptionv3-20-regular.pth',
                'densenet121': './checkpoint/densenet121/Wednesday_25_December_2024_10h_52m_48s/densenet121-14-best.pth',
                'attention56': './checkpoint/attention56/Tuesday_24_December_2024_22h_48m_18s/attention56-51-best.pth'}
    args.weights = pth_list[args.net]

    img_list = {'vgg16': 15,
                'xception': 18,
                'stochasticdepth34': 19,
                'squeezenet': 27,
                'shufflenetv2': 30,
                'resnet34': 37,
                'mobilenetv2': 54,
                'inceptionv3': 66,
                'densenet121': 72,
                'attention56': 79}
    args.weights = pth_list[args.net]


    net.load_state_dict(torch.load(args.weights))
    print(net)
    net.eval()
    output_path = f"output/trained/single_b/{args.net}/img_plot.png"

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(cifar100_test_loader):
            if n_iter < img_list[args.net]:
                continue
            elif n_iter == img_list[args.net]:
                print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))

                if (args.b == 1):
                    # 获取第一个图像，调整维度顺序，移动到 CPU，转换为 NumPy 数组，并进行归一化
                    image_to_show = image[0, :, :, :].permute(1, 2, 0).cpu().to(torch.float32)

                    # 或者，如果你的数据已经在 [0, 1] 范围内：
                    # image_to_show = image[0, :, :, :].permute(1, 2, 0).cpu()

                    # 使用 Matplotlib 显示图像
                    plt.imshow(image_to_show)
                    plt.title(f"{n_iter}")
                    plt.axis('off')
                    plt.savefig(output_path)
                    plt.show()
                    plt.close()  # 关闭图像，释放资源

                if args.gpu:
                    image = image.cuda()
                    label = label.cuda()
                    print('GPU INFO.....')
                    print(torch.cuda.memory_summary(), end='')

                output = net(image)
                _, pred = output.topk(5, 1, largest=True, sorted=True)

                label = label.view(label.size(0), -1).expand_as(pred)
                correct = pred.eq(label).float()

                # compute top 5
                correct_5 += correct[:, :5].sum()

                # compute top1
                correct_1 += correct[:, :1].sum()
                break

    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')

    print()
    print("Top 1 err: ", 1 - correct_1 / len(cifar100_test_loader.dataset))
    print("Top 5 err: ", 1 - correct_5 / len(cifar100_test_loader.dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
