import argparse
import logging
import os
import random
import shutil
import time
from pathlib import Path

import math
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import test  # import test.py to get mAP after each epoch
from models.yolo import Model
from utils.datasets import create_dataloader
from utils.general import (
    torch_distributed_zero_first, labels_to_class_weights, plot_labels, check_anchors, labels_to_image_weights,
    compute_loss, plot_images, fitness, strip_optimizer, plot_results, get_latest_run, check_dataset, check_file,
    check_git_status, check_img_size, increment_dir, print_mutation, plot_evolution, set_logging, init_seeds)
from utils.google_utils import attempt_download
from utils.torch_utils import ModelEMA, select_device, intersect_dicts

logger = logging.getLogger(__name__)


def train(hyp, opt, device, tb_writer=None):
    logger.info(f'Hyperparameters {hyp}')#太够意思了，训练时候参数,各epoch情况,损失,测试集的结果全部保存
    log_dir = Path(tb_writer.log_dir) if tb_writer else Path(opt.logdir) / 'evolve'  # logging directory
    wdir = log_dir / 'weights'  # weights directory
    os.makedirs(wdir, exist_ok=True)#保存路径
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = str(log_dir / 'results.txt')#训练过程中各种指标
    epochs, batch_size, total_batch_size, weights, rank = \
        opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank

    # Save run settings 保存当前参数
    with open(log_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False) # 这行代码将超参数(hyp)保存到一个名为'hyp.yaml'的文件中
    with open(log_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False) # 这行代码将训练参数(opt)保存到一个名为'opt.yaml'的文件中

    # Configure
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)#随机种子 用于保证训练的可复现性
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict 数据集的定义，通过yaml读取进来
    with torch_distributed_zero_first(rank):#所有进程都一起
        check_dataset(data_dict)  # check
    train_path = data_dict['train']#数据路径与类别名字
    test_path = data_dict['val']
    nc, names = (1, ['item']) if opt.single_cls else (int(data_dict['nc']), data_dict['names'])  # number classes, names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # Model
    pretrained = weights.endswith('.pt') # 预训练文件
    if pretrained:#有预训练模型的话，会自动下载，最好在github下载好 然后放到对应位置
        with torch_distributed_zero_first(rank):
            attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=device)  # load checkpoint加载预训练文件
        if hyp.get('anchors'):
            ckpt['model'].yaml['anchors'] = round(hyp['anchors'])  # force autoanchor
        model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc).to(device)  # create根据配置文件(opt.cfg)或预训练模型的配置信息(ckpt['model'].yaml)创建一个新的模型(Model)，其中ch为输入图像的通道数，nc为类别数
        exclude = ['anchor'] if opt.cfg or hyp.get('anchors') else []  # exclude keys
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)  # load 加载权重
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
    else:
        model = Model(opt.cfg, ch=3, nc=nc).to(device)  # create 就是咱们之前讲的创建模型那块

    # Freeze 要不要冻结一些层，做迁移学习.感觉没必要。。。
    freeze = ['', ]  # parameter names to freeze (full or partial)
    if any(freeze):
        for k, v in model.named_parameters():
            if any(x in k for x in freeze):
                print('freezing %s' % k)
                v.requires_grad = False

    # Optimizer
    nbs = 64  # nominal batch size 累计多少次更新一次模型，咱们的话就是64/16=4次，相当于扩大batch
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups 设置了个优化组：权重，偏置，其他参数
    for k, v in model.named_parameters(): #遍历模型的所有参数，其中k为参数的名称，v为参数的值
        v.requires_grad = True
        if '.bias' in k:
            pg2.append(v)  # biases偏置
        elif '.weight' in k and '.bn' not in k:
            pg1.append(v)  # apply weight decay权重
        else:
            pg0.append(v)  # all else

    if opt.adam: #优化器与学习率衰减
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - hyp['lrf']) + hyp['lrf']  # cosine定义了一个余弦学习率调度函数(lf)，根据当前迭代次数(x)和总迭代次数(epochs)来计算学习率。公式中使用了余弦函数，并根据超参数中的学习率衰减因子(lrf)进行调整。
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # Resume 这个best_fitness是sum([0.0, 0.0, 0.1, 0.9]*[精确度, 召回率, mAP@0.5, mAP@0.5:0.95])
    # 相当于一个综合指标来判断每一次的得分
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer 优化器
        if ckpt['optimizer'] is not None: # 这行代码检查预训练模型中是否存在优化器状态字典(ckpt['optimizer'])，如果存在，则将其加载到当前优化器中，以便恢复训练时的优化器状态。
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # Results结果
        if ckpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(ckpt['training_results'])  # write results.txt

        # Epochs 训练了多少次了已经
        start_epoch = ckpt['epoch'] + 1
        if opt.resume:#又保存了一份？新训练的会覆盖之前旧的？
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
            shutil.copytree(wdir, wdir.parent / f'weights_backup_epoch{start_epoch - 1}')  # save previous weights
        if epochs < start_epoch:#就是你设置的epoch为100 但是现在模型已经训练了150 那就再训练100 表示需要进行微调额外的训练轮数。
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, state_dict

    # Image sizes stride是总的下采样比例 目的是看下数据的大小能不能整除这个比例
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    # DP mode 如果你的机器里面有过个GPU，需要改一些参数。官网教程：https://github.com/ultralytics/yolov5/issues/475
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm 多卡同步做BN
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    # Exponential moving average 滑动平均能让参数更新的更平滑一点不至于波动太大
    # 参考博客：https://www.jianshu.com/p/f99f982ad370
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # DDP mode 多机多卡，有时候DP可能会出现负载不均衡，这个能直接解决该问题。DP用的时候 经常ID为0的GPU干满，其他的没咋用
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank)

    # 数据预处理部分 Trainloader 创建dataloader就是我们一开始讲的部分
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect,
                                            rank=rank, world_size=opt.world_size, workers=opt.workers)
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class 判断类别数是否正常获取数据集中标签的最大类别(mlc)。通过将数据集中的标签连接起来，并取第一列的最大值来获取
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # Process 0
    if rank in [-1, 0]:
        ema.updates = start_epoch * nb // accumulate  # set EMA updates
        testloader = create_dataloader(test_path, imgsz_test, total_batch_size, gs, opt,
                                       hyp=hyp, augment=False, cache=opt.cache_images and not opt.notest, rect=True,
                                       rank=-1, world_size=opt.world_size, workers=opt.workers)[0]  # testloader用于加载测试数据

        if not opt.resume:
            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            plot_labels(labels, save_dir=log_dir)
            if tb_writer:
                # tb_writer.add_hparams(hyp, {})  # causes duplicate https://github.com/ultralytics/yolov5/pull/384
                tb_writer.add_histogram('classes', c, 0)

            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)

    # Model parameters 类别个数，
    hyp['cls'] *= nc / 80.  # scale coco-tuned hyp['cls'] to current dataset
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    #根据标签设置各类别数据初始权重 根据数据集的标签计算各类别的初始权重，并将其作为类别权重(model.class_weights)附加到模型上。
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    #热身持续多少个epoch
    nw = max(round(hyp['warmup_epochs'] * nb), 1e3)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # 根据超参数中的热身训练轮数(warmup_epochs)和批次数(nb)计算热身训练的迭代次数(nw)。热身训练用于在训练初期逐渐增加学习率，以帮助模型更好地收敛。
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    # 日志要保存的结果，先初始化
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)存储训练过程中的结果，包括精确度、召回率、mAP@0.5、mAP@0.5-0.95以及验证集上的损失
    scheduler.last_epoch = start_epoch - 1  # do not move
    #混合精度训练，参考官网说明：https://pytorch.org/docs/stable/amp.html 1.6新功能 fp32与fp16混合 提速比较多
    scaler = amp.GradScaler(enabled=cuda)
    #打印信息
    logger.info('Image sizes %g train, %g test\n'
                'Using %g dataloader workers\nLogging results to %s\n'
                'Starting training for %g epochs...' % (imgsz, imgsz_test, dataloader.num_workers, log_dir, epochs))
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        # Update image weights (optional)
        if opt.image_weights:
            # Generate indices
            if rank in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights调整每个类别的图像权重
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights根据标签的类别和类别权重计算每个图像的权重
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx 根据图像权重，具有较高权重的图像更有可能被选择
            # Broadcast if DDP
            if rank != -1:
                indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(4, device=device)  # mean losses
        if rank != -1: #DDP模式每次取数据的随机种子都不同
            dataloader.sampler.set_epoch(epoch)
        #创建进度条
        pbar = enumerate(dataloader) # 这行代码使用enumerate函数创建一个可迭代的进度条对象(pbar)，用于遍历数据加载器中的批次
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'targets', 'img_size'))
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar 如果当前进程的排名(rank)为-1或0（主进程），则使用tqdm函数创建一个带有进度条的可迭代对象(pbar)，并设置进度条的总长度为批次数(nb)。
        optimizer.zero_grad() # 梯度清零
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)训练开始以来的批次数
            # 归一化
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0将图像(imgs)转移到设备(device)上

            # Warmup 热身
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round()) #
                for j, x in enumerate(optimizer.param_groups): # 根据插值结果，对优化器(optimizer)中的每个参数组进行学习率更新
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0 lf就是余弦衰退函数
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:# 对优化器(optimizer)中具有动量参数的参数组进行动量更新，动量到底时多走一点
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale 各种输入的大小，也是随机的范围[imgsz * 0.5, imgsz * 1.5 + gs] 其中gs=32
            if opt.multi_scale: # 多尺度训练(opt.multi_scale)，则随机选择一个尺度(sz)。尺度范围为原始图像尺寸(imgsz)的0.5倍到1.5倍之间，并且是以步长(gs)的倍数进行取整
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1: #得到新的输入大小
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False) # 线性插值方法将图像(imgs)的尺寸调整为新的图像尺寸(ns)

            # Forward
            with amp.autocast(enabled=cuda):# 用到了1.6新特性 混合精度
                pred = model(imgs)  # forward 将图片送入网络得到预测结果
                #总损失，分类损失，回归损失，置信度损失
                loss, loss_items = compute_loss(pred, targets.to(device), model)  # loss scaled by batch_size
                if rank != -1:
                    loss *= opt.world_size  # gradient averaged between devices in DDP mode

            # Backward
            scaler.scale(loss).backward() # 使用梯度缩放器(scaler)将损失(loss)进行反向传播，并计算相对于模型参数的梯度。梯度缩放器会自动缩放梯度，以防止梯度溢出或下溢。

            # Optimize 相当于Backward多次才更新一次参数
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step 度更新使用梯度缩放器(scaler)的step方法来执行，将缩放后的梯度应用于优化器(optimizer)的参数更新
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # Print 展示信息
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses计算平均损失(mloss)，通过累计计算平均损失的方式更新平均损失值。用于在进度条中展示平均损失
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

                # Plot
                if ni < 3:
                    f = str(log_dir / ('train_batch%g.jpg' % ni))  # filename
                    result = plot_images(images=imgs, targets=targets, paths=paths, fname=f) # 调用plot_images函数绘制训练结果图像，并返回绘制的结果(result)
                    if tb_writer and result is not None:
                        tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                        # tb_writer.add_graph(model, imgs)  # add model to tensorboard

            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler 学习率衰减
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step() #调用学习率调度器(scheduler)的step方法，根据当前训练轮次(epoch)来更新学习率。

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # mAP 更新EMA
            if ema:
                ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride'])
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:  # Calculate mAP 如果未禁用测试(opt.notest为False)或为最后一个训练轮次(final_epoch)，则调用test.test函数进行测试
                results, maps, times = test.test(opt.data,
                                                 batch_size=total_batch_size,
                                                 imgsz=imgsz_test,
                                                 model=ema.ema,
                                                 single_cls=opt.single_cls,
                                                 dataloader=testloader,
                                                 save_dir=log_dir,
                                                 plots=epoch == 0 or final_epoch)  # plot first and last

            # Write
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
            if len(opt.name) and opt.bucket:#这个整不了，涉及上传
                os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))

            # Tensorboard
            if tb_writer:
                tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                        'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                        'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                        'x/lr0', 'x/lr1', 'x/lr2']  # params
                for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                    tb_writer.add_scalar(tag, x, epoch)

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi

            # Save model
            save = (not opt.nosave) or (final_epoch and not opt.evolve)
            if save:
                with open(results_file, 'r') as f:  # create checkpoint打开结果文件(results_file)并读取内容，用于创建检查点(ckpt)。检查点包括当前训练轮次(epoch)、最佳适应度(best_fitness)、训练结果(training_results)、EMA模型(ema.ema)和优化器(optimizer)的状态字典
                    ckpt = {'epoch': epoch,
                            'best_fitness': best_fitness,
                            'training_results': f.read(),
                            'model': ema.ema,
                            'optimizer': None if final_epoch else optimizer.state_dict()}

                # Save last, best and delete
                torch.save(ckpt, last) # 将检查点(ckpt)保存为最新的模型文件(last)。
                if best_fitness == fi:
                    torch.save(ckpt, best) # 如果当前适应度(fi)等于最佳适应度(best_fitness)，则将检查点(ckpt)保存为最佳模型文件(best)
                del ckpt
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training
   # 收尾工作，包括重命名模型文件、剥离优化器、保存训练结果图、打印训练耗时、销毁分布式进程组和清空GPU缓存。
    if rank in [-1, 0]:
        # Strip optimizers
        n = opt.name if opt.name.isnumeric() else ''
        fresults, flast, fbest = log_dir / f'results{n}.txt', wdir / f'last{n}.pt', wdir / f'best{n}.pt'
        for f1, f2 in zip([wdir / 'last.pt', wdir / 'best.pt', results_file], [flast, fbest, fresults]):
            if os.path.exists(f1):
                os.rename(f1, f2)  # rename
                if str(f2).endswith('.pt'):  # is *.pt
                    strip_optimizer(f2)  # strip optimizer
                    os.system('gsutil cp %s gs://%s/weights' % (f2, opt.bucket)) if opt.bucket else None  # upload
        # Finish
        if not opt.evolve:
            plot_results(save_dir=log_dir)  # save as results.png
        logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))

    dist.destroy_process_group() if rank not in [-1, 0] else None
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='预训练文件')
    parser.add_argument('--cfg', type=str, default='/root/zzy/YOLOV5/yolov5-master/models/yolov5s.yaml', help='model.yaml path')#网络配置
    parser.add_argument('--data', type=str, default='/root/zzy/YOLOV5/MaskDataSet/data.yaml', help='data.yaml path')#数据
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path') # 超参数，学习率
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=8, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')#矩形训练
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')#接着之前的训练
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')#不保存
    parser.add_argument('--notest', action='store_true', help='only test final epoch')#不测试
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')#是否调整候选框
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')#超参数更新
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')#缓存图片
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--name', default='', help='renames experiment folder exp{N} to exp{N}_{name} if supplied')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')#是否多尺度训练
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')#是否一个类别
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')#优化器选择
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')#跨GPU的BN
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')#GPU ID
    parser.add_argument('--logdir', type=str, default='runs/', help='logging directory')
    parser.add_argument('--workers', type=int, default=0, help='maximum number of dataloader workers')#windows的同学别改
    opt = parser.parse_args()

    # Set DDP variables WORLD_SIZE：进程数 RANK：进程编号
    opt.total_batch_size = opt.batch_size
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)
    if opt.global_rank in [-1, 0]:
        check_git_status()

    # Resume
    if opt.resume:  # resume an interrupted run 是否继续训练 设置了继续训练(opt.resume)，则根据指定的检查点路径或最新的训练路径来恢复训练。
        # 通过读取检查点文件中的配置参数(opt.yaml)和模型路径(opt.weights)，更新全局变量opt，并设置opt.resume为True。
        #传入模型的路径或者最后一次跑的模型（在runs中有last.pt）
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        log_dir = Path(ckpt).parent.parent  # runs/exp0
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(log_dir / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))  # replace
        opt.cfg, opt.weights, opt.resume = '', ckpt, True
        logger.info('Resuming training from %s' % ckpt)

    else:#加载之前配置好的参数 如果没有设置继续训练，则加载配置参数。通过检查文件路径(opt.data, opt.cfg, opt.hyp)，确保配置文件存在
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
        log_dir = increment_dir(Path(opt.logdir) / 'exp', opt.name)  # runs/exp1

    device = select_device(opt.device, batch_size=opt.batch_size)

    # DDP mode 分布式训练，没有多卡的同学略过
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)#选择GPU
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size

    logger.info(opt)
    with open(opt.hyp,encoding='gb2312') as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps 加载超参数文件

    # Train
    if not opt.evolve:
        tb_writer = None
        if opt.global_rank in [-1, 0]:
            logger.info(f'Start Tensorboard with "tensorboard --logdir {opt.logdir}", view at http://localhost:6006/')
            tb_writer = SummaryWriter(log_dir=log_dir)  # runs/exp0 ，用于将训练过程中的指标和可视化结果写入TensorBoard日志目录

        train(hyp, opt, device, tb_writer) # train函数进行训练
    # 参数搜索与突变
    # Evolve hyperparameters (optional) 参考github issue:https://github.com/ultralytics/yolov3/issues/392
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                'mixup': (1, 0.0, 1.0)}  # image mixup (probability)
        #进化优化(evolve)，通过选择最佳超参数并进行突变(mutate)来改进模型的性能。
        assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
        opt.notest, opt.nosave = True, True  # only test/save final epoch 只在最后一个训练轮次进行测试和保存。
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices 构建保存最佳结果的超参数文件路径(yaml_file)
        yaml_file = Path(opt.logdir) / 'evolve' / 'hyp_evolved.yaml'  # save best result here
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

        for _ in range(300):  # generations to evolve
            if os.path.exists('evolve.txt'):  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min()  # weights
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, device)

            # Write mutation results
            print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

        # Plot results
        plot_evolution(yaml_file)
        print(f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
              f'Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}')
