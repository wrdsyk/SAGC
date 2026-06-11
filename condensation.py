import math
import os
from collections import Counter

import torch
import torch.nn.functional as F
import torch.optim as optim

from models.mlp import MLP


def generate_labels_syn(labels_train, feat_train, args, result_dir, device='cuda'):
    counter = Counter(labels_train.cpu().numpy())
    sorted_counter = sorted(counter.items(), key=lambda x: x[1])

    num_class_dict = {}
    labels_syn = []
    for c, num in sorted_counter:
        num_class_dict[c] = math.ceil(num * args.reduction_rate)
        labels_syn += [c] * num_class_dict[c]

    labels_syn = torch.LongTensor(labels_syn).to(device)
    return labels_syn, num_class_dict


def build_teacher_path(save_dir, args):
    return os.path.join(
        save_dir, 'teacher',
        f'teacher_{args.dataset}_s{args.seed}_K{args.feat_prop_k}'
        f'_h{args.hidden}_do{args.dropout}_T{args.teacher_model_loop}.pt')


def train_teacher(feat_train, labels_train, feat_test, labels_test,
                  d, nclass, args, save_dir, device='cuda'):
    model = MLP(
        channel_list=[d, args.hidden, args.hidden, args.hidden, nclass],
        dropout=[args.dropout] * 4,
        num_layers=4, norm='BatchNorm', act='relu',
    ).to(device)

    teacher_path = build_teacher_path(save_dir, args)
    os.makedirs(os.path.dirname(teacher_path), exist_ok=True)

    if not os.path.exists(teacher_path):
        model.initialize()
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
        for _ in range(args.teacher_model_loop):
            optimizer.zero_grad()
            output = model.forward(feat_train)
            loss = F.nll_loss(output, labels_train)
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), teacher_path)

    model.load_state_dict(torch.load(teacher_path))
    return model


def build_cache_path(save_dir, args):
    return os.path.join(
        save_dir, 'feat',
        f'feat_{args.dataset}_r{args.reduction_rate}_s{args.seed}'
        f'_K{args.feat_prop_k}_b{args.sap_beta}_rr{args.sap_residual_reg}'
        f'_dw{args.diversity_weight}_km{args.sap_kcenter_mass}'
        f'_fa{args.feat_alpha}_T{args.condensing_loop}_lf{args.lr_feat}.pt')
