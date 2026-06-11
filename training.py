import json
import os

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import accuracy


INDUCTIVE_DATASETS = {'reddit', 'reddit2'}


def _dataset_setting(dataset):
    return 'inductive' if dataset in INDUCTIVE_DATASETS else 'transductive'


def _paper_rate(args, idx_train, labels):
    code_rate = float(args.reduction_rate)
    if _dataset_setting(args.dataset) == 'inductive':
        return code_rate
    n_total = int(labels.shape[0]) if hasattr(labels, 'shape') else len(labels)
    n_train = int(len(idx_train))
    if n_total <= 0:
        return None
    return code_rate * n_train / n_total


def build_student_tag(args):
    return (f'{args.dataset}_r{args.reduction_rate}_s{args.seed}'
            f'_K{args.feat_prop_k}_b{args.sap_beta}')


def _forward_inductive(model, inductive_eval, ind_loaders, device):
    outs = {}
    for split in ('train', 'val', 'test'):
        feat_s = inductive_eval[f'feat_{split}']
        outs[split] = model.inference(feat_s, ind_loaders[split], device,
                                      output_device=device)
    return outs['train'], outs['val'], outs['test']


def train_on_syn_graph(model, feat_syn, edge_index_syn, edge_weight_syn,
                       labels_syn, feat, adj, labels,
                       idx_train, idx_val, idx_test, args,
                       save_dir, result_dir, device='cuda',
                       inductive_eval=None):
    optimizer = optim.Adam(model.parameters(), lr=args.lr_model,
                           weight_decay=args.weight_decay)

    ind_loaders = None
    if inductive_eval is not None:
        from torch_geometric.loader import NeighborSampler
        ind_loaders = {}
        for split in ('train', 'val', 'test'):
            adj_s = inductive_eval[f'adj_{split}']
            n_s = inductive_eval[f'feat_{split}'].shape[0]
            ind_loaders[split] = NeighborSampler(
                adj_s, sizes=[-1],
                batch_size=args.batch_size,
                num_workers=0,
                return_e_id=False,
                num_nodes=n_s,
                shuffle=False)

    best_val = 0
    best_test = 0
    labels_val = labels[idx_val]
    labels_test = labels[idx_test]

    ckpt_dir = os.path.join(save_dir, 'student')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f'{build_student_tag(args)}.pt')

    print('[Student] training on the condensed graph...')

    for j in range(args.student_model_loop + 1):
        model.train()
        optimizer.zero_grad()
        logits_syn = model.forward(
            feat_syn, edge_index_syn, edge_weight=edge_weight_syn)
        loss = F.cross_entropy(logits_syn, labels_syn)
        loss.backward()
        optimizer.step()

        if j % args.student_val_stage == 0:
            if inductive_eval is None:
                output = model.predict(feat.to(device), adj.to(device))
                out_val = output[idx_val]
                out_test = output[idx_test]
            else:
                _, out_val, out_test = _forward_inductive(
                    model, inductive_eval, ind_loaders, device)

            acc_val = accuracy(out_val, labels_val)
            acc_test = accuracy(out_test, labels_test)

            if acc_val > best_val:
                best_val = acc_val
                best_test = acc_test
                torch.save(model.state_dict(), ckpt_path)

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path))

    model.eval()
    with torch.no_grad():
        if inductive_eval is None:
            output = model.predict(feat.to(device), adj.to(device))
            out_test = output[idx_test]
        else:
            _, _, out_test = _forward_inductive(
                model, inductive_eval, ind_loaders, device)

    preds_test = out_test.max(1)[1].cpu().numpy()
    labels_test_np = labels[idx_test].cpu().numpy()
    overall_acc = float((preds_test == labels_test_np).sum()) / len(labels_test_np)
    print(f'[Student] best test accuracy: {overall_acc:.4f}')

    result_subdir = os.path.join(result_dir, args.dataset, 'sagc')
    os.makedirs(result_subdir, exist_ok=True)
    json_name = f'r{args.reduction_rate}_s{args.seed}.json'

    result = {
        'schema_version': 3,
        'dataset': args.dataset,
        'setting': _dataset_setting(args.dataset),
        'reduction_rate': args.reduction_rate,
        'paper_reduction_rate': _paper_rate(args, idx_train, labels),
        'seed': args.seed,
        'best_test_acc': overall_acc,
        'best_val_acc': float(best_val),
        'hparams': {
            'feat_prop_k': args.feat_prop_k,
            'sap_beta': args.sap_beta,
            'sap_residual_reg': args.sap_residual_reg,
            'diversity_weight': args.diversity_weight,
            'sap_kcenter_mass': args.sap_kcenter_mass,
            'feat_alpha': args.feat_alpha,
            'condensing_loop': args.condensing_loop,
            'teacher_model_loop': args.teacher_model_loop,
            'student_model_loop': args.student_model_loop,
            'lr_feat': args.lr_feat,
            'lr_model': args.lr_model,
            'nlayers': args.nlayers,
            'hidden': args.hidden,
            'dropout': args.dropout,
            'activation': args.activation,
        },
    }

    with open(os.path.join(result_subdir, json_name), 'w') as f:
        json.dump(result, f, indent=2)

    return best_test
