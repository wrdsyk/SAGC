import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.checkpoint import checkpoint


def sparsemax(z, dim=-1):
    if dim != -1 and dim != z.dim() - 1:
        z = z.transpose(dim, -1)
    z_sorted, _ = torch.sort(z, dim=-1, descending=True)
    K = z.size(-1)
    arange = torch.arange(1, K + 1, device=z.device, dtype=z.dtype)
    cumsum = torch.cumsum(z_sorted, dim=-1)
    support = 1 + arange * z_sorted > cumsum
    k_star = support.sum(dim=-1, keepdim=True).clamp(min=1)
    tau_numer = torch.gather(cumsum, -1, k_star - 1) - 1
    tau = tau_numer / k_star.to(z.dtype)
    p = torch.clamp(z - tau, min=0)
    if dim != -1 and dim != z.dim() - 1:
        p = p.transpose(dim, -1)
    return p


def _row_orthogonality_loss_lowmem(rows):
    n = rows.shape[0]
    norms = rows.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    rows_norm = rows / norms
    trace_row_gram = (rows_norm * rows_norm).sum()
    if rows_norm.shape[0] <= rows_norm.shape[1]:
        gram = rows_norm @ rows_norm.t()
    else:
        gram = rows_norm.t() @ rows_norm
    return (gram ** 2).sum() - 2.0 * trace_row_gram + n


def _batchnorm_modules(model):
    return [
        module for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]


def _set_train_with_bn_eval(model):
    model.train()
    for module in _batchnorm_modules(model):
        module.eval()


def _calibrate_bn_from_synthetic(model, feat_syn_full, momentum=1.0):
    bn_layers = _batchnorm_modules(model)
    if not bn_layers:
        return
    old_momenta = [bn.momentum for bn in bn_layers]
    model.train()
    for bn in bn_layers:
        bn.momentum = momentum
    with torch.no_grad():
        model.forward(feat_syn_full)
    for bn, old_momentum in zip(bn_layers, old_momenta):
        bn.momentum = old_momentum


class AttributiveFeatSyn(nn.Module):

    def __init__(self, labels_syn, X_orig, labels_train, device, beta=0.0):
        super().__init__()
        self.register_buffer('X_orig', X_orig.detach())
        self.device = device
        self.d = X_orig.shape[1]
        self.n_syn = len(labels_syn)
        self.beta = float(beta)
        self.chunk_size = 128

        self.class_real_idx = {}
        self.class_syn_idx = {}
        for c in labels_syn.unique().tolist():
            real_idx = torch.where(labels_train == c)[0]
            syn_idx = torch.where(labels_syn == c)[0]
            if len(real_idx) == 0 or len(syn_idx) == 0:
                continue
            self.class_real_idx[c] = real_idx
            self.class_syn_idx[c] = syn_idx

        self.alphas = nn.ParameterDict()
        for c, real_idx in self.class_real_idx.items():
            n_syn_c = len(self.class_syn_idx[c])
            self.alphas[str(c)] = nn.Parameter(
                torch.zeros(n_syn_c, len(real_idx), device=device))

        if self.beta > 0:
            self.residual = nn.Parameter(
                torch.zeros(self.n_syn, self.d, device=device))
        else:
            self.register_buffer(
                'residual',
                torch.zeros(self.n_syn, self.d, device=device))

    def init_kcenter(self, kcenter_selected_per_class, kcenter_mass=0.9):
        for c, sel_idx in kcenter_selected_per_class.items():
            if c not in self.alphas:
                continue
            real_idx = self.class_real_idx[c].tolist()
            real_idx_map = {j: pos for pos, j in enumerate(real_idx)}
            n_syn_c, N_c = self.alphas[str(c)].shape
            with torch.no_grad():
                eps = 1e-3
                low = (1.0 - kcenter_mass) / max(N_c - 1, 1) + eps
                target = torch.full((n_syn_c, N_c), low, device=self.device)
                for i, j_global in enumerate(sel_idx):
                    if i >= n_syn_c or j_global not in real_idx_map:
                        continue
                    target[i, real_idx_map[j_global]] = kcenter_mass
                target = target / target.sum(dim=-1, keepdim=True).clamp(min=eps)
                self.alphas[str(c)].data.copy_(target.log())

    def _activate(self, logits):
        return sparsemax(logits, dim=-1)

    def _compute_sap_only(self):
        chunks = []
        for c, real_idx in self.class_real_idx.items():
            alpha_logits = self.alphas[str(c)]
            X_real_c = self.X_orig[real_idx]
            syn_idx = self.class_syn_idx[c]
            for start in range(0, alpha_logits.shape[0], self.chunk_size):
                end = min(start + self.chunk_size, alpha_logits.shape[0])
                logits_chunk = alpha_logits[start:end]

                def _dense_chunk(logits, X_real=X_real_c):
                    return self._activate(logits) @ X_real

                if torch.is_grad_enabled() and logits_chunk.requires_grad:
                    X_syn_c = checkpoint(
                        _dense_chunk, logits_chunk, use_reentrant=True)
                else:
                    X_syn_c = _dense_chunk(logits_chunk)
                chunks.append((int(syn_idx[start]), X_syn_c))
        chunks.sort(key=lambda item: item[0])
        return torch.cat([chunk for _, chunk in chunks], dim=0)

    def compute_feat_syn(self):
        out = self._compute_sap_only()
        if self.beta > 0:
            out = out + self.beta * self.residual
        return out

    def forward(self):
        return self.compute_feat_syn()

    def diversity_reg(self):
        total = 0.0
        count = 0
        for c, _ in self.class_real_idx.items():
            logits = self.alphas[str(c)]
            A = self._activate(logits)
            n_syn_c = A.shape[0]
            if n_syn_c < 2:
                continue
            total = total + _row_orthogonality_loss_lowmem(A)
            count += n_syn_c * n_syn_c
        return total / max(count, 1)

    def joint_diversity_reg(self, X_syn):
        total = 0.0
        count = 0
        for c, _ in self.class_real_idx.items():
            syn_idx = self.class_syn_idx[c]
            if len(syn_idx) < 2:
                continue
            X_c = X_syn[syn_idx]
            total = total + _row_orthogonality_loss_lowmem(X_c)
            count += X_c.shape[0] ** 2
        return total / max(count, 1)

    def residual_reg(self):
        if self.beta <= 0:
            return torch.tensor(0.0, device=self.device)
        return (self.residual ** 2).mean()


def kcenter_selected_indices_per_class(feat_train, labels_train,
                                       num_class_dict):
    per_class = {}
    for class_id, cnt in num_class_dict.items():
        idx = torch.where(labels_train == class_id)[0]
        feature = feat_train[idx]
        mean = torch.mean(feature, dim=0, keepdim=True)
        dis = torch.cdist(feature, mean)[:, 0]
        rank = torch.argsort(dis)
        idx_centers = rank[:1].tolist()
        for _ in range(cnt - 1):
            feature_centers = feature[idx_centers]
            dis_center = torch.cdist(feature, feature_centers)
            dis_min, _ = torch.min(dis_center, dim=-1)
            id_max = torch.argmax(dis_min).item()
            idx_centers.append(id_max)
        per_class[class_id] = idx[idx_centers].cpu().tolist()
    return per_class


def _fair_coeff_from_labels_syn(labels_syn, nclass):
    counts = [int((labels_syn == c).sum()) for c in range(nclass)]
    max_count = max(max(counts), 1)
    return [cnt / max_count for cnt in counts]


def node_condensation_sap(feat_syn_module, labels_syn, feat_train,
                          labels_train, index, index_syn, validation_model,
                          nclass, d, args, save_path, device='cuda'):
    optimizer = optim.Adam(feat_syn_module.parameters(), lr=args.lr_feat)
    loss_fn = nn.MSELoss()

    diversity_w = args.diversity_weight
    residual_reg_w = args.sap_residual_reg
    is_hybrid = feat_syn_module.beta > 0

    fair_coeff = _fair_coeff_from_labels_syn(labels_syn, nclass)
    fair_coeff_sum = torch.tensor(sum(fair_coeff)).to(device)

    def _compute_loss():
        feat_syn_inner = feat_syn_module()
        output_syn_batch = validation_model.forward(feat_syn_inner)
        loss_i = F.nll_loss(output_syn_batch, labels_syn)
        feat_loss = torch.tensor(0.0, device=device)
        for c in range(nclass):
            if fair_coeff[c] > 0:
                feat_train_c = feat_train[index[c]]
                feat_syn_c = feat_syn_inner[index_syn[c]]
                feat_loss += (fair_coeff[c] * loss_fn(
                    feat_train_c.mean(dim=0),
                    feat_syn_c.mean(dim=0)))
        feat_loss = feat_loss / fair_coeff_sum
        loss_i = loss_i + args.feat_alpha * feat_loss
        if diversity_w > 0:
            if is_hybrid:
                div = feat_syn_module.joint_diversity_reg(feat_syn_inner)
            else:
                div = feat_syn_module.diversity_reg()
            loss_i = loss_i + diversity_w * div
        if is_hybrid and residual_reg_w > 0:
            loss_i = loss_i + residual_reg_w * feat_syn_module.residual_reg()
        return loss_i

    for i in range(args.condensing_loop + 1):
        _set_train_with_bn_eval(validation_model)
        optimizer.zero_grad(set_to_none=True)
        loss = _compute_loss()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with torch.no_grad():
        feat_syn_final = feat_syn_module().detach()
    torch.save(feat_syn_final, save_path)
    return feat_syn_final


def node_condensation_sap_streaming(feat_syn_module, labels_syn, feat_train,
                                    labels_train, index, index_syn,
                                    validation_model, nclass, d, args,
                                    save_path, device='cuda'):
    classes = list(feat_syn_module.class_real_idx.keys())
    is_hybrid = feat_syn_module.beta > 0

    feat_syn_module.to(device)
    alphas_data = {c: feat_syn_module.alphas[str(c)].data for c in classes}
    residual_data = feat_syn_module.residual.data if is_hybrid else None

    adam_alpha = {c: {'m': torch.zeros_like(alphas_data[c]),
                      'v': torch.zeros_like(alphas_data[c]),
                      'step': 0} for c in classes}
    adam_residual = None
    if is_hybrid:
        adam_residual = {'m': torch.zeros_like(residual_data),
                         'v': torch.zeros_like(residual_data),
                         'step': 0}

    lr = args.lr_feat
    b1, b2, eps = 0.9, 0.999, 1e-8

    diversity_w = args.diversity_weight
    residual_reg_w = args.sap_residual_reg
    target_gb = args.sap_streaming_target_gb
    max_chunk_size = args.sap_streaming_max_chunk_size
    bn_calib_interval = args.sap_streaming_bn_calib_interval
    stat_interval = args.sap_streaming_stat_interval

    fair_coeff = _fair_coeff_from_labels_syn(labels_syn, nclass)

    div_count = sum(alphas_data[c].shape[0] ** 2
                    for c in classes if alphas_data[c].shape[0] >= 2)
    div_count = max(div_count, 1)
    n_syn_total = sum(alphas_data[c].shape[0] for c in classes)
    residual_numel = n_syn_total * d

    validation_model.to(device)
    _set_train_with_bn_eval(validation_model)
    fair_coeff_dev = torch.tensor(fair_coeff, device=device)
    fair_coeff_sum_dev = torch.tensor(sum(fair_coeff)).to(device)
    labels_syn_dev = labels_syn.to(device) if not labels_syn.is_cuda else labels_syn
    teacher_requires_grad = [p.requires_grad
                             for p in validation_model.parameters()]
    validation_model.requires_grad_(False)

    x_real_cache = {}
    try:
        x_src = feat_syn_module.X_orig
        for cls in classes:
            real_idx_cls = feat_syn_module.class_real_idx[cls].to(device)
            x_real_cache[cls] = x_src[real_idx_cls]
    except RuntimeError as err:
        if 'out of memory' not in str(err).lower():
            raise
        x_real_cache = {}
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()

    def _is_cuda_oom(err):
        return isinstance(err, RuntimeError) and (
            'out of memory' in str(err).lower())

    def _initial_class_chunk_size(cls):
        n_rows, n_cols = alphas_data[cls].shape
        bytes_per_alpha_elem = 32.0
        target_bytes = target_gb * (1024 ** 3)
        est = int(target_bytes // max(n_cols * bytes_per_alpha_elem, 1.0))
        return max(1, min(n_rows, max_chunk_size, max(est, 1)))

    def _halve_chunk_after_oom(chunk):
        if chunk <= 1:
            raise RuntimeError('CUDA OOM with chunk_size=1')
        new_chunk = max(1, chunk // 2)
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()
        return new_chunk

    def _get_x_real(cls):
        if cls in x_real_cache:
            return x_real_cache[cls], False
        real_idx_cls = feat_syn_module.class_real_idx[cls].to(device)
        return feat_syn_module.X_orig[real_idx_cls], True

    def _adam_step_rows_inplace(param_tensor, grad, state, row_index, t):
        p_slice = param_tensor[row_index]
        m_slice = state['m'][row_index]
        v_slice = state['v'][row_index]
        m_slice.mul_(b1).add_(grad, alpha=1.0 - b1)
        v_slice.mul_(b2).addcmul_(grad, grad, value=1.0 - b2)
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        denom = (v_slice.sqrt() / (bc2 ** 0.5)).add_(eps)
        p_slice.addcdiv_(m_slice, denom, value=-lr / bc1)
        state['m'][row_index] = m_slice
        state['v'][row_index] = v_slice
        param_tensor[row_index] = p_slice

    stat_cache = {}

    def _assemble_full_feat_syn():
        n_syn_full = feat_syn_module.n_syn
        feat_syn_full = torch.empty(n_syn_full, d, device=device)
        with torch.no_grad():
            for cls in classes:
                syn_idx_cls = feat_syn_module.class_syn_idx[cls]
                X_real_cls, free_x_real = _get_x_real(cls)
                chunk = _initial_class_chunk_size(cls)
                start_row = 0
                while start_row < alphas_data[cls].shape[0]:
                    end = min(start_row + chunk, alphas_data[cls].shape[0])
                    alpha_chunk = None
                    A_chunk = None
                    X_syn_chunk = None
                    try:
                        alpha_chunk = alphas_data[cls][start_row:end]
                        A_chunk = sparsemax(alpha_chunk, dim=-1)
                        X_syn_chunk = A_chunk @ X_real_cls
                        syn_chunk = syn_idx_cls[start_row:end].to(device)
                        if is_hybrid:
                            X_syn_chunk = (
                                X_syn_chunk
                                + feat_syn_module.beta
                                * residual_data[syn_chunk])
                        feat_syn_full[syn_chunk] = X_syn_chunk
                        del alpha_chunk, A_chunk, X_syn_chunk
                        start_row = end
                    except RuntimeError as err:
                        if not _is_cuda_oom(err):
                            raise
                        alpha_chunk = A_chunk = X_syn_chunk = None
                        chunk = _halve_chunk_after_oom(chunk)
                if free_x_real:
                    del X_real_cls
        return feat_syn_full

    for epoch in range(args.condensing_loop + 1):
        if epoch % bn_calib_interval == 0:
            feat_syn_full = _assemble_full_feat_syn()
            _calibrate_bn_from_synthetic(
                validation_model, feat_syn_full, momentum=1.0)
            _set_train_with_bn_eval(validation_model)
            del feat_syn_full

        residual_t = None
        if is_hybrid:
            adam_residual['step'] += 1
            residual_t = adam_residual['step']

        for c in classes:
            syn_idx_c = feat_syn_module.class_syn_idx[c]
            n_syn_c = len(syn_idx_c)

            X_real_c, free_x_real_c = _get_x_real(c)
            chunk_c = _initial_class_chunk_size(c)
            use_alg = float(fair_coeff_dev[c].detach().cpu()) > 0
            use_div = diversity_w > 0 and n_syn_c >= 2

            alg_grad_mu = None
            alg_scale = None
            div_gram = None
            need_stats = use_alg or (use_div and is_hybrid)
            refresh_stats = (
                need_stats
                and (epoch % stat_interval == 0 or c not in stat_cache))
            cached_stats = stat_cache.get(c)
            if need_stats and refresh_stats:
                sum_feat = torch.zeros(d, device=device)
                if use_div and is_hybrid:
                    div_gram = torch.zeros(d, d, device=device)
                with torch.no_grad():
                    start_row = 0
                    while start_row < n_syn_c:
                        end_row = min(start_row + chunk_c, n_syn_c)
                        alpha_chunk = None
                        A_chunk = None
                        feat_chunk = None
                        rows_norm = None
                        sum_update = None
                        gram_update = None
                        try:
                            syn_chunk = syn_idx_c[start_row:end_row].to(device)
                            alpha_chunk = alphas_data[c][start_row:end_row]
                            A_chunk = sparsemax(alpha_chunk, dim=-1)
                            feat_chunk = A_chunk @ X_real_c
                            if is_hybrid:
                                feat_chunk = (
                                    feat_chunk
                                    + feat_syn_module.beta
                                    * residual_data[syn_chunk])
                            if use_alg:
                                sum_update = feat_chunk.sum(dim=0)
                            if use_div and is_hybrid:
                                rows_norm = feat_chunk / feat_chunk.norm(
                                    dim=-1, keepdim=True).clamp(min=1e-8)
                                gram_update = rows_norm.t() @ rows_norm
                                del rows_norm
                            if sum_update is not None:
                                sum_feat.add_(sum_update)
                            if gram_update is not None:
                                div_gram.add_(gram_update)
                            del alpha_chunk, A_chunk, feat_chunk
                            del sum_update, gram_update
                            start_row = end_row
                        except RuntimeError as err:
                            if not _is_cuda_oom(err):
                                raise
                            alpha_chunk = A_chunk = feat_chunk = None
                            rows_norm = None
                            sum_update = gram_update = None
                            chunk_c = _halve_chunk_after_oom(chunk_c)
                if use_alg:
                    mu_syn = sum_feat / n_syn_c
                    mu_train = X_real_c.mean(dim=0)
                    alg_grad_mu = (2.0 / d) * (mu_syn - mu_train).detach()
                    alg_scale = (
                        args.feat_alpha
                        * fair_coeff_dev[c] / fair_coeff_sum_dev)
                    del mu_syn, mu_train
                stat_cache[c] = {
                    'alg_grad_mu': (alg_grad_mu.detach()
                                    if alg_grad_mu is not None else None),
                    'alg_scale': (float(alg_scale.detach().cpu())
                                  if torch.is_tensor(alg_scale)
                                  else alg_scale),
                    'div_gram': (div_gram.detach()
                                 if div_gram is not None else None),
                }
                del sum_feat
            elif need_stats and cached_stats is not None:
                if cached_stats['alg_grad_mu'] is not None:
                    alg_grad_mu = cached_stats['alg_grad_mu']
                    alg_scale = cached_stats['alg_scale']
                if cached_stats['div_gram'] is not None:
                    div_gram = cached_stats['div_gram']

            state_a = adam_alpha[c]
            state_a['step'] += 1
            alpha_t = state_a['step']

            start_row = 0
            while start_row < n_syn_c:
                end_row = min(start_row + chunk_c, n_syn_c)
                per_sample = None
                rows_norm_grad = None
                div_sur = None
                div_c = None

                alpha_gpu = None
                A_chunk = None
                feat_syn_chunk = None
                labels_syn_chunk = None
                output_syn_chunk = None
                loss = None
                residual_gpu = None
                try:
                    syn_chunk = syn_idx_c[start_row:end_row].to(device)
                    alpha_gpu = (alphas_data[c][start_row:end_row]
                                 .detach()
                                 .clone()
                                 .requires_grad_(True))
                    A_chunk = sparsemax(alpha_gpu, dim=-1)
                    feat_syn_chunk = A_chunk @ X_real_c

                    if is_hybrid:
                        residual_gpu = (residual_data[syn_chunk]
                                        .detach()
                                        .clone()
                                        .requires_grad_(True))
                        feat_syn_chunk = (
                            feat_syn_chunk
                            + feat_syn_module.beta * residual_gpu)

                    labels_syn_chunk = labels_syn_dev[syn_chunk]
                    output_syn_chunk = validation_model.forward(feat_syn_chunk)
                    per_sample = F.nll_loss(
                        output_syn_chunk, labels_syn_chunk,
                        reduction='none')
                    loss = per_sample.sum() / n_syn_total

                    if alg_grad_mu is not None:
                        loss = loss + alg_scale * (
                            feat_syn_chunk.sum(dim=0) * alg_grad_mu
                        ).sum() / n_syn_c

                    if use_div:
                        if is_hybrid and div_gram is not None:
                            rows_norm_grad = (
                                feat_syn_chunk / feat_syn_chunk.norm(
                                    dim=-1, keepdim=True).clamp(min=1e-8))
                            div_sur = 2.0 * (
                                (rows_norm_grad @ div_gram.detach())
                                * rows_norm_grad
                            ).sum()
                            loss = loss + diversity_w * (div_sur / div_count)
                        elif not is_hybrid:
                            div_c = _row_orthogonality_loss_lowmem(A_chunk)
                            loss = loss + diversity_w * (div_c / div_count)

                    if is_hybrid and residual_reg_w > 0:
                        loss = loss + residual_reg_w * (
                            (residual_gpu ** 2).sum() / residual_numel)

                    loss.backward()
                    _adam_step_rows_inplace(
                        alphas_data[c], alpha_gpu.grad, state_a,
                        slice(start_row, end_row), alpha_t)

                    if is_hybrid:
                        _adam_step_rows_inplace(
                            residual_data, residual_gpu.grad, adam_residual,
                            syn_chunk, residual_t)

                    del (alpha_gpu, A_chunk, feat_syn_chunk,
                         labels_syn_chunk, output_syn_chunk, loss)
                    if is_hybrid and residual_gpu is not None:
                        del residual_gpu
                    del per_sample, rows_norm_grad, div_sur, div_c
                    start_row = end_row
                except RuntimeError as err:
                    if not _is_cuda_oom(err):
                        raise
                    alpha_gpu = A_chunk = feat_syn_chunk = None
                    labels_syn_chunk = output_syn_chunk = loss = None
                    residual_gpu = None
                    per_sample = rows_norm_grad = None
                    div_sur = div_c = None
                    chunk_c = _halve_chunk_after_oom(chunk_c)

            if free_x_real_c:
                del X_real_c
            if alg_grad_mu is not None:
                del alg_grad_mu, alg_scale
            if div_gram is not None:
                del div_gram

        if epoch % 10 == 0:
            torch.cuda.empty_cache()

    n_syn_full = feat_syn_module.n_syn
    feat_syn_final = torch.zeros(n_syn_full, d)
    with torch.no_grad():
        for c in classes:
            syn_idx_c = feat_syn_module.class_syn_idx[c]
            X_real_c, free_x_real_c = _get_x_real(c)
            n_syn_c = len(syn_idx_c)
            chunk_c = _initial_class_chunk_size(c)
            start_row = 0
            while start_row < n_syn_c:
                end_row = min(start_row + chunk_c, n_syn_c)
                alpha_chunk = None
                A_chunk = None
                X_syn_chunk = None
                try:
                    syn_chunk = syn_idx_c[start_row:end_row].to(device)
                    alpha_chunk = alphas_data[c][start_row:end_row]
                    A_chunk = sparsemax(alpha_chunk, dim=-1)
                    X_syn_chunk = A_chunk @ X_real_c
                    if is_hybrid:
                        X_syn_chunk = (
                            X_syn_chunk + feat_syn_module.beta
                            * residual_data[syn_chunk])
                    feat_syn_final[syn_chunk.cpu()] = X_syn_chunk.cpu()
                    del alpha_chunk, A_chunk, X_syn_chunk
                    start_row = end_row
                except RuntimeError as err:
                    if not _is_cuda_oom(err):
                        raise
                    alpha_chunk = A_chunk = X_syn_chunk = None
                    chunk_c = _halve_chunk_after_oom(chunk_c)
            if free_x_real_c:
                del X_real_c

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(feat_syn_final, save_path)

    for param, requires_grad in zip(validation_model.parameters(),
                                    teacher_requires_grad):
        param.requires_grad_(requires_grad)

    return feat_syn_final
