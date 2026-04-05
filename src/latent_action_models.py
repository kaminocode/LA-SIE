import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import src.resnet as resnet
from src import models as model_utils
from src.latent_action import (
    DirectFullMatrixOperator,
    DirectSkewExpOperator,
    LatentCodeToFullMatrixOperator,
    SharedGeneratorOperator,
    apply_operator,
    composition_error,
    identity_error,
    inverse_error,
    operator_norm,
    vector_norm,
)
from src.latent_action_online_eval import cross_image_transfer_stats


def _mlp_dims(args):
    return np.array([int(dim) for dim in args.mlp.split("-")])


def _build_split_projectors(args, repr_size, equi_repr_size):
    inv_repr_size = repr_size - equi_repr_size
    mlp_dims = _mlp_dims(args)

    if inv_repr_size > 0:
        ratio_inv = inv_repr_size / float(repr_size)
        mlp_inv = [str(elt) for elt in list(np.round((mlp_dims * ratio_inv)).astype(int))]
        inv_emb_size = int(mlp_inv[-1])
        projector_inv = model_utils.Projector(inv_repr_size, "-".join(mlp_inv))
    else:
        mlp_inv = []
        inv_emb_size = 0
        projector_inv = nn.Identity()

    ratio_equi = equi_repr_size / float(repr_size)
    mlp_equi = [str(elt) for elt in list(np.round((mlp_dims * ratio_equi)).astype(int))]
    equi_emb_size = int(mlp_equi[-1])
    projector_equi = model_utils.Projector(equi_repr_size, "-".join(mlp_equi))

    return {
        "inv_repr_size": inv_repr_size,
        "inv_emb_size": inv_emb_size,
        "equi_emb_size": equi_emb_size,
        "mlp_inv": mlp_inv,
        "mlp_equi": mlp_equi,
        "projector_inv": projector_inv,
        "projector_equi": projector_equi,
    }


def _full_batch_stats(args, projected_views):
    gathered_views = [torch.cat(model_utils.FullGatherLayer.apply(view), dim=0) for view in projected_views]
    std_loss = projected_views[0].new_zeros(())
    cov_loss = projected_views[0].new_zeros(())
    denom = float(len(gathered_views))
    for view in gathered_views:
        centered = view - view.mean(dim=0)
        std = torch.sqrt(centered.var(dim=0) + 0.0001)
        std_loss = std_loss + torch.mean(F.relu(1 - std)) / 2
        cov = (centered.T @ centered) / (args.batch_size - 1)
        cov_loss = cov_loss + model_utils.off_diagonal(cov).pow_(2).sum().div(cov.shape[0])
    return std_loss / denom, cov_loss / denom


def _predicted_std_loss(predicted_views):
    gathered_views = [torch.cat(model_utils.FullGatherLayer.apply(view), dim=0) for view in predicted_views]
    pred_std_loss = predicted_views[0].new_zeros(())
    denom = float(len(gathered_views))
    for view in gathered_views:
        centered = view - view.mean(dim=0)
        std = torch.sqrt(centered.var(dim=0) + 0.0001)
        pred_std_loss = pred_std_loss + torch.mean(F.relu(1 - std)) / 2
    return pred_std_loss / denom


def _maybe_predicted_std(stats, args, predicted_views):
    if not args.latent_enable_pred_std:
        return stats, predicted_views[0].new_zeros(())
    with torch.no_grad():
        stats = model_utils.std_losses(stats, args, "_pred", torch.cat(predicted_views, dim=0))
    pred_std_loss = _predicted_std_loss(predicted_views)
    return stats, pred_std_loss


def _average_mse(tensors):
    reference = tensors[0]
    loss = reference.new_zeros(())
    count = 0
    for i in range(len(tensors)):
        for j in range(i + 1, len(tensors)):
            loss = loss + F.mse_loss(tensors[i], tensors[j])
            count += 1
    return loss / float(count)


class PairLatentActionBase(nn.Module):
    def __init__(
        self,
        args,
        operator_builder,
        *,
        enable_identity=False,
        enable_inverse=False,
        num_classes=55,
    ):
        super().__init__()
        self.args = args
        self.equi_repr_size = self.args.equi
        self.backbone, self.repr_size = resnet.__dict__[args.arch](zero_init_residual=True)
        dims = _build_split_projectors(args, self.repr_size, self.equi_repr_size)
        self.inv_repr_size = dims["inv_repr_size"]
        self.inv_emb_size = dims["inv_emb_size"]
        self.equi_emb_size = dims["equi_emb_size"]
        self.projector_inv = dims["projector_inv"]
        self.projector_equi = dims["projector_equi"]
        print("Invariant projector dims: ", dims["mlp_inv"])
        print("Equivariant projector dims: ", dims["mlp_equi"])

        self.operator = operator_builder(self.equi_repr_size, args)
        self.default_identity = enable_identity
        self.default_inverse = enable_inverse
        self.evaluator = model_utils.OnlineEvaluator(
            self.inv_repr_size,
            self.equi_repr_size,
            self.inv_emb_size,
            self.equi_emb_size,
            num_classes=num_classes,
        )

    def _alignment_weight(self):
        if self.args.latent_align_weight is not None:
            return self.args.latent_align_weight
        return self.args.equi_factor * self.args.sim_coeff

    def _identity_enabled(self):
        return self.default_identity or self.args.latent_enable_identity

    def _inverse_enabled(self):
        return self.default_inverse or self.args.latent_enable_inverse

    def _split_views(self, embeddings):
        inv_views = []
        equi_views = []
        inv_proj_views = []
        equi_proj_views = []
        full_proj_views = []
        for embedding in embeddings:
            if self.inv_repr_size > 0:
                inv_view = embedding[..., :self.inv_repr_size]
                inv_proj = self.projector_inv(inv_view)
            else:
                inv_view = embedding.new_zeros(embedding.shape[0], 0)
                inv_proj = inv_view
            equi_view = embedding[..., self.inv_repr_size:]
            equi_proj = self.projector_equi(equi_view)
            full_proj = torch.cat((inv_proj, equi_proj), dim=1)
            inv_views.append(inv_view)
            equi_views.append(equi_view)
            inv_proj_views.append(inv_proj)
            equi_proj_views.append(equi_proj)
            full_proj_views.append(full_proj)
        return inv_views, equi_views, inv_proj_views, equi_proj_views, full_proj_views

    def _base_stats(self, embeddings, projected_views):
        stats = {}
        with torch.no_grad():
            stats = model_utils.std_losses(stats, self.args, "_view1", embeddings[0], proj_out=projected_views[0])
            stats = model_utils.std_losses(stats, self.args, "_view2", embeddings[1], proj_out=projected_views[1])
        return stats

    def _identity_loss(self, prediction_self):
        return identity_error(prediction_self.operator)

    def _inverse_loss(self, prediction_forward, prediction_backward):
        return inverse_error(prediction_forward.operator, prediction_backward.operator)

    def _maybe_add_generator_stats(self, stats):
        if hasattr(self.operator, "generator_matrices"):
            stats["Latent/generator_norm"] = operator_norm(self.operator.generator_matrices())
        return stats

    def _log_prediction_stats(self, stats, primary_prediction):
        stats["Latent/operator_norm"] = operator_norm(primary_prediction.operator)
        if primary_prediction.code is not None:
            stats["Latent/code_norm"] = vector_norm(primary_prediction.code)
        if primary_prediction.coefficients is not None:
            stats["Latent/coeff_norm"] = vector_norm(primary_prediction.coefficients)
        return self._maybe_add_generator_stats(stats)

    def forward(self, x, y, z, labels):
        x_emb = self.backbone(x)
        y_emb = self.backbone(y)
        embeddings = [x_emb, y_emb]
        inv_views, equi_views, inv_proj_views, equi_proj_views, full_proj_views = self._split_views(embeddings)

        loss_eval, stats_eval = self.evaluator(
            [x_emb.detach(), y_emb.detach()],
            [full_proj_views[0].detach(), full_proj_views[1].detach()],
            labels,
            z,
        )

        stats = self._base_stats(embeddings, full_proj_views)

        if self.inv_repr_size > 0:
            repr_loss_inv = F.mse_loss(inv_proj_views[0], inv_proj_views[1])
        else:
            repr_loss_inv = x_emb.new_zeros(())

        prediction_xy = self.operator(equi_views[0], equi_views[1])
        prediction_yx = self.operator(equi_views[1], equi_views[0])
        prediction_xx = self.operator(equi_views[0], equi_views[0])
        y_equi_pred = apply_operator(prediction_xy.operator, equi_views[0])
        repr_loss_equi = F.mse_loss(y_equi_pred, equi_views[1].float())
        stats, pred_std_loss = _maybe_predicted_std(stats, self.args, [y_equi_pred])

        identity_loss_value = self._identity_loss(prediction_xx)
        inverse_loss_value = self._inverse_loss(prediction_xy, prediction_yx)

        std_loss, cov_loss = _full_batch_stats(self.args, full_proj_views)

        loss = (
            self.args.sim_coeff * repr_loss_inv
            + self._alignment_weight() * repr_loss_equi
            + self.args.std_coeff * std_loss
            + self.args.cov_coeff * cov_loss
        )
        if self.args.latent_enable_pred_std:
            loss = loss + self.args.std_coeff * pred_std_loss
        if self._identity_enabled():
            loss = loss + self.args.latent_identity_weight * identity_loss_value
        if self._inverse_enabled():
            loss = loss + self.args.latent_inverse_weight * inverse_loss_value

        stats["repr_loss_inv"] = repr_loss_inv
        stats["repr_loss_equi"] = repr_loss_equi
        stats["std_loss"] = std_loss
        if self.args.latent_enable_pred_std:
            stats["pred_std_loss"] = pred_std_loss
        stats["cov_loss"] = cov_loss
        stats["loss_identity"] = identity_loss_value
        stats["loss_inverse"] = inverse_loss_value
        stats["loss"] = loss
        stats["Latent/alignment_loss"] = repr_loss_equi
        if self.args.latent_enable_pred_std:
            stats["Latent/pred_std_loss"] = pred_std_loss
        stats["Latent/identity_loss"] = identity_loss_value
        stats["Latent/inverse_loss"] = inverse_loss_value
        stats["Latent/composition_loss"] = x_emb.new_zeros(())
        stats["Latent/identity_error"] = identity_loss_value
        stats["Latent/inverse_error"] = inverse_loss_value
        stats["Latent/composition_error"] = x_emb.new_zeros(())
        stats = self._log_prediction_stats(stats, prediction_xy)
        stats.update(
            cross_image_transfer_stats(
                self.args,
                equi_views[0],
                equi_views[1],
                prediction_xy,
                labels,
                z,
            )
        )
        return loss, loss_eval, stats, stats_eval


class TripletLatentActionBase(nn.Module):
    def __init__(
        self,
        args,
        operator_builder,
        *,
        enable_identity=False,
        enable_inverse=True,
        enable_composition=False,
        num_classes=55,
    ):
        super().__init__()
        self.args = args
        self.equi_repr_size = self.args.equi
        self.backbone, self.repr_size = resnet.__dict__[args.arch](zero_init_residual=True)
        dims = _build_split_projectors(args, self.repr_size, self.equi_repr_size)
        self.inv_repr_size = dims["inv_repr_size"]
        self.inv_emb_size = dims["inv_emb_size"]
        self.equi_emb_size = dims["equi_emb_size"]
        self.projector_inv = dims["projector_inv"]
        self.projector_equi = dims["projector_equi"]
        print("Invariant projector dims: ", dims["mlp_inv"])
        print("Equivariant projector dims: ", dims["mlp_equi"])

        self.operator = operator_builder(self.equi_repr_size, args)
        self.default_identity = enable_identity
        self.default_inverse = enable_inverse
        self.default_composition = enable_composition
        self.evaluator = model_utils.OnlineEvaluator(
            self.inv_repr_size,
            self.equi_repr_size,
            self.inv_emb_size,
            self.equi_emb_size,
            num_classes=num_classes,
        )

    def _alignment_weight(self):
        if self.args.latent_align_weight is not None:
            return self.args.latent_align_weight
        return self.args.equi_factor * self.args.sim_coeff

    def _identity_enabled(self):
        return self.default_identity or self.args.latent_enable_identity

    def _inverse_enabled(self):
        return self.default_inverse or self.args.latent_enable_inverse

    def _composition_enabled(self):
        return self.default_composition or self.args.latent_enable_composition

    def _split_views(self, embeddings):
        inv_views = []
        equi_views = []
        inv_proj_views = []
        equi_proj_views = []
        full_proj_views = []
        for embedding in embeddings:
            if self.inv_repr_size > 0:
                inv_view = embedding[..., :self.inv_repr_size]
                inv_proj = self.projector_inv(inv_view)
            else:
                inv_view = embedding.new_zeros(embedding.shape[0], 0)
                inv_proj = inv_view
            equi_view = embedding[..., self.inv_repr_size:]
            equi_proj = self.projector_equi(equi_view)
            full_proj = torch.cat((inv_proj, equi_proj), dim=1)
            inv_views.append(inv_view)
            equi_views.append(equi_view)
            inv_proj_views.append(inv_proj)
            equi_proj_views.append(equi_proj)
            full_proj_views.append(full_proj)
        return inv_views, equi_views, inv_proj_views, equi_proj_views, full_proj_views

    def _base_stats(self, embeddings, projected_views):
        stats = {}
        with torch.no_grad():
            stats = model_utils.std_losses(stats, self.args, "_view1", embeddings[0], proj_out=projected_views[0])
            stats = model_utils.std_losses(stats, self.args, "_view2", embeddings[1], proj_out=projected_views[1])
        return stats

    def _maybe_add_generator_stats(self, stats):
        if hasattr(self.operator, "generator_matrices"):
            stats["Latent/generator_norm"] = operator_norm(self.operator.generator_matrices())
        return stats

    def _primary_norm_stats(self, stats, predictions):
        operator_norms = [operator_norm(pred.operator) for pred in predictions]
        stats["Latent/operator_norm"] = torch.stack(operator_norms).mean()
        code_norms = [vector_norm(pred.code) for pred in predictions if pred.code is not None]
        if code_norms:
            stats["Latent/code_norm"] = torch.stack(code_norms).mean()
        coeff_norms = [vector_norm(pred.coefficients) for pred in predictions if pred.coefficients is not None]
        if coeff_norms:
            stats["Latent/coeff_norm"] = torch.stack(coeff_norms).mean()
        return self._maybe_add_generator_stats(stats)

    def forward(self, x0, x1, x2, z01, z12, z02, labels):
        emb0 = self.backbone(x0)
        emb1 = self.backbone(x1)
        emb2 = self.backbone(x2)
        embeddings = [emb0, emb1, emb2]
        inv_views, equi_views, inv_proj_views, equi_proj_views, full_proj_views = self._split_views(embeddings)

        loss_eval, stats_eval = self.evaluator(
            [emb0.detach(), emb1.detach()],
            [full_proj_views[0].detach(), full_proj_views[1].detach()],
            labels,
            z01,
        )

        stats = self._base_stats(embeddings, full_proj_views)

        if self.inv_repr_size > 0:
            repr_loss_inv = _average_mse(inv_proj_views)
        else:
            repr_loss_inv = emb0.new_zeros(())

        pred01 = self.operator(equi_views[0], equi_views[1])
        pred12 = self.operator(equi_views[1], equi_views[2])
        pred02 = self.operator(equi_views[0], equi_views[2])
        pred10 = self.operator(equi_views[1], equi_views[0])
        pred21 = self.operator(equi_views[2], equi_views[1])
        pred20 = self.operator(equi_views[2], equi_views[0])
        pred00 = self.operator(equi_views[0], equi_views[0])
        pred11 = self.operator(equi_views[1], equi_views[1])
        pred22 = self.operator(equi_views[2], equi_views[2])

        y01_pred = apply_operator(pred01.operator, equi_views[0])
        y12_pred = apply_operator(pred12.operator, equi_views[1])
        y02_pred = apply_operator(pred02.operator, equi_views[0])
        align01 = F.mse_loss(y01_pred, equi_views[1].float())
        align12 = F.mse_loss(y12_pred, equi_views[2].float())
        align02 = F.mse_loss(y02_pred, equi_views[2].float())
        repr_loss_equi = (align01 + align12 + align02) / 3.0
        stats, pred_std_loss = _maybe_predicted_std(stats, self.args, [y01_pred, y12_pred, y02_pred])

        identity_loss_value = (
            identity_error(pred00.operator)
            + identity_error(pred11.operator)
            + identity_error(pred22.operator)
        ) / 3.0
        inverse_loss_value = (
            inverse_error(pred01.operator, pred10.operator)
            + inverse_error(pred12.operator, pred21.operator)
            + inverse_error(pred02.operator, pred20.operator)
        ) / 3.0
        composition_loss_value = composition_error(pred02.operator, pred12.operator, pred01.operator)

        std_loss, cov_loss = _full_batch_stats(self.args, full_proj_views)

        loss = (
            self.args.sim_coeff * repr_loss_inv
            + self._alignment_weight() * repr_loss_equi
            + self.args.std_coeff * std_loss
            + self.args.cov_coeff * cov_loss
        )
        if self.args.latent_enable_pred_std:
            loss = loss + self.args.std_coeff * pred_std_loss
        if self._identity_enabled():
            loss = loss + self.args.latent_identity_weight * identity_loss_value
        if self._inverse_enabled():
            loss = loss + self.args.latent_inverse_weight * inverse_loss_value
        if self._composition_enabled():
            loss = loss + self.args.latent_composition_weight * composition_loss_value

        stats["repr_loss_inv"] = repr_loss_inv
        stats["repr_loss_equi"] = repr_loss_equi
        stats["std_loss"] = std_loss
        if self.args.latent_enable_pred_std:
            stats["pred_std_loss"] = pred_std_loss
        stats["cov_loss"] = cov_loss
        stats["loss_identity"] = identity_loss_value
        stats["loss_inverse"] = inverse_loss_value
        stats["loss_composition"] = composition_loss_value
        stats["loss"] = loss
        stats["Latent/alignment_loss"] = repr_loss_equi
        if self.args.latent_enable_pred_std:
            stats["Latent/pred_std_loss"] = pred_std_loss
        stats["Latent/identity_loss"] = identity_loss_value
        stats["Latent/inverse_loss"] = inverse_loss_value
        stats["Latent/composition_loss"] = composition_loss_value
        stats["Latent/identity_error"] = identity_loss_value
        stats["Latent/inverse_error"] = inverse_loss_value
        stats["Latent/composition_error"] = composition_loss_value
        stats = self._primary_norm_stats(stats, [pred01, pred12, pred02])
        stats.update(
            cross_image_transfer_stats(
                self.args,
                equi_views[0],
                equi_views[1],
                pred01,
                labels,
                z01,
            )
        )
        return loss, loss_eval, stats, stats_eval


def _direct_full_builder(feature_dim, args):
    return DirectFullMatrixOperator(
        feature_dim,
        hidden_dim=args.latent_operator_hidden_dim,
    )


def _direct_skewexp_builder(feature_dim, args):
    return DirectSkewExpOperator(
        feature_dim,
        hidden_dim=args.latent_operator_hidden_dim,
    )


def _latentcode_builder(feature_dim, args):
    return LatentCodeToFullMatrixOperator(
        feature_dim,
        args.latent_action_dim,
    )


def _sharedgen_fixed_builder(feature_dim, args):
    return SharedGeneratorOperator(
        feature_dim,
        args.latent_action_dim,
        args.num_generators,
        learnable_generators=False,
    )


def _sharedgen_learned_builder(feature_dim, args):
    return SharedGeneratorOperator(
        feature_dim,
        args.latent_action_dim,
        args.num_generators,
        learnable_generators=True,
    )


class direct_full_matrix_2v(PairLatentActionBase):
    def __init__(self, args, num_classes=55):
        super().__init__(args, _direct_full_builder, num_classes=num_classes)


class direct_skewexp_2v(PairLatentActionBase):
    def __init__(self, args, num_classes=55):
        super().__init__(args, _direct_skewexp_builder, num_classes=num_classes)


class latentcode_to_full_matrix_2v(PairLatentActionBase):
    def __init__(self, args, num_classes=55):
        super().__init__(args, _latentcode_builder, num_classes=num_classes)


class sharedgen_fixed_2v(PairLatentActionBase):
    def __init__(self, args, num_classes=55):
        super().__init__(args, _sharedgen_fixed_builder, num_classes=num_classes)


class sharedgen_learned_2v(PairLatentActionBase):
    def __init__(self, args, num_classes=55):
        super().__init__(args, _sharedgen_learned_builder, num_classes=num_classes)


class sharedgen_learned_2v_identity(PairLatentActionBase):
    def __init__(self, args, num_classes=55):
        super().__init__(
            args,
            _sharedgen_learned_builder,
            enable_identity=True,
            num_classes=num_classes,
        )


class sharedgen_learned_2v_inverse(PairLatentActionBase):
    def __init__(self, args, num_classes=55):
        super().__init__(
            args,
            _sharedgen_learned_builder,
            enable_inverse=True,
            num_classes=num_classes,
        )


class sharedgen_learned_3v_no_comp(TripletLatentActionBase):
    def __init__(self, args, num_classes=55):
        super().__init__(
            args,
            _sharedgen_learned_builder,
            enable_inverse=True,
            num_classes=num_classes,
        )


class sharedgen_learned_3v_comp(TripletLatentActionBase):
    def __init__(self, args, num_classes=55):
        super().__init__(
            args,
            _sharedgen_learned_builder,
            enable_inverse=True,
            enable_composition=True,
            num_classes=num_classes,
        )
