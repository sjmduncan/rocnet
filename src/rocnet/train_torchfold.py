import torch

from rocnet.model import Decoder, Encoder
from rocnet.octree import Octree
from rocnet.torchfold import Fold


def encode_tree_fold(fold: Fold, encoder: Encoder, tree: Octree):
    def encode_node(node, level):
        if node.is_leaf():
            if node.is_empty_leaf():
                return fold.add("_tf_encode_empty_leaf", node.leaf_feature.cuda())
            else:
                return fold.add("_tf_encode_leaf", node.leaf_feature.cuda())
        elif node.is_non_leaf():
            c = [encode_node(c, level + 1) for c in node.children]
            return fold.add(f"_tf_encode_node_lvl{level}", c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7])

    encoding = encode_node(tree.root, 0)
    if encoder.has_root_encoder:
        root_code = fold.add("_tf_encode_root", encoding)
        return root_code
    else:
        return encoding


def decode_loss_fold(fold: Fold, decoder: Decoder, code: torch.Tensor, tgt_tree: Octree):
    def decode_node_leaf(tgt, est, l):
        label_prob = fold.add("_tf_classify_node", est)
        label_loss = fold.add("label_loss_fn", label_prob, tgt.node_type.cuda())
        if tgt.is_leaf():
            if tgt.is_empty_leaf():
                return fold.add("_tf_decode_empty_leaf"), label_loss
            else:
                fea = fold.add("_tf_decode_leaf", est)
                recon_loss = fold.add("recon_loss_fn", fea, tgt.leaf_feature.cuda())
                return recon_loss, label_loss

        elif tgt.is_non_leaf():
            children = fold.add(f"_tf_decode_node_lvl{l}", est).split(8)
            child_losses = [decode_node_leaf(tgt_child, est_child, l + 1) for (tgt_child, est_child) in zip(tgt.children, children)]

            child_recon_loss = fold.add("_tf_vec_sum_8", *[loss[0] for loss in child_losses])
            child_label_loss = fold.add("_tf_vec_sum_8", *[loss[1] for loss in child_losses])

            return child_recon_loss, fold.add("_tf_vec_sum_2", child_label_loss, label_loss)

    if decoder.has_root_decoder:
        est_root_node = fold.add("_tf_decode_root", code)
    else:
        est_root_node = code.reshape(decoder.cfg.node_channels, 4, 4, 4).unsqueeze(0)
    recon_loss, label_loss = decode_node_leaf(tgt_tree.root, est_root_node, 0)
    return recon_loss, label_loss


def batch_loss_fold(model, batch):
    """"""
    encoder_fold = Fold(cuda=True)
    encoder_fold_nodes = [encode_tree_fold(encoder_fold, model.encoder, tree) for tree in batch]
    codes = torch.split(encoder_fold.apply(model.encoder, [encoder_fold_nodes])[0], 1, 0)

    decoder_fold = Fold(cuda=True)
    decoder_fold_nodes = [decode_loss_fold(decoder_fold, model.decoder, code, tree) for (code, tree) in zip(codes, batch)]
    recon_loss_nodes, label_loss_nodes = map(list, zip(*decoder_fold_nodes))
    recon_loss = decoder_fold.apply(model.decoder, [recon_loss_nodes])
    label_loss = decoder_fold.apply(model.decoder, [label_loss_nodes])
    full_loss = torch.cat([recon_loss[0].unsqueeze(0), label_loss[0].unsqueeze(0), (recon_loss[0] + label_loss[0]).unsqueeze(0)]).T
    return full_loss, full_loss.mean(axis=0)
