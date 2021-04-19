import torch


def hard_mining(neg_output, neg_labels, num_hard):
    _, idcs = torch.topk(neg_output, min(num_hard, len(neg_output)))
    neg_output = torch.index_select(neg_output, 0, idcs)
    neg_labels = torch.index_select(neg_labels, 0, idcs)
    return neg_output, neg_labels


def get_hard_samples(logits, labels, neg_more=2, neg_least_ratio=0.5, neg_max_ratio=0.7):
    logits = logits.view(-1)
    labels = labels.view(-1)

    pos_idcs = labels > 0
    pos_output = logits[pos_idcs]
    pos_labels = labels[pos_idcs]

    neg_idcs = labels <= 0
    neg_output = logits[neg_idcs]
    neg_labels = labels[neg_idcs]

    neg_at_least=max(neg_more,int(neg_least_ratio * neg_output.size(0)))
    hard_num = min(neg_output.size(0), pos_output.size(0) + neg_at_least, int(neg_max_ratio * neg_output.size(0)) + neg_more)
    if hard_num > 0:
        neg_output, neg_labels = hard_mining(neg_output, neg_labels, hard_num)

    logits=torch.cat([pos_output,neg_output])
    labels = torch.cat([pos_labels, neg_labels])
    return logits, labels


def hard_mining_pos(pos_output, pos_labels, num_hard):
    _, idcs = torch.topk(-pos_output, min(num_hard, len(pos_output)))
    pos_output = torch.index_select(pos_output, 0, idcs)
    pos_labels = torch.index_select(pos_labels, 0, idcs)
    return pos_output, pos_labels


def get_hard_samples_soft_symmetric(logits, labels, soft_high_conf_label_threshold=0.25, min_count=2,
                                    preservation_ratio=0.2):
    logits = logits.view(-1)
    labels = labels.view(-1)

    pos_idcs = labels > 1 - soft_high_conf_label_threshold
    pos_output = logits[pos_idcs]
    pos_labels = labels[pos_idcs]

    neg_idcs = labels < soft_high_conf_label_threshold
    neg_output = logits[neg_idcs]
    neg_labels = labels[neg_idcs]

    neg_at_least = max(min_count, int(preservation_ratio * neg_output.size(0)))
    hard_num = min(neg_output.size(0), neg_at_least)
    if hard_num > 0:
        neg_output, neg_labels = hard_mining(neg_output, neg_labels, hard_num)

    pos_at_least = max(min_count, int(preservation_ratio * pos_output.size(0)))
    pos_hard_num = min(pos_output.size(0), pos_at_least)
    if pos_hard_num > 0:
        pos_output, pos_labels = hard_mining_pos(pos_output, pos_labels, pos_hard_num)

    logits = torch.cat([pos_output, neg_output])
    labels = torch.cat([pos_labels, neg_labels])
    return logits, labels


def get_hard_samples_symmetric(logits, labels, min_count=2,
                                    preservation_ratio=0.2):
    logits = logits.view(-1)
    labels = labels.view(-1)

    pos_idcs = labels > 0
    pos_output = logits[pos_idcs]
    pos_labels = labels[pos_idcs]

    neg_idcs = labels <= 0
    neg_output = logits[neg_idcs]
    neg_labels = labels[neg_idcs]

    neg_at_least = max(min_count, int(preservation_ratio * neg_output.size(0)))
    hard_num = min(neg_output.size(0), neg_at_least)
    if hard_num > 0:
        neg_output, neg_labels = hard_mining(neg_output, neg_labels, hard_num)

    pos_at_least = max(min_count, int(preservation_ratio * pos_output.size(0)))
    pos_hard_num = min(pos_output.size(0), pos_at_least)
    if pos_hard_num > 0:
        pos_output, pos_labels = hard_mining_pos(pos_output, pos_labels, pos_hard_num)

    logits = torch.cat([pos_output, neg_output])
    labels = torch.cat([pos_labels, neg_labels])
    return logits, labels
