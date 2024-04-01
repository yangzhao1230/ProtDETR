import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

class HungarianMatcher(nn.Module):

    def __init__(self, cost_class: float = 1,):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])

        cost_class = -out_prob[:, tgt_ids]

        # Final cost matrix
        C = self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["labels"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class,)
