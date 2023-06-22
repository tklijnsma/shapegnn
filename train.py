import os, copy
import numpy as np
import tqdm

import torch
import torch.nn as nn
from torch_geometric.data import Data, InMemoryDataset as Dataset, DataLoader
import torch_geometric.nn
from torch_cluster import knn_graph
from torch_scatter import scatter_max, scatter_add, scatter_mean
from torch.optim.lr_scheduler import ReduceLROnPlateau


import utils
import generate_data

scripter = utils.Scripter()


def huber(d, delta):
    """
    See: https://en.wikipedia.org/wiki/Huber_loss#Definition
    Multiplied by 2 w.r.t Wikipedia version (aligning with Jan's definition)
    """
    return torch.where(
        torch.abs(d) <= delta, d**2, 2.0 * delta * (torch.abs(d) - delta)
    )


def scatter_count(input: torch.Tensor) -> torch.LongTensor:
    """
    Returns ordered counts over an index array

    Example:
    >>> scatter_count(torch.Tensor([0, 0, 0, 1, 1, 2, 2])) # input
    >>> tensor([3, 2, 2])

    Index assumptions work like in torch_scatter, so:
    >>> scatter_count(torch.Tensor([1, 1, 1, 2, 2, 4, 4]))
    >>> tensor([0, 3, 2, 0, 2])
    """
    return scatter_add(torch.ones_like(input, dtype=torch.long), input.long())


class ShapeDataset(Dataset):
    def __init__(self, n=10000):
        self.n = n
        self.shapes = [None for _ in range(self.n)]
        super().__init__()

    def len(self):
        return self.n

    def get(self, i):
        if self.shapes[i] is None:
            X, y = generate_data.event()
            self.shapes[i] = Data(x=torch.FloatTensor(X), y=torch.LongTensor(y))
        return self.shapes[i]


class ShapeGNN(nn.Module):
    """
    Model to recognize basic shapes in N-dimensional point clouds.

    A model based on a standard subclass of a PyTorch module that
    uses consecutive EdgeConv layers (from the PyTorch Geometric
    package) in order to learn what basic shapes in a point cloud
    look like.


    Attributes
    ----------

    input_dim : int
        Number of columns of the input data.

    output_dim : int
        Number of columns of the output matrix.

    hidden_dim : int
        Dimension of the hidden (latent) space.

    k : int
        Number of neighbors to generate edges with when running
        the k-nearest neighbors algorithm.

    n_edgeconvs : int
        Number of consecutive EdgeConv layers of which the model
        should consist.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 3,
        hidden_dim: int = 16,
        k: int = 16,
        n_edgeconvs: int = 4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_edgeconvs = n_edgeconvs
        self.k = k

        self.embedding = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())

        self.edgeconvs = nn.ModuleList()
        for _ in range(n_edgeconvs):
            self.edgeconvs.append(
                torch_geometric.nn.conv.EdgeConv(
                    nn=nn.Sequential(
                        nn.Linear(2 * hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                    ),
                    aggr="mean",
                )
            )

        self.output = nn.Linear(n_edgeconvs * hidden_dim, output_dim)

    def forward(self, data: torch_geometric.data.Data) -> torch.FloatTensor:
        x = data.x
        x = self.embedding(x)

        intermediate_outputs = []
        for edgeconv in self.edgeconvs:
            edge_index = knn_graph(x, k=self.k, batch=data.batch)
            x = edgeconv(x, edge_index=edge_index)
            intermediate_outputs.append(x)

        x = torch.cat(intermediate_outputs, dim=-1)
        x = self.output(x)
        return x


class LossResult:
    def __init__(self, **components):
        self.components = components
        self.offset = 1.0

    @property
    def loss(self):
        return self.offset + sum(self.components.values())

    def __repr__(self):
        loss = self.loss
        r = [f"final loss:     {loss:.10f}"]
        for c, v in self.components.items():
            perc = 100.0 * v / (loss - self.offset)
            r.append(f"  {c:20} {v:15.10f}   {perc:5.2f}%")
        return "\n".join(r)

    def __add__(self, o):
        lr = LossResult(**self.components)
        for c in o.components.keys():
            if not c in lr.components:
                lr.components[c] = 0.0
            lr.components[c] += o.components[c]
        return lr

    def __truediv__(self, num):
        lr = LossResult(**self.components)
        for c in lr.components.keys():
            lr.components[c] /= num
        return lr

    def backward(self):
        return self.loss.backward()


class ObjectCondensation:
    def __init__(self, x, data, qmin=1.0, sB=0.1):
        assert not torch.isnan(x).any()
        self.data = data

        self.qmin = qmin
        self.sB = sB

        # Optional huberization for V_att distances
        self.huberize_norm_for_V_att = True

        # Quantities that will surely be used can be calculated now:
        self.beta = torch.sigmoid(x[:, 0])
        self.x = x[:, 1:]
        self.y = data.y

        self.n = x.size(0)
        self.n_real = (data.y > 0).sum()
        self.n_cond = torch.max(data.y)

        self.is_noise = data.y == 0
        self.is_real = ~self.is_noise

        self.x_real = self.x[self.is_real]
        self.y_real = (
            data.y[self.is_real] - 1
        )  # Make it 0-indexed; normally 0 is the noise cluster

        # Number of points per cluster / cond point
        self.n_per_cond = scatter_count(self.y_real)

    def calc_q_paper(self):
        self.q = self.beta.arctanh() ** 2 + self.qmin
        self.calc_q_dependent_quantities()

    def calc_q_betaclip(self):
        """
        Performs a clip on beta before calling arctanh**2.
        Very often necessary to avoid NaN's in the arctanh calc.
        Known as "soft q scaling".
        """
        self.q = (self.beta.clip(0.0, 1 - 1e-4) / 1.002).arctanh() ** 2 + self.qmin
        self.calc_q_dependent_quantities()

    def calc_q_dependent_quantities(self):
        self.q_real = self.q[self.is_real]
        self.q_cond, self.i_cond = scatter_max(self.q_real, self.y_real)
        self.x_cond = self.x_real[self.i_cond]
        self.beta_cond = self.beta[self.i_cond]

    @property
    def d(self):
        # Distance matrix of every point to every condensation point
        # (n_hits, 1, cluster_space_dim) - (1, n_cond, cluster_space_dim)
        #   gives (n_hits, n_cond, cluster_space_dim)
        if not hasattr(self, "_d"):
            self._d = (self.x.unsqueeze(1) - self.x_cond.unsqueeze(0)).norm(dim=-1)
        return self._d

    @property
    def M(self):
        # Connectivity matrix for real hits: Only 1 of hit belongs to cond point, otherwise 0
        return torch.nn.functional.one_hot(self.y_real).long()

    @property
    def V_att(self):
        if self.huberize_norm_for_V_att:
            # Parabolic (like normal L2-norm) where distances < threshold,
            # but linear outside.
            # This prevents unreasonably high losses when misassigning
            # singular points, and allows the network to space clusters
            # more.
            d = huber(self.d[self.is_real] + 1e-5, 4.0)
        else:
            d = self.d[self.is_real] ** 2
        V = self.M * self.q_real.unsqueeze(-1) * self.q_cond.unsqueeze(0) * d
        assert V.size() == (self.n_real, self.n_cond)
        V = V.sum() / self.n
        return V

    @property
    def V_rep(self):
        M_inv = 1 - torch.nn.functional.one_hot(self.y).long()
        M_inv = M_inv[
            :, 1:
        ]  # Throw away the noise column; there is no cond point for noise

        # Power-scale the norms: Gaussian scaling term instead of a cone
        d = torch.exp(-4.0 * self.d**2)

        # (n, 1) * (1, n_cond) * (n, n_cond)
        V = self.q.unsqueeze(1) * self.q_cond.unsqueeze(0) * M_inv * d
        assert V.size() == (self.n, self.n_cond)

        V = torch.maximum(V, torch.tensor(0.0)).sum() / self.n
        return V

    @property
    def L_beta_noise(self):
        return self.sB * self.beta[self.is_noise].mean()

    @property
    def L_beta_sig(self):
        return (1 - self.beta[self.i_cond]).mean()

    @property
    def L_beta_sig_short_range_potential(self):
        # (N, 1): Inverse scaled distance to the cond point every hit belongs to
        # low d -> high closeness, and vice versa
        # Keep only distances w.r.t. belonging cond point
        # Then sum over _hits_, so the result is (n_cond,)
        closeness = (1.0 / (20.0 * self.d[self.is_real] ** 2 + 1.0) * self.M).sum(dim=0)
        assert torch.all(closeness >= 1.0) and torch.all(closeness <= self.n_per_cond)

        # closeness of the cond point w.r.t. itself will be 1., by definition
        # Remove that one, then divide by number of hits per cluster
        # to obtain average closeness per cluster
        closeness = (closeness - 1.0) / self.n_per_cond
        assert torch.all(closeness >= 0.0) and torch.all(closeness <= 1.0)

        # Multiply by the beta of the cond point and take the average, invert
        L = -(closeness * self.beta_cond).mean()
        assert -1 <= L <= 0.0

        # Summary: For a good prediction,
        # small d -> large closeness -> more neg L -> low L
        # high beta_cond -> more neg L -> low L
        return L

    @property
    def L_beta_sig_logterm(self):
        # For a good prediction: large beta_cond -> more negative -> low L
        return (-0.2 * torch.log(self.beta_cond + 1e-9)).mean()


def loss_fn(x, data):
    try:
        oc = ObjectCondensation(x, data)
        # oc.sB = .35
        oc.sB = 1.0
        oc.calc_q_betaclip()
        return LossResult(
            V_att=oc.V_att,
            V_rep=oc.V_rep,
            # L_beta_sig = oc.L_beta_sig,
            L_beta_sig_srp=oc.L_beta_sig_short_range_potential,
            L_beta_sig_logterm=oc.L_beta_sig_logterm,
            L_beta_noise=oc.L_beta_noise,
        )
    except Exception:
        utils.logger.error("Caught exception when evaluating loss fn; saving event")
        torch.save(data, "debug_data.pt")
        torch.save(x, "debug_x.pt")
        raise


@scripter
def train():
    seed = utils.pull_arg("-s", "--seed", type=int, default=1001).seed
    np.random.seed(seed)

    model = ShapeGNN(2, 3, hidden_dim=32)

    if ckpt := utils.pull_arg("-i", "--ckpt", type=str).ckpt:
        utils.logger.info(f"Loading initial model weights from ckpt {ckpt}")
        model.load_state_dict(torch.load(ckpt)["model"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.05, patience=10, verbose=True
    )

    n_train = 20000
    fixed_test_ds = ShapeDataset(int(0.125 * n_train))

    def train():
        train_ds = ShapeDataset(n_train)
        model.train()
        for data in tqdm.tqdm(DataLoader(train_ds, batch_size=1, shuffle=True)):
            optimizer.zero_grad()
            x = model(data)
            assert x.size() == (data.x.size(0), 3)
            assert not torch.isnan(x).any()
            loss = loss_fn(x, data)
            loss.backward()
            optimizer.step()
        utils.logger.info(f"Final train loss:\n{loss}")
        scheduler.step(loss.loss)

    best_test_loss = torch.inf

    def test():
        nonlocal best_test_loss
        with torch.no_grad():
            model.eval()

            loss = LossResult()
            for data in tqdm.tqdm(
                DataLoader(fixed_test_ds, batch_size=1, shuffle=False)
            ):
                x = model(data)
                loss += loss_fn(x, data)
            print(f"test loss:\n{loss/len(fixed_test_ds)}")

            valid_ds = ShapeDataset(int(0.125 * n_train))
            valid_loss = LossResult()
            for data in tqdm.tqdm(DataLoader(valid_ds, batch_size=1, shuffle=False)):
                x = model(data)
                valid_loss += loss_fn(x, data)
            print(f"valid loss:\n{valid_loss/len(valid_ds)}")

        if loss.loss < best_test_loss:
            ckpt = "models/gnn_best.pth.tar"
            print(f"Saving to {ckpt}")
            best_test_loss = loss.loss
            os.makedirs("models", exist_ok=True)
            torch.save(dict(model=model.state_dict()), ckpt)

    for i_epoch in range(400):
        print(f"Training epoch {i_epoch}")
        train()
        test()


@scripter
def plot():
    np.random.seed(utils.pull_arg("-s", "--seed", type=int, default=1002).seed)
    n = utils.pull_arg("-n", type=int, default=1002).n
    do_clustering = utils.pull_arg("-c", "--cluster", action="store_true").cluster
    ds = ShapeDataset(n)

    # plt.scatter(x, y, s=80, facecolors='none', edgecolors='r')

    with torch.no_grad():
        model = ShapeGNN(2, 3, hidden_dim=32)
        ckpt = utils.pull_arg("ckpt", type=str).ckpt
        utils.logger.info(f"Loading initial model weights from ckpt {ckpt}")
        model.load_state_dict(torch.load(ckpt)["model"])
        model.eval()

        for data in DataLoader(ds, batch_size=1, shuffle=False):
            x = model(data)

            oc = ObjectCondensation(x, data)
            oc.sB = 2.0
            oc.calc_q_betaclip()

            x = x[:, 1:].numpy()
            s = oc.q / oc.q.mean() * 20

            if do_clustering:
                # Produce the truth clustering now
                truth_clusters = []
                y_to_truth_cluster = {}
                for y in np.unique(data.y):
                    c = Cluster(0, data.y == y, data.y == y)
                    c.y = y
                    truth_clusters.append(c)
                    y_to_truth_cluster[y] = c

            n_cols = 3
            n_rows = 2 if do_clustering else 1

            with utils.quick_fig((8 * n_cols, 8 * n_rows)) as fig:
                ax = fig.add_subplot(n_rows, n_cols, 1)
                for y in np.unique(data.y):
                    sel = data.y == y
                    ax.scatter(data.x[sel, 0], data.x[sel, 1])
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.set_title("Normal space (truth-colored)")

                ax = fig.add_subplot(n_rows, n_cols, 2)
                for y in np.unique(data.y):
                    sel = data.y == y
                    pc = ax.scatter(x[sel, 0], x[sel, 1], label=f"{int(y)}", s=s[sel])
                    if do_clustering:
                        y_to_truth_cluster[y].color = pc.get_facecolor()
                ax.set_title("Clustering space (truth-colored)")

                ax = fig.add_subplot(n_rows, n_cols, 3)
                is_noise = data.y == 0
                bins = np.linspace(0, 1, 40)
                ax.hist(oc.beta[is_noise], bins=bins, label="noise", density=True)
                ax.hist(
                    oc.beta[~is_noise],
                    bins=bins,
                    label="shapes",
                    density=True,
                    alpha=0.5,
                )
                ax.legend()
                ax.set_xlabel("beta")
                ax.set_title("beta distribution")

                if do_clustering:

                    def dist(c1, c2):
                        """
                        Distance between two clusters
                        """
                        x1 = x[c1.in_cluster].mean(axis=0)
                        x2 = x[c2.in_cluster].mean(axis=0)
                        return np.sqrt(((x1 - x2) ** 2).sum())

                    clusters = make_clusters(oc.beta, x)

                    # Find the closest truth cluster color (None if nothing is close)
                    for c in clusters:
                        d_closest = 0.5
                        c.color = None
                        for truth_cluster in truth_clusters:
                            if (d := dist(c, truth_cluster)) < d_closest:
                                c.color = truth_cluster.color
                                d_closest = d

                    is_pred_noise = np.ones(oc.beta.shape[0], dtype=bool)
                    for c in clusters:
                        is_pred_noise[c.in_cluster] = False

                    ax = fig.add_subplot(n_rows, n_cols, 4)
                    ax.scatter(data.x[is_pred_noise, 0], data.x[is_pred_noise, 1])
                    for c in clusters:
                        ax.scatter(
                            data.x[c.in_cluster, 0], data.x[c.in_cluster, 1], c=c.color
                        )
                    ax.set_xlim(-1, 1)
                    ax.set_ylim(-1, 1)
                    ax.set_title("Normal space (pred-colored)")

                    ax = fig.add_subplot(n_rows, n_cols, 5)
                    ax.scatter(
                        x[is_pred_noise, 0], x[is_pred_noise, 1], s=s[is_pred_noise]
                    )
                    for c in clusters:
                        ax.scatter(
                            x[c.in_cluster, 0],
                            x[c.in_cluster, 1],
                            c=c.color,
                            s=s[c.in_cluster],
                        )
                    ax.set_title("Clustering space (pred-colored)")


class Cluster:
    def __init__(self, i_center, in_core, in_cluster):
        self.i_center = i_center
        self.in_core = in_core
        self.in_cluster = in_cluster


def make_clusters(
    beta, x, cluster_core_radius=0.1, min_core_charge=1.5, cluster_radius=0.2
):
    beta = copy.deepcopy(beta)
    x = copy.deepcopy(x)

    # Output array
    clusters = []

    # While there points available to be potentially clustered
    while True:
        # Pick max beta
        i = np.argmax(beta)

        if beta[i] == 0.0:
            # No more maxima left, quit
            break

        x_center = x[i]

        # (N,3) - (1,3) = (N,3)
        d_squared = ((x - x_center[np.newaxis, :]) ** 2).sum(axis=-1)
        in_core = d_squared < cluster_core_radius**2

        # Collect charge in the core; if sufficient, consider it an instance
        q_core = beta[in_core].sum()

        if q_core < min_core_charge:
            # Not a valid cluster; disable the point, go to next maximum
            beta[i] = 0.0
            continue

        # Grab all available points around all points in the core
        x_core = x[in_core]

        # (N,1,3) - (1,n_core,3) = (N, n_core, 3)
        d_squared = ((x[:, np.newaxis, :] - x_core[np.newaxis, :, :]) ** 2).sum(axis=-1)

        in_cluster = (d_squared <= cluster_radius).any(axis=-1)

        clusters.append(Cluster(i, in_core, in_cluster))

        # Disable the used points
        x[in_cluster] -= 1000.0
        beta[in_cluster] = 0.0

    return clusters


@scripter
def debug():
    utils.debug()
    # data = Data(
    #     x = torch.rand((100,2)),
    #     y = torch.arange(5).repeat_interleave(torch.LongTensor([51, 9, 15, 17, 8]))
    #     )
    # out = torch.rand((100, 3))
    # out[-1,0] = 1.1
    data = torch.load("debug_data.pt")
    out = torch.load("debug_x.pt")
    loss_fn(out, data)


if __name__ == "__main__":
    scripter.run()
