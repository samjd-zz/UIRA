import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from typing import Optional, Sequence


class DeviceSelector:
    def select(self, requested: str = "auto") -> torch.device:
        if requested == "auto":
            xpu = getattr(torch, "xpu", None)
            if xpu is not None and xpu.is_available():
                return torch.device("xpu")
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")

        device = torch.device(requested)
        if device.type == "xpu":
            xpu = getattr(torch, "xpu", None)
            if xpu is None or not xpu.is_available():
                raise RuntimeError("Intel XPU was requested, but no XPU device is available.")
        elif device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available.")
        elif device.type not in {"cpu", "cuda", "xpu"}:
            raise ValueError(f"Unsupported device type: {device.type}")
        return device


def select_device(requested: str = "auto") -> torch.device:
    return DeviceSelector().select(requested)

class AdamicIsomorphicLoss(nn.Module):
    """Enforces metric preservation between environment distance d_X and latent distance d_Z."""
    def __init__(self, c_scale: float = 1.0):
        super().__init__()
        self.c_scale = c_scale

    def forward(self, z_representations: torch.Tensor, d_x_grounding: torch.Tensor) -> torch.Tensor:
        # Compute pairwise Euclidean distances in latent space Z -> d_Z(Phi(x_i), Phi(x_j))
        d_z = torch.cdist(z_representations, z_representations, p=2)
        
        # L_iso = E [ | d_Z - c * d_X |^2 ]
        loss_iso = F.mse_loss(d_z, self.c_scale * d_x_grounding)
        return loss_iso


class YetzirahPermutationEngine(nn.Module):
    """Compiles representations via tensor-product combinations of atomic root embeddings."""
    def __init__(self, num_atomic_roots: int = 32, embed_dim: int = 512, root_combination_size: int = 2):
        super().__init__()
        if type(root_combination_size) is not int or root_combination_size != 2:
            raise ValueError("Only root_combination_size=2 is supported.")
        if type(num_atomic_roots) is not int or num_atomic_roots <= 0:
            raise ValueError("num_atomic_roots must be a positive integer.")
        if type(embed_dim) is not int or embed_dim <= 0:
            raise ValueError("embed_dim must be a positive integer.")
        self.embed_dim = embed_dim
        self.root_combination_size = root_combination_size
        self.atomic_roots = nn.Parameter(torch.randn(num_atomic_roots, embed_dim))
        
        # Project tensor products back down to hidden dimension
        self.permutation_projection = nn.Linear(embed_dim ** root_combination_size, embed_dim)

    def forward(self, root_tuples: torch.Tensor) -> torch.Tensor:
        # root_tuples shape: [batch_size, seq_len, root_combination_size]
        if (
            root_tuples.ndim != 3 or root_tuples.shape[-1] != 2
            or any(size == 0 for size in root_tuples.shape)
        ):
            raise ValueError("root_tuples must have nonempty shape [batch, sequence, 2].")
        if root_tuples.dtype not in (torch.int32, torch.int64):
            raise TypeError("root_tuples must have int32 or int64 dtype.")
        if root_tuples.device != self.atomic_roots.device:
            raise ValueError("root_tuples and model parameters must be on the same device.")
        if torch.any(root_tuples < 0) or torch.any(root_tuples >= self.atomic_roots.size(0)):
            raise ValueError("root_tuples contain an index outside the root vocabulary.")
        batch_size, seq_len, k = root_tuples.shape
        
        # Retrieve root vectors: [batch_size, seq_len, k, embed_dim]
        selected_roots = self.atomic_roots[root_tuples]
        
        # Calculate Tensor Outer Product across the k root dimensions
        # For k=2: e_v1 (tensor_prod) e_v2 -> shape [batch, seq, embed_dim * embed_dim]
        r1 = selected_roots[:, :, 0, :]
        r2 = selected_roots[:, :, 1, :]
        tensor_prod = torch.bmm(
            r1.reshape(-1, self.embed_dim, 1),
            r2.reshape(-1, 1, self.embed_dim)
        ).reshape(batch_size, seq_len, -1)
        
        # Project hypergraph permutation back to standard latent dimension
        h_concept = self.permutation_projection(tensor_prod)
        return h_concept


class BabelMultiViewPartitioner(nn.Module):
    """Projects continuous latent space into K disentangled cognitive sub-manifolds."""
    def __init__(self, embed_dim: int = 512, num_projections: int = 4):
        super().__init__()
        self.num_projections = num_projections
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU()
            ) for _ in range(num_projections)
        ])

    def forward(self, z: torch.Tensor) -> list[torch.Tensor]:
        return [proj(z) for proj in self.projections]


class UIRAFullArchitecture(nn.Module):
    """Complete Universal Isomorphic Representation Architecture (UIRA)."""
    def __init__(
        self, 
        num_atomic_roots: int = 32, 
        embed_dim: int = 512, 
        num_projections: int = 4,
        vocab_size: int = 1000,
        lambda_iso: float = 0.1,
        lambda_consensus: float = 0.1
    ):
        super().__init__()
        self.lambda_iso = lambda_iso
        self.lambda_consensus = lambda_consensus
        
        # Core Components
        self.yetzirah_engine = YetzirahPermutationEngine(num_atomic_roots, embed_dim)
        self.isomorphic_loss_fn = AdamicIsomorphicLoss(c_scale=1.0)
        self.babel_partitioner = BabelMultiViewPartitioner(embed_dim, num_projections)
        
        # Reconciliation Gate
        self.consensus_gate = nn.Linear(embed_dim * num_projections, embed_dim)
        
        # Downstream Task Head (e.g., Autoregressive Token Decoder)
        self.task_head = nn.Linear(embed_dim, vocab_size)

    def forward(
        self, 
        root_tuples: torch.Tensor, 
        env_distance_matrix: Optional[torch.Tensor] = None,
        target_tokens: Optional[torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        
        # 1. Permutation Synthesis (Yetzirah Engine)
        z_adamic = self.yetzirah_engine(root_tuples)  # [batch, seq, embed_dim]
        
        # 2. Compute Isomorphic Metric Loss (Adamic Layer)
        loss_iso = z_adamic.new_zeros(())
        if env_distance_matrix is not None:
            # Flatten spatial dims to evaluate pairwise distance on pooled sequence vectors
            z_pooled = z_adamic.mean(dim=1)
            loss_iso = self.isomorphic_loss_fn(z_pooled, env_distance_matrix)
            
        # 3. Disentangled Sub-Manifold Projections (Babel Partition)
        views = self.babel_partitioner(z_adamic)  # K tensors of shape [batch, seq, embed_dim]
        
        # 4. Cross-Manifold Consensus Invariance Loss
        loss_consensus = z_adamic.new_zeros(())
        num_views = len(views)
        for i in range(num_views):
            for j in range(i + 1, num_views):
                loss_consensus += F.mse_loss(views[i], views[j])
                
        # 5. Linear multi-view reconciliation, not factual verification
        concat_views = torch.cat(views, dim=-1)
        z_reconciled = self.consensus_gate(concat_views)
        
        # 6. Per-position token prediction
        logits = self.task_head(z_reconciled)
        loss_task = z_adamic.new_zeros(())
        if target_tokens is not None:
            loss_task = F.cross_entropy(logits.view(-1, logits.size(-1)), target_tokens.view(-1))
            
        # Total Objective Function: L_UIRA = L_task + λ1 * L_iso + λ2 * L_consensus
        total_loss = loss_task + (self.lambda_iso * loss_iso) + (self.lambda_consensus * loss_consensus)
        
        return {
            "logits": logits,
            "z_final": z_reconciled,
            "total_loss": total_loss,
            "loss_task": loss_task,
            "loss_iso": loss_iso,
            "loss_consensus": loss_consensus
        }


class SyntheticTrainingExample:
    def __init__(self, device_selector: Optional[DeviceSelector] = None) -> None:
        self.device_selector = device_selector if device_selector is not None else DeviceSelector()

    def run(self, argv: Optional[Sequence[str]] = None) -> dict[str, torch.Tensor]:
        parser = argparse.ArgumentParser(description="Run a UIRA synthetic training step.")
        parser.add_argument(
            "--device", default="auto", choices=("auto", "cpu", "cuda", "xpu"),
            help="Execution device; auto prefers Intel XPU, then CUDA, then CPU.",
        )
        device = self.device_selector.select(parser.parse_args(argv).device)
        batch_size, seq_len, num_roots, vocab_size = 4, 8, 16, 250
        model = UIRAFullArchitecture(
            num_atomic_roots=num_roots, embed_dim=128,
            num_projections=4, vocab_size=vocab_size,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        root_tuples = torch.randint(0, num_roots, (batch_size, seq_len, 2), device=device)
        distances = torch.rand(batch_size, batch_size, device=device)
        distances = (distances + distances.T) / 2.0
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        optimizer.zero_grad()
        outputs = model(root_tuples, distances, targets)
        outputs["total_loss"].backward()
        optimizer.step()
        print("--- UIRA Pipeline Execution Successful ---")
        print(f"Device:           {device}")
        print(f"Total Loss:      {outputs['total_loss'].item():.4f}")
        print(f"  ├─ Task Loss:      {outputs['loss_task'].item():.4f}")
        print(f"  ├─ Iso Loss:       {outputs['loss_iso'].item():.4f}")
        print(f"  └─ Consensus Loss: {outputs['loss_consensus'].item():.4f}")
        print(f"Output Logits Shape: {outputs['logits'].shape}")
        return outputs


if __name__ == "__main__":
    SyntheticTrainingExample().run()
