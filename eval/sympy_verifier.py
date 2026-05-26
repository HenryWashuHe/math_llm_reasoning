"""Programmatic verification for construction-style linear algebra problems.

Many V4-style tasks (and a large fraction of the 4K corpus) are answerable
by exhibiting a concrete matrix / vector / polynomial that satisfies a
property. For those, LLM-as-judge is expensive and noisy — sympy can check
the answer directly with 0.0 $ and perfect consistency.

Coverage plan:
  - construct_rank_k_generalized_vector: verify (T-λI)^k v = 0, (T-λI)^(k-1) v ≠ 0
  - kernel_power_two_larger:             verify ker((T+aI)^2) ≠ ker(T+aI)
  - char_min_poly_compute:               parse polys from answer, compare to expected
  - nilpotent_construct:                 verify T^k = 0
  - eigenvalue_find:                     compare solved vs expected eigenvalues

Each verifier returns a VerifyResult with label ∈ {"correct", "incorrect",
"unknown"}. "unknown" means the parser couldn't pull a usable answer — fall
back to gpt-4o.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Optional

import sympy as sp


@dataclass
class VerifyResult:
    label: str
    reason: str
    parsed: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------- matrix parsing ----------

_MATRIX_ENV_RE = re.compile(
    r"\\begin\{(?:pmatrix|bmatrix|matrix)\}(.*?)\\end\{(?:pmatrix|bmatrix|matrix)\}",
    re.DOTALL,
)


def _parse_latex_matrix(body: str) -> Optional[sp.Matrix]:
    rows_raw = [r for r in body.split(r"\\") if r.strip()]
    rows: list[list[sp.Expr]] = []
    for r in rows_raw:
        cells_raw = [c.strip() for c in r.split("&")]
        cells: list[sp.Expr] = []
        for c in cells_raw:
            c = c.replace(r"\,", "").replace(r"\;", "").strip()
            if not c:
                return None
            try:
                cells.append(sp.sympify(c))
            except Exception:
                return None
        rows.append(cells)
    if not rows or any(len(r) != len(rows[0]) for r in rows):
        return None
    return sp.Matrix(rows)


def extract_matrices(text: str) -> list[sp.Matrix]:
    out: list[sp.Matrix] = []
    for m in _MATRIX_ENV_RE.finditer(text):
        mat = _parse_latex_matrix(m.group(1))
        if mat is not None:
            out.append(mat)
    return out


# ---------- verifiers ----------

def verify_rank_k_generalized_vector(
    answer: str,
    lam: sp.Expr,
    k: int,
    dim: int = 2,
) -> VerifyResult:
    """Verify the candidate answer constructs T (dim×dim) and v such that
    (T - λI)^k v = 0 and (T - λI)^(k-1) v ≠ 0.
    """
    mats = extract_matrices(answer)
    if len(mats) < 2:
        return VerifyResult("unknown", f"need T and v; found {len(mats)} matrices")

    T = None
    v = None
    for m in mats:
        if m.shape == (dim, dim) and T is None:
            T = m
        elif m.shape == (dim, 1) and v is None:
            v = m
        elif m.shape == (1, dim) and v is None:
            v = m.T
    if T is None or v is None:
        return VerifyResult("unknown", "could not identify T (dim×dim) and v (dim×1)")

    try:
        M = T - lam * sp.eye(dim)
        Mk = M ** k
        Mkm1 = M ** (k - 1)
        vk = sp.simplify(Mk * v)
        vkm1 = sp.simplify(Mkm1 * v)
    except Exception as e:
        return VerifyResult("unknown", f"sympy eval failed: {e}")

    if vk != sp.zeros(*vk.shape):
        return VerifyResult("incorrect", f"(T-λI)^{k} v ≠ 0: {vk.tolist()}",
                            parsed={"T": str(T), "v": str(v)})
    if vkm1 == sp.zeros(*vkm1.shape):
        return VerifyResult("incorrect", f"(T-λI)^{k-1} v = 0 (v is a rank-{k-1} vector, not rank-{k})",
                            parsed={"T": str(T), "v": str(v)})
    return VerifyResult("correct", "passed both conditions",
                        parsed={"T": str(T), "v": str(v)})


def verify_nilpotent(answer: str, k: int, dim: Optional[int] = None) -> VerifyResult:
    """Verify candidate T satisfies T^k = 0 and T^(k-1) ≠ 0 (strictly nilpotent of index k)."""
    mats = extract_matrices(answer)
    cands = [m for m in mats if m.shape[0] == m.shape[1]]
    if dim is not None:
        cands = [m for m in cands if m.shape[0] == dim]
    if not cands:
        return VerifyResult("unknown", "no square matrix found in answer")

    for T in cands:
        try:
            Tk = sp.simplify(T ** k)
            Tkm1 = sp.simplify(T ** (k - 1)) if k >= 1 else None
        except Exception:
            continue
        if Tk != sp.zeros(*Tk.shape):
            continue
        if k >= 2 and Tkm1 == sp.zeros(*Tkm1.shape):
            return VerifyResult("incorrect",
                                f"T^{k-1} also = 0; not strictly index-{k} nilpotent",
                                parsed={"T": str(T)})
        return VerifyResult("correct", f"T^{k}=0 as required", parsed={"T": str(T)})

    return VerifyResult("incorrect", f"no candidate matrix satisfies T^{k}=0")


def verify_kernel_power_two_larger(answer: str, a: sp.Expr) -> VerifyResult:
    """Verify candidate T (2x2) proves ker((T+aI)^2) ⊋ ker(T+aI) as counterexample."""
    mats = extract_matrices(answer)
    cands = [m for m in mats if m.shape == (2, 2)]
    if not cands:
        return VerifyResult("unknown", "no 2×2 matrix in answer")

    for T in cands:
        try:
            M = T + a * sp.eye(2)
            M2 = sp.simplify(M ** 2)
            k1 = M.nullspace()
            k2 = M2.nullspace()
        except Exception:
            continue
        d1 = len(k1)
        d2 = len(k2)
        if d2 > d1:
            return VerifyResult("correct",
                                f"dim ker(M)={d1} < dim ker(M^2)={d2}",
                                parsed={"T": str(T)})

    return VerifyResult("incorrect", "no candidate yields dim ker(M^2) > dim ker(M)")


# ---------- router hook ----------

def dispatch(task_family: str, answer: str, problem_meta: dict) -> VerifyResult:
    """Route (task_family, problem_meta) to the right verifier.

    problem_meta carries parameters extracted from the problem text that the
    verifier needs (e.g., eigenvalue λ, nilpotent index k). Populated by the
    dataset-builder stage, not inferred here.
    """
    if task_family == "generalized_eigenvector_construction":
        lam = sp.sympify(problem_meta["lambda"])
        k = int(problem_meta.get("power", 2))
        dim = int(problem_meta.get("dim", 2))
        return verify_rank_k_generalized_vector(answer, lam, k, dim)

    if task_family == "nilpotent_construction":
        k = int(problem_meta["k"])
        dim = problem_meta.get("dim")
        return verify_nilpotent(answer, k, dim if dim is None else int(dim))

    if task_family == "kernel_power_two_larger":
        a = sp.sympify(problem_meta["a"])
        return verify_kernel_power_two_larger(answer, a)

    return VerifyResult("unknown", f"no verifier registered for task_family={task_family}")
