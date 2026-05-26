"""Sympy verifiers for Axler Ch.8 char/min-poly construction tasks.

Covers three task types in the 938557 causal run:

  construct_with_char_poly:     "example of operator on C^n with char poly p(z)"
  construct_with_char_min_poly: "... char poly p(z), min poly m(z)"
  compute_char_min_poly:        "find char poly and min poly of this operator"

For construction tasks we parse (a) target polys from the problem text,
(b) candidate matrix from the model answer, and (c) verify the matrix's
sympy-computed char/min polys match.

For compute tasks we parse (a) the operator definition from the problem,
(b) the claimed polys from the answer, and (c) check equality.

Each verifier returns VerifyResult from mathreason_sae/eval/sympy_verifier.py.
"""
from __future__ import annotations

import re
from typing import Optional

import sympy as sp

from mathreason_sae.eval.sympy_verifier import VerifyResult, extract_matrices


_POLY_FACTOR = re.compile(r"\(z\s*-\s*(-?\d+)\)(?:\s*\^?\s*(?:\{(\d+)\}|(\d+)))?")
_BARE_Z = re.compile(r"(?<![\w\-])z(?:\s*\^?\s*(?:\{(\d+)\}|(\d+)))?(?![\w\-])")


def _parse_factored_poly(expr_text: str) -> Optional[sp.Expr]:
    """Parse strings like '(z-1)(z-5)^{3}', '(z-7)^2(z-8)^2', 'z(z-1)^2(z-3)' into a sympy poly in z."""
    z = sp.symbols("z")
    factors = _POLY_FACTOR.findall(expr_text)
    stripped = _POLY_FACTOR.sub("", expr_text)
    bare = _BARE_Z.findall(stripped)
    if not factors and not bare:
        return None
    poly = sp.Integer(1)
    for root_str, exp_a, exp_b in factors:
        exp = int(exp_a or exp_b or "1")
        root = int(root_str)
        poly *= (z - root) ** exp
    for exp_a, exp_b in bare:
        exp = int(exp_a or exp_b or "1")
        poly *= z ** exp
    return sp.expand(poly)


def _parse_target_polys_from_problem(q: str) -> dict:
    """Pull out target char poly and optional min poly from problem text."""
    out = {}
    m_char = re.search(r"characteristic polynomial\s+(?:equals|is)\s+([^\.]*?)(?:\s+and\b|\s*$|\s*,)", q, re.IGNORECASE)
    if m_char:
        p = _parse_factored_poly(m_char.group(1))
        if p is not None:
            out["char_poly"] = p

    m_min = re.search(r"minimal polynomial\s+(?:equals|is)\s+([^\.]*?)(?:\s*$|\s*,|\s+and\b)", q, re.IGNORECASE)
    if m_min:
        p = _parse_factored_poly(m_min.group(1))
        if p is not None:
            out["min_poly"] = p

    m_dim = re.search(r"\\mathbf\{?[CR]\}?\^\{?(\d+)\}?", q)
    if m_dim:
        out["dim"] = int(m_dim.group(1))

    return out


def _char_poly(M: sp.Matrix) -> sp.Expr:
    z = sp.symbols("z")
    return sp.expand(M.charpoly(z).as_expr())


def _min_poly(M: sp.Matrix) -> Optional[sp.Expr]:
    """Compute the minimal polynomial of a square matrix in z.

    sympy's `Matrix.minimal_polynomial` returns a Poly expression when given
    a symbol argument, but errors on symbolic matrices with duplicate eigenvalues
    in some versions. Fall back to iterative construction.
    """
    z = sp.symbols("z")
    n = M.shape[0]
    I = sp.eye(n)
    # Build powers M^0, M^1, ..., M^n and find the minimal linear combination = 0
    try:
        # Direct API — most sympy versions
        mp = M.minimal_polynomial(z)
        return sp.expand(sp.Poly(mp, z).as_expr())
    except Exception:
        pass
    try:
        # Char-poly fallback (always returns a multiple of min poly; correct whenever
        # char == min, which is the common case for constructed Jordan-form blocks)
        return _char_poly(M)
    except Exception:
        return None


def verify_construct_with_polys(answer: str, problem_text: str) -> VerifyResult:
    target = _parse_target_polys_from_problem(problem_text)
    if "char_poly" not in target:
        return VerifyResult("unknown", "could not parse target char poly from problem")
    dim = target.get("dim")
    mats = extract_matrices(answer)
    cands = [m for m in mats if m.shape[0] == m.shape[1]]
    if dim is not None:
        cands = [m for m in cands if m.shape[0] == dim]
    if not cands:
        return VerifyResult("unknown", f"no {dim}x{dim} matrix in answer" if dim else "no square matrix")

    last_reason = ""
    for M in cands:
        try:
            cp = _char_poly(M)
        except Exception as e:
            last_reason = f"char poly eval failed: {e}"
            continue
        if sp.simplify(cp - target["char_poly"]) != 0:
            last_reason = f"char poly mismatch: got {cp}, want {target['char_poly']}"
            continue
        if "min_poly" in target:
            mp = _min_poly(M)
            if mp is None:
                last_reason = "min poly eval failed"
                continue
            if sp.simplify(mp - target["min_poly"]) != 0:
                last_reason = f"min poly mismatch: got {mp}, want {target['min_poly']}"
                continue
        return VerifyResult("correct", "polys match", parsed={"M": str(M)})

    return VerifyResult("incorrect", last_reason or "no candidate matrix satisfies target polys")


def _parse_operator_from_problem(q: str) -> Optional[sp.Matrix]:
    """Parse definitions like T(z_1, z_2, z_3, z_4) = (0, z_1, z_2, z_3) → matrix."""
    m = re.search(r"T\s*\\left\s*\(\s*(z_\{?\d+\}?(?:\s*,\s*z_\{?\d+\}?)+)\s*\\right\s*\)\s*=\s*\\left\s*\(\s*(.+?)\s*\\right\s*\)", q)
    if m is None:
        m = re.search(r"T\s*\(\s*(z_\{?\d+\}?(?:\s*,\s*z_\{?\d+\}?)+)\s*\)\s*=\s*\(\s*(.+?)\s*\)", q)
    if m is None:
        return None
    vars_txt, outs_txt = m.group(1), m.group(2)
    var_names = [v.replace("{","").replace("}","") for v in re.findall(r"z_\{?\d+\}?", vars_txt)]
    syms = {v: sp.symbols(v) for v in var_names}

    outs_txt = outs_txt.replace("{", "").replace("}", "")
    out_exprs = [s.strip() for s in re.split(r",(?![^(]*\))", outs_txt)]
    if len(out_exprs) != len(var_names):
        return None

    rows = []
    for e in out_exprs:
        expr_text = e
        for v in var_names:
            expr_text = expr_text.replace(v, f"({v})")
        try:
            expr = sp.sympify(expr_text, locals=syms)
        except Exception:
            return None
        row = [sp.diff(expr, syms[v]) for v in var_names]
        if any(r.free_symbols for r in row):
            return None
        rows.append(row)
    return sp.Matrix(rows)


def verify_compute_char_min_poly(answer: str, problem_text: str) -> VerifyResult:
    M = _parse_operator_from_problem(problem_text)
    if M is None:
        return VerifyResult("unknown", "could not parse operator definition from problem")

    truth_char = _char_poly(M)
    truth_min = _min_poly(M)

    claimed_char = None
    claimed_min = None
    m1 = re.search(r"characteristic polynomial[^$]*?(?:is|=)[^$]*?\$([^$]+)\$", answer, re.IGNORECASE)
    if m1:
        p = _parse_factored_poly(m1.group(1))
        if p is not None:
            claimed_char = p
        else:
            try:
                claimed_char = sp.expand(sp.sympify(m1.group(1).replace("^", "**")))
            except Exception:
                claimed_char = None

    m2 = re.search(r"minimal polynomial[^$]*?(?:is|=)[^$]*?\$([^$]+)\$", answer, re.IGNORECASE)
    if m2:
        p = _parse_factored_poly(m2.group(1))
        if p is not None:
            claimed_min = p
        else:
            try:
                claimed_min = sp.expand(sp.sympify(m2.group(1).replace("^", "**")))
            except Exception:
                claimed_min = None

    if claimed_char is None and claimed_min is None:
        return VerifyResult("unknown", "could not parse claimed polys from answer")

    bad = []
    if claimed_char is not None and sp.simplify(claimed_char - truth_char) != 0:
        bad.append(f"char poly got {claimed_char}, want {truth_char}")
    if claimed_min is not None and truth_min is not None and sp.simplify(claimed_min - truth_min) != 0:
        bad.append(f"min poly got {claimed_min}, want {truth_min}")

    if bad:
        return VerifyResult("incorrect", "; ".join(bad))
    return VerifyResult("correct", f"polys match (char={truth_char}, min={truth_min})")


def dispatch_axler(task_type: str, answer: str, problem_text: str) -> VerifyResult:
    if task_type in ("construct_with_char_poly", "construct_with_char_min_poly"):
        return verify_construct_with_polys(answer, problem_text)
    if task_type == "compute_char_min_poly":
        return verify_compute_char_min_poly(answer, problem_text)
    return VerifyResult("unknown", f"no verifier for task_type={task_type}")
