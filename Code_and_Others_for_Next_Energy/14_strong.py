"""Step 14. Stronger estimators under the same leave-one-domain-out protocol (tests whether
the conclusions depend on estimator strength). Four additional configurations, all scored with the same certificate:
  gbrtuned_self : gradient boosting, hyperparameters chosen inside every outer fold by grouped
                  (domain-wise) 3-fold cross-validation on the 12 source domains (16-point grid).
  gbrfull_self  : the main GBR configuration trained on every cycle (stride 1) instead of every 6th.
  cnn_self      : causal 1D convolutional sequence model over a window of the last W cycles of
                  self-referenced features; W and learning rate chosen on two held-out source domains.
  pinn          : JAX port of the physics-informed network of Wang et al. (2024, PINN4SOH: Solution_u
                  encoder/predictor, dynamical_F network, data + PDE + monotonic physics losses, same
                  widths, sine activations, dropout and loss weights). Deployable preprocessing: z-score
                  of the 16 features with source statistics and cycle index / 1000.
  pinnpub       : the same network with the published per-cell whole-life min-max scaling of features and cycle
                  index; it uses future cycles of the test cell and is reported only as a non-deployable reference.
Usage: python 14_strong.py [gbr|cnn|pinn|pinnpub|all]"""
import sys, time, warnings, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from common import load_cells, cell_rows, OUT, SEED, QUICK
warnings.filterwarnings('ignore')
D, F = load_cells(); XS = ['d_' + c for c in F]; DOMS = sorted(D.dom.unique())
if QUICK: DOMS = DOMS[:3]
WHAT = sys.argv[1] if len(sys.argv) > 1 else 'all'


def save(cfg, res):
    pd.concat([r[0] for r in res]).to_csv(f'{OUT}/pred_{cfg}.csv.gz', index=False, float_format='%.6f')
    C = pd.DataFrame([x for r in res for x in r[1]]); C.to_csv(f'{OUT}/cells_{cfg}.csv', index=False)
    print(f'{cfg}: median RMSE {C.rmse.median():.4f} after projection {C.rmse_q.median():.4f} '
          f'LB<=RMSE {np.mean(C.LB <= C.rmse + 1e-12):.3f}', flush=True)


def zs(tr, te, X):
    mu, sd = tr[X].mean(), tr[X].std() + 1e-12
    return ((tr[X] - mu) / sd).values.astype('float32'), ((te[X] - mu) / sd).values.astype('float32')


# ---------------------------------------------------------------- tuned and full-data GBR
GRID = [dict(learning_rate=lr, max_leaf_nodes=ml, max_iter=mi, l2_regularization=l2)
        for lr in (0.05, 0.1) for ml in (15, 63) for mi in (200, 600) for l2 in (0.0, 1.0)]


def gbr_tuned(tgt):
    tr = D[D.dom != tgt]; te = D[D.dom == tgt]
    inner = tr.iloc[::12]; Zi, _ = zs(inner, inner, XS)
    cv = []
    for h in GRID:
        e = []
        for a, b in GroupKFold(3).split(Zi, groups=inner.dom):
            m = HistGradientBoostingRegressor(random_state=SEED, **h).fit(Zi[a], inner.soh.values[a])
            e.append(np.sqrt(np.mean((m.predict(Zi[b]) - inner.soh.values[b]) ** 2)))
        cv.append(np.mean(e))
    best = GRID[int(np.argmin(cv))]
    tr6 = tr.iloc[::6]; Ztr, Zte = zs(tr6, te, XS)
    p = HistGradientBoostingRegressor(random_state=SEED, **best).fit(Ztr, tr6.soh.values).predict(Zte)
    te = te[['cell', 'dataset', 'k', 'soh']].assign(p=p)
    return te[['cell', 'k', 'soh', 'p']], [dict(r, **{f'hp_{k}': v for k, v in best.items()}) for r in cell_rows(te, tgt)]


def gbr_full(tgt):
    tr = D[D.dom != tgt]; te = D[D.dom == tgt]; Ztr, Zte = zs(tr, te, XS)
    p = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=SEED).fit(Ztr, tr.soh.values).predict(Zte)
    te = te[['cell', 'dataset', 'k', 'soh']].assign(p=p)
    return te[['cell', 'k', 'soh', 'p']], cell_rows(te, tgt)


# ---------------------------------------------------------------- JAX models
def jax_imports():
    global jax, jnp, optax
    import jax, jax.numpy as jnp, optax
    jax.config.update('jax_platform_name', 'cpu')


def dense_init(key, sizes):
    ps = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        k = jax.random.fold_in(key, i)
        ps.append((jax.random.normal(k, (a, b)) * np.sqrt(2.0 / (a + b)), jnp.zeros(b)))  # Xavier normal
    return ps


def windows(Z, cells, W):
    """Causal windows: for cycle k of a cell, rows max(0, k-W+1)..k, left-padded with the first cycle."""
    out = np.empty((len(Z), W, Z.shape[1]), 'float32'); start = 0
    for n in pd.Series(cells).groupby(cells, sort=False).size().values:
        z = Z[start:start + n]; pad = np.concatenate([np.repeat(z[:1], W - 1, 0), z])
        idx = np.arange(n)[:, None] + np.arange(W)[None, :]
        out[start:start + n] = pad[idx]; start += n
    return out


def cnn_fns():
    def init(key, d):
        k1, k2, k3 = jax.random.split(key, 3)
        return dict(c1=(jax.random.normal(k1, (5, d, 32)) * np.sqrt(2 / (5 * d)), jnp.zeros(32)),
                    c2=(jax.random.normal(k2, (5, 32, 32)) * np.sqrt(2 / (5 * 32)), jnp.zeros(32)),
                    out=dense_init(k3, [64, 32, 1]))

    def apply(P, x):  # x: (B, W, d)
        conv = lambda h, wb: jax.nn.relu(jax.lax.conv_general_dilated(h, wb[0], (1,), 'SAME', dimension_numbers=('NWC', 'WIO', 'NWC')) + wb[1])
        h = conv(conv(x, P['c1']), P['c2'])
        h = jnp.concatenate([h[:, -1, :], h.mean(1)], -1)  # last-step state and window mean
        (w1, b1), (w2, b2) = P['out']
        return (jax.nn.relu(h @ w1 + b1) @ w2 + b2)[:, 0]
    return init, apply


def train_nn(init, apply, Xtr, ytr, Xva, yva, lr, key, epochs=60, patience=8, bs=256):
    P = init(key, Xtr.shape[-1]); opt = optax.adam(lr); S = opt.init(P)
    loss = lambda P, x, y: jnp.mean((apply(P, x) - y) ** 2)

    @jax.jit
    def step(P, S, x, y):
        l, g = jax.value_and_grad(loss)(P, x, y); u, S = opt.update(g, S, P); return optax.apply_updates(P, u), S, l
    pred = jax.jit(apply)
    best, bestP, wait, rng = np.inf, P, 0, np.random.default_rng(SEED)
    for ep in range(epochs):
        perm = rng.permutation(len(Xtr))
        for i in range(0, len(perm) - bs + 1, bs):
            j = perm[i:i + bs]; P, S, _ = step(P, S, Xtr[j], ytr[j])
        v = float(np.sqrt(np.mean((np.concatenate([np.asarray(pred(P, Xva[i:i + 4096])) for i in range(0, len(Xva), 4096)]) - yva) ** 2)))
        if v < best - 1e-5: best, bestP, wait = v, P, 0
        else:
            wait += 1
            if wait >= patience: break
    return bestP, best, ep + 1


def predict_batched(apply, P, X):
    f = jax.jit(apply); return np.concatenate([np.asarray(f(P, X[i:i + 4096])) for i in range(0, len(X), 4096)])


def cnn_fold(tgt):
    init, apply = cnn_fns(); key = jax.random.PRNGKey(SEED)
    src = [d for d in DOMS if d != tgt] if not QUICK else [d for d in sorted(D.dom.unique()) if d != tgt]
    va_doms = list(np.random.default_rng(SEED + DOMS.index(tgt)).choice(src, 2, replace=False))
    tr = D[D.dom != tgt]; te = D[D.dom == tgt]
    mu, sd = tr[XS].mean(), tr[XS].std() + 1e-12
    Z = lambda d: ((d[XS] - mu) / sd).values.astype('float32')
    best = None
    for W in (10, 30):
        Xall = windows(Z(tr), tr.cell.values, W); yall = tr.soh.values.astype('float32')
        isva = tr.dom.isin(va_doms).values; sub = np.zeros(len(tr), bool); sub[::6] = True
        for lr in (1e-3, 3e-3):
            P, v, ep = train_nn(init, apply, Xall[sub & ~isva], yall[sub & ~isva], Xall[isva][::3], yall[isva][::3], lr, key)
            if best is None or v < best[0]: best = (v, W, lr, ep)
    v, W, lr, _ = best
    # refit on all source domains with the chosen W and lr; early stopping on the same validation split
    Xall = windows(Z(tr), tr.cell.values, W); yall = tr.soh.values.astype('float32')
    isva = tr.dom.isin(va_doms).values; sub = np.zeros(len(tr), bool); sub[::6] = True
    P, _, ep = train_nn(init, apply, Xall[sub & ~isva], yall[sub & ~isva], Xall[isva][::3], yall[isva][::3], lr, key)
    p = predict_batched(apply, P, windows(Z(te), te.cell.values, W))
    te = te[['cell', 'dataset', 'k', 'soh']].assign(p=p)
    return te[['cell', 'k', 'soh', 'p']], [dict(r, hp_W=W, hp_lr=lr, epochs=ep) for r in cell_rows(te, tgt)]


def pinn_fold(tgt, epochs=120, warm=15, patience=20, bs=256, norm='causal'):
    """JAX port of PINN4SOH (Wang et al., 2024). Architecture and loss as in Model/Model.py:
    u = Solution_u([x, t]) with encoder MLP(17-60-60-32, sine, dropout 0.2) and predictor (dropout 0.2, 32-32 sine, 32-1);
    F = MLP(35-60-60-1, sine, dropout 0.2) on [x, t, u, u_x, u_t]; f = u_t - F;
    loss = MSE(u, y) + 0.7 MSE(f, 0) + 0.2 sum relu((u2 - u1)(y1 - y2)) over consecutive-cycle pairs;
    Adam, warm-up then cosine learning rate for Solution_u (0.002 -> 0.01 -> 0.0002), lr_F = 0.001,
    early stopping on a random 20% of source cells. Epochs shortened from 200 to 120 for CPU time."""
    key = jax.random.PRNGKey(SEED)
    ku, kp, kf = jax.random.split(key, 3)
    P = dict(enc=dense_init(ku, [17, 60, 60, 32]), pre=dense_init(kp, [32, 32, 1]), F=dense_init(kf, [35, 60, 60, 1]))
    tr = D[D.dom != tgt]; te = D[D.dom == tgt]
    mu, sd = tr[F].mean(), tr[F].std() + 1e-12
    def X(d):
        if norm == 'causal':
            return np.column_stack([((d[F] - mu) / sd).values, d.k.values / 1000.0]).astype('float32')
        # published PINN4SOH preprocessing: per-cell min-max to [-1, 1] over the whole life (features and cycle index);
        # uses the test cell's future cycles, so it is reported only as a non-deployable reference
        Z = d[F + ['k']].astype(float); g = Z.groupby(d.cell.values, sort=False)
        Z = 2 * (Z - g.transform('min')) / (g.transform('max') - g.transform('min')).replace(0, np.nan) - 1
        return Z.fillna(0.0).values.astype('float32')
    rng = np.random.default_rng(SEED + 7)
    cells = tr.cell.unique(); va_cells = set(rng.choice(cells, int(0.2 * len(cells)), replace=False))
    isva = tr.cell.isin(va_cells).values
    Xtr, ytr = X(tr), tr.soh.values.astype('float32')
    nxt = np.r_[tr.cell.values[1:] == tr.cell.values[:-1], False]  # pair (k, k+1) within the same cell
    i1 = np.where(nxt & ~isva)[0][::6]
    Xva, yva = Xtr[isva][::3], ytr[isva][::3]

    def mlp(ps, h, keys, drop, rate=0.2, hidden_dropout=True):
        for i, (w, b) in enumerate(ps):
            h = h @ w + b
            if i < len(ps) - 1:
                h = jnp.sin(h)
                if drop and hidden_dropout and i == len(ps) - 2:
                    h = h * jax.random.bernoulli(keys[i], 1 - rate, h.shape) / (1 - rate)
        return h

    def u_single(P, x, t, key, drop):
        k = jax.random.split(key, 4)
        e = mlp(P['enc'], jnp.concatenate([x, t[None]]), k, drop)
        if drop: e = e * jax.random.bernoulli(k[3], 0.8, e.shape) / 0.8
        return mlp(P['pre'], e, k[2:], drop, hidden_dropout=False)[0]

    def forward(P, xt, key, drop=True):
        ks = jax.random.split(key, xt.shape[0] + 1)
        def one(x, t, k):
            u, (ux, ut) = jax.value_and_grad(u_single, argnums=(1, 2))(P, x, t, k, drop)
            return u, ux, ut
        u, ux, ut = jax.vmap(one)(xt[:, :-1], xt[:, -1], ks[1:])
        kF = jax.random.split(ks[0], 3)
        Fv = mlp(P['F'], jnp.concatenate([xt, u[:, None], ux, ut[:, None]], 1), kF, drop)[:, 0]
        return u, ut - Fv

    def loss(P, x1, x2, y1, y2, key):
        k1, k2 = jax.random.split(key)
        u1, f1 = forward(P, x1, k1); u2, f2 = forward(P, x2, k2)
        l1 = 0.5 * jnp.mean((u1 - y1) ** 2) + 0.5 * jnp.mean((u2 - y2) ** 2)
        l2 = 0.5 * jnp.mean(f1 ** 2) + 0.5 * jnp.mean(f2 ** 2)
        l3 = jnp.sum(jax.nn.relu((u2 - u1) * (y1 - y2)))
        return l1 + 0.7 * l2 + 0.2 * l3

    sched = np.concatenate([np.linspace(0.002, 0.01, warm), 0.0002 + 0.5 * (0.01 - 0.0002) * (1 + np.cos(np.pi * np.arange(epochs - warm) / (epochs - warm)))])
    optU = optax.inject_hyperparams(optax.adam)(learning_rate=0.002); optF = optax.adam(0.001)
    U = dict(enc=P['enc'], pre=P['pre']); SU, SF = optU.init(U), optF.init(P['F'])

    @jax.jit
    def step(U, Fp, SU, SF, x1, x2, y1, y2, key, lr):
        g = jax.grad(loss)(dict(U, F=Fp), x1, x2, y1, y2, key)
        SU.hyperparams['learning_rate'] = lr
        uu, SU = optU.update(dict(enc=g['enc'], pre=g['pre']), SU, U); U = optax.apply_updates(U, uu)
        uf, SF = optF.update(g['F'], SF, Fp); Fp = optax.apply_updates(Fp, uf)
        return U, Fp, SU, SF

    pred_u = jax.jit(jax.vmap(lambda Pp, x: u_single(Pp, x[:-1], x[-1], jax.random.PRNGKey(0), False), in_axes=(None, 0)))
    pv = lambda Pp, Xm: np.concatenate([np.asarray(pred_u(Pp, Xm[i:i + 8192])) for i in range(0, len(Xm), 8192)])
    Fp = P['F']; best, bestU, wait, rngb = np.inf, U, 0, np.random.default_rng(SEED)
    for ep in range(epochs):
        perm = rngb.permutation(i1)
        for b in range(0, len(perm) - bs + 1, bs):
            j = perm[b:b + bs]
            U, Fp, SU, SF = step(U, Fp, SU, SF, Xtr[j], Xtr[j + 1], ytr[j], ytr[j + 1], jax.random.PRNGKey(ep * 100000 + b), float(sched[ep]))
        v = float(np.mean((pv(U, Xva) - yva) ** 2))
        if v < best: best, bestU, wait = v, U, 0
        else:
            wait += 1
            if wait > patience: break
    p = pv(bestU, X(te))
    te = te[['cell', 'dataset', 'k', 'soh']].assign(p=p)
    return te[['cell', 'k', 'soh', 'p']], [dict(r, epochs=ep + 1) for r in cell_rows(te, tgt)]


if __name__ == '__main__':
    t0 = time.time()
    if WHAT in ('gbr', 'all'):
        save('gbrtuned_self', Parallel(n_jobs=2)(delayed(gbr_tuned)(t) for t in DOMS)); print('t', time.time() - t0, flush=True)
        save('gbrfull_self', Parallel(n_jobs=2)(delayed(gbr_full)(t) for t in DOMS)); print('t', time.time() - t0, flush=True)
    if WHAT in ('cnn', 'all'):
        jax_imports(); res = []
        for t in DOMS: res.append(cnn_fold(t)); print(t, res[-1][1][0].get('hp_W'), round(time.time() - t0), flush=True)
        save('cnn_self', res)
    if WHAT in ('pinn', 'all'):
        jax_imports(); res = []
        for t in DOMS: res.append(pinn_fold(t)); print(t, res[-1][1][0].get('epochs'), round(time.time() - t0), flush=True)
        save('pinn', res)
    if WHAT in ('pinnpub', 'all'):
        jax_imports(); res = []
        for t in DOMS: res.append(pinn_fold(t, norm='percell')); print(t, res[-1][1][0].get('epochs'), round(time.time() - t0), flush=True)
        save('pinnpub', res)
