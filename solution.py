import numpy as np

from baseline import fit_fm


def _field_prior_adjustment(encoded, target_split, base):
    train_pack = encoded.get('train')
    target_pack = encoded.get(target_split)
    if train_pack is None or target_pack is None or len(train_pack) < 2:
        return base
    try:
        x_train = np.asarray(train_pack[0])
        y_train = np.asarray(train_pack[1], dtype=np.float64).reshape(-1)
        x_target = np.asarray(target_pack[0])
    except (TypeError, ValueError):
        return base
    if x_train.ndim != 2 or x_target.ndim != 2 or y_train.size != x_train.shape[0]:
        return base
    if x_target.shape[0] != base.size or x_train.shape[1] != x_target.shape[1]:
        return base
    if not np.all(np.isfinite(y_train)):
        return base
    rate = float(np.clip(np.mean(y_train), 1e-5, 1.0 - 1e-5))
    gl = np.log(rate / (1.0 - rate))
    adj = np.zeros(base.size, dtype=np.float64)
    used = 0
    for column in range(x_train.shape[1]):
        try:
            a = x_train[:, column]
            b = x_target[:, column]
            va = np.ones(a.size, dtype=bool)
            vb = np.ones(b.size, dtype=bool)
            if np.issubdtype(a.dtype, np.number):
                va &= np.isfinite(a)
                vb &= np.isfinite(b)
            keys, inv = np.unique(a[va], return_inverse=True)
            if keys.size == 0 or keys.size > 100000:
                continue
            cnt = np.bincount(inv, minlength=keys.size).astype(np.float64)
            pos = np.bincount(inv, weights=y_train[va], minlength=keys.size)
            loc = np.searchsorted(keys, b[vb])
            ok = loc < keys.size
            if keys.size:
                ok &= keys[np.minimum(loc, keys.size - 1)] == b[vb]
            rows = np.flatnonzero(vb)[ok]
            loc = loc[ok]
            posterior = (pos[loc] + 20.0 * rate) / (cnt[loc] + 20.0)
            adj[rows] += np.log(posterior / (1.0 - posterior)) - gl
            used += 1
        except (TypeError, ValueError, IndexError):
            continue
    return base if used == 0 else base + 0.35 * adj / float(used)


def _key(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return str(value)


def _number(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _value(row, name, index):
    if isinstance(row, dict):
        return row.get(name)
    if hasattr(row, '_asdict'):
        return row._asdict().get(name)
    try:
        return row[name]
    except (TypeError, KeyError, IndexError):
        try:
            return row[index]
        except (TypeError, IndexError):
            return None


def _watch_residual(data_access, target_split, n_target):
    columns = ('user_id', 'video_id', 'duration_ms', 'play_time_ms')
    train_file = 'log_standard_4_08_to_4_21_pure.csv'
    target_file = 'log_standard_4_08_to_4_21_pure.csv' if target_split == 'train' else 'log_standard_4_22_to_5_08_pure.csv'
    bounds = np.asarray([0.0, 10000.0, 20000.0, 40000.0, 80000.0, 120000.0, 180000.0, 260000.0, 400000.0, 700000.0, np.inf])
    sums = {}
    counts = {}
    bin_sum = np.zeros(10, dtype=np.float64)
    bin_count = np.zeros(10, dtype=np.float64)
    try:
        for row in data_access.iter_rows(train_file, columns, split='train'):
            duration = _number(_value(row, 'duration_ms', 2))
            play = _number(_value(row, 'play_time_ms', 3))
            video = _value(row, 'video_id', 1)
            if not np.isfinite(duration) or not np.isfinite(play) or duration <= 0.0 or video is None:
                continue
            bucket = int(np.searchsorted(bounds, duration, side='right') - 1)
            bucket = max(0, min(9, bucket))
            quality = float(np.clip(play / duration, 0.0, 1.5))
            key = _key(video)
            sums[key] = sums.get(key, 0.0) + quality
            counts[key] = counts.get(key, 0.0) + 1.0
            bin_sum[bucket] += quality
            bin_count[bucket] += 1.0
        global_mean = float(np.sum(bin_sum) / max(1.0, np.sum(bin_count)))
        bin_mean = (bin_sum + 30.0 * global_mean) / (bin_count + 30.0)
        item_residual = {}
        for key, total in sums.items():
            item_mean = (total + 30.0 * global_mean) / (counts[key] + 30.0)
            item_residual[key] = item_mean - global_mean
        output = np.zeros(n_target, dtype=np.float64)
        for index, row in enumerate(data_access.iter_rows(target_file, columns, split=target_split)):
            if index >= n_target:
                break
            duration = _number(_value(row, 'duration_ms', 2))
            video = _value(row, 'video_id', 1)
            if np.isfinite(duration) and duration > 0.0:
                bucket = int(np.searchsorted(bounds, duration, side='right') - 1)
                bucket = max(0, min(9, bucket))
                value = bin_mean[bucket] - global_mean
            else:
                value = 0.0
            if video is not None:
                value += item_residual.get(_key(video), 0.0)
            output[index] = value
        scale = float(np.std(output))
        if not np.isfinite(scale) or scale < 1e-6:
            return np.zeros(n_target, dtype=np.float64)
        return np.clip(output / scale, -3.0, 3.0)
    except Exception:
        return np.zeros(n_target, dtype=np.float64)


def _codes(matrix):
    array = np.asarray(matrix)
    if array.ndim != 2:
        return None
    if np.issubdtype(array.dtype, np.number):
        values = np.asarray(array, dtype=np.float64)
        values = np.nan_to_num(values, nan=-9.0e15, posinf=9.0e15, neginf=-9.0e15)
        return np.rint(values).astype(np.int64)
    result = np.empty(array.shape, dtype=np.int64)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            try:
                result[i, j] = int(float(array[i, j]))
            except (TypeError, ValueError):
                result[i, j] = hash(str(array[i, j]))
    return result


def _pairwise_delta(x_train, y_train, base_train, x_target, seed, config):
    x_train = np.asarray(x_train)
    x_target = np.asarray(x_target)
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
    base_train = np.asarray(base_train, dtype=np.float64).reshape(-1)
    if x_train.ndim != 2 or x_target.ndim != 2:
        return np.zeros(x_target.shape[0] if x_target.ndim else 0, dtype=np.float64)
    if x_train.shape[0] != y_train.size or x_train.shape[0] != base_train.size:
        return np.zeros(x_target.shape[0], dtype=np.float64)
    if x_train.shape[1] != x_target.shape[1] or x_train.shape[0] < 4:
        return np.zeros(x_target.shape[0], dtype=np.float64)
    codes_train = _codes(x_train)
    codes_target = _codes(x_target)
    if codes_train is None or codes_target is None:
        return np.zeros(x_target.shape[0], dtype=np.float64)
    n, width = codes_train.shape
    if width > 1:
        max_cols = max(1, int(config.get('pair_max_cols', 32)))
        active = list(range(1, min(width, max_cols + 1)))
    else:
        active = [0]
    if not active:
        return np.zeros(x_target.shape[0], dtype=np.float64)
    labels = np.isfinite(y_train) & (y_train > 0.5)
    users = codes_train[:, 0]
    order = np.argsort(users, kind='mergesort')
    sorted_users = users[order]
    starts = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    tables = [dict() for _ in active]
    rng = np.random.default_rng(int(seed) + 104729)
    epochs = max(1, int(config.get('pair_epochs', 3)))
    max_pairs = max(1, int(config.get('pair_max_pairs', 6)))
    learning_rate = float(config.get('pair_lr', 0.03))
    regularization = float(config.get('pair_reg', 0.001))
    groups = np.arange(starts.size)
    for _ in range(epochs):
        rng.shuffle(groups)
        for group in groups:
            ids = order[starts[group]:ends[group]]
            positives = ids[labels[ids]]
            negatives = ids[~labels[ids]]
            if positives.size == 0 or negatives.size == 0:
                continue
            count = min(max_pairs, positives.size, negatives.size)
            if positives.size > count:
                positives = rng.choice(positives, size=count, replace=False)
            if negatives.size > count:
                negatives = rng.choice(negatives, size=count, replace=False)
            for positive, negative in zip(positives, negatives):
                margin = float(base_train[positive] - base_train[negative])
                for table_index, column in enumerate(active):
                    pos_key = int(codes_train[positive, column])
                    neg_key = int(codes_train[negative, column])
                    if pos_key == neg_key:
                        continue
                    pos_value = tables[table_index].get(pos_key, 0.0)
                    neg_value = tables[table_index].get(neg_key, 0.0)
                    margin += pos_value - neg_value
                gradient = 1.0 / (1.0 + np.exp(np.clip(margin, -20.0, 20.0)))
                step = learning_rate * gradient
                for table_index, column in enumerate(active):
                    pos_key = int(codes_train[positive, column])
                    neg_key = int(codes_train[negative, column])
                    if pos_key == neg_key:
                        continue
                    table = tables[table_index]
                    pos_value = table.get(pos_key, 0.0)
                    neg_value = table.get(neg_key, 0.0)
                    table[pos_key] = pos_value + step - learning_rate * regularization * pos_value
                    table[neg_key] = neg_value - step - learning_rate * regularization * neg_value
    output = np.zeros(x_target.shape[0], dtype=np.float64)
    for table_index, column in enumerate(active):
        unique, inverse = np.unique(codes_target[:, column], return_inverse=True)
        values = np.asarray([tables[table_index].get(int(key), 0.0) for key in unique], dtype=np.float64)
        output += values[inverse]
    train_delta = np.zeros(n, dtype=np.float64)
    for table_index, column in enumerate(active):
        unique, inverse = np.unique(codes_train[:, column], return_inverse=True)
        values = np.asarray([tables[table_index].get(int(key), 0.0) for key in unique], dtype=np.float64)
        train_delta += values[inverse]
    scale = float(np.std(train_delta))
    if not np.isfinite(scale) or scale < 1e-6:
        return np.zeros(x_target.shape[0], dtype=np.float64)
    return np.clip(output / scale, -4.0, 4.0)


def score(splits, data_access, target_split: str, seed: int, config: dict) -> np.ndarray:
    params = {name: config[name] for name in ('k', 'lr', 'epochs', 'bs', 'patience') if name in config}
    params['seed'] = seed
    params.setdefault('verbose', False)
    model, encoded = fit_fm(splits, **params)
    x_train, y_train, _ = encoded['train']
    x_target, _, _ = encoded[target_split]
    base_train = np.asarray(model.predict(x_train), dtype=np.float64).reshape(-1)
    base_target = np.asarray(model.predict(x_target), dtype=np.float64).reshape(-1)
    base_train = np.nan_to_num(base_train, nan=0.0, posinf=30.0, neginf=-30.0)
    base_target = np.nan_to_num(base_target, nan=0.0, posinf=30.0, neginf=-30.0)
    residual = _field_prior_adjustment(encoded, target_split, base_target)
    watch = _watch_residual(data_access, target_split, base_target.size)
    pairwise = _pairwise_delta(x_train, y_train, base_train, x_target, seed, config)
    blend = float(config.get('watch_blend', 0.085))
    pair_blend = float(config.get('pair_blend', 0.8))
    result = residual + blend * watch + pair_blend * pairwise
    return np.nan_to_num(result, nan=0.0, posinf=30.0, neginf=-30.0)
